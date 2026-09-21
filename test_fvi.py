"""
test_fvi.py -- Experimental comparison of a candidate vegetation index (FVI)
against the current production VARI + NDWI pipeline.

FVI = (NIR + Green - Red - Blue) / (NIR + Green + Red + Blue)

Rationale: VARI is RGB-only and structurally cannot see that water absorbs
near-infrared strongly, which is why the production pipeline needs a
SEPARATE NDWI water mask bolted on. FVI folds NIR directly into a
VARI-style normalized ratio, so the hypothesis under test is: can FVI
separate vegetation from water on its own, without an external mask step?

This script does NOT touch the production app or its data. It's a
standalone experiment you run against your own already-fetched images.

Usage:
    python test_fvi.py <city> <year>

Reads from ./static/satellite_{city}_{year}.png and ./static/nir_{city}_{year}.png
(same files satellite_fetch.py already produced -- nothing new to fetch).

Writes comparison images to ./static/fvi_test_{city}_{year}_*.png and
prints a quantitative comparison to the console.
"""
import sys
import os
import cv2
import numpy as np

IMAGE_SIZE = 1024
STATIC_DIR = "static"


def load_satellite(city, year):
    path = os.path.join(STATIC_DIR, f"satellite_{city}_{year}.png")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path} -- run satellite_fetch.py first.")
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_nir(city, year):
    path = os.path.join(STATIC_DIR, f"nir_{city}_{year}.png")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path} -- run satellite_fetch.py first.")
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def otsu_threshold_uint8(values):
    if values.size == 0:
        return 255
    hist_input = values.reshape(-1, 1).astype(np.uint8)
    thresh_val, _ = cv2.threshold(hist_input, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(thresh_val)


def to_uint8(index_vals, lo=-1.0, hi=1.0):
    clipped = np.clip(index_vals, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)


def compute_production_pipeline(img_rgb, nir_gray):
    """Current production method: VARI classification + separate NDWI water mask + Otsu."""
    img = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    nir = cv2.resize(nir_gray, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    ndwi = (G - nir) / (G + nir + 1e-6)
    water = ndwi > 0.0

    vari = (G - R) / (G + R - B + 1e-6)
    brightness = (R + G + B) / 3.0
    vari_u8 = to_uint8(vari)
    land_pixels = vari_u8[~water]
    thresh = otsu_threshold_uint8(land_pixels)

    veg = (vari_u8 > thresh) & (brightness > 30) & (~water)
    kernel = np.ones((5, 5), np.uint8)
    veg_u8 = cv2.morphologyEx(veg.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    veg_u8 = cv2.morphologyEx(veg_u8, cv2.MORPH_CLOSE, kernel)
    return veg_u8 > 127, water, vari


def compute_fvi(img_rgb, nir_gray):
    """FVI = (NIR + Green - Red - Blue) / (NIR + Green + Red + Blue)"""
    img = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    nir = cv2.resize(nir_gray, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    fvi = (nir + G - R - B) / (nir + G + R + B + 1e-6)
    return fvi, (R, G, B)


def multi_otsu_2threshold(values_u8):
    """
    Generalizes Otsu's method from 2 classes to 3: finds two thresholds
    (t1 < t2) that split a histogram into three groups, maximizing
    between-class variance. Brute-force over all (t1, t2) pairs using
    cumulative histogram statistics -- 256x256 combinations, cheap.

    Used to test whether ONE index (FVI) with two thresholds can do what
    the production pipeline currently needs two separate indices for:
    class 0 (lowest FVI) = water, class 1 (middle) = bare land/urban,
    class 2 (highest) = vegetation.
    """
    hist = np.bincount(values_u8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0, 255
    prob = hist / total
    bins = np.arange(256)

    cum_p = np.cumsum(prob)
    cum_mean = np.cumsum(prob * bins)
    global_mean = cum_mean[-1]

    best_score = -1.0
    best_t1, best_t2 = 0, 255
    for t1 in range(0, 254):
        w0 = cum_p[t1]
        if w0 <= 1e-9:
            continue
        m0 = cum_mean[t1] / w0
        for t2 in range(t1 + 1, 255):
            w1 = cum_p[t2] - cum_p[t1]
            if w1 <= 1e-9:
                continue
            w2 = 1.0 - cum_p[t2]
            if w2 <= 1e-9:
                continue
            m1 = (cum_mean[t2] - cum_mean[t1]) / w1
            m2 = (global_mean - cum_mean[t2]) / w2
            score = w0 * (m0 - global_mean) ** 2 + w1 * (m1 - global_mean) ** 2 + w2 * (m2 - global_mean) ** 2
            if score > best_score:
                best_score = score
                best_t1, best_t2 = t1, t2
    return best_t1, best_t2


def compute_fvi_pipeline(img_rgb, nir_gray):
    """
    FVI classification using multi-level (3-class) Otsu -- ONE index,
    TWO thresholds, no separate water-mask formula needed. Lowest class
    is treated as water, highest as vegetation.
    """
    fvi, (R, G, B) = compute_fvi(img_rgb, nir_gray)
    brightness = (R + G + B) / 3.0
    fvi_u8 = to_uint8(fvi)

    t1, t2 = multi_otsu_2threshold(fvi_u8.ravel())
    water = fvi_u8 <= t1
    veg = (fvi_u8 > t2) & (brightness > 30)

    kernel = np.ones((5, 5), np.uint8)
    veg_u8 = cv2.morphologyEx(veg.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    veg_u8 = cv2.morphologyEx(veg_u8, cv2.MORPH_CLOSE, kernel)
    return veg_u8 > 127, fvi, (t1, t2), water


def compute_hybrid_pipeline(img_rgb, nir_gray):
    """
    Hybrid: keeps NDWI for water exclusion (proven reliable across both
    scarce water like Bengaluru's lakes and abundant water like Chennai's
    ocean), but swaps FVI in as the vegetation-classifying formula in
    place of VARI, using its own single-class Otsu threshold computed
    ONLY over land pixels -- same two-stage structure as production,
    just a NIR-aware greenness signal instead of an RGB-only one.
    """
    img = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    nir = cv2.resize(nir_gray, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    ndwi = (G - nir) / (G + nir + 1e-6)
    water = ndwi > 0.0

    fvi = (nir + G - R - B) / (nir + G + R + B + 1e-6)
    brightness = (R + G + B) / 3.0
    fvi_u8 = to_uint8(fvi)
    land_pixels = fvi_u8[~water]
    thresh = otsu_threshold_uint8(land_pixels)

    veg = (fvi_u8 > thresh) & (brightness > 30) & (~water)
    kernel = np.ones((5, 5), np.uint8)
    veg_u8 = cv2.morphologyEx(veg.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    veg_u8 = cv2.morphologyEx(veg_u8, cv2.MORPH_CLOSE, kernel)
    return veg_u8 > 127, thresh


def make_mask_visual(mask_bool):
    return (mask_bool.astype(np.uint8)) * 255


def run(city, year):
    print(f"\n=== FVI experimental comparison: {city} {year} ===\n")

    img_rgb = load_satellite(city, year)
    nir_gray = load_nir(city, year)

    prod_mask, ndwi_water, vari_vals = compute_production_pipeline(img_rgb, nir_gray)
    fvi_mask, fvi_vals, (t1, t2), fvi_water = compute_fvi_pipeline(img_rgb, nir_gray)
    hybrid_mask, hybrid_thresh = compute_hybrid_pipeline(img_rgb, nir_gray)

    total_px = prod_mask.size

    # --- Core hypothesis test ---
    water_px = int(np.sum(ndwi_water))
    fvi_false_positive_on_water = int(np.sum(fvi_mask & ndwi_water))
    vari_u8 = to_uint8(vari_vals)
    vari_alone_thresh = otsu_threshold_uint8(vari_u8[~ndwi_water])
    vari_alone_mask = vari_u8 > vari_alone_thresh
    vari_only_false_positive_on_water = int(np.sum(vari_alone_mask & ndwi_water))

    print(f"Reference water pixels (per NDWI): {water_px} ({water_px/total_px*100:.2f}% of frame)")
    print(f"  VARI alone (no water mask) misclassifies as forest: {vari_only_false_positive_on_water} px "
          f"({vari_only_false_positive_on_water/max(water_px,1)*100:.2f}% of water)")
    print(f"  FVI alone (no water mask) misclassifies as forest:  {fvi_false_positive_on_water} px "
          f"({fvi_false_positive_on_water/max(water_px,1)*100:.2f}% of water)")

    # --- Does FVI's OWN water class (from multi-Otsu) agree with NDWI's? ---
    water_agreement = np.sum(fvi_water == ndwi_water) / total_px * 100
    print(f"\nFVI's own multi-Otsu water class vs. NDWI reference: {water_agreement:.2f}% pixel agreement")
    print(f"FVI thresholds (0-255 scale): water<={t1}  |  land  |  forest>{t2}")

    # --- Agreement on land (excluding the water question entirely) ---
    land = ~ndwi_water
    agree_on_land = np.sum((prod_mask == fvi_mask)[land])
    land_px = int(np.sum(land))
    print(f"\nAgreement between production mask and FVI mask, land pixels only: "
          f"{agree_on_land/land_px*100:.2f}%")

    print(f"\nProduction forest area (px): {int(np.sum(prod_mask))}")
    print(f"FVI-only forest area (px):   {int(np.sum(fvi_mask))}")

    # --- Hybrid: NDWI water mask (trusted) + FVI as the vegetation classifier ---
    hybrid_water_fp = int(np.sum(hybrid_mask & ndwi_water))
    agree_hybrid_land = np.sum((prod_mask == hybrid_mask)[land])
    print(f"\n--- Hybrid (NDWI water mask + FVI classifier) ---")
    print(f"Hybrid forest area (px): {int(np.sum(hybrid_mask))}")
    print(f"Hybrid misclassifies as forest on NDWI water: {hybrid_water_fp} px "
          f"({hybrid_water_fp/max(water_px,1)*100:.2f}% of water)")
    print(f"Agreement between production mask and hybrid mask, land pixels only: "
          f"{agree_hybrid_land/land_px*100:.2f}%")
    print(f"Hybrid's own land-only Otsu threshold (FVI scale): {(hybrid_thresh/255*2-1):.4f}")

    # --- Save visuals ---
    os.makedirs(STATIC_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(STATIC_DIR, f"fvi_test_{city}_{year}_production_mask.png"), make_mask_visual(prod_mask))
    cv2.imwrite(os.path.join(STATIC_DIR, f"fvi_test_{city}_{year}_fvi_mask.png"), make_mask_visual(fvi_mask))
    cv2.imwrite(os.path.join(STATIC_DIR, f"fvi_test_{city}_{year}_hybrid_mask.png"), make_mask_visual(hybrid_mask))

    diff = np.zeros((*prod_mask.shape, 3), dtype=np.uint8)
    both = prod_mask & fvi_mask
    only_prod = prod_mask & ~fvi_mask
    only_fvi = ~prod_mask & fvi_mask
    diff[both] = [58, 107, 26]      # both agree it's forest -- green
    diff[only_prod] = [43, 57, 192]  # only production method says forest -- red
    diff[only_fvi] = [0, 180, 232]   # only FVI says forest -- gold
    cv2.imwrite(os.path.join(STATIC_DIR, f"fvi_test_{city}_{year}_diff.png"), diff)

    hybrid_diff = np.zeros((*prod_mask.shape, 3), dtype=np.uint8)
    hboth = prod_mask & hybrid_mask
    only_prod_h = prod_mask & ~hybrid_mask
    only_hybrid = ~prod_mask & hybrid_mask
    hybrid_diff[hboth] = [58, 107, 26]
    hybrid_diff[only_prod_h] = [43, 57, 192]
    hybrid_diff[only_hybrid] = [0, 180, 232]
    cv2.imwrite(os.path.join(STATIC_DIR, f"fvi_test_{city}_{year}_hybrid_diff.png"), hybrid_diff)

    print(f"\nSaved comparison images to {STATIC_DIR}/fvi_test_{city}_{year}_*.png")
    print("  - _production_mask.png : current VARI+NDWI+Otsu result")
    print("  - _fvi_mask.png        : FVI-only result (no external water mask)")
    print("  - _diff.png            : production vs FVI-only (green=agree, red=only production, gold=only FVI)")
    print("  - _hybrid_mask.png     : NDWI water mask + FVI classifier result")
    print("  - _hybrid_diff.png     : production vs hybrid (green=agree, red=only production, gold=only hybrid)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_fvi.py <city> <year>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])