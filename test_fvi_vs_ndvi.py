"""
test_fvi_vs_ndvi.py -- Fair comparison: FVI (one formula) vs NDVI+NDWI
(the real literature-standard two-step approach), not vs VARI (which has
no NIR channel at all and was already known-broken).

Usage: python test_fvi_vs_ndvi.py <city> <year>
"""
import sys
import os
import cv2
import numpy as np

IMAGE_SIZE = 1024
STATIC_DIR = "static"


def load_satellite(city, year):
    img = cv2.imread(os.path.join(STATIC_DIR, f"satellite_{city}_{year}.png"))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_nir(city, year):
    return cv2.imread(os.path.join(STATIC_DIR, f"nir_{city}_{year}.png"), cv2.IMREAD_GRAYSCALE)


def to_uint8(index_vals, lo=-1.0, hi=1.0):
    clipped = np.clip(index_vals, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)


def otsu_threshold_uint8(values):
    if values.size == 0:
        return 255
    thresh_val, _ = cv2.threshold(values.reshape(-1, 1).astype(np.uint8), 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(thresh_val)


def multi_otsu_2threshold(values_u8):
    hist = np.bincount(values_u8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    prob = hist / total
    bins = np.arange(256)
    cum_p = np.cumsum(prob)
    cum_mean = np.cumsum(prob * bins)
    global_mean = cum_mean[-1]
    best_score, best_t1, best_t2 = -1.0, 0, 255
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
                best_score, best_t1, best_t2 = score, t1, t2
    return best_t1, best_t2


def run(city, year):
    img_rgb = load_satellite(city, year)
    nir_gray = load_nir(city, year)

    img = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    nir = cv2.resize(nir_gray, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    brightness = (R + G + B) / 3.0

    # Reference water mask: standard NDWI (McFeeters 1996), Green/NIR
    ndwi = (G - nir) / (G + nir + 1e-6)
    water_ref = ndwi > 0.0
    total_px = water_ref.size
    water_px = int(np.sum(water_ref))

    # --- Standard two-step: NDVI (vegetation) + NDWI (water), combined ---
    ndvi = (nir - R) / (nir + R + 1e-6)
    ndvi_u8 = to_uint8(ndvi)
    land_pixels = ndvi_u8[~water_ref]
    ndvi_thresh = otsu_threshold_uint8(land_pixels)
    ndvi_veg = (ndvi_u8 > ndvi_thresh) & (brightness > 30) & (~water_ref)

    # Does NDVI alone (no NDWI mask) already separate water from veg?
    ndvi_alone_thresh = otsu_threshold_uint8(ndvi_u8.ravel())
    ndvi_alone_veg = (ndvi_u8 > ndvi_alone_thresh) & (brightness > 30)
    ndvi_alone_fp_on_water = int(np.sum(ndvi_alone_veg & water_ref))

    # --- FVI: one formula, 3-class Otsu, no separate water mask ---
    fvi = (nir + G - R - B) / (nir + G + R + B + 1e-6)
    fvi_u8 = to_uint8(fvi)
    t1, t2 = multi_otsu_2threshold(fvi_u8.ravel())
    fvi_veg = (fvi_u8 > t2) & (brightness > 30)
    fvi_fp_on_water = int(np.sum(fvi_veg & water_ref))

    kernel = np.ones((5, 5), np.uint8)
    ndvi_veg_u8 = cv2.morphologyEx((ndvi_veg.astype(np.uint8)) * 255, cv2.MORPH_OPEN, kernel)
    ndvi_veg_u8 = cv2.morphologyEx(ndvi_veg_u8, cv2.MORPH_CLOSE, kernel) > 127
    fvi_veg_u8 = cv2.morphologyEx((fvi_veg.astype(np.uint8)) * 255, cv2.MORPH_OPEN, kernel)
    fvi_veg_u8 = cv2.morphologyEx(fvi_veg_u8, cv2.MORPH_CLOSE, kernel) > 127

    land = ~water_ref
    land_px = int(np.sum(land))
    agree_on_land = int(np.sum((ndvi_veg_u8 == fvi_veg_u8)[land]))

    print(f"\n=== FVI vs NDVI+NDWI: {city} {year} ===")
    print(f"Water reference (NDWI): {water_px} px ({water_px/total_px*100:.2f}% of frame)\n")
    print(f"NDVI alone, NO water mask -- misclassifies as forest: "
          f"{ndvi_alone_fp_on_water} px ({ndvi_alone_fp_on_water/max(water_px,1)*100:.2f}% of water)")
    print(f"FVI alone,  NO water mask -- misclassifies as forest: "
          f"{fvi_fp_on_water} px ({fvi_fp_on_water/max(water_px,1)*100:.2f}% of water)\n")
    print(f"Standard NDVI+NDWI pipeline forest area (px): {int(np.sum(ndvi_veg_u8))}")
    print(f"FVI-only (one formula) pipeline forest area (px): {int(np.sum(fvi_veg_u8))}")
    print(f"Agreement between NDVI+NDWI and FVI, land pixels only: {agree_on_land/land_px*100:.2f}%")

    # --- Visual diff: where do the two methods actually disagree? ---
    diff = np.zeros((*ndvi_veg_u8.shape, 3), dtype=np.uint8)
    both = ndvi_veg_u8 & fvi_veg_u8
    only_ndvi = ndvi_veg_u8 & ~fvi_veg_u8
    only_fvi = ~ndvi_veg_u8 & fvi_veg_u8
    diff[both] = [58, 107, 26]        # both agree -- green
    diff[only_ndvi] = [43, 57, 192]   # only NDVI+NDWI says forest -- red
    diff[only_fvi] = [0, 180, 232]    # only FVI says forest -- gold
    os.makedirs(STATIC_DIR, exist_ok=True)
    out_path = os.path.join(STATIC_DIR, f"fvi_vs_ndvi_diff_{city}_{year}.png")
    cv2.imwrite(out_path, diff)
    print(f"\nSaved diff visual: {out_path}")
    print("  green = both agree it's forest | red = only NDVI+NDWI says forest | gold = only FVI says forest")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
