import os
import json
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="ForestWatch API")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Cities are hardcoded for now.
# NOTE: this dict must stay in sync with the CITIES dict in satellite_fetch.py.
CITIES = {
    "bengaluru": {"label": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "buffer_km": 6},
    "chennai":   {"label": "Chennai",   "lat": 13.0827, "lon": 80.2707, "buffer_km": 6},
}

IMAGE_SIZE = 1024
BASE_URL = "http://127.0.0.1:8000/static"

# ---------------------------------------------------------------------------
# Global threshold cache
#
# Per-image Otsu thresholds drift across years (different atmospheric
# conditions, sun angle, haze cause the FVI histogram to shift). This
# inflates apparent pixel transitions and produces unrealistically high
# Markov transition rates. Fix: compute ONE global threshold per city
# from ALL years' FVI values pooled together, applied consistently for
# cross-year comparison in /predict.
#
# /analyze and /compare keep per-image thresholds -- correct and standard
# for single-year detection. Only the temporal transition model needs
# cross-year consistency.
# ---------------------------------------------------------------------------
_global_threshold_cache: dict = {}

# ---------------------------------------------------------------------------
# Minimum transition gap
#
# With annual Sentinel-2 composites, boundary pixels oscillate year-to-year
# due to seasonal greenness variation in the Jan-May compositing window,
# atmospheric residuals, and sub-pixel edge effects. Each oscillating pixel
# contributes a loss AND a gain event every cycle, inflating transition
# rates to ~18%/yr even when the real net 9-year change is a GAIN (Chennai:
# 608ha in 2016 -> 872ha in 2025, yet uncorrected annual pairing measured
# 18.8%/yr loss).
#
# Fix: only count pixel pairs separated by at least MIN_TRANSITION_GAP_YEARS.
# Oscillating pixels typically return to their original state within 1-2
# years, so a 3-year minimum gap filters noise while preserving all real
# multi-year structural transitions. This matches the original 3-year fetch
# cadence (2016/2019/2022/2025) that produced realistic rates before annual
# data was added.
# ---------------------------------------------------------------------------
MIN_TRANSITION_GAP_YEARS = 3

# NOTE: this constant is currently unused in production. It was defined
# for a temporal-trend CA-ANN feature (density change vs. N years earlier)
# that was implemented, tested, and found to make walk-forward results
# WORSE with only 10 years of source data -- see multiscale_features()'s
# docstring for the full negative-result writeup and the reasoning for
# why it should be revisited once a longer time series is available.
# Kept here (rather than deleted) so the reasoning and the value used are
# both on record for the thesis methodology section.
TREND_LOOKBACK_YEARS = MIN_TRANSITION_GAP_YEARS


def pixel_area_ha(buffer_km: float, dimensions: int = IMAGE_SIZE) -> float:
    """
    Real ground area represented by one pixel, in hectares.
    Derived from the actual GEE thumbnail region size and pixel count.
    """
    side_m = 2 * buffer_km * 1000
    pixel_side_m = side_m / dimensions
    return (pixel_side_m ** 2) / 10000


def compute_fvi(img_rgb: np.ndarray, nir_gray: np.ndarray) -> np.ndarray:
    """
    ForestWatch Vegetation Index:
    FVI = (NIR + Green - Red - Blue) / (NIR + Green + Red + Blue)

    Folds NIR directly into the normalized ratio so the index distinguishes
    water (absorbs NIR) from vegetation (reflects NIR) without a separate
    water mask formula.
    """
    img = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    nir = cv2.resize(nir_gray, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return (nir + G - R - B) / (nir + G + R + B + 1e-6)


def index_to_uint8(values: np.ndarray) -> np.ndarray:
    """Maps normalized index range [-1, 1] to 0-255 for Otsu thresholding."""
    clipped = np.clip(values, -1.0, 1.0)
    return ((clipped + 1.0) / 2.0 * 255).astype(np.uint8)


def multi_otsu_2threshold(values_u8: np.ndarray) -> tuple:
    """
    Generalizes Otsu's method [Otsu, 1979] from 2 classes to 3: finds two
    thresholds (t1 < t2) that split a histogram into three groups by
    maximizing between-class variance.
    """
    hist = np.bincount(values_u8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0, 255
    prob = hist / total
    bins = np.arange(256)

    cum_p    = np.cumsum(prob)
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
            score = (w0 * (m0 - global_mean) ** 2
                   + w1 * (m1 - global_mean) ** 2
                   + w2 * (m2 - global_mean) ** 2)
            if score > best_score:
                best_score = score
                best_t1, best_t2 = t1, t2
    return best_t1, best_t2


def compute_fvi_mask(img_rgb: np.ndarray, nir_gray: np.ndarray,
                     fixed_thresholds: tuple | None = None):
    """
    Single-image vegetation mask using FVI + multi-level (3-class) Otsu.

    fixed_thresholds=(t1, t2) on the 0-255 scale: if supplied, skip
    per-image Otsu and use these directly. Used by /predict to apply a
    consistent global threshold across all years.

    Returns (vegetation_mask_uint8, (t1_fvi_scale, t2_fvi_scale)).
    """
    img = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    brightness = (R + G + B) / 3.0

    fvi   = compute_fvi(img_rgb, nir_gray)
    fvi_u8 = index_to_uint8(fvi)

    if fixed_thresholds is not None:
        t1, t2 = fixed_thresholds
    else:
        t1, t2 = multi_otsu_2threshold(fvi_u8.ravel())

    veg = ((fvi_u8 > t2) & (brightness > 30)).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN,  kernel)
    veg = cv2.morphologyEx(veg, cv2.MORPH_CLOSE, kernel)

    t1_scale = round((t1 / 255.0 * 2.0) - 1.0, 4)
    t2_scale = round((t2 / 255.0 * 2.0) - 1.0, 4)
    return veg, (t1_scale, t2_scale)


def get_global_thresholds(city: str) -> tuple:
    """
    Computes and caches one global (t1, t2) pair per city by pooling FVI
    values from ALL available years and running multi-level Otsu once on
    the combined histogram. Prevents per-image threshold drift.
    """
    if city in _global_threshold_cache:
        return _global_threshold_cache[city]

    all_fvi_u8 = []
    for f in sorted(os.listdir(STATIC_DIR)):
        if f.startswith(f"satellite_{city}_") and f.endswith(".png"):
            year = f.replace(f"satellite_{city}_", "").replace(".png", "")
            try:
                img = _load_satellite_raw(city, year)
                nir = _load_nir_raw(city, year)
                fvi = compute_fvi(img, nir)
                all_fvi_u8.append(index_to_uint8(fvi).ravel())
            except Exception:
                pass

    if not all_fvi_u8:
        return (100, 180)

    pooled = np.concatenate(all_fvi_u8)
    t1, t2 = multi_otsu_2threshold(pooled)
    _global_threshold_cache[city] = (t1, t2)
    print(f"  [global threshold] {city}: t1={t1} t2={t2} "
          f"(pooled from {len(all_fvi_u8)} years)")
    return t1, t2


# ---------------------------------------------------------------------------
# Minimum Mapping Unit
# ---------------------------------------------------------------------------
MIN_MAPPING_UNIT_HA = 0.1


def filter_min_mapping_unit(mask_bool: np.ndarray, pixel_ha: float,
                             min_ha: float = MIN_MAPPING_UNIT_HA) -> np.ndarray:
    """Removes connected regions smaller than min_ha from a boolean change mask."""
    min_px = max(1, int(round(min_ha / pixel_ha)))
    mask_u8 = mask_bool.astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    keep = np.zeros_like(mask_bool)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_px:
            keep |= (labels == label_id)
    return keep


def carbon_metrics(mask: np.ndarray, pixel_ha: float) -> dict:
    forest_px = int(np.sum(mask > 127))
    total_px  = mask.size
    area_ha   = round(forest_px * pixel_ha, 2)
    carbon_t  = round(area_ha * 190.0, 2)
    co2_t     = round(carbon_t * 3.67, 2)
    pct       = round((forest_px / total_px) * 100, 1)
    return {"forest_area_ha": area_ha, "carbon_t": carbon_t,
            "co2_equiv_t": co2_t, "coverage_pct": pct}


def compute_fragmentation_metrics(mask: np.ndarray, pixel_ha: float) -> dict:
    """
    Landscape fragmentation metrics via connected-component analysis of
    the forest mask -- standard FRAGSTATS-style indices used throughout
    landscape ecology literature (McGarigal & Marks, 1995).

      patch_count             -- number of distinct forest patches (8-conn)
      mean_patch_size_ha      -- average patch area
      largest_patch_index_pct -- largest single patch's area as % of total
                                  forest (a high value = one dominant core
                                  block; a low value = forest scattered
                                  across many small patches)
      edge_density_m_per_ha   -- total patch perimeter / total forest area
                                  (higher = more fragmented, more edge
                                  relative to interior)

    Why this matters alongside raw hectares: two masks can report
    IDENTICAL total forest area while representing very different
    ecological realities -- one contiguous 500ha block vs. the same 500ha
    scattered across 200 disconnected slivers. The latter has far less
    viable core habitat, more edge-effect exposure (temperature,
    moisture, invasive-species penetration all increase near edges), and
    less resilience to further loss. Raw area alone hides this; these
    metrics surface it.

    Can be computed on measured masks (from /analyze) AND on the CA-ANN's
    projected future masks (from /predict) -- showing not just how much
    forest is lost by a given horizon, but how much MORE FRAGMENTED the
    remaining forest becomes, which is often the more ecologically
    meaningful story.
    """
    mask_bool = mask > 127
    mask_u8   = mask_bool.astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    patch_areas_px = stats[1:, cv2.CC_STAT_AREA]  # label 0 is background
    patch_count    = len(patch_areas_px)

    if patch_count == 0:
        return {
            "patch_count": 0, "mean_patch_size_ha": 0.0,
            "largest_patch_index_pct": 0.0, "edge_density_m_per_ha": 0.0,
        }

    total_forest_px = int(patch_areas_px.sum())
    patch_areas_ha  = patch_areas_px * pixel_ha
    mean_patch_size_ha = float(np.mean(patch_areas_ha))
    largest_patch_index_pct = round(
        (int(patch_areas_px.max()) / total_forest_px) * 100, 2)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_perimeter_px = sum(cv2.arcLength(c, closed=True) for c in contours)
    pixel_side_m   = (pixel_ha * 10000) ** 0.5   # ha -> m^2 -> side length
    total_perimeter_m = total_perimeter_px * pixel_side_m
    total_forest_ha   = total_forest_px * pixel_ha
    edge_density_m_per_ha = (
        round(total_perimeter_m / total_forest_ha, 2) if total_forest_ha > 0 else 0.0
    )

    return {
        "patch_count":             patch_count,
        "mean_patch_size_ha":      round(mean_patch_size_ha, 3),
        "largest_patch_index_pct": largest_patch_index_pct,
        "edge_density_m_per_ha":   edge_density_m_per_ha,
    }


def make_overlay(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = bgr.copy()
    fg  = mask > 127
    out[fg] = (bgr[fg] * 0.45 + np.array([0, 200, 80], np.float32) * 0.55).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Transition rate computation
# ---------------------------------------------------------------------------

def compute_annual_transition_rates(masks_by_year: dict,
                                    min_gap_years: int = MIN_TRANSITION_GAP_YEARS
                                    ) -> tuple:
    """
    Estimates per-year p_gain (non-forest -> forest) and p_loss
    (forest -> non-forest) from historical masks.

    Only year-pairs separated by >= min_gap_years are counted, filtering
    annual oscillation noise (see MIN_TRANSITION_GAP_YEARS docstring above
    for the full diagnostic history: uncorrected annual pairing gave
    p_loss=18.8%/yr on Chennai data despite the real 9yr net change being
    a +264ha GAIN).
    """
    years = sorted(masks_by_year.keys(), key=int)
    gain_events, gain_denominator = 0, 0.0
    loss_events, loss_denominator = 0, 0.0

    for i, y0 in enumerate(years):
        for y1 in years[i + 1:]:
            elapsed = int(y1) - int(y0)
            if elapsed < min_gap_years:
                continue
            m0, m1 = masks_by_year[y0], masks_by_year[y1]
            non_forest_0 = ~m0
            forest_0     =  m0

            gain_events      += int(np.sum(non_forest_0 & m1))
            gain_denominator += int(np.sum(non_forest_0)) * elapsed
            loss_events      += int(np.sum(forest_0 & ~m1))
            loss_denominator += int(np.sum(forest_0)) * elapsed

    p_gain = (gain_events / gain_denominator) if gain_denominator > 0 else 0.0
    p_loss = (loss_events / loss_denominator) if loss_denominator > 0 else 0.0
    return p_gain, p_loss


def compute_local_forest_density(mask: np.ndarray, window: int = 15) -> np.ndarray:
    """
    Fraction of forest pixels in a window x window neighbourhood.
    Fallback spatial-suitability signal used when no CA-ANN model exists.
    """
    m = (mask > 0).astype(np.float32)
    return cv2.boxFilter(m, ddepth=-1, ksize=(window, window), normalize=True)


# ---------------------------------------------------------------------------
# CA-ANN spatial suitability model
# ---------------------------------------------------------------------------

CA_ANN_MODEL_PATH = os.path.join(BASE_DIR, "ca_ann_model.joblib")
CA_ANN_STATS_PATH = os.path.join(BASE_DIR, "ca_ann_model_stats.json")
_ca_ann_cache = {"model": None, "beta": None, "loaded": False}


def multiscale_features(mask_bool: np.ndarray) -> np.ndarray:
    """
    Four features per pixel: forest density at 5x5, 15x15, 31x31 windows,
    plus normalized distance to nearest forest/non-forest boundary.

    METHODOLOGY NOTE -- a temporal trend feature was tried and rejected:
    a 5th feature (change in 15x15 density vs. 3 years earlier) was added
    to give the CA-ANN information the single-snapshot density heuristic
    structurally cannot access. Walk-forward validation showed it made
    results WORSE, not better: 1-year spatial-ranking wins dropped from
    3/8 to 1/8, and the 6-year projection horizon's mean Kappa delta
    flipped from +0.0015 to -0.0034. Likely cause: with only 10 annual
    years, ~1/3 of training transitions (years 2016-2018) have no valid
    3-years-earlier mask, so "trend=0" conflates "genuinely stable" with
    "unknown" for a third of the training data; separately, with only
    ~9 independent transitions per city, spatially-autocorrelated pixels
    give far less independent signal than the raw pixel count suggests,
    likely too little effective sample size for a new feature to prove
    itself cleanly. This is a data-volume ceiling, not a rejected
    hypothesis -- worth revisiting once a longer time series is fetched
    (Landsat pre-2019 extension, already on the roadmap) or once external
    layers (Hansen GFC, LST) are incorporated, per the validation-stage
    roadmap item.
    """
    m = mask_bool.astype(np.float32)
    d5  = cv2.boxFilter(m, -1, (5,  5))
    d15 = cv2.boxFilter(m, -1, (15, 15))
    d31 = cv2.boxFilter(m, -1, (31, 31))

    mask_u8 = mask_bool.astype(np.uint8) * 255
    dist_to_nonforest = cv2.distanceTransform(mask_u8,       cv2.DIST_L2, 5)
    dist_to_forest    = cv2.distanceTransform(255 - mask_u8, cv2.DIST_L2, 5)
    edge_dist      = np.minimum(dist_to_nonforest, dist_to_forest)
    edge_dist_norm = np.clip(edge_dist / 30.0, 0, 1)

    feats = np.stack([d5, d15, d31, edge_dist_norm], axis=-1)
    return feats.reshape(-1, 4)


def calibrate_proba(p_resampled: np.ndarray, beta: float) -> np.ndarray:
    """
    Prior-correction for probabilities from a model trained on a
    negative-class-undersampled set (Dal Pozzolo et al., 2015).
    """
    p = np.clip(p_resampled, 1e-6, 1 - 1e-6)
    odds = p / (1 - p)
    corrected_odds = odds * beta
    return corrected_odds / (1 + corrected_odds)


def load_ca_ann_model():
    """
    Loads the offline-trained CA-ANN model (produced by train_ca_ann.py),
    cached after first load. Returns (None, None) if no model exists yet.
    """
    if not _ca_ann_cache["loaded"]:
        if os.path.exists(CA_ANN_MODEL_PATH):
            import joblib
            saved = joblib.load(CA_ANN_MODEL_PATH)
            _ca_ann_cache["model"] = saved["model"]
            _ca_ann_cache["beta"]  = saved["beta"]
        _ca_ann_cache["loaded"] = True
    return _ca_ann_cache["model"], _ca_ann_cache["beta"]


def load_ca_ann_stats():
    """Walk-forward validation stats saved by train_ca_ann.py."""
    if os.path.exists(CA_ANN_STATS_PATH):
        with open(CA_ANN_STATS_PATH) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Risk heatmap & conservation priority ranking
#
# The CA-ANN's calibrated per-pixel change probability (ann_score) is
# already computed inside project_future_mask_ca_ann() purely as an
# internal ranking signal for a specific bounded projection. These two
# features surface that same score DIRECTLY -- as a continuous risk
# visualization, and as an actionable ranked list -- independent of any
# particular target year, turning the model's internal reasoning into
# something a viewer can actually see and act on.
# ---------------------------------------------------------------------------

def render_risk_heatmap(img_rgb: np.ndarray, ann_score: np.ndarray,
                        current_forest: np.ndarray) -> np.ndarray:
    """
    Visualizes the CA-ANN's raw calibrated change-probability as a
    continuous heatmap (via OpenCV's TURBO colormap: dark blue = low
    RELATIVE risk through green/yellow to red = high RELATIVE risk),
    applied only over CURRENT forest pixels.

    Uses min-max contrast stretching WITHIN the forest mask, not a raw
    linear [0,1]->colour mapping. Raw single-year change probabilities
    cluster low for almost every pixel (real annual change is rare), so
    a literal linear mapping would render nearly the entire forest in a
    single dull dark-blue shade with no visible variation. Stretching to
    the actual observed min-max range shows RELATIVE risk clearly --
    which is the meaningful question here ("where, within this city's
    forest, is risk concentrated"), not an absolute probability reading.

    This is deliberately different from /predict's striped gain/loss
    overlay: that overlay shows a BOUNDED outcome at one specific target
    year (how many pixels, decided by the Markov rate). This shows the
    model's continuous, relative risk ranking for the existing forest
    right now, independent of any horizon.
    """
    img = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = (bgr.astype(np.float32) * 0.55).astype(np.uint8)  # dim background

    forest_scores = ann_score[current_forest]
    if forest_scores.size > 0:
        lo, hi = float(forest_scores.min()), float(forest_scores.max())
        if hi > lo:
            norm_score = np.clip((ann_score - lo) / (hi - lo), 0, 1)
        else:
            norm_score = np.zeros_like(ann_score)
    else:
        norm_score = np.zeros_like(ann_score)

    # Soften into a visible "glow" rather than hard per-pixel edges/dots.
    # Purely cosmetic for this rendered PNG -- the underlying ann_score
    # array used for high_risk_area_ha / conservation-priority stats
    # elsewhere is completely untouched by this blur.
    norm_score = cv2.GaussianBlur(norm_score, (0, 0), sigmaX=3)
    norm_score = np.clip(norm_score, 0, 1)

    score_u8 = (norm_score * 255).astype(np.uint8)
    heat = cv2.applyColorMap(score_u8, cv2.COLORMAP_TURBO)

    fg = current_forest
    out[fg] = (bgr[fg].astype(np.float32) * 0.35
               + heat[fg].astype(np.float32) * 0.65).astype(np.uint8)
    return out


def compute_conservation_priority(mask: np.ndarray, ann_score: np.ndarray,
                                  pixel_ha: float, min_patch_ha: float = 0.5,
                                  top_n: int = 10) -> list:
    """
    Ranks individual forest patches by a transparent priority score
    combining SIZE (larger, more ecologically valuable patches) and RISK
    (the CA-ANN's mean calibrated change-probability within that patch):

        priority_score = normalized_area * mean_risk

    both terms in [0, 1], so a large AND at-risk patch ranks highest;
    a small low-risk patch or a huge stable core patch both rank lower.
    Patches under min_patch_ha are excluded as likely noise / isolated
    street trees rather than meaningful habitat.

    This is what turns detection + prediction into a DECISION-SUPPORT
    output: not just "here's what's happening" but "here are the specific
    N patches that most warrant conservation attention right now, and
    why" -- an explicit, ranked, actionable result.

    centroid/bbox are reported in PIXEL space (not lat/lon) -- converting
    accurately to geographic coordinates requires the exact GEE buffer
    region geometry, which is a natural follow-up once map-based
    (Leaflet/Mapbox) display is added per the roadmap.
    """
    mask_bool = mask > 127
    mask_u8   = mask_bool.astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    min_px = max(1, int(round(min_patch_ha / pixel_ha)))

    max_area_px = 0
    for label_id in range(1, num_labels):
        area_px = stats[label_id, cv2.CC_STAT_AREA]
        if area_px >= min_px:
            max_area_px = max(max_area_px, area_px)

    candidates = []
    for label_id in range(1, num_labels):
        area_px = stats[label_id, cv2.CC_STAT_AREA]
        if area_px < min_px:
            continue

        patch_pixels = (labels == label_id)
        mean_risk = float(np.mean(ann_score[patch_pixels]))
        norm_area = (area_px / max_area_px) if max_area_px > 0 else 0.0
        priority_score = round(norm_area * mean_risk, 4)

        x, y, w, h, _ = stats[label_id]
        cx, cy = centroids[label_id]

        candidates.append({
            "patch_id":       int(label_id),
            "area_ha":        round(area_px * pixel_ha, 2),
            "mean_risk":      round(mean_risk, 4),
            "priority_score": priority_score,
            "bbox_px":        {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "centroid_px":    {"x": round(float(cx), 1), "y": round(float(cy), 1)},
        })

    candidates.sort(key=lambda c: c["priority_score"], reverse=True)
    return candidates[:top_n]


def _prepare_city_risk_context(city: str):
    """
    Shared setup for /risk and /conservation-priority: builds the
    consistent-threshold historical masks, loads the trained CA-ANN, and
    computes the calibrated per-pixel risk score for the latest year.
    Raises HTTPException if the city is unknown, too little history
    exists, or no trained model is available yet.
    """
    if city not in CITIES:
        raise HTTPException(400, f"Unknown city: {city}")

    known_years = sorted(
        (f.replace(f"satellite_{city}_", "").replace(".png", "")
         for f in os.listdir(STATIC_DIR)
         if f.startswith(f"satellite_{city}_") and f.endswith(".png")),
        key=int,
    )
    if len(known_years) < 3:
        raise HTTPException(400, "Need at least 3 known years of data before this can be computed.")

    latest_year = known_years[-1]
    global_t    = get_global_thresholds(city)
    masks_by_year = {}
    for y in known_years:
        img = load_satellite(city, y)
        nir = load_nir(city, y)
        mask, _ = compute_fvi_mask(img, nir, fixed_thresholds=global_t)
        masks_by_year[y] = mask > 127

    ca_ann_model, ca_ann_beta = load_ca_ann_model()
    if ca_ann_model is None:
        raise HTTPException(400, "No trained CA-ANN model found -- run train_ca_ann.py first.")

    latest_mask = masks_by_year[latest_year]
    X     = multiscale_features(latest_mask)
    raw_p = ca_ann_model.predict_proba(X)[:, 1]
    ann_score = calibrate_proba(raw_p, ca_ann_beta).reshape(latest_mask.shape)

    return latest_year, latest_mask, ann_score


def project_future_mask_ca_ann(model, beta: float,
                                masks_by_year: dict, target_year: int):
    """
    CA-ANN projection.

    DESIGN (mirrors project_future_mask() exactly, see that function):
      - HOW MANY pixels change is decided by the historical Markov rate
        (compute_annual_transition_rates, with the >=3yr minimum-gap
        filter), compounded over the projection horizon -- identical
        maths to the density-heuristic fallback.
      - WHICH pixels change is decided by the trained CA-ANN's calibrated
        change-probability, used purely as a RANKING score within the
        forest pool (loss candidates) and non-forest pool (gain
        candidates) separately.

    BUG THIS REPLACES (kept here for the thesis methodology record):
    an earlier version thresholded each pixel's own time-compounded
    probability independently at 0.5 and XOR'd the result against the
    current mask. Two visible failures resulted:
      1. Every patch boundary showed a symmetric ring of predicted LOSS
         on the forest side and predicted GAIN on the non-forest side
         simultaneously -- because boundary pixels on BOTH sides
         legitimately score a high "near an edge" probability from the
         same multiscale density/edge-distance features, and nothing
         capped how many pixels on each side could flip.
      2. Because per-pixel probabilities compound with elapsed years and
         there was no shared pixel budget, by ~9 years out nearly every
         edge pixel independently crossed 0.5 and entire small patches
         were erased, regardless of the true historical rate of change.
    Separating "how many" (bounded by the real Markov rate) from "which"
    (CA-ANN ranking) fixes both: total changed pixels stays proportional
    to the actual historical rate (~1-3%/yr, not an unbounded per-pixel
    cascade), and a patch can only lose as many edge pixels as that rate
    supports, so interior cores persist even at long horizons.
    """
    years       = sorted(masks_by_year.keys(), key=int)
    latest_year = years[-1]
    latest_mask = masks_by_year[latest_year]
    elapsed     = target_year - int(latest_year)
    if elapsed <= 0:
        raise ValueError("target_year must be after the latest known year")

    # HOW MANY: same Markov rate calculation as the heuristic fallback.
    p_gain, p_loss = compute_annual_transition_rates(masks_by_year)

    non_forest_mask  = ~latest_mask
    forest_mask      =  latest_mask
    non_forest_count = int(np.sum(non_forest_mask))
    forest_count     = int(np.sum(forest_mask))

    p_gain_compound = 1 - (1 - p_gain) ** elapsed
    p_loss_compound = 1 - (1 - p_loss) ** elapsed

    expected_gain_px = min(non_forest_count, int(round(p_gain_compound * non_forest_count)))
    expected_loss_px = min(forest_count,     int(round(p_loss_compound * forest_count)))

    # WHICH: CA-ANN calibrated score, used only as a ranking signal.
    X         = multiscale_features(latest_mask)
    raw_p     = model.predict_proba(X)[:, 1]
    ann_score = calibrate_proba(raw_p, beta).reshape(latest_mask.shape)

    # Gain candidates: non-forest pixels, ranked by CA-ANN score
    # descending (bare land adjacent to existing forest scores highest --
    # edge growth / infill is the plausible gain pattern).
    gain_scores = np.where(non_forest_mask, ann_score, -1.0)
    gain_order  = np.argsort(gain_scores.ravel())[::-1]
    gain_mask   = np.zeros_like(latest_mask)
    if expected_gain_px > 0:
        chosen = gain_order[:expected_gain_px]
        chosen = chosen[gain_scores.ravel()[chosen] > 0]
        gain_mask.ravel()[chosen] = True

    # Loss candidates: forest pixels, ranked by CA-ANN score descending
    # (the most edge-like / least stable forest pixels are lost first).
    loss_scores = np.where(forest_mask, ann_score, -1.0)
    loss_order  = np.argsort(loss_scores.ravel())[::-1]
    loss_mask   = np.zeros_like(latest_mask)
    if expected_loss_px > 0:
        chosen = loss_order[:expected_loss_px]
        loss_mask.ravel()[chosen] = True

    projected = (latest_mask | gain_mask) & ~loss_mask

    meta = {
        "base_year":         latest_year,
        "elapsed_years":     elapsed,
        "p_gain_annual":     round(p_gain, 5),
        "p_loss_annual":     round(p_loss, 5),
        "predicted_gain_px": int(np.sum(gain_mask)),
        "predicted_loss_px": int(np.sum(loss_mask)),
        "method":            "ca_ann_ranked_markov_count",
        "threshold_mode":    "global_pooled",
        "min_gap_years":     MIN_TRANSITION_GAP_YEARS,
    }
    return projected, meta


def project_future_mask(masks_by_year: dict, target_year: int) -> tuple:
    """
    Density-heuristic fallback projection used when no trained CA-ANN
    model is available. Historical Markov rate decides HOW MUCH; local
    forest density decides WHICH pixels. project_future_mask_ca_ann()
    above mirrors this exact structure, swapping density for the trained
    CA-ANN score as the ranking signal.
    """
    years       = sorted(masks_by_year.keys(), key=int)
    latest_year = years[-1]
    latest_mask = masks_by_year[latest_year]
    elapsed     = target_year - int(latest_year)
    if elapsed <= 0:
        raise ValueError("target_year must be after the latest known year")

    p_gain, p_loss = compute_annual_transition_rates(masks_by_year)

    non_forest_mask  = ~latest_mask
    forest_mask      =  latest_mask
    non_forest_count = int(np.sum(non_forest_mask))
    forest_count     = int(np.sum(forest_mask))

    p_gain_compound = 1 - (1 - p_gain) ** elapsed
    p_loss_compound = 1 - (1 - p_loss) ** elapsed

    expected_gain_px = min(non_forest_count, int(round(p_gain_compound * non_forest_count)))
    expected_loss_px = min(forest_count,     int(round(p_loss_compound * forest_count)))

    density = compute_local_forest_density(latest_mask)

    gain_scores = np.where(non_forest_mask, density, -1.0)
    gain_order  = np.argsort(gain_scores.ravel())[::-1]
    gain_mask   = np.zeros_like(latest_mask)
    if expected_gain_px > 0:
        chosen = gain_order[:expected_gain_px]
        chosen = chosen[gain_scores.ravel()[chosen] > 0]
        gain_mask.ravel()[chosen] = True

    loss_scores = np.where(forest_mask, -density, -1.0)
    loss_order  = np.argsort(loss_scores.ravel())[::-1]
    loss_mask   = np.zeros_like(latest_mask)
    if expected_loss_px > 0:
        chosen = loss_order[:expected_loss_px]
        loss_mask.ravel()[chosen] = True

    projected = (latest_mask | gain_mask) & ~loss_mask

    meta = {
        "base_year":         latest_year,
        "elapsed_years":     elapsed,
        "p_gain_annual":     round(p_gain, 5),
        "p_loss_annual":     round(p_loss, 5),
        "predicted_gain_px": int(np.sum(gain_mask)),
        "predicted_loss_px": int(np.sum(loss_mask)),
        "threshold_mode":    "global_pooled",
        "min_gap_years":     MIN_TRANSITION_GAP_YEARS,
    }
    return projected, meta


def validate_prediction(masks_by_year: dict) -> dict:
    """
    Holds out the most recent year, projects from earlier years only,
    and reports pixel agreement and Cohen's Kappa.
    """
    years = sorted(masks_by_year.keys(), key=int)
    if len(years) < 3:
        reason = "Need at least 3 known years to validate a projection."
        return {"available": False, "reason": reason, "summary": reason}

    held_out_year  = years[-1]
    training_masks = {y: masks_by_year[y] for y in years[:-1]}

    predicted, _ = project_future_mask(training_masks, int(held_out_year))
    actual = masks_by_year[held_out_year]

    agreement_pct   = float(np.mean(predicted == actual)) * 100
    p_o             = agreement_pct / 100
    p_pred_forest   = float(np.mean(predicted))
    p_actual_forest = float(np.mean(actual))
    p_e   = (p_pred_forest * p_actual_forest
             + (1 - p_pred_forest) * (1 - p_actual_forest))
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 1e-9 else 0.0

    summary = (
        f"Single held-out-year check: trained on {', '.join(years[:-1])}, "
        f"tested against real {held_out_year} data -- "
        f"{round(agreement_pct, 2)}% pixel agreement, Kappa = {round(float(kappa), 4)}."
    )

    return {
        "available":           True,
        "held_out_year":       held_out_year,
        "trained_on_years":    years[:-1],
        "pixel_agreement_pct": round(agreement_pct, 2),
        "kappa":               round(float(kappa), 4),
        "summary":             summary,
    }


def select_multiyear_validation(stats: dict, elapsed_years: int) -> dict:
    """
    Picks the pre-computed multi-year walk-forward validation result
    (produced offline by train_ca_ann.py's run_multiyear_validation)
    whose tested horizon is closest to this request's elapsed_years, and
    builds a display-ready summary string.

    Why this exists: the CA-ANN's original walk-forward validation only
    ever tested 1-year-ahead transitions (see "spatial_ranking_validation"
    in the stats file), which validates the model's spatial ranking
    quality but says nothing about accuracy at the 3/6/9-year horizons
    /predict actually serves. This selects from a SEPARATE validation
    block that backtests those real horizons directly -- training on data
    up to a cutoff year, projecting forward exactly N years, and comparing
    to the real held-out mask at that horizon.

    With only 10 years of source data (2016-2025), a 9-year-horizon
    backtest has zero valid test points (it would require training data
    starting before the earliest fetched year). That horizon's entry
    reports "available": False honestly rather than fabricating a number,
    and will self-activate once more years are fetched.
    """
    mv = stats.get("multiyear_projection_validation")
    if not mv or not mv.get("by_horizon"):
        reason = "No multi-year validation found -- re-run train_ca_ann.py to generate it."
        return {"available": False, "reason": reason, "summary": reason}

    by_horizon = mv["by_horizon"]
    available_horizons = [int(h) for h, v in by_horizon.items() if v.get("available")]
    if not available_horizons:
        reason = "Not enough historical years yet to backtest any multi-year horizon."
        return {"available": False, "reason": reason, "summary": reason}

    closest_h = min(available_horizons, key=lambda h: abs(h - elapsed_years))
    entry = by_horizon[str(closest_h)]

    horizon_note = "" if closest_h == elapsed_years else (
        f" (closest available backtest horizon is {closest_h}yr "
        f"vs this {elapsed_years}yr projection)"
    )
    summary = (
        f"Walk-forward tested at a {closest_h}-year horizon across "
        f"{entry['n_test_points']} independent train/test splits{horizon_note}: "
        f"{entry['mean_ca_ann_agree']:.2f}% mean pixel agreement, "
        f"mean Kappa = {entry['mean_ca_ann_kappa']:.4f} "
        f"(vs {entry['mean_old_kappa']:.4f} for the density-heuristic baseline)."
    )

    return {
        "available":              True,
        "kappa":                  round(entry["mean_ca_ann_kappa"], 4),
        "pixel_agreement_pct":    round(entry["mean_ca_ann_agree"], 2),
        "tested_horizon_years":   closest_h,
        "requested_horizon_years": elapsed_years,
        "n_test_points":          entry["n_test_points"],
        "summary":                summary,
    }


def make_prediction_overlay(img_rgb: np.ndarray, current_forest: np.ndarray,
                             gain_pixels: np.ndarray,
                             loss_pixels: np.ndarray) -> np.ndarray:
    """
    Solid green = current measured forest.
    Diagonal stripes = predicted change (gold=gain, red=loss).
    """
    img = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = bgr.copy()

    stable = current_forest & ~loss_pixels
    out[stable] = (bgr[stable] * 0.45
                   + np.array([26, 107, 58], np.float32) * 0.55).astype(np.uint8)

    yy, xx = np.indices(out.shape[:2])
    stripe = ((xx + yy) % 12) < 5

    out[gain_pixels & stripe] = [0,  180, 232]   # gold
    out[loss_pixels & stripe] = [43,  57, 192]   # red

    return out


# ---------------------------------------------------------------------------
# Image loaders
# ---------------------------------------------------------------------------

def _load_satellite_raw(city: str, year: str) -> np.ndarray:
    path = os.path.join(STATIC_DIR, f"satellite_{city}_{year}.png")
    img  = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Missing {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_nir_raw(city: str, year: str) -> np.ndarray:
    path = os.path.join(STATIC_DIR, f"nir_{city}_{year}.png")
    img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Missing {path}")
    return img


def load_satellite(city: str, year: str) -> np.ndarray:
    try:
        return _load_satellite_raw(city, year)
    except FileNotFoundError:
        raise HTTPException(404, f"No satellite image for {city} {year}. Run satellite_fetch.py first.")


def load_nir(city: str, year: str) -> np.ndarray:
    try:
        return _load_nir_raw(city, year)
    except FileNotFoundError:
        raise HTTPException(404, f"No NIR image for {city} {year}. Run satellite_fetch.py first.")


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/cities")
def list_cities():
    return {"cities": [{"key": k, "label": v["label"]} for k, v in CITIES.items()]}


@app.get("/years/{city}")
def list_years(city: str):
    years = set()
    for f in os.listdir(STATIC_DIR):
        if f.startswith(f"satellite_{city}_") and f.endswith(".png"):
            y = f.replace(f"satellite_{city}_", "").replace(".png", "")
            years.add(y)
    return {"years": sorted(years)}


@app.get("/analyze/{city}/{year}")
def analyze(city: str, year: str):
    """
    Single-year analysis. Per-image Otsu thresholds -- correct for
    single-scene detection where no cross-year comparison is needed.
    """
    if city not in CITIES:
        raise HTTPException(400, f"Unknown city: {city}")

    img_rgb  = load_satellite(city, year)
    nir      = load_nir(city, year)
    mask, (t1, t2) = compute_fvi_mask(img_rgb, nir)
    pixel_ha = pixel_area_ha(CITIES[city]["buffer_km"])
    metrics       = carbon_metrics(mask, pixel_ha)
    fragmentation = compute_fragmentation_metrics(mask, pixel_ha)
    overlay  = make_overlay(img_rgb, mask)

    cv2.imwrite(os.path.join(STATIC_DIR, f"mask_{city}_{year}.png"),    mask)
    cv2.imwrite(os.path.join(STATIC_DIR, f"overlay_{city}_{year}.png"), overlay)

    return {
        "city": city, "year": year,
        "satellite_url": f"{BASE_URL}/satellite_{city}_{year}.png",
        "mask_url":      f"{BASE_URL}/mask_{city}_{year}.png",
        "overlay_url":   f"{BASE_URL}/overlay_{city}_{year}.png",
        "fvi_thresholds_used": {"water_max": t1, "forest_min": t2},
        "fragmentation": fragmentation,
        **metrics,
    }


@app.get("/compare/{city}/{year_a}/{year_b}")
def compare(city: str, year_a: str, year_b: str):
    """
    Two-year change detection. Per-image Otsu per year; MMU filter
    suppresses boundary-flip noise in the change masks.
    """
    if city not in CITIES:
        raise HTTPException(400, f"Unknown city: {city}")

    img_a  = load_satellite(city, year_a)
    img_b  = load_satellite(city, year_b)
    nir_a  = load_nir(city, year_a)
    nir_b  = load_nir(city, year_b)

    mask_a, (t1_a, t2_a) = compute_fvi_mask(img_a, nir_a)
    mask_b, (t1_b, t2_b) = compute_fvi_mask(img_b, nir_b)
    pixel_ha = pixel_area_ha(CITIES[city]["buffer_km"])

    forest_a = mask_a > 127
    forest_b = mask_b > 127

    lost_raw   = forest_a & ~forest_b
    gained_raw = ~forest_a & forest_b
    stable     = forest_a & forest_b

    lost   = filter_min_mapping_unit(lost_raw,   pixel_ha)
    gained = filter_min_mapping_unit(gained_raw, pixel_ha)

    diff_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    diff_img[stable] = [58, 107,  26]
    diff_img[lost]   = [43,  57, 192]
    diff_img[gained] = [ 0, 180, 232]

    cv2.imwrite(os.path.join(STATIC_DIR, f"diff_{city}_{year_a}_{year_b}.png"), diff_img)

    metrics_a = carbon_metrics(mask_a, pixel_ha)
    metrics_b = carbon_metrics(mask_b, pixel_ha)

    lost_ha   = round(int(np.sum(lost))   * pixel_ha, 2)
    gained_ha = round(int(np.sum(gained)) * pixel_ha, 2)
    net_ha    = round(gained_ha - lost_ha, 2)

    return {
        "city": city,
        "year_a": year_a, "year_b": year_b,
        "diff_url": f"{BASE_URL}/diff_{city}_{year_a}_{year_b}.png",
        "min_mapping_unit_ha": MIN_MAPPING_UNIT_HA,
        "year_a_data": {**metrics_a, "fvi_thresholds_used": {"water_max": t1_a, "forest_min": t2_a}},
        "year_b_data": {**metrics_b, "fvi_thresholds_used": {"water_max": t1_b, "forest_min": t2_b}},
        "change": {
            "lost_ha":       lost_ha,
            "gained_ha":     gained_ha,
            "net_ha":        net_ha,
            "lost_carbon_t": round(lost_ha * 190.0, 2),
            "lost_co2_t":    round(lost_ha * 190.0 * 3.67, 2),
        }
    }


@app.get("/predict/{city}/{target_year}")
def predict(city: str, target_year: int):
    """
    Future projection with three fixes applied (see docstrings above for
    full diagnostic history):

    1. GLOBAL THRESHOLDS: all historical masks are built with one pooled
       threshold per city, preventing classification drift from creating
       phantom transitions.

    2. MIN TRANSITION GAP: only year-pairs >= MIN_TRANSITION_GAP_YEARS
       apart contribute to the Markov rate, filtering annual oscillation
       noise that inflated rates ~10x with annual data.

    3. RANKED-COUNT CA-ANN: the CA-ANN projection uses the Markov rate to
       bound how many pixels change and its own calibrated score only to
       rank which ones -- fixing the symmetric loss/gain ring artifact
       and the total-erosion-by-long-horizon bug that came from
       thresholding each pixel independently.

    /analyze and /compare are unchanged -- they use per-image thresholds,
    correct for single-scene detection.
    """
    if city not in CITIES:
        raise HTTPException(400, f"Unknown city: {city}")

    known_years = sorted(
        (f.replace(f"satellite_{city}_", "").replace(".png", "")
         for f in os.listdir(STATIC_DIR)
         if f.startswith(f"satellite_{city}_") and f.endswith(".png")),
        key=int,
    )
    if len(known_years) < 3:
        raise HTTPException(400,
            "Need at least 3 known years of data before a projection can be made.")

    latest_year = known_years[-1]
    if target_year <= int(latest_year):
        raise HTTPException(400,
            f"target_year must be after the latest known year ({latest_year}).")

    pixel_ha = pixel_area_ha(CITIES[city]["buffer_km"])

    global_t = get_global_thresholds(city)
    masks_by_year = {}
    for y in known_years:
        img = load_satellite(city, y)
        nir = load_nir(city, y)
        mask, _ = compute_fvi_mask(img, nir, fixed_thresholds=global_t)
        masks_by_year[y] = mask > 127

    ca_ann_model, ca_ann_beta = load_ca_ann_model()
    if ca_ann_model is not None:
        projected_mask, meta = project_future_mask_ca_ann(
            ca_ann_model, ca_ann_beta, masks_by_year, target_year)
        stats = load_ca_ann_stats()
        if stats is not None:
            validation = select_multiyear_validation(stats, meta["elapsed_years"])
        else:
            reason = "Model loaded but no validation stats found -- re-run train_ca_ann.py."
            validation = {"available": False, "reason": reason, "summary": reason}
    else:
        projected_mask, meta = project_future_mask(masks_by_year, target_year)
        validation = validate_prediction(masks_by_year)
        meta["method"] = "density_heuristic_fallback"

    latest_img  = load_satellite(city, latest_year)
    latest_mask = masks_by_year[latest_year]
    gain_pixels = projected_mask & ~latest_mask
    loss_pixels = ~projected_mask & latest_mask

    overlay = make_prediction_overlay(latest_img, latest_mask, gain_pixels, loss_pixels)
    cv2.imwrite(os.path.join(STATIC_DIR, f"predict_{city}_{target_year}.png"), overlay)

    projected_metrics = carbon_metrics(projected_mask.astype(np.uint8) * 255, pixel_ha)
    latest_metrics    = carbon_metrics(latest_mask.astype(np.uint8) * 255, pixel_ha)

    # Fragmentation isn't just measured for the present -- it's projected
    # forward too, since a city can lose very little total area while its
    # remaining forest becomes far more fragmented (more, smaller,
    # disconnected patches), which is often the more ecologically
    # meaningful story than hectares alone.
    projected_fragmentation = compute_fragmentation_metrics(
        projected_mask.astype(np.uint8) * 255, pixel_ha)
    latest_fragmentation = compute_fragmentation_metrics(
        latest_mask.astype(np.uint8) * 255, pixel_ha)

    return {
        "city":            city,
        "base_year":       latest_year,
        "target_year":     target_year,
        "predict_url":     f"{BASE_URL}/predict_{city}_{target_year}.png",
        "base_year_data":  {**latest_metrics,    "fragmentation": latest_fragmentation},
        "projected_data":  {**projected_metrics, "fragmentation": projected_fragmentation},
        "projection_meta": meta,
        "validation":      validation,
    }


@app.get("/risk/{city}")
def risk_heatmap(city: str, high_risk_percentile: float = 90.0):
    """
    Continuous deforestation risk heatmap for the latest available year,
    using the CA-ANN's own calibrated per-pixel change probability
    directly -- not bounded by any target-year pixel budget the way
    /predict's projection is. Shows the model's spatial risk RANKING for
    the forest that exists right now.

    "High risk" is defined as the top `high_risk_percentile`% of forest
    BY SCORE (default: top 10%), not an absolute probability cutoff. A
    fixed threshold like ">0.5" would report ~0 for almost any city,
    because raw single-year change probabilities cluster low for the
    vast majority of pixels (real annual change is rare) -- that
    threshold only meant something in the OLD, since-replaced design
    where probabilities were compounded over up to 9 years and routinely
    crossed 0.5. A percentile cutoff always returns a meaningful,
    non-trivial result and matches how ann_score is used everywhere else
    in this codebase: as a RANKING signal, never as an absolute
    real-world probability.

    NOTE ON INTERPRETATION: this score is trained on raw 1-year
    consecutive-pair transitions (deliberately, for ranking quality --
    see multiscale_features docstring), so it reflects annual pixel
    volatility INCLUDING oscillation noise. It should not be compared
    directly to /predict's p_loss_annual, which is the >=3yr-minimum-gap
    filtered Markov rate representing real structural loss with noise
    removed. The two numbers measure different things by design.
    """
    latest_year, latest_mask, ann_score = _prepare_city_risk_context(city)
    pixel_ha = pixel_area_ha(CITIES[city]["buffer_km"])

    # Restrict to patches at or above the same MIN_MAPPING_UNIT_HA
    # threshold already used elsewhere (/compare's change-noise filter),
    # for consistency. Tiny sub-threshold fragments (a handful of pixels)
    # are far more likely to be classification noise or individual street
    # trees than meaningful forest units -- including them both clutters
    # the visual with isolated "hot" dots carrying little real ecological
    # weight, and skews the percentile-based risk statistics. Note this
    # means significant_forest_area_ha below will be slightly less than
    # /analyze's total forest_area_ha for the same city/year, since that
    # endpoint reports the raw unfiltered total.
    significant_mask = filter_min_mapping_unit(latest_mask, pixel_ha)

    latest_img = load_satellite(city, latest_year)
    heatmap = render_risk_heatmap(latest_img, ann_score, significant_mask)
    cv2.imwrite(os.path.join(STATIC_DIR, f"risk_{city}_{latest_year}.png"), heatmap)

    forest_scores = ann_score[significant_mask]
    forest_px = int(np.sum(significant_mask))

    if forest_px > 0:
        risk_cutoff = float(np.percentile(forest_scores, high_risk_percentile))
        high_risk_mask = (ann_score >= risk_cutoff) & significant_mask
        high_risk_px = int(np.sum(high_risk_mask))
    else:
        high_risk_px = 0

    return {
        "city": city,
        "year": latest_year,
        "risk_heatmap_url": f"{BASE_URL}/risk_{city}_{latest_year}.png",
        "significant_forest_area_ha": round(forest_px * pixel_ha, 2),
        "high_risk_percentile":  high_risk_percentile,
        "high_risk_area_ha":     round(high_risk_px * pixel_ha, 2),
        "high_risk_pct_of_forest": round((high_risk_px / forest_px) * 100, 1) if forest_px > 0 else 0.0,
        "mean_relative_risk_score": round(float(np.mean(forest_scores)), 4) if forest_px > 0 else 0.0,
        "note": (
            "Risk score is a RELATIVE ranking within this city's forest "
            "(trained on raw 1-year transitions), not directly comparable "
            "to /predict's filtered multi-year Markov rate -- see endpoint "
            "docstring."
        ),
    }


@app.get("/conservation-priority/{city}")
def conservation_priority(city: str, min_patch_ha: float = 0.5, top_n: int = 10):
    """
    Ranked list of the top_n forest patches most warranting conservation
    attention, combining patch size with the CA-ANN's mean risk score.
    Turns detection + prediction into an actionable output. See
    compute_conservation_priority() for the scoring methodology.
    """
    latest_year, latest_mask, ann_score = _prepare_city_risk_context(city)
    pixel_ha = pixel_area_ha(CITIES[city]["buffer_km"])

    priority_patches = compute_conservation_priority(
        latest_mask.astype(np.uint8) * 255, ann_score, pixel_ha,
        min_patch_ha=min_patch_ha, top_n=top_n)

    return {
        "city": city,
        "year": latest_year,
        "min_patch_ha": min_patch_ha,
        "priority_patches": priority_patches,
    }



# ---------------------------------------------------------------------------
# Hansen Global Forest Change (GFC) external validation
#
# Everything validated so far checks our methodology against ITSELF --
# held-out years compared to our own earlier detections/projections.
# This section validates our FVI + per-image-Otsu detection methodology
# (the exact methodology /analyze uses) against an INDEPENDENT, widely
# cited external reference: Hansen et al. 2013 (Science 342:850-853),
# the standard benchmark product in global forest-change literature.
#
# See hansen_fetch.py's module docstring for the full methodology note
# on Hansen's gain-tracking limitation (stops in 2012) -- summarized
# again in compute_hansen_forest_mask below and in the endpoint response.
# ---------------------------------------------------------------------------

HANSEN_CANOPY_THRESHOLD_DEFAULT = 30.0  # % canopy cover; a common literature default for "forest"


def load_hansen_treecover(city: str) -> np.ndarray:
    path = os.path.join(STATIC_DIR, f"hansen_treecover_{city}.png")
    if not os.path.exists(path):
        raise HTTPException(404, f"No Hansen treecover data for {city}. Run hansen_fetch.py first.")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
        # GEE's getThumbURL, given "dimensions" as a single int, scales to
        # that as a MAXIMUM while preserving the region's native aspect
        # ratio -- it does not force an exact square. This can differ
        # between source datasets even for the identical region (observed:
        # Hansen's static mosaic returned 1004x1024 where Sentinel-2
        # returns a clean 1024x1024 for the same city). Resize defensively
        # so pixel grids always align with our own masks for comparison.
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    return (img.astype(np.float32) / 255.0) * 100.0  # back to 0-100%


def load_hansen_lossyear(city: str) -> np.ndarray:
    path = os.path.join(STATIC_DIR, f"hansen_lossyear_{city}.png")
    if not os.path.exists(path):
        raise HTTPException(404, f"No Hansen lossyear data for {city}. Run hansen_fetch.py first.")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
        # Same defensive resize as load_hansen_treecover (see its comment
        # for why this can happen). NEAREST rather than LINEAR here:
        # lossyear values are effectively categorical (discrete year
        # buckets), and linear interpolation could invent spurious
        # blended values at boundaries between two different loss years.
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    return (img.astype(np.float32) / 255.0) * 25.0  # back to 0-25 (years since 2000)


def compute_hansen_forest_mask(treecover_pct: np.ndarray, lossyear: np.ndarray,
                                target_year: int,
                                canopy_threshold: float = HANSEN_CANOPY_THRESHOLD_DEFAULT
                                ) -> np.ndarray:
    """
    Reconstructs Hansen's implied forest mask for target_year:
        forest_2000 = treecover2000 >= canopy_threshold
        minus any pixel whose recorded loss year is <= (target_year - 2000)

    LIMITATION (see hansen_fetch.py's module docstring for the full
    explanation): Hansen's "gain" band, which tracks regrowth, only
    covers 2000-2012 and is deliberately NOT used here since it was
    never updated for later years. For any year after 2012 (i.e. every
    year this project uses), this reconstruction cannot see real
    regrowth that happened after 2012 -- the result is a systematic
    UNDER-estimate of true forest extent for recent years. This is
    expected, not a bug in this reconstruction.
    """
    forest_2000 = treecover_pct >= canopy_threshold
    years_since_2000 = target_year - 2000
    # lossyear==0 means "no loss ever recorded"; round to correct for the
    # small quantization error introduced by the 0-255 -> 0-25 PNG
    # round-trip (~0.1yr per grey level).
    lossyear_rounded = np.round(lossyear).astype(np.int32)
    lost_by_target = (lossyear_rounded > 0) & (lossyear_rounded <= years_since_2000)
    return forest_2000 & ~lost_by_target


def compute_validation_metrics(our_mask: np.ndarray, reference_mask: np.ndarray) -> dict:
    """
    Confusion-matrix-based validation of our_mask against reference_mask
    (Hansen, treated as ground truth) -- standard remote-sensing accuracy
    assessment: precision, recall, F1, overall accuracy, Cohen's Kappa.

    NOTE: this STRICT pixel-exact metric can look misleadingly harsh when
    comparing two detection methods with different spatial GRANULARITY --
    see compute_buffered_agreement below, and the /validate-hansen
    endpoint docstring, for why both are reported together here.
    """
    our_bool = our_mask.astype(bool)
    ref_bool = reference_mask.astype(bool)

    tp = int(np.sum(our_bool & ref_bool))
    fp = int(np.sum(our_bool & ~ref_bool))
    fn = int(np.sum(~our_bool & ref_bool))
    tn = int(np.sum(~our_bool & ~ref_bool))
    total = tp + fp + fn + tn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    p_o = accuracy
    p_our_forest = (tp + fp) / total if total > 0 else 0.0
    p_ref_forest = (tp + fn) / total if total > 0 else 0.0
    p_e = p_our_forest * p_ref_forest + (1 - p_our_forest) * (1 - p_ref_forest)
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 1e-9 else 0.0

    return {
        "true_positive_px":  tp, "false_positive_px": fp,
        "false_negative_px": fn, "true_negative_px":  tn,
        "precision":        round(precision, 4),
        "recall":           round(recall, 4),
        "f1_score":         round(f1, 4),
        "overall_accuracy": round(accuracy, 4),
        "kappa":            round(float(kappa), 4),
    }


def compute_buffered_agreement(our_mask: np.ndarray, reference_mask: np.ndarray,
                                pixel_side_m: float, buffer_m: float = 30.0) -> dict:
    """
    Positional-tolerance ("buffered") accuracy assessment: standard
    practice in remote-sensing validation when comparing two detection
    methods with different spatial GRANULARITY, not just possible
    registration error (see e.g. buffer-based accuracy assessment
    literature for land-cover change products). Rather than requiring
    exact pixel-for-pixel agreement, this checks whether each dataset's
    detections fall within `buffer_m` of the other.

    WHY THIS IS NEEDED HERE (found during real testing, not a
    hypothetical concern): visual inspection of the fetched Hansen data
    showed a recognisable, correctly-located Chennai/Bengaluru coastline
    (ocean correctly masked, water-body outlines visible), ruling out a
    fetch/alignment bug -- but Hansen's classifier produces SPARSE,
    scattered, tree-level detections in this urban context (consistent
    with published findings that global tree-cover products underestimate
    fragmented/scattered urban tree cover), while our FVI method produces
    DENSE, contiguous patch-level detections. A strict pixel-exact
    confusion matrix penalises this granularity difference heavily even
    where the two datasets broadly agree on WHERE forest exists -- e.g.
    Chennai 2025 strict precision was ~3% despite the two datasets'
    general spatial patterns visibly corresponding. Buffered assessment
    answers the more meaningful question: "does Hansen's sparse detection
    fall near/within what we detect, and vice versa" -- at a tolerance
    (default: Hansen's own native ~30m resolution) that's a principled,
    citable choice rather than an arbitrary one.

    Returns BOTH directions:
      buffered_recall    -- fraction of Hansen's detected pixels that
                             fall within buffer_m of OUR detected forest
      buffered_precision -- fraction of OUR detected pixels that fall
                             within buffer_m of a Hansen-detected pixel
    """
    buffer_px = max(1, int(round(buffer_m / pixel_side_m)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * buffer_px + 1, 2 * buffer_px + 1))

    our_u8 = our_mask.astype(np.uint8) * 255
    ref_u8 = reference_mask.astype(np.uint8) * 255

    our_dilated = cv2.dilate(our_u8, kernel) > 127
    ref_dilated = cv2.dilate(ref_u8, kernel) > 127

    ref_px = int(np.sum(reference_mask))
    our_px = int(np.sum(our_mask))

    buffered_recall    = (float(np.sum(reference_mask & our_dilated)) / ref_px) if ref_px > 0 else 0.0
    buffered_precision = (float(np.sum(our_mask & ref_dilated)) / our_px) if our_px > 0 else 0.0
    buffered_f1 = (
        2 * buffered_precision * buffered_recall / (buffered_precision + buffered_recall)
        if (buffered_precision + buffered_recall) > 0 else 0.0
    )

    return {
        "buffer_m":           buffer_m,
        "buffered_precision": round(buffered_precision, 4),
        "buffered_recall":    round(buffered_recall, 4),
        "buffered_f1":        round(buffered_f1, 4),
    }


@app.get("/validate-hansen/{city}/{year}")
def validate_hansen(city: str, year: str,
                    canopy_threshold: float = HANSEN_CANOPY_THRESHOLD_DEFAULT,
                    buffer_m: float = 30.0):
    """
    Validates our FVI + per-image-Otsu detection methodology (the SAME
    methodology /analyze uses) against Hansen Global Forest Change -- an
    independent, peer-reviewed, widely cited external reference dataset
    (Hansen et al. 2013, Science). Every other validation in this project
    (validate_prediction, spatial_ranking_validation,
    multiyear_projection_validation) checks our methodology against
    ITSELF; this is the one check against an INDEPENDENT source.

    Reports TWO complementary metrics:
      - validation_metrics: strict pixel-exact confusion matrix
      - buffered_agreement: positional-tolerance agreement at `buffer_m`
        (default 30m, Hansen's own native resolution)

    Both are reported because they answer different questions and neither
    alone is the "correct" one to cite -- strict pixel-exact accuracy is
    the standard remote-sensing metric, but Hansen's classifier produces
    sparse, scattered, tree-level detections in this urban context while
    our FVI method produces dense, contiguous patch-level detections
    (see compute_buffered_agreement's docstring for the full reasoning
    and how this was diagnosed during testing). Buffered agreement is the
    more appropriate comparison given that granularity mismatch.

    canopy_threshold: % canopy cover Hansen's treecover2000 must meet to
    count as forest baseline (default 30%, a common literature choice --
    adjustable here since published studies vary, commonly using
    10/25/30/50%; the FAO's 2015 Forest Resources Assessment uses 10%).

    IMPORTANT CAVEAT: Hansen's regrowth ("gain") tracking stops in 2012,
    so its reference for any later year systematically UNDERESTIMATES
    true forest extent. Our FVI mask reporting MORE forest than Hansen
    is a partially EXPECTED direction of disagreement for this reason,
    not necessarily an error in our own methodology -- see
    compute_hansen_forest_mask's docstring and hansen_fetch.py for the
    full explanation.
    """
    if city not in CITIES:
        raise HTTPException(400, f"Unknown city: {city}")

    img_rgb = load_satellite(city, year)
    nir     = load_nir(city, year)
    our_mask, _ = compute_fvi_mask(img_rgb, nir)  # per-image Otsu, same as /analyze
    our_bool = our_mask > 127

    treecover = load_hansen_treecover(city)
    lossyear  = load_hansen_lossyear(city)
    hansen_mask = compute_hansen_forest_mask(treecover, lossyear, int(year), canopy_threshold)

    metrics = compute_validation_metrics(our_bool, hansen_mask)

    pixel_ha = pixel_area_ha(CITIES[city]["buffer_km"])
    pixel_side_m = (pixel_ha * 10000) ** 0.5
    buffered = compute_buffered_agreement(our_bool, hansen_mask, pixel_side_m, buffer_m)

    our_area_ha    = round(int(np.sum(our_bool)) * pixel_ha, 2)
    hansen_area_ha = round(int(np.sum(hansen_mask)) * pixel_ha, 2)

    # Same colour scheme as /compare's diff map, reused for visual consistency.
    diff_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    both        = our_bool & hansen_mask
    only_ours   = our_bool & ~hansen_mask
    only_hansen = ~our_bool & hansen_mask
    diff_img[both]        = [58, 107,  26]   # both agree -- green
    diff_img[only_ours]   = [ 0, 180, 232]   # we say forest, Hansen doesn't -- gold
    diff_img[only_hansen] = [43,  57, 192]   # Hansen says forest, we don't -- red

    diff_path = os.path.join(STATIC_DIR, f"hansen_diff_{city}_{year}.png")
    cv2.imwrite(diff_path, diff_img)

    return {
        "city": city,
        "year": year,
        "canopy_threshold_pct":   canopy_threshold,
        "our_forest_area_ha":     our_area_ha,
        "hansen_forest_area_ha":  hansen_area_ha,
        "diff_url": f"{BASE_URL}/hansen_diff_{city}_{year}.png",
        "validation_metrics": metrics,
        "buffered_agreement": buffered,
        "methodology_note": (
            "Hansen's classifier produces sparse, scattered, tree-level "
            "detections in this dense urban context (visually confirmed: "
            "the fetched data shows a correctly-located coastline and "
            "water-body outlines, ruling out a fetch/alignment error -- "
            "the sparsity is a genuine, documented characteristic of "
            "global tree-cover products in fragmented urban settings), "
            "while our FVI method detects dense, contiguous patches. This "
            "granularity mismatch makes strict pixel-exact precision/"
            "recall look misleadingly harsh even where the two datasets "
            "broadly agree on WHERE forest is located -- buffered_agreement "
            "(positional tolerance at Hansen's own native ~30m resolution) "
            "is the more appropriate comparison for that reason."
        ),
        "caveat": (
            "Hansen's forest-GAIN tracking only covers 2000-2012 and was "
            "never updated in later dataset releases -- its reference for "
            "any year after 2012 cannot see real post-2012 regrowth, making "
            "it a systematic UNDERESTIMATE of true forest extent for every "
            "year this project uses (2016-2025). Our detection reporting "
            "MORE forest than Hansen is a partially expected direction of "
            "disagreement for this reason, not necessarily a detection "
            "error -- see Hansen et al. 2013 (Science 342:850-853) and "
            "hansen_fetch.py for the full methodology note."
        ),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)