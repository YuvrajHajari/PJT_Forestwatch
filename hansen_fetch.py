"""
hansen_fetch.py -- One-time fetch of Hansen Global Forest Change (GFC)
reference data for external ground-truth validation.

Unlike satellite_fetch.py (which pulls a NEW composite per city per
year), Hansen GFC is a single global dataset covering 2000-2025 via two
bands:
  - treecover2000: % canopy cover as of year 2000 (0-100)
  - lossyear: year a pixel's forest was lost, encoded as an offset from
    2000 (0 = no loss ever recorded, 1-25 = loss detected in 2001-2025)

So this script fetches ONE pair of images per city (not per year) -- any
year's forest state can be reconstructed from these two bands together
(see main.py's compute_hansen_forest_mask): start from the year-2000
baseline, then remove any pixel whose recorded loss year is at or before
the target year.

IMPORTANT LIMITATION (also documented in main.py's /validate-hansen
endpoint -- worth knowing before presenting these numbers): Hansen's
"gain" band, which tracks forest REGROWTH, only covers 2000-2012 and was
never updated in any later dataset release. It is deliberately NOT used
here. For any year after 2012 (i.e. every year this project uses,
2016-2025), the treecover2000-minus-loss reconstruction cannot see any
real regrowth that happened after 2012 -- meaning the Hansen reference
for recent years is a systematic UNDERESTIMATE of true forest extent.
If our FVI-detected forest area exceeds the Hansen reference, that is a
partially EXPECTED direction of disagreement, not necessarily a
detection error on our part. This is a well-known, widely-discussed
limitation of the Hansen product in the remote sensing literature
(see Hansen et al. 2013, Science 342(6160):850-853), not something
specific to this project.

Usage:
    python hansen_fetch.py

Produces (one pair per city, not per year):
    static/hansen_treecover_{city}.png  -- treecover2000, 0-100% -> 0-255
    static/hansen_lossyear_{city}.png   -- lossyear, 0-25 -> 0-255
"""
import os
import ee
import requests
import numpy as np
import cv2
from dotenv import load_dotenv

load_dotenv()

GEE_PROJECT = os.environ.get("GEE_PROJECT")

# Must match the CITIES dict in main.py / satellite_fetch.py.
CITIES = {
    "chennai":   {"lat": 13.0827, "lon": 80.2707, "buffer_km": 6},
    "bengaluru": {"lat": 12.9716, "lon": 77.5946, "buffer_km": 6},
}

# Must match main.py's IMAGE_SIZE -- fetching at the SAME region and
# dimensions as the existing Sentinel-2 imagery gives pixel-aligned
# output, so Hansen and our own masks can be compared pixel-for-pixel
# without a separate reprojection step.
IMAGE_SIZE = 1024

# Try the newest available release first, fall back to the previous
# well-established release if the newest asset isn't reachable (new GEE
# catalog releases occasionally have propagation delays or access quirks
# shortly after publication).
HANSEN_ASSET_CANDIDATES = [
    "UMD/hansen/global_forest_change_2025_v1_13",  # loss years through 2025
    "UMD/hansen/global_forest_change_2024_v1_12",  # loss years through 2024
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def init():
    if not GEE_PROJECT:
        raise RuntimeError(
            "GEE_PROJECT is not set. Copy .env.example to .env and fill in "
            "your Google Earth Engine project ID."
        )
    ee.Initialize(project=GEE_PROJECT)
    print("\u2705 GEE initialized")


def load_hansen_image():
    """Tries each candidate Hansen asset in order, returns the first that loads."""
    last_err = None
    for asset_id in HANSEN_ASSET_CANDIDATES:
        try:
            img = ee.Image(asset_id)
            _ = img.bandNames().getInfo()  # force evaluation to confirm it actually loads
            print(f"  using Hansen asset: {asset_id}")
            return img
        except Exception as e:
            last_err = e
            print(f"  {asset_id} unavailable ({e}), trying next candidate...")
    raise RuntimeError(f"No Hansen GFC asset could be loaded. Last error: {last_err}")


def fetch_hansen_bands(cfg, hansen_img):
    point = ee.Geometry.Point([cfg["lon"], cfg["lat"]])
    region = point.buffer(cfg["buffer_km"] * 1000)

    # IMPORTANT: "dimensions" is passed as a bare int (IMAGE_SIZE), matching
    # satellite_fetch.py's EXACT parameter pattern for Sentinel-2 -- not an
    # explicit "WxH" string, and without an explicit "scale". An earlier
    # version of this function used the "WxH" string (to force an exact
    # square output) and later a "scale"-only variant (to try fixing a
    # near-zero-values symptom) -- testing against real data showed BOTH
    # of those choices caused genuinely poor precision/recall against our
    # own Sentinel-2-derived masks (~3% precision, ~20% recall even at a
    # threshold Hansen confidently detects), consistent with GEE applying
    # a DIFFERENT internal rendering/projection path for those parameter
    # forms than it does for satellite_fetch.py's plain-int form -- i.e. a
    # genuine pixel-grid MISALIGNMENT between the two datasets, not a real
    # accuracy problem. Matching the exact working pattern is the fix: it
    # gives GEE the best chance of applying the SAME default rendering
    # behaviour to both fetches. Any resulting shape mismatch (e.g.
    # 1004x1024 instead of exactly 1024x1024, as seen previously) is
    # handled by the defensive resize already present in main.py's
    # load_hansen_treecover / load_hansen_lossyear.
    treecover_url = hansen_img.select(["treecover2000"]).getThumbURL({
        "region": region, "dimensions": IMAGE_SIZE,
        "format": "png", "min": 0, "max": 100,
    })
    lossyear_url = hansen_img.select(["lossyear"]).getThumbURL({
        "region": region, "dimensions": IMAGE_SIZE,
        "format": "png", "min": 0, "max": 25,
    })

    r1 = requests.get(treecover_url, timeout=60)
    r1.raise_for_status()
    treecover = cv2.imdecode(np.frombuffer(r1.content, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    r2 = requests.get(lossyear_url, timeout=60)
    r2.raise_for_status()
    lossyear = cv2.imdecode(np.frombuffer(r2.content, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    return treecover, lossyear


def fetch_all():
    init()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    hansen_img = load_hansen_image()

    for city_key, cfg in CITIES.items():
        tc_out = os.path.join(OUTPUT_DIR, f"hansen_treecover_{city_key}.png")
        ly_out = os.path.join(OUTPUT_DIR, f"hansen_lossyear_{city_key}.png")

        if os.path.exists(tc_out) and os.path.exists(ly_out):
            print(f"  skip {city_key} (Hansen data already fetched)")
            continue

        print(f"\U0001F4E1 Fetching Hansen GFC bands for {city_key}...")
        try:
            treecover, lossyear = fetch_hansen_bands(cfg, hansen_img)
            cv2.imwrite(tc_out, treecover)
            cv2.imwrite(ly_out, lossyear)
            print(f"  \u2705 saved")
        except Exception as e:
            print(f"  \u274c {e}")


if __name__ == "__main__":
    fetch_all()