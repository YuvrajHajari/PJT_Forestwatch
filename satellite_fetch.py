import os
import ee
import requests
import numpy as np
import cv2
from dotenv import load_dotenv

load_dotenv()

GEE_PROJECT = os.environ.get("GEE_PROJECT")

# Cities are hardcoded for now (dropdown-based city selection) -- geocoding
# to support arbitrary city names is a planned later upgrade, not built yet.
#
# NOTE: this dict must stay in sync with the CITIES dict in main.py.
# De-duplicating into a single shared config file is a known roadmap item.
CITIES = {
    "chennai":   {"lat": 13.0827, "lon": 80.2707, "buffer_km": 6},
    "bengaluru": {"lat": 12.9716, "lon": 77.5946, "buffer_km": 6},
}

YEARS = list(range(2016, 2026))  # annual, 2016-2025 -- was every 3rd year (2016, 2019, 2022, 2025)

# IMAGE_SIZE must match main.py's IMAGE_SIZE.
#
# Sentinel-2 native ground resolution is 10m/pixel. With buffer_km=6 (a 12km
# x 12km region), requesting 1024x1024 gives ~11.7m/pixel -- close to native,
# so VARI can resolve small parks/tree clusters instead of blurring them
# together. Going higher doesn't add real detail (GEE would just be
# interpolating past the sensor's actual resolution), it only slows the
# fetch and increases file size for no accuracy gain.
IMAGE_SIZE = 1024

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def init():
    if not GEE_PROJECT:
        raise RuntimeError(
            "GEE_PROJECT is not set. Copy .env.example to .env and fill in "
            "your Google Earth Engine project ID."
        )
    ee.Initialize(project=GEE_PROJECT)
    print("✅ GEE initialized")


def fetch_image(city_key, cfg, year):
    point = ee.Geometry.Point([cfg["lon"], cfg["lat"]])
    region = point.buffer(cfg["buffer_km"] * 1000)
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(f"{year}-01-01", f"{year}-05-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .select(["B4", "B3", "B2"])
        .median()
    )
    url = collection.getThumbURL({
        "region": region, "dimensions": IMAGE_SIZE,
        "format": "png", "min": 0, "max": 3000, "gamma": 1.4,
    })
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fetch_nir(city_key, cfg, year):
    """
    Fetches Sentinel-2's B8 (near-infrared) band as a single-channel
    grayscale image, same region/composite window as fetch_image().

    Needed for NDWI water masking: water absorbs NIR strongly (dark in this
    band) while vegetation reflects it strongly (bright), which is a much
    more reliable land/water separator than anything derivable from RGB
    alone -- RGB-only water misclassification under VARI is exactly the
    coastal bug this exists to fix. Same band also usable later for a real
    NDVI index (roadmap item), not just water masking.
    """
    point = ee.Geometry.Point([cfg["lon"], cfg["lat"]])
    region = point.buffer(cfg["buffer_km"] * 1000)
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(f"{year}-01-01", f"{year}-05-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .select(["B8"])
        .median()
    )
    url = collection.getThumbURL({
        "region": region, "dimensions": IMAGE_SIZE,
        "format": "png", "min": 0, "max": 3000, "gamma": 1.0,
    })
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    arr = np.frombuffer(r.content, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def fetch_all():
    init()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for city_key, cfg in CITIES.items():
        for year in YEARS:
            rgb_out = os.path.join(OUTPUT_DIR, f"satellite_{city_key}_{year}.png")
            nir_out = os.path.join(OUTPUT_DIR, f"nir_{city_key}_{year}.png")

            if not os.path.exists(rgb_out):
                print(f"📡 {city_key} {year} RGB...")
                try:
                    img = fetch_image(city_key, cfg, year)
                    cv2.imwrite(rgb_out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    print(f"  ✅ saved")
                except Exception as e:
                    print(f"  ❌ {e}")
            else:
                print(f"  skip {city_key} {year} RGB (exists)")

            if not os.path.exists(nir_out):
                print(f"📡 {city_key} {year} NIR (B8)...")
                try:
                    nir = fetch_nir(city_key, cfg, year)
                    cv2.imwrite(nir_out, nir)
                    print(f"  ✅ saved")
                except Exception as e:
                    print(f"  ❌ {e}")
            else:
                print(f"  skip {city_key} {year} NIR (exists)")


if __name__ == "__main__":
    fetch_all()