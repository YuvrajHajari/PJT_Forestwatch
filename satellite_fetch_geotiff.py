import os
import ee
import requests
from dotenv import load_dotenv

load_dotenv()

GEE_PROJECT = os.environ.get("GEE_PROJECT")

# NOTE: unchanged from the original -- still needs de-duplicating into a
# shared config file (cities_config.py, from the multi-city expansion).
CITIES = {
    "chennai":   {"lat": 13.0827, "lon": 80.2707, "buffer_km": 6},
    "bengaluru": {"lat": 12.9716, "lon": 77.5946, "buffer_km": 6},
}

YEARS = list(range(2016, 2026))

# IMAGE_SIZE kept for continuity with the existing pipeline (still governs
# the requested pixel dimensions, still ~11.7m/pixel at buffer_km=6).
IMAGE_SIZE = 1024

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def init():
    if not GEE_PROJECT:
        raise RuntimeError(
            "GEE_PROJECT is not set. Copy .env.example to .env and fill in "
            "your Google Earth Engine project ID."
        )
    ee.Initialize(project=GEE_PROJECT)
    print("\u2705 GEE initialized")


def _region_for(cfg):
    point = ee.Geometry.Point([cfg["lon"], cfg["lat"]])
    return point.buffer(cfg["buffer_km"] * 1000).bounds()  # bounds() -> real rectangular AOI, matches getDownloadURL's expectations


def fetch_rgb_geotiff(city_key, cfg, year):
    """
    Real, unclipped Sentinel-2 SR reflectance (R,G,B), as a GeoTIFF --
    replaces the old getThumbURL() PNG path. No min/max clip, no gamma,
    no 8-bit flattening: this is the actual SR product, scaled by 10000
    per Sentinel-2's own convention (see raster_io.py's docstring).
    """
    region = _region_for(cfg)
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(f"{year}-01-01", f"{year}-05-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .select(["B4", "B3", "B2"])  # R, G, B -- order preserved through to the GeoTIFF bands
        .median()
    )
    url = collection.getDownloadURL({
        "region": region,
        "dimensions": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        "format": "GEO_TIFF",
    })
    r = requests.get(url, timeout=120)  # real exports are larger than thumbnails; longer timeout
    r.raise_for_status()
    return r.content


def fetch_nir_geotiff(city_key, cfg, year):
    """Real Sentinel-2 B8 (NIR) reflectance, as a single-band GeoTIFF."""
    region = _region_for(cfg)
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(f"{year}-01-01", f"{year}-05-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .select(["B8"])
        .median()
    )
    url = collection.getDownloadURL({
        "region": region,
        "dimensions": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        "format": "GEO_TIFF",
    })
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def fetch_all():
    init()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for city_key, cfg in CITIES.items():
        for year in YEARS:
            rgb_out = os.path.join(OUTPUT_DIR, f"satellite_{city_key}_{year}.tif")
            nir_out = os.path.join(OUTPUT_DIR, f"nir_{city_key}_{year}.tif")

            if not os.path.exists(rgb_out):
                print(f"\U0001f4e1 {city_key} {year} RGB (GeoTIFF)...")
                try:
                    tif_bytes = fetch_rgb_geotiff(city_key, cfg, year)
                    with open(rgb_out, "wb") as f:
                        f.write(tif_bytes)
                    print(f"  \u2705 saved ({len(tif_bytes)/1024:.0f} KB)")
                except Exception as e:
                    print(f"  \u274c {e}")
            else:
                print(f"  skip {city_key} {year} RGB (exists)")

            if not os.path.exists(nir_out):
                print(f"\U0001f4e1 {city_key} {year} NIR (B8, GeoTIFF)...")
                try:
                    tif_bytes = fetch_nir_geotiff(city_key, cfg, year)
                    with open(nir_out, "wb") as f:
                        f.write(tif_bytes)
                    print(f"  \u2705 saved ({len(tif_bytes)/1024:.0f} KB)")
                except Exception as e:
                    print(f"  \u274c {e}")
            else:
                print(f"  skip {city_key} {year} NIR (exists)")


if __name__ == "__main__":
    fetch_all()