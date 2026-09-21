"""
raster_io.py -- Shared GeoTIFF read/write utility.

Replaces the PNG-based cv2.imread/imwrite calls scattered across
satellite_fetch.py, main.py, test_fvi.py, and train_ca_ann.py.

WHY THIS EXISTS (read before touching call sites):
satellite_fetch.py previously used ee.Image.getThumbURL(), which is a
VISUALIZATION endpoint: it clips reflectance to [0, 3000], applies a
gamma curve (1.4), and flattens to an 8-bit PNG. That means every
downstream computation -- FVI, Otsu thresholds, NDWI -- was running on
cosmetically-stretched display values, not real reflectance.

getDownloadURL(format='GEO_TIFF') instead returns:
  - Real Sentinel-2 SR reflectance, scaled by 10000 (i.e. a pixel value
    of 3000 means 0.30 reflectance, not "30% of a gamma-corrected 0-255
    display range"). This scaling is Sentinel-2's own SR product
    convention, not something we invented.
  - Actual per-pixel georeferencing (CRS + transform), so pixels have
    real lat/lon, not just an implicit grid.
  - Uncapped values -- bright surfaces (concrete, sand) can exceed 3000,
    which the old thumbnail pipeline silently clipped away.

Every function below returns float32 arrays in REFLECTANCE units
(roughly 0-1 for typical land surfaces, occasionally higher for very
bright targets) -- not uint8. Code written against the old 0-255 uint8
convention (FVI thresholds, brightness gates, index_to_uint8 mappings)
will need its constants re-derived against this new scale; that
migration is intentionally NOT done inside this file, since it touches
methodology (Otsu thresholds, brightness cutoffs) that deserves its own
validation pass rather than a silent unit change.
"""
import os
import numpy as np
import rasterio
from rasterio.transform import Affine

SR_SCALE = 10000.0  # Sentinel-2 SR Harmonized: pixel_value / 10000 = reflectance


def load_geotiff(path: str) -> tuple[np.ndarray, dict]:
    """
    Reads a GeoTIFF, returns (array, meta).

    array: float32, shape (bands, H, W) for multi-band or (H, W) for
    single-band, already divided by SR_SCALE into real reflectance units.

    meta: rasterio's profile dict (crs, transform, dtype, etc.) -- pass
    this straight to save_geotiff() if you need to write a derived
    product (e.g. a forest mask) that stays georeferenced.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32) / SR_SCALE
        meta = src.profile.copy()
    if arr.shape[0] == 1:
        arr = arr[0]
    return arr, meta


def load_satellite(city: str, year: str, static_dir: str) -> tuple[np.ndarray, dict]:
    """RGB reflectance, shape (H, W, 3), channel order R,G,B (NOT B,G,R --
    unlike the old cv2 pipeline, there is no BGR swap here; every caller
    that assumed cv2's BGR order needs updating, not this function)."""
    path = os.path.join(static_dir, f"satellite_{city}_{year}.tif")
    arr, meta = load_geotiff(path)  # (3, H, W)
    rgb = np.transpose(arr, (1, 2, 0))  # -> (H, W, 3)
    return rgb, meta


def load_nir(city: str, year: str, static_dir: str) -> tuple[np.ndarray, dict]:
    """NIR reflectance, shape (H, W), single band."""
    path = os.path.join(static_dir, f"nir_{city}_{year}.tif")
    return load_geotiff(path)


def save_geotiff(path: str, array: np.ndarray, meta: dict, dtype="float32"):
    """
    Writes array back out as a GeoTIFF, reusing the source's
    georeferencing (meta from load_geotiff/load_satellite/load_nir).

    array: (H, W) or (bands, H, W). If (H, W, bands) is passed (the
    orientation load_satellite returns), it's transposed automatically.
    """
    if array.ndim == 3 and array.shape[2] in (3, 4) and array.shape[0] != array.shape[2]:
        array = np.transpose(array, (2, 0, 1))  # (H,W,C) -> (C,H,W)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]

    out_meta = meta.copy()
    out_meta.update({
        "count": array.shape[0],
        "dtype": dtype,
        "driver": "GTiff",
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", **out_meta) as dst:
        dst.write(array.astype(dtype))


def pixel_latlon(meta: dict, row: int, col: int) -> tuple[float, float]:
    """Real lat/lon for a given pixel -- previously impossible, since PNGs
    carried no georeferencing at all. Useful for e.g. reporting a
    conservation-priority patch's actual map coordinates instead of only
    its pixel bounding box."""
    transform: Affine = meta["transform"]
    lon, lat = transform * (col, row)
    return lat, lon