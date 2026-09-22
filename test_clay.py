"""One-patch Clay smoke test. Run from the repo root:
    python test_clay.py
Needs: checkpoints/clay-v1.5.ckpt and configs/metadata.yaml downloaded per
clay_features.py's setup instructions, plus a real fetched .tif
(satellite_chennai_2025.tif / nir_chennai_2025.tif in static/).
"""
import datetime
import numpy as np

from raster_io import load_satellite, load_nir
import clay_features as cf

rgb, meta = load_satellite("chennai", "2025", "static")
nir, _ = load_nir("chennai", "2025", "static")
print("full image:", rgb.shape, nir.shape)

# one 256x256 patch from the top-left corner
patch = np.concatenate(
    [np.transpose(rgb[:256, :256, :], (2, 0, 1)), nir[np.newaxis, :256, :256]], axis=0
)
print("patch:", patch.shape, patch.dtype, "reflectance range:", patch.min(), patch.max())

from raster_io import pixel_latlon
lat, lon = pixel_latlon(meta, 128, 128)
print("patch center lat/lon:", lat, lon)

GSD = (2 * 6 * 1000) / 1024  # 6km buffer, 1024px fetch -> ~11.72 m/pixel
date = datetime.datetime(2025, 3, 1)  # matches the Jan-May fetch window

embedding = cf.extract_patch_embedding(patch, lat, lon, date, GSD)
print("embedding shape:", embedding.shape)
print("embedding[:10]:", embedding[:10])
print("embedding mean/std:", embedding.mean(), embedding.std())
