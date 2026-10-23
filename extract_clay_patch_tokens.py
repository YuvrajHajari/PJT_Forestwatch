"""extract_clay_patch_tokens.py -- Diagnostic extraction: instead of Clay's
single CLS token per 256x256 tile (the current production approach, one
vector broadcast to all 65,536 pixels in that tile), this pulls the FULL
grid of patch tokens Clay's encoder produces internally (32x32 tokens per
256x256 tile, i.e. one token per 8x8 pixel block), to test whether the
current CLS-only approach is leaving real per-pixel spatial signal on the
table.

Scoped to ONE city (Bengaluru, the only city where Clay's CLS-only
embeddings currently show a validated benefit) to keep this a quick,
cheap diagnostic rather than a multi-hour full-dataset run. No GPU
available on this machine, confirmed via torch.cuda.is_available() ==
False, and a single-patch forward pass measured at ~10-13s on this CPU,
so this script's wall-clock is dominated by n_city_years * 16 tiles *
~12s/tile.

Caches the RAW (128, 128, 1024) per-city-year token grid to
static/clay_patch_raw/ -- 4x4 tiles of 32x32 tokens each, NOT reduced yet.
A separate script (clay_patch_token_experiment.py) pools these, fits PCA,
and runs the same quick single-holdout diagnostic pattern already used in
clay_dim_experiment.py.

Usage:
    python extract_clay_patch_tokens.py
"""
import os
import sys
import time
import datetime
import numpy as np
import torch

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import raster_io
import clay_features as cf
from satellite_fetch_geotiff import CITIES

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
OUT_DIR = os.path.join(STATIC_DIR, "clay_patch_raw")

IMAGE_SIZE = 1024
TILE_SIZE = cf.PATCH_SIZE  # 256
N_TILES_SIDE = IMAGE_SIZE // TILE_SIZE  # 4
FETCH_MONTH_MID = 3

TARGET_CITY = "bengaluru"


def find_years(city):
    years = []
    prefix, suffix = f"satellite_{city}_", ".tif"
    for f in sorted(os.listdir(STATIC_DIR)):
        if f.startswith(prefix) and f.endswith(suffix):
            year = f[len(prefix):-len(suffix)]
            if os.path.exists(os.path.join(STATIC_DIR, f"nir_{city}_{year}.tif")):
                years.append(year)
    return sorted(years)


def extract_patch_token_grid(rgb_patch, nir_patch, lat, lon, date, gsd):
    """Like clay_features.extract_patch_embedding, but returns the full
    32x32x1024 patch-token grid instead of just the CLS token."""
    patch = np.concatenate(
        [np.transpose(rgb_patch, (2, 0, 1)), nir_patch[np.newaxis, :, :]], axis=0
    )
    model = cf.load_clay()
    datacube = cf.build_datacube(patch, lat, lon, date, gsd)
    with torch.no_grad():
        unmsk_patch, unmsk_idx, msk_idx, msk_matrix = model.model.encoder(datacube)
    tokens = unmsk_patch[:, 1:, :].squeeze(0).cpu().numpy()  # drop CLS, keep 1024 patch tokens
    n_side = int(round(tokens.shape[0] ** 0.5))  # 32
    return tokens.reshape(n_side, n_side, tokens.shape[-1])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    years = find_years(TARGET_CITY)
    print(f"{TARGET_CITY}: {len(years)} fetched years: {years}")

    cfg = CITIES[TARGET_CITY]
    gsd = (2 * cfg["buffer_km"] * 1000) / IMAGE_SIZE

    t_start = time.time()
    for yi, year in enumerate(years):
        out_path = os.path.join(OUT_DIR, f"{TARGET_CITY}_{year}.npy")
        if os.path.exists(out_path):
            print(f"  skip {TARGET_CITY} {year} (already extracted)")
            continue

        print(f"[{yi+1}/{len(years)}] Extracting patch tokens: {TARGET_CITY} {year}...")
        rgb, meta = raster_io.load_satellite(TARGET_CITY, year, STATIC_DIR)
        nir, _ = raster_io.load_nir(TARGET_CITY, year, STATIC_DIR)
        date = datetime.datetime(int(year), FETCH_MONTH_MID, 1)

        token_side = None
        full_grid = None
        t0 = time.time()
        for row in range(0, IMAGE_SIZE - TILE_SIZE + 1, TILE_SIZE):
            for col in range(0, IMAGE_SIZE - TILE_SIZE + 1, TILE_SIZE):
                rgb_patch = rgb[row:row+TILE_SIZE, col:col+TILE_SIZE, :]
                nir_patch = nir[row:row+TILE_SIZE, col:col+TILE_SIZE]
                lat, lon = raster_io.pixel_latlon(meta, row + TILE_SIZE // 2, col + TILE_SIZE // 2)
                tile_tokens = extract_patch_token_grid(rgb_patch, nir_patch, lat, lon, date, gsd)

                if full_grid is None:
                    token_side = tile_tokens.shape[0]
                    full_side = token_side * N_TILES_SIDE
                    full_grid = np.zeros((full_side, full_side, tile_tokens.shape[-1]), dtype=np.float32)

                tr, tc = row // TILE_SIZE, col // TILE_SIZE
                full_grid[tr*token_side:(tr+1)*token_side, tc*token_side:(tc+1)*token_side] = tile_tokens

        np.save(out_path, full_grid)
        elapsed = time.time() - t0
        total_elapsed = time.time() - t_start
        print(f"  saved {out_path}  shape={full_grid.shape}  ({elapsed:.0f}s this year, {total_elapsed/60:.1f}min total so far)")

    print("\nDone.")


if __name__ == "__main__":
    main()
