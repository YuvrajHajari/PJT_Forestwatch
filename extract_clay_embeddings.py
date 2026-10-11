"""
extract_clay_embeddings.py -- Offline, one-time extraction of frozen Clay
v1.5 tile embeddings, PCA-reduced and cached to disk so main.py /
train_ca_ann.py never need to import torch/claymodel or touch the 5GB
checkpoint at request-serving time.

Two-pass design:
  Pass 1 (extract): runs Clay's frozen encoder once per 256x256 tile per
  fetched city/year GeoTIFF pair, caches the RAW 1024-dim embeddings per
  city/year to static/clay_raw/.
  Pass 2 (reduce): pools ALL cached raw embeddings across every city/year
  fetched so far, fits ONE PCA(n_components=N_PCA_COMPONENTS) on the pooled
  set, and writes out PCA-reduced per-tile grids (static/clay_reduced/) plus
  the fitted PCA transform itself (static/clay_pca.joblib) so it can be
  reapplied consistently to any future fetch.

WHY PCA to ~8 components, not the raw 1024 dims: with only ~18 independent
transitions to train the CA-ANN on (see main.py's
compute_annual_transition_rates docstring), concatenating 1024 raw Clay
dims onto the 4 existing hand-engineered features would swamp them and
almost certainly overfit inside a 16-unit first hidden layer. 8 components
keeps Clay's contribution the same order of magnitude as the existing
features -- an explicit, documented tradeoff, not a silent default.

NOTE: PCA is refit from scratch every time this script runs, using
whatever city/years are cached at that point -- rerun this after fetching
more years so the PCA basis reflects the full dataset, not a subset.

Usage:
    python extract_clay_embeddings.py
"""
import os
import sys
import datetime
import numpy as np
import joblib
from sklearn.decomposition import PCA

if sys.stdout.encoding != "utf-8":  # Windows cp1252 console can't print the emoji below
    sys.stdout.reconfigure(encoding="utf-8")

import raster_io
import clay_features as cf
from satellite_fetch_geotiff import CITIES  # shared city lat/lon/buffer_km

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
RAW_DIR = os.path.join(STATIC_DIR, "clay_raw")
REDUCED_DIR = os.path.join(STATIC_DIR, "clay_reduced")
PCA_PATH = os.path.join(STATIC_DIR, "clay_pca.joblib")

IMAGE_SIZE = 1024
TILE_SIZE = cf.PATCH_SIZE  # 256, matches Clay's tested chip size -> 4x4 = 16 tiles/image
N_PCA_COMPONENTS = 8

# Midpoint of the Jan-May compositing window satellite_fetch_geotiff.py uses
# for its median composite -- matches test_clay.py's convention.
FETCH_MONTH_MID = 3


def find_available_geotiffs():
    """(city, year) pairs that have both satellite_*.tif and nir_*.tif fetched."""
    pairs = []
    for city in CITIES:
        prefix, suffix = f"satellite_{city}_", ".tif"
        for f in sorted(os.listdir(STATIC_DIR)):
            if f.startswith(prefix) and f.endswith(suffix):
                year = f[len(prefix):-len(suffix)]
                if os.path.exists(os.path.join(STATIC_DIR, f"nir_{city}_{year}.tif")):
                    pairs.append((city, year))
    return sorted(pairs)


def extract_raw(pairs):
    os.makedirs(RAW_DIR, exist_ok=True)
    for city, year in pairs:
        out_path = os.path.join(RAW_DIR, f"{city}_{year}.npy")
        if os.path.exists(out_path):
            print(f"  skip {city} {year} (already extracted)")
            continue

        print(f"Extracting Clay embeddings: {city} {year}...")
        rgb, meta = raster_io.load_satellite(city, year, STATIC_DIR)
        nir, _ = raster_io.load_nir(city, year, STATIC_DIR)
        cfg = CITIES[city]
        gsd = (2 * cfg["buffer_km"] * 1000) / IMAGE_SIZE
        date = datetime.datetime(int(year), FETCH_MONTH_MID, 1)

        tile_dict = cf.extract_features_for_mask_grid(rgb, nir, meta, date, gsd, stride=TILE_SIZE)
        emb_dim = next(iter(tile_dict.values())).shape[0]
        n_side = IMAGE_SIZE // TILE_SIZE
        grid = np.zeros((n_side, n_side, emb_dim), dtype=np.float32)
        for (row, col), emb in tile_dict.items():
            grid[row // TILE_SIZE, col // TILE_SIZE] = emb

        np.save(out_path, grid)
        print(f"  saved {out_path}  shape={grid.shape}")


def fit_and_reduce(pairs):
    os.makedirs(REDUCED_DIR, exist_ok=True)
    grids = {(city, year): np.load(os.path.join(RAW_DIR, f"{city}_{year}.npy"))
             for city, year in pairs}

    pooled = np.concatenate([g.reshape(-1, g.shape[-1]) for g in grids.values()], axis=0)
    n_components = min(N_PCA_COMPONENTS, pooled.shape[0])
    if n_components < N_PCA_COMPONENTS:
        print(f"  WARNING: only {pooled.shape[0]} tiles pooled -- reducing PCA "
              f"components from {N_PCA_COMPONENTS} to {n_components}. Refit once "
              f"more city/years are fetched.")

    print(f"\nFitting PCA({n_components}) on {pooled.shape[0]} pooled tiles "
          f"({pooled.shape[1]}-dim each)...")
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(pooled)
    print(f"  explained variance ratio: {pca.explained_variance_ratio_.round(4)}")
    print(f"  total explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    joblib.dump(pca, PCA_PATH)
    print(f"  saved PCA transform to {PCA_PATH}")

    for (city, year), grid in grids.items():
        n_side = grid.shape[0]
        reduced = pca.transform(grid.reshape(-1, grid.shape[-1])).astype(np.float32)
        reduced_grid = reduced.reshape(n_side, n_side, n_components)
        out_path = os.path.join(REDUCED_DIR, f"{city}_{year}.npy")
        np.save(out_path, reduced_grid)
        print(f"  saved {out_path}  shape={reduced_grid.shape}")


def main():
    pairs = find_available_geotiffs()
    if not pairs:
        print("No fetched GeoTIFFs found in static/ -- run satellite_fetch_geotiff.py first.")
        return
    print(f"Found {len(pairs)} city/year GeoTIFF pairs.\n")

    extract_raw(pairs)
    fit_and_reduce(pairs)
    print(f"\nDone. multiscale_features_v2() reads PCA-reduced tile grids from {REDUCED_DIR}")


if __name__ == "__main__":
    main()
