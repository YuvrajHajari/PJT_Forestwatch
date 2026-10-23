"""clay_patch_token_experiment.py -- Quick diagnostic: does Clay's full
32x32 patch-token grid per tile (one token per 8x8 pixel block) beat the
current production approach (a single CLS token per 256x256 tile,
broadcast to all 65,536 pixels in it)?

Scoped to Bengaluru ONLY, and trained on Bengaluru-only walk-forward
splits (not main.py's normal cross-city pooled training), so the
comparison is strictly apples-to-apples: same training data for every
configuration, only the Clay feature representation changes. This is a
narrower, faster diagnostic than train_ca_ann.py's full harness, same
spirit as clay_dim_experiment.py's single-holdout pattern but using every
available Bengaluru walk-forward test point (not just the latest one) for
a sturdier signal while staying fast.

Requires static/clay_patch_raw/bengaluru_*.npy to exist first --
run extract_clay_patch_tokens.py.

Usage:
    python clay_patch_token_experiment.py
"""
import os
import sys
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import main as fw
import train_ca_ann as tca

CITY = "bengaluru"
STATIC_DIR = fw.STATIC_DIR
PATCH_RAW_DIR = os.path.join(STATIC_DIR, "clay_patch_raw")
CLS_RAW_DIR = os.path.join(STATIC_DIR, "clay_raw")
TILE_SIZE = 256
N_COMPONENTS = 8  # match production's N_PCA_COMPONENTS for a fair comparison

_patch_cache, _cls_cache = {}, {}
_pca_cache = {}


def patch_grids():
    if not _patch_cache:
        for f in os.listdir(PATCH_RAW_DIR):
            if f.startswith(CITY) and f.endswith(".npy"):
                _patch_cache[f[:-4]] = np.load(os.path.join(PATCH_RAW_DIR, f))
    return _patch_cache


def cls_grids():
    if not _cls_cache:
        for f in os.listdir(CLS_RAW_DIR):
            if f.startswith(CITY) and f.endswith(".npy"):
                _cls_cache[f[:-4]] = np.load(os.path.join(CLS_RAW_DIR, f))
    return _cls_cache


def get_pca(key, grids_fn):
    if key not in _pca_cache:
        grids = grids_fn()
        pooled = np.concatenate([g.reshape(-1, g.shape[-1]) for g in grids.values()], axis=0)
        pca = PCA(n_components=N_COMPONENTS, random_state=0)
        pca.fit(pooled)
        print(f"  PCA({N_COMPONENTS}) fit on {pooled.shape[0]} pooled vectors "
              f"({key}), explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        _pca_cache[key] = pca
    return _pca_cache[key]


def make_features_fn(mode):
    """mode: 'none', 'cls' (current production approach), 'patch' (new,
    finer-grained token-grid approach). Falls back to hand-engineered-only
    features for any non-Bengaluru city, since this diagnostic only
    extracted Clay data for Bengaluru."""
    def fn(mask_bool, city, year):
        base = fw.multiscale_features(mask_bool)
        if mode == "none" or city != CITY:
            return base
        h, w = mask_bool.shape
        if mode == "cls":
            grid = cls_grids()[f"{city}_{year}"]
            pca = get_pca("cls", cls_grids)
            n_side = grid.shape[0]
            reduced = pca.transform(grid.reshape(-1, grid.shape[-1])).astype(np.float32)
            reduced_grid = reduced.reshape(n_side, n_side, N_COMPONENTS)
            broadcast = np.repeat(np.repeat(reduced_grid, TILE_SIZE, axis=0), TILE_SIZE, axis=1)
        else:  # patch
            grid = patch_grids()[f"{city}_{year}"]  # (128, 128, 1024)
            pca = get_pca("patch", patch_grids)
            n_side = grid.shape[0]
            reduced = pca.transform(grid.reshape(-1, grid.shape[-1])).astype(np.float32)
            reduced_grid = reduced.reshape(n_side, n_side, N_COMPONENTS)
            factor = h // n_side  # 1024/128 = 8
            broadcast = np.repeat(np.repeat(reduced_grid, factor, axis=0), factor, axis=1)
        broadcast = broadcast[:h, :w]
        return np.concatenate([base, broadcast.reshape(-1, N_COMPONENTS)], axis=1)
    return fn


def bengaluru_only_splits(masks, n_test_points=4):
    """Same walk-forward logic as train_ca_ann.make_walkforward_splits,
    restricted to Bengaluru's own consecutive transitions for BOTH the
    test points and the training pool, so every configuration in this
    diagnostic trains on identical data."""
    city_years = sorted({y for (c, y) in masks if c == CITY}, key=int)
    transitions = [(CITY, a, b) for a, b in zip(city_years[:-1], city_years[1:])]
    n_test = min(n_test_points, max(0, len(transitions) - 2))
    test_points = transitions[-n_test:] if n_test > 0 else []
    splits = []
    for test_t in test_points:
        ya = test_t[1]
        train_pool = [t for t in transitions if t[2] <= ya]
        splits.append((test_t, train_pool))
    return splits


def run_config(name, mode, masks, splits):
    orig_fn = fw.multiscale_features_v2
    fw.multiscale_features_v2 = make_features_fn(mode)
    try:
        t0 = time.time()
        deltas = []
        print(f"\n--- {name} ---")
        for test_t, train_pool in splits:
            X_train, y_train = tca.build_training_set(masks, train_pool)
            model, beta = tca.train_mlp(X_train, y_train)
            scores = tca.evaluate_split(model, beta, masks, test_t)
            city, ya, yb = test_t
            old_k, ann_k = scores["old"][1], scores["ca_ann"][1]
            delta = ann_k - old_k
            deltas.append((ann_k, delta))
            print(f"  {city} {ya}->{yb}: old={old_k:.4f}  ca_ann={ann_k:.4f}  delta={delta:+.4f}")
        mean_kappa = float(np.mean([k for k, _ in deltas]))
        mean_delta = float(np.mean([d for _, d in deltas]))
        print(f"  mean ca_ann kappa: {mean_kappa:.4f}   mean delta vs baseline: {mean_delta:+.4f}   ({time.time()-t0:.0f}s)")
        return mean_kappa, mean_delta
    finally:
        fw.multiscale_features_v2 = orig_fn


def main():
    print("Loading cached masks...")
    masks = tca.load_all_masks()
    splits = bengaluru_only_splits(masks)
    print(f"Bengaluru-only walk-forward test points: {[t for t, _ in splits]}")

    results = {}
    results["no_clay (pre-Clay baseline)"] = run_config("no_clay (pre-Clay baseline)", "none", masks, splits)
    results["clay_cls_8 (current production, tile-level)"] = run_config("clay_cls_8 (current production, tile-level)", "cls", masks, splits)
    results["clay_patch_8 (new, 8x8-pixel-block-level)"] = run_config("clay_patch_8 (new, 8x8-pixel-block-level)", "patch", masks, splits)

    print("\n=== Summary (Bengaluru-only walk-forward, 1-year horizon) ===")
    for name, (kappa, delta) in results.items():
        print(f"  {name:42s}  mean kappa={kappa:.4f}  delta vs baseline={delta:+.4f}")


if __name__ == "__main__":
    main()
