"""
clay_dim_experiment.py -- Quick, cheap diagnostic: does the CA-ANN benefit
from Clay's embedding when using FEWER PCA dimensions and/or a stronger
regularization penalty, compared to the current production default
(8 components, sklearn's default alpha=0.0001)?

NOT a replacement for train_ca_ann.py's full walk-forward validation
(19 model fits, ~3 hours). This uses a SINGLE held-out test per city (the
most recent year-pair, e.g. 2024->2025) -- same single-holdout pattern
main.py's validate_prediction() already uses elsewhere in this project --
so only 2 model fits per configuration, not 8-19.

Also uses a much larger MLP batch_size (4096, vs sklearn's default
min(200, n_samples)) purely for wall-clock speed in this diagnostic --
applied uniformly across every config, so the comparison between configs
stays fair even though absolute Kappa may differ slightly from
train_ca_ann.py's production run.

Touches NO production file: reads only the already-cached raw Clay
embeddings (static/clay_raw/*.npy), fits scratch PCA objects in memory,
and monkeypatches fw.multiscale_features_v2 for the duration of each
config's evaluation only (restored immediately after).

Usage:
    python clay_dim_experiment.py
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

RAW_DIR = os.path.join(fw.STATIC_DIR, "clay_raw")
TILE_SIZE = 256
BATCH_SIZE = 4096  # speed-only choice for this diagnostic, see module docstring

_raw_cache = None
_pca_cache = {}


def raw_grids():
    global _raw_cache
    if _raw_cache is None:
        _raw_cache = {
            f[:-4]: np.load(os.path.join(RAW_DIR, f))
            for f in os.listdir(RAW_DIR) if f.endswith(".npy")
        }
    return _raw_cache


def get_pca(n_components):
    if n_components not in _pca_cache:
        grids = raw_grids()
        pooled = np.concatenate([g.reshape(-1, g.shape[-1]) for g in grids.values()], axis=0)
        pca = PCA(n_components=n_components, random_state=0)
        pca.fit(pooled)
        _pca_cache[n_components] = pca
    return _pca_cache[n_components]


def make_features_fn(n_clay):
    """Returns a fw.multiscale_features_v2-compatible function using n_clay
    PCA components (0 = no Clay at all, the pre-Clay baseline)."""
    def fn(mask_bool, city, year):
        base = fw.multiscale_features(mask_bool)
        if n_clay == 0:
            return base
        grid = raw_grids()[f"{city}_{year}"]
        pca = get_pca(n_clay)
        n_side = grid.shape[0]
        reduced = pca.transform(grid.reshape(-1, grid.shape[-1])).astype(np.float32)
        reduced_grid = reduced.reshape(n_side, n_side, n_clay)
        h, w = mask_bool.shape
        broadcast = np.repeat(np.repeat(reduced_grid, TILE_SIZE, axis=0), TILE_SIZE, axis=1)
        broadcast = broadcast[:h, :w]
        return np.concatenate([base, broadcast.reshape(-1, n_clay)], axis=1)
    return fn


def train_mlp(X, y, alpha, seed=0):
    """Same balancing logic as train_ca_ann.py's train_mlp(), plus a
    configurable alpha and a much larger batch_size for speed."""
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_neg_sample = min(len(idx_neg), len(idx_pos) * 3)
    beta = n_neg_sample / len(idx_neg)
    idx_neg_sample = rng.choice(idx_neg, size=n_neg_sample, replace=False)
    idx_train = np.concatenate([idx_pos, idx_neg_sample])
    rng.shuffle(idx_train)

    clf = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                        max_iter=400, random_state=seed, early_stopping=True,
                        alpha=alpha, batch_size=BATCH_SIZE)
    clf.fit(X[idx_train], y[idx_train])
    return clf, beta


def latest_split_per_city(masks):
    """Just the most-recent 1-year-ahead test point per city (e.g.
    2024->2025), from train_ca_ann.py's existing walk-forward split
    builder -- reused rather than reimplemented."""
    splits = tca.make_walkforward_splits(masks)
    latest = {}
    for test_t, pool in splits:
        city, ya, yb = test_t
        if city not in latest or int(yb) > int(latest[city][0][2]):
            latest[city] = (test_t, pool)
    return list(latest.values())


def run_config(name, n_clay, alpha, masks, splits):
    orig_fn = fw.multiscale_features_v2
    fw.multiscale_features_v2 = make_features_fn(n_clay)
    try:
        t0 = time.time()
        deltas = []
        print(f"\n--- {name} (n_clay={n_clay}, alpha={alpha}) ---")
        for test_t, train_pool in splits:
            X_train, y_train = tca.build_training_set(masks, train_pool)
            model, beta = train_mlp(X_train, y_train, alpha)
            scores = tca.evaluate_split(model, beta, masks, test_t)
            city, ya, yb = test_t
            old_k, ann_k = scores["old"][1], scores["ca_ann"][1]
            delta = ann_k - old_k
            deltas.append(delta)
            print(f"  {city} {ya}->{yb}: old={old_k:.4f}  ca_ann={ann_k:.4f}  delta={delta:+.4f}")
        mean_delta = float(np.mean(deltas))
        print(f"  mean delta: {mean_delta:+.4f}   ({time.time()-t0:.0f}s)")
        return mean_delta
    finally:
        fw.multiscale_features_v2 = orig_fn


def main():
    print("Loading cached masks...")
    masks = tca.load_all_masks()
    splits = latest_split_per_city(masks)
    print(f"\nUsing {len(splits)} held-out test points: "
          f"{[(t[0], t[1], t[2]) for t, _ in splits]}")

    results = {}
    results["no_clay (pre-Clay baseline)"] = run_config(
        "no_clay (pre-Clay baseline)", 0, 0.0001, masks, splits)
    results["clay_8 (current production)"] = run_config(
        "clay_8 (current production)", 8, 0.0001, masks, splits)
    results["clay_3 (fewer dims)"] = run_config(
        "clay_3 (fewer dims)", 3, 0.0001, masks, splits)
    results["clay_8_regularized (alpha=0.01)"] = run_config(
        "clay_8_regularized (alpha=0.01)", 8, 0.01, masks, splits)
    results["clay_3_regularized (alpha=0.01)"] = run_config(
        "clay_3_regularized (alpha=0.01)", 3, 0.01, masks, splits)

    print("\n=== Summary: mean (ca_ann_kappa - baseline_kappa) across the 2 held-out points ===")
    for name, delta in results.items():
        flag = "BETTER than baseline" if delta > 0 else "still behind baseline"
        print(f"  {name:38s}  {delta:+.4f}   ({flag})")


if __name__ == "__main__":
    main()
