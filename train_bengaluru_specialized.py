"""
train_bengaluru_specialized.py -- Trains a Bengaluru-ONLY CA-ANN, with
Clay's PCA basis fit on Bengaluru's own embeddings alone (not pooled
across all 4 cities).

WHY: across every full run so far (2/3/4-city pooled training), Bengaluru
was the only city where Clay's embedding consistently helped the CA-ANN
beat the density-heuristic baseline; Chennai/Delhi/Kolkata consistently
hurt. Two things could be diluting Bengaluru's own result in the pooled
production model:
  1. The shared PCA basis is fit across all 4 cities' tile embeddings --
     its 8 components have to represent Delhi/Kolkata/Chennai's variance
     too, not just Bengaluru's.
  2. The MLP itself is trained on all 4 cities' transitions pooled
     together, so Bengaluru's spatial pattern has to share model capacity
     with 3 other, apparently-less-Clay-friendly cities.
This script isolates both: PCA fit on Bengaluru's own 160 cached tiles
only, and the final MLP trained ONLY on Bengaluru's own 9 consecutive
annual transitions (2016-2025) -- no cross-city pooling at all.

Does NOT touch the production ca_ann_model.joblib / ca_ann_model_stats.json
/ static/clay_pca.joblib / static/clay_reduced/ -- everything here is
saved under separate *_bengaluru-suffixed filenames so the existing
4-city production model stays intact until a decision is made to
promote this one.

Usage:
    python train_bengaluru_specialized.py
"""
import os
import sys
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import cohen_kappa_score
import joblib

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import main as fw
import train_ca_ann as tca

CITY = "bengaluru"
RAW_DIR = os.path.join(fw.STATIC_DIR, "clay_raw")
TILE_SIZE = 256
N_CLAY = 8

MODEL_PATH = os.path.join(fw.BASE_DIR, "ca_ann_model_bengaluru.joblib")
STATS_PATH = os.path.join(fw.BASE_DIR, "ca_ann_model_stats_bengaluru.json")
PCA_PATH = os.path.join(fw.STATIC_DIR, "clay_pca_bengaluru.joblib")


def load_bengaluru_masks():
    """Reuses fw's own mask-building (global-pooled FVI threshold, same
    methodology as production) but restricted to Bengaluru only."""
    global_t = fw.get_global_thresholds(CITY)
    print(f"  [{CITY}] global threshold: t1={global_t[0]}  t2={global_t[1]}")
    years = sorted(
        (f.replace(f"satellite_{CITY}_", "").replace(".tif", "")
         for f in os.listdir(fw.STATIC_DIR)
         if f.startswith(f"satellite_{CITY}_") and f.endswith(".tif")),
        key=int,
    )
    masks = {}
    for year in years:
        img = fw.load_satellite(CITY, year)
        nir = fw.load_nir(CITY, year)
        mask, _ = fw.compute_fvi_mask(img, nir, fixed_thresholds=global_t)
        masks[(CITY, year)] = mask > 127
        print(f"    loaded {CITY} {year}")
    return masks


def fit_bengaluru_pca():
    """PCA fit ONLY on Bengaluru's cached raw 1024-dim tile embeddings."""
    grids = [np.load(os.path.join(RAW_DIR, f))
             for f in os.listdir(RAW_DIR) if f.startswith(f"{CITY}_")]
    pooled = np.concatenate([g.reshape(-1, g.shape[-1]) for g in grids], axis=0)
    print(f"Fitting Bengaluru-only PCA({N_CLAY}) on {pooled.shape[0]} tiles...")
    pca = PCA(n_components=N_CLAY, random_state=0)
    pca.fit(pooled)
    print(f"  explained variance ratio: {pca.explained_variance_ratio_.round(4)}")
    print(f"  total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    joblib.dump(pca, PCA_PATH)
    return pca


def make_bengaluru_features_fn(pca):
    raw_cache = {}

    def fn(mask_bool, city, year):
        base = fw.multiscale_features(mask_bool)
        key = f"{city}_{year}"
        if key not in raw_cache:
            raw_cache[key] = np.load(os.path.join(RAW_DIR, f"{key}.npy"))
        grid = raw_cache[key]
        n_side = grid.shape[0]
        reduced = pca.transform(grid.reshape(-1, grid.shape[-1])).astype(np.float32)
        reduced_grid = reduced.reshape(n_side, n_side, N_CLAY)
        h, w = mask_bool.shape
        broadcast = np.repeat(np.repeat(reduced_grid, TILE_SIZE, axis=0), TILE_SIZE, axis=1)[:h, :w]
        return np.concatenate([base, broadcast.reshape(-1, N_CLAY)], axis=1)
    return fn


def train_mlp(X, y, seed=0):
    """Same balancing logic as train_ca_ann.py's train_mlp()."""
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_neg_sample = min(len(idx_neg), len(idx_pos) * 3)
    beta = n_neg_sample / len(idx_neg)
    idx_neg_sample = rng.choice(idx_neg, size=n_neg_sample, replace=False)
    idx_train = np.concatenate([idx_pos, idx_neg_sample])
    rng.shuffle(idx_train)
    clf = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                        max_iter=400, random_state=seed, early_stopping=True)
    clf.fit(X[idx_train], y[idx_train])
    return clf, beta


def main():
    print(f"=== Training Bengaluru-SPECIALIZED CA-ANN (no cross-city pooling) ===\n")
    masks = load_bengaluru_masks()
    print(f"\nLoaded {len(masks)} Bengaluru year masks.\n")

    pca = fit_bengaluru_pca()
    features_fn = make_bengaluru_features_fn(pca)
    orig_fn = fw.multiscale_features_v2
    fw.multiscale_features_v2 = features_fn

    try:
        years = sorted({y for (c, y) in masks}, key=int)
        transitions = [(CITY, a, b) for a, b in zip(years[:-1], years[1:])]
        print(f"\nBengaluru-only consecutive transitions: {len(transitions)}")

        # --- Walk-forward validation: same single-holdout-per-split pattern
        # as train_ca_ann.py's make_walkforward_splits, restricted to just
        # this one city (passing a Bengaluru-only masks dict means every
        # OTHER city in fw.CITIES naturally yields zero transitions and is
        # skipped -- no code changes needed to that shared function).
        splits = tca.make_walkforward_splits(masks)
        print(f"Walk-forward test points: {len(splits)}\n")

        results = []
        for test_t, train_pool in splits:
            X_train, y_train = tca.build_training_set(masks, train_pool)
            model, beta = train_mlp(X_train, y_train)
            scores = tca.evaluate_split(model, beta, masks, test_t)
            city, ya, yb = test_t
            old_k, ann_k = scores["old"][1], scores["ca_ann"][1]
            delta = ann_k - old_k
            results.append(delta)
            print(f"  {city} {ya}->{yb}: old={old_k:.4f}  ca_ann={ann_k:.4f}  delta={delta:+.4f}")

        wins = sum(d > 0 for d in results)
        mean_delta = float(np.mean(results))
        print(f"\nSpecialized Bengaluru-only: {wins}/{len(results)} wins, mean delta {mean_delta:+.4f}")

        # --- Multi-year horizons (3/6/9yr), Bengaluru only ---
        print("\nRunning multi-year projection validation (Bengaluru only)...\n")
        mv = tca.run_multiyear_validation(masks)
        for h in tca.HORIZONS_TO_TEST:
            entry = mv["by_horizon"][str(h)]
            if entry.get("available"):
                print(f"  {h}yr: {entry['n_test_points']} pts, wins {entry['wins']}, "
                      f"mean Kappa {entry['mean_old_kappa']:.4f} -> {entry['mean_ca_ann_kappa']:.4f}")
            else:
                print(f"  {h}yr: unavailable -- {entry['reason']}")

        # --- Final model on all Bengaluru transitions ---
        print("\nTraining final Bengaluru-specialized model on all data...")
        X_all, y_all = tca.build_training_set(masks, transitions)
        print(f"  Training set: {X_all.shape[0]:,} pixels, {int(y_all.sum()):,} change ({y_all.mean()*100:.1f}%)")
        final_model, final_beta = train_mlp(X_all, y_all)
        joblib.dump({"model": final_model, "beta": final_beta}, MODEL_PATH)
        print(f"Saved specialized model to {MODEL_PATH}")

        stats = {
            "city": CITY,
            "spatial_ranking_wins": wins,
            "spatial_ranking_n": len(results),
            "spatial_ranking_mean_delta": mean_delta,
            "multiyear_projection_validation": mv,
            "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        }
        with open(STATS_PATH, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved stats to {STATS_PATH}")

    finally:
        fw.multiscale_features_v2 = orig_fn


if __name__ == "__main__":
    main()
