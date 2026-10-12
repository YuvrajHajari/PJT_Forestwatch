"""
bengaluru_ensemble_experiment.py -- Tests whether averaging predictions
across multiple randomly-seeded CA-ANN fits (a seed ensemble) stabilizes
and/or improves Bengaluru's results, compared to any single seed.

WHY: the cross-run consistency check showed Bengaluru's CA-ANN-vs-baseline
delta flipping sign across different training runs (6yr delta: -0.023,
+0.008, +0.019, -0.005 across four different pooling configurations) with
IDENTICAL baseline numbers each time -- strong evidence of high variance
from random MLP initialization / random negative-class undersampling,
not a stable underlying effect. Averaging several independently-seeded
models' raw probabilities before calibrating is a standard variance
-reduction technique for exactly this situation, and is a genuine
improvement if it helps (not a presentation trick): it changes what the
model actually predicts, using data we already have.

Uses the PRODUCTION feature set (4-city pooled Clay-PCA, the
already-decided configuration), restricted to Bengaluru-only evaluation
-- NOT the earlier Bengaluru-specialized PCA/training experiment, which
already showed a regression at 6yr and was not promoted.

Does NOT touch production ca_ann_model.joblib/stats.json. Saves its own
ensemble artifacts under separate filenames pending a decision to promote.

Usage:
    python bengaluru_ensemble_experiment.py
"""
import os
import sys
import json
import numpy as np
from sklearn.neural_network import MLPClassifier
import joblib

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import main as fw
import train_ca_ann as tca

CITY = "bengaluru"
N_SEEDS = 8
MODEL_PATH = os.path.join(fw.BASE_DIR, "ca_ann_model_bengaluru_ensemble.joblib")
STATS_PATH = os.path.join(fw.BASE_DIR, "ca_ann_model_stats_bengaluru_ensemble.json")


def load_bengaluru_masks():
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


def train_mlp(X, y, seed):
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


def train_ensemble(X, y, n_seeds=N_SEEDS):
    """Trains n_seeds independently-seeded models on the same data,
    returns [(model, beta), ...]."""
    return [train_mlp(X, y, seed) for seed in range(n_seeds)]


def project_future_mask_ca_ann_ensemble(models_and_betas, city, masks_by_year, target_year):
    """Mirrors main.py's project_future_mask_ca_ann() exactly, except the
    CA-ANN ranking score is the AVERAGE of raw predict_proba across every
    ensemble member, calibrated once on that average -- rather than a
    single model's score. The Markov-rate pixel BUDGET (how many pixels
    change) is identical to the single-model version; only WHICH pixels
    get picked, via the averaged ranking, differs."""
    years = sorted(masks_by_year.keys(), key=int)
    latest_year = years[-1]
    latest_mask = masks_by_year[latest_year]
    elapsed = target_year - int(latest_year)
    if elapsed <= 0:
        raise ValueError("target_year must be after the latest known year")

    p_gain, p_loss = fw.compute_annual_transition_rates(masks_by_year)
    non_forest_mask = ~latest_mask
    forest_mask = latest_mask
    non_forest_count = int(np.sum(non_forest_mask))
    forest_count = int(np.sum(forest_mask))
    p_gain_compound = 1 - (1 - p_gain) ** elapsed
    p_loss_compound = 1 - (1 - p_loss) ** elapsed
    expected_gain_px = min(non_forest_count, int(round(p_gain_compound * non_forest_count)))
    expected_loss_px = min(forest_count, int(round(p_loss_compound * forest_count)))

    X = fw.multiscale_features_v2(latest_mask, city, latest_year)
    raw_p_list = [model.predict_proba(X)[:, 1] for model, beta in models_and_betas]
    raw_p_avg = np.mean(raw_p_list, axis=0)
    beta_avg = float(np.mean([beta for _, beta in models_and_betas]))
    ann_score = fw.calibrate_proba(raw_p_avg, beta_avg).reshape(latest_mask.shape)

    gain_scores = np.where(non_forest_mask, ann_score, -1.0)
    gain_order = np.argsort(gain_scores.ravel())[::-1]
    gain_mask = np.zeros_like(latest_mask)
    if expected_gain_px > 0:
        chosen = gain_order[:expected_gain_px]
        chosen = chosen[gain_scores.ravel()[chosen] > 0]
        gain_mask.ravel()[chosen] = True

    loss_scores = np.where(forest_mask, ann_score, -1.0)
    loss_order = np.argsort(loss_scores.ravel())[::-1]
    loss_mask = np.zeros_like(latest_mask)
    if expected_loss_px > 0:
        chosen = loss_order[:expected_loss_px]
        loss_mask.ravel()[chosen] = True

    return (latest_mask | gain_mask) & ~loss_mask


def score(pred, actual):
    from sklearn.metrics import cohen_kappa_score
    agree = float(np.mean(pred == actual) * 100)
    kappa = float(cohen_kappa_score(actual.ravel(), pred.ravel()))
    return agree, kappa


def evaluate_split_ensemble(masks, test_transition, train_pool):
    city, ya, yb = test_transition
    train_years = sorted({y for (c, y) in masks if c == city and y <= ya}, key=int)
    train_masks_bool = {y: masks[(city, y)] for y in train_years}

    X_train, y_train = tca.build_training_set(masks, train_pool)
    ensemble = train_ensemble(X_train, y_train)

    old_pred, _ = fw.project_future_mask(train_masks_bool, int(yb))
    ann_pred = project_future_mask_ca_ann_ensemble(ensemble, city, train_masks_bool, int(yb))
    actual = masks[(city, yb)]

    old_agree, old_kappa = score(old_pred, actual)
    ann_agree, ann_kappa = score(ann_pred, actual)
    return old_kappa, ann_kappa


def main():
    print(f"=== Bengaluru seed-ensemble experiment ({N_SEEDS} seeds) ===\n")
    masks = load_bengaluru_masks()
    print(f"\nLoaded {len(masks)} Bengaluru year masks.\n")

    # --- 1yr spatial-ranking walk-forward, ensembled ---
    splits = tca.make_walkforward_splits(masks)
    print(f"1yr walk-forward test points: {len(splits)}\n")
    results_1yr = []
    for test_t, train_pool in splits:
        old_k, ann_k = evaluate_split_ensemble(masks, test_t, train_pool)
        delta = ann_k - old_k
        results_1yr.append(delta)
        city, ya, yb = test_t
        print(f"  {city} {ya}->{yb}: old={old_k:.4f}  ensemble_ca_ann={ann_k:.4f}  delta={delta:+.4f}")
    print(f"\n1yr ensemble: {sum(d>0 for d in results_1yr)}/{len(results_1yr)} wins, "
          f"mean delta {np.mean(results_1yr):+.4f}\n")

    # --- Multi-year horizons (3/6/9yr), ensembled ---
    print("Running multi-year projection validation (ensembled)...\n")
    points_by_horizon = tca.generate_multiyear_test_points(masks)
    all_deltas_by_horizon = {}
    for h in tca.HORIZONS_TO_TEST:
        points = points_by_horizon[h]
        if not points:
            print(f"  {h}yr: unavailable (no valid test points)")
            continue
        deltas = []
        for city, train_end, target_year in points:
            city_years_all = sorted({y for (c, y) in masks if c == city}, key=int)
            train_years = [y for y in city_years_all if int(y) <= int(train_end)]
            train_masks_bool = {y: masks[(city, y)] for y in train_years}
            consecutive = [(city, a, b) for a, b in zip(train_years[:-1], train_years[1:])]
            if len(consecutive) < 2:
                continue
            X_train, y_train = tca.build_training_set(masks, consecutive)
            ensemble = train_ensemble(X_train, y_train)
            old_pred, _ = fw.project_future_mask(train_masks_bool, int(target_year))
            ann_pred = project_future_mask_ca_ann_ensemble(ensemble, city, train_masks_bool, int(target_year))
            actual = masks[(city, target_year)]
            old_agree, old_kappa = score(old_pred, actual)
            ann_agree, ann_kappa = score(ann_pred, actual)
            delta = ann_kappa - old_kappa
            deltas.append(delta)
            print(f"  [h={h}yr] {city} {train_end}->{target_year}: "
                  f"old={old_kappa:.4f}  ensemble_ca_ann={ann_kappa:.4f}  delta={delta:+.4f}")
        if deltas:
            all_deltas_by_horizon[h] = deltas
            print(f"  {h}yr ensemble: {sum(d>0 for d in deltas)}/{len(deltas)} wins, "
                  f"mean delta {np.mean(deltas):+.4f}\n")

    # --- Final ensemble model on all Bengaluru transitions ---
    print("Training final Bengaluru ensemble model on all data...")
    years = sorted({y for (c, y) in masks}, key=int)
    all_transitions = [(CITY, a, b) for a, b in zip(years[:-1], years[1:])]
    X_all, y_all = tca.build_training_set(masks, all_transitions)
    final_ensemble = train_ensemble(X_all, y_all)
    joblib.dump({"ensemble": final_ensemble, "n_seeds": N_SEEDS}, MODEL_PATH)
    print(f"Saved final ensemble model to {MODEL_PATH}")

    stats = {
        "city": CITY,
        "n_seeds": N_SEEDS,
        "spatial_ranking_1yr_deltas": results_1yr,
        "spatial_ranking_1yr_mean_delta": float(np.mean(results_1yr)),
        "multiyear_deltas_by_horizon": {str(h): d for h, d in all_deltas_by_horizon.items()},
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {STATS_PATH}")

    print("\n=== Summary: ensemble mean delta vs single-seed production numbers ===")
    print("Production (single seed, 4-city pooled model) Bengaluru deltas were:")
    print("  1yr: -0.0144, +0.0016, -0.0017, +0.0019 (mean -0.0032)")
    print("  3yr: +0.0057, -0.0186, +0.0018, +0.0038 (mean -0.0018)")
    print("  6yr: +0.0189 (single point)")
    print(f"\nEnsemble ({N_SEEDS} seeds) results are printed above -- compare directly.")


if __name__ == "__main__":
    main()
