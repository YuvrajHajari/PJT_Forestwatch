"""
train_ca_ann.py -- Trains and saves the CA-ANN prediction model that
main.py's /predict endpoint loads and serves.

Run this whenever you have new years of data to retrain on the latest
available history. This is an offline step -- main.py never retrains
on a live request.

Usage:
    python train_ca_ann.py

Produces:
    ca_ann_model.joblib       -- the trained model + calibration beta
    ca_ann_model_stats.json   -- walk-forward validation results

WHAT THE CA-ANN LEARNS vs WHAT MARKOV DECIDES (read this before touching
the transition-pair logic below):

  The CA-ANN learns SPATIAL suitability only: given a pixel's neighbourhood
  context, how likely is it (relatively) to be one that changes? It trains
  on CONSECUTIVE annual pairs (1yr gap) deliberately, because:
    - More training pairs -> better generalisation of the spatial pattern
    - A boundary pixel that oscillates annually IS genuinely more volatile
      than a deep-interior pixel -- labeling it "changed" is a correct
      training signal for ranking, even though it doesn't represent real
      permanent deforestation on its own
    - Ranking quality (AUC) is what was validated to matter (mean Kappa
      0.68->0.79 across 8 walk-forward tests); it is robust to this kind
      of label noise, unlike absolute rate estimates

  The MARKOV rate (HOW MUCH changes) is computed separately in
  main.py's compute_annual_transition_rates() with a >=3yr minimum gap,
  which filters the oscillation noise that inflated rates to ~18%/yr on
  annual data. project_future_mask_ca_ann() in main.py uses this corrected
  rate to set the pixel budget, and the CA-ANN score here purely to rank
  candidates within that budget. See main.py's docstrings for the full
  before/after diagnostic history (the symmetric red/gold ring artifact
  and the total-erosion-by-2034 bug this fixes).

Also applies the global-pooled-threshold fix so training masks exactly
match what /predict builds at serving time.
"""
import os
import sys
import json
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import cohen_kappa_score
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as fw

MODEL_PATH = os.path.join(fw.BASE_DIR, "ca_ann_model.joblib")
STATS_PATH = os.path.join(fw.BASE_DIR, "ca_ann_model_stats.json")


def load_all_masks():
    """
    Recomputes masks from source imagery for every city/year available,
    using the GLOBAL pooled threshold (not per-image Otsu) so that the
    masks the model trains on are identical to what /predict serves.
    """
    masks = {}
    for city in fw.CITIES:
        global_t = fw.get_global_thresholds(city)
        print(f"  [{city}] global threshold: t1={global_t[0]}  t2={global_t[1]}")

        years = sorted(
            (f.replace(f"satellite_{city}_", "").replace(".png", "")
             for f in os.listdir(fw.STATIC_DIR)
             if f.startswith(f"satellite_{city}_") and f.endswith(".png")),
            key=int,
        )
        for year in years:
            try:
                img = fw.load_satellite(city, year)
                nir = fw.load_nir(city, year)
                mask, _ = fw.compute_fvi_mask(img, nir, fixed_thresholds=global_t)
                masks[(city, year)] = mask > 127
                print(f"    loaded {city} {year}")
            except Exception as e:
                print(f"    skip {city} {year}: {e}")
    return masks


# Matches index.html's target-year dropdown (2028/2031/2034 relative to
# base year 2025 = +3/+6/+9). Self-adapting: as more years get fetched in
# future work, longer horizons automatically gain valid test points
# without any code change here.
HORIZONS_TO_TEST = [3, 6, 9]


def generate_multiyear_test_points(masks, horizons=HORIZONS_TO_TEST):
    """
    For each city and each candidate horizon h, finds all valid
    (train_end_year, target_year) pairs where:
      - target_year = train_end_year + h, and target_year is an actual
        fetched year (<= the latest year available)
      - train_end_year leaves enough history for the Markov rate to be
        computable, i.e. train_end_year - first_year >= MIN_TRANSITION_GAP_YEARS
        (otherwise compute_annual_transition_rates has zero valid pairs
        and returns a meaningless 0.0 rate)

    Returns {horizon: [(city, train_end_year, target_year), ...]}.
    """
    min_gap = fw.MIN_TRANSITION_GAP_YEARS
    points_by_horizon = {h: [] for h in horizons}

    for city in fw.CITIES:
        city_years = sorted({y for (c, y) in masks if c == city}, key=int)
        if not city_years:
            continue
        first_year = int(city_years[0])
        year_set   = {int(y) for y in city_years}
        min_valid_train_end = first_year + min_gap

        for h in horizons:
            for train_end in city_years:
                train_end_i = int(train_end)
                if train_end_i < min_valid_train_end:
                    continue
                target_i = train_end_i + h
                if target_i not in year_set:
                    continue
                target_year = next(y for y in city_years if int(y) == target_i)
                points_by_horizon[h].append((city, train_end, target_year))

    return points_by_horizon


def run_multiyear_validation(masks):
    """
    Walk-forward validation at the horizons /predict actually serves
    (3/6/9 years), unlike the 1-year-ahead spatial-ranking check above.
    For each valid (train_end, target) pair: trains a fresh CA-ANN using
    only data up to train_end, projects forward exactly `horizon` years,
    and compares both the CA-ANN and the density-heuristic baseline
    against the real held-out mask at target_year.

    Retrains the model from scratch for every test point (expensive, but
    this is an offline script -- main.py never repeats this work).

    Returns a dict with "by_horizon" (summary stats per horizon, honestly
    marked unavailable if zero test points exist for that horizon) and
    "per_test_results" (full detail, kept for the thesis appendix).
    """
    points_by_horizon = generate_multiyear_test_points(masks)
    by_horizon_summary = {}
    all_per_test = []

    for h in HORIZONS_TO_TEST:
        points = points_by_horizon[h]
        if not points:
            by_horizon_summary[str(h)] = {
                "available": False,
                "reason": (
                    f"Not enough historical years to backtest a {h}-year horizon "
                    f"(needs training data starting at least {fw.MIN_TRANSITION_GAP_YEARS}yr "
                    f"before a cutoff that still leaves {h}yr of held-out years to test "
                    f"against). Will become available once more years are fetched."
                ),
            }
            continue

        results = []
        for city, train_end, target_year in points:
            city_years_all = sorted({y for (c, y) in masks if c == city}, key=int)
            train_years = [y for y in city_years_all if int(y) <= int(train_end)]
            train_masks_bool = {y: masks[(city, y)] for y in train_years}

            consecutive = list(zip(train_years[:-1], train_years[1:]))
            consecutive = [(city, ya, yb) for ya, yb in consecutive]
            if len(consecutive) < 2:
                continue  # not enough pairs yet to fit a spatial model

            X_train, y_train = build_training_set(masks, consecutive)
            model, beta = train_mlp(X_train, y_train)

            old_pred, _ = fw.project_future_mask(train_masks_bool, int(target_year))
            ann_pred, _ = fw.project_future_mask_ca_ann(model, beta, train_masks_bool, int(target_year))
            actual = masks[(city, target_year)]

            def score(pred):
                agree = float(np.mean(pred == actual) * 100)
                kappa = float(cohen_kappa_score(actual.ravel(), pred.ravel()))
                return agree, kappa

            old_agree, old_kappa = score(old_pred)
            ann_agree, ann_kappa = score(ann_pred)

            result = {
                "city": city, "train_end_year": train_end, "target_year": target_year,
                "horizon_years": h,
                "old_agree": old_agree, "old_kappa": old_kappa,
                "ca_ann_agree": ann_agree, "ca_ann_kappa": ann_kappa,
            }
            results.append(result)
            all_per_test.append(result)
            print(f"  [h={h}yr] {city} {train_end}\u2192{target_year}: "
                  f"old Kappa {old_kappa:.4f}  |  CA-ANN Kappa {ann_kappa:.4f}  "
                  f"[{ann_kappa - old_kappa:+.4f}]")

        if results:
            deltas = [r["ca_ann_kappa"] - r["old_kappa"] for r in results]
            by_horizon_summary[str(h)] = {
                "available":          True,
                "n_test_points":      len(results),
                "wins":               sum(d > 0 for d in deltas),
                "mean_old_kappa":     float(np.mean([r["old_kappa"]    for r in results])),
                "mean_ca_ann_kappa":  float(np.mean([r["ca_ann_kappa"] for r in results])),
                "mean_old_agree":     float(np.mean([r["old_agree"]    for r in results])),
                "mean_ca_ann_agree":  float(np.mean([r["ca_ann_agree"] for r in results])),
                "mean_kappa_delta":   float(np.mean(deltas)),
            }
        else:
            by_horizon_summary[str(h)] = {
                "available": False,
                "reason": f"No valid {h}-year test points could be evaluated.",
            }

    return {"by_horizon": by_horizon_summary, "per_test_results": all_per_test}


def get_consecutive_transitions(masks):
    """
    Returns all consecutive annual (city, year_a, year_b) pairs.
    Deliberately NOT filtered by min-gap -- see module docstring for why
    the CA-ANN's spatial training is a separate concern from the Markov
    rate's min-gap filter.
    """
    transitions = []
    for city in fw.CITIES:
        city_years = sorted({y for (c, y) in masks if c == city}, key=int)
        for ya, yb in zip(city_years[:-1], city_years[1:]):
            transitions.append((city, ya, yb))
    return transitions


def build_training_set(masks, transitions):
    """
    masks: {(city, year): bool mask}, transitions: [(city, ya, yb), ...].
    Features computed from mA (the "before" state) via fw.multiscale_features.
    """
    X_list, y_list = [], []
    for city, ya, yb in transitions:
        mA, mB = masks[(city, ya)], masks[(city, yb)]
        X_list.append(fw.multiscale_features(mA))
        y_list.append((mA != mB).astype(np.int32).reshape(-1))
    return np.concatenate(X_list), np.concatenate(y_list)


def train_mlp(X, y, seed=0):
    """
    Balances classes for training (change pixels are rare); returns
    (model, beta) where beta is the fraction of true negatives kept,
    needed by calibrate_proba at inference time.
    """
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


def make_walkforward_splits(masks, min_train_transitions=2, n_test_points=4):
    """
    Walk-forward splits using consecutive annual transitions. Test points
    are consecutive 1yr pairs; training pool uses all consecutive
    transitions ending before the test year. This matches how
    build_training_set()/train_mlp() are used for the final model, so the
    validation numbers are honest.
    """
    city_transitions = {}
    for city in fw.CITIES:
        city_years = sorted({y for (c, y) in masks if c == city}, key=int)
        pairs = list(zip(city_years[:-1], city_years[1:]))
        city_transitions[city] = [(city, a, b) for a, b in pairs]

    splits = []
    for city in fw.CITIES:
        transitions = city_transitions[city]
        if len(transitions) <= min_train_transitions:
            continue
        n_test = min(n_test_points, len(transitions) - min_train_transitions)
        test_points = transitions[-n_test:] if n_test > 0 else []
        for test_t in test_points:
            ya = test_t[1]
            train_pool = [
                t2 for c2 in fw.CITIES
                for t2 in city_transitions[c2]
                if t2[2] <= ya
            ]
            splits.append((test_t, train_pool))
    return splits


def evaluate_split(model, beta, masks, test_transition):
    """
    Old density-heuristic pipeline vs CA-ANN, using the exact functions
    main.py will serve. Both now use the same >=3yr-min-gap Markov rate
    internally (via compute_annual_transition_rates), so this comparison
    isolates the effect of CA-ANN ranking vs density ranking -- the thing
    that was actually validated to help (mean Kappa 0.68 -> 0.79).
    """
    city, ya, yb = test_transition
    train_years = sorted({y for (c, y) in masks if c == city and y <= ya}, key=int)
    train_masks_bool = {y: masks[(city, y)] for y in train_years}

    old_pred, _ = fw.project_future_mask(train_masks_bool, int(yb))
    ann_pred, _ = fw.project_future_mask_ca_ann(model, beta, train_masks_bool, int(yb))
    actual = masks[(city, yb)]

    def score(pred):
        agree = float(np.mean(pred == actual) * 100)
        kappa = float(cohen_kappa_score(actual.ravel(), pred.ravel()))
        return agree, kappa

    return {"old": score(old_pred), "ca_ann": score(ann_pred)}


def main():
    print("=== Training CA-ANN prediction model ===")
    print(f"    Markov rate min gap: {fw.MIN_TRANSITION_GAP_YEARS}yr "
          f"(applied inside compute_annual_transition_rates)")
    print(f"    CA-ANN spatial training: consecutive annual pairs "
          f"(ranking signal only, see module docstring)\n")

    masks = load_all_masks()
    print(f"\nLoaded {len(masks)} city-year masks.\n")

    all_transitions = get_consecutive_transitions(masks)
    print(f"Training transitions (consecutive annual pairs): {len(all_transitions)}")

    if len(all_transitions) < 3:
        print("Not enough transitions yet to train. Fetch more years first.")
        return

    # --- Validation 1: 1-year-ahead spatial ranking check (original design) ---
    # Tests whether the CA-ANN correctly ranks WHICH pixels are volatile.
    # Does NOT validate accuracy at the 3/6/9yr horizons /predict serves --
    # see Validation 2 below for that.
    print("\nRunning spatial-ranking validation (1-year-ahead transitions)...\n")
    splits = make_walkforward_splits(masks)
    results = []
    for test_t, train_pool in splits:
        X_train, y_train = build_training_set(masks, train_pool)
        model, beta = train_mlp(X_train, y_train)
        scores = evaluate_split(model, beta, masks, test_t)
        results.append({
            "test":         list(test_t),
            "old_kappa":    scores["old"][1],
            "ca_ann_kappa": scores["ca_ann"][1],
            "old_agree":    scores["old"][0],
            "ca_ann_agree": scores["ca_ann"][0],
        })
        city, ya, yb = test_t
        print(f"{city} {ya}\u2192{yb}: "
              f"old Kappa {scores['old'][1]:.4f}  |  "
              f"CA-ANN Kappa {scores['ca_ann'][1]:.4f}  "
              f"[{scores['ca_ann'][1]-scores['old'][1]:+.4f}]")

    kappa_deltas = [r["ca_ann_kappa"] - r["old_kappa"] for r in results]
    wins = sum(d > 0 for d in kappa_deltas)
    spatial_ranking_validation = {
        "n_test_points":     len(results),
        "wins":              wins,
        "mean_kappa_delta":  float(np.mean(kappa_deltas)),
        "mean_old_kappa":    float(np.mean([r["old_kappa"]    for r in results])),
        "mean_ca_ann_kappa": float(np.mean([r["ca_ann_kappa"] for r in results])),
        "per_test_results":  results,
        "note": (
            "Consecutive 1-year-ahead transitions. Validates the CA-ANN's "
            "SPATIAL RANKING quality (does it correctly identify which "
            "pixels are most likely to change), not multi-year projection "
            "accuracy -- see multiyear_projection_validation for that. A "
            "ranking-only test does not benefit as clearly from the "
            "Markov-bounded-count design used in production, since the "
            "shared pixel budget caps how much a ranking improvement can "
            "move the final Kappa at a 1-year horizon."
        ),
    }
    print(f"\nSpatial ranking summary: CA-ANN better in {wins}/{len(results)} tests, "
          f"mean Kappa {spatial_ranking_validation['mean_old_kappa']:.4f} -> "
          f"{spatial_ranking_validation['mean_ca_ann_kappa']:.4f}")

    # --- Validation 2: multi-year projection accuracy (what /predict needs) ---
    print("\nRunning multi-year projection validation (3/6/9-year horizons)...\n")
    multiyear_projection_validation = run_multiyear_validation(masks)
    for h in HORIZONS_TO_TEST:
        entry = multiyear_projection_validation["by_horizon"][str(h)]
        if entry.get("available"):
            print(f"  {h}yr horizon: {entry['n_test_points']} test points, "
                  f"CA-ANN better in {entry['wins']}, mean Kappa "
                  f"{entry['mean_old_kappa']:.4f} -> {entry['mean_ca_ann_kappa']:.4f}")
        else:
            print(f"  {h}yr horizon: unavailable -- {entry['reason']}")

    stats = {
        "spatial_ranking_validation":      spatial_ranking_validation,
        "multiyear_projection_validation": multiyear_projection_validation,
        "method":               "ca_ann_ranked_markov_count",
        "markov_min_gap_years": fw.MIN_TRANSITION_GAP_YEARS,
        "threshold_mode":       "global_pooled",
    }

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved validation stats to {STATS_PATH}")

    print("\nTraining final model on all available data...")
    X_all, y_all = build_training_set(masks, all_transitions)
    print(f"  Training set: {X_all.shape[0]:,} pixels, "
          f"{int(y_all.sum()):,} change ({y_all.mean()*100:.1f}%)")
    final_model, final_beta = train_mlp(X_all, y_all)

    joblib.dump({"model": final_model, "beta": final_beta}, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")
    print("\nDone. Restart uvicorn to start serving the updated CA-ANN predictions.")


if __name__ == "__main__":
    main()