"""
test_ca_ann.py -- Experimental CA-ANN extension to the prediction module.

Replaces the hand-built "local forest density" spatial-suitability heuristic
in project_future_mask() with a small trained neural network (an MLP), while
keeping the Markov transition rate (how much changes) exactly as validated
before. This is the "ANN" half of a CA-ANN model -- a well-precedented
extension of CA-Markov in the literature (Talcher/Odisha, Mumbai, Delhi).

Does NOT touch main.py or your live app. Standalone experiment, run against
your own already-computed masks, mirroring how test_fvi.py was developed.

Usage:
    python test_ca_ann.py

Reads mask_{city}_{year}.png from ./static/ for every city/year you have
(falls back to recomputing from satellite_*/nir_* via main.py's
compute_fvi_mask if a mask PNG is missing). Trains on your earliest
transitions, validates on the most recent one, and prints a head-to-head
comparison against the old density-heuristic baseline.
"""
import os
import sys
import numpy as np
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, cohen_kappa_score

IMAGE_SIZE = 1024
STATIC_DIR = "static"
CITIES = ["chennai", "bengaluru"]
YEARS = [str(y) for y in range(2016, 2026)]  # annual, matches the updated satellite_fetch.py


def load_mask(city, year):
    """Prefers the actual production mask PNG; falls back to recomputing
    via main.py's compute_fvi_mask so results always match the live pipeline."""
    path = os.path.join(STATIC_DIR, f"mask_{city}_{year}.png")
    if os.path.exists(path):
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return m > 127
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import main as fw
    img = fw.load_satellite(city, year)
    nir = fw.load_nir(city, year)
    mask, _ = fw.compute_fvi_mask(img, nir)
    return mask > 127


def multiscale_features(mask_bool):
    """
    Four features per pixel: forest density at 5x5, 15x15, 31x31 windows,
    plus distance to the nearest forest/non-forest boundary. The network's
    job is learning how to weigh these scales together -- something a
    single fixed window size (the old heuristic) structurally can't do.
    """
    m = mask_bool.astype(np.float32)
    d5 = cv2.boxFilter(m, -1, (5, 5))
    d15 = cv2.boxFilter(m, -1, (15, 15))
    d31 = cv2.boxFilter(m, -1, (31, 31))

    mask_u8 = mask_bool.astype(np.uint8) * 255
    dist_to_nonforest = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    dist_to_forest = cv2.distanceTransform(255 - mask_u8, cv2.DIST_L2, 5)
    edge_dist = np.minimum(dist_to_nonforest, dist_to_forest)
    # Cap and normalize the distance feature so it's on a comparable scale to densities (0-1)
    edge_dist_norm = np.clip(edge_dist / 30.0, 0, 1)

    feats = np.stack([d5, d15, d31, edge_dist_norm], axis=-1)
    return feats.reshape(-1, 4)


def build_training_set(masks, transitions):
    """masks: {(city, year): bool mask}. transitions: [(city, year_a, year_b), ...]."""
    X_list, y_list = [], []
    for city, ya, yb in transitions:
        mA, mB = masks[(city, ya)], masks[(city, yb)]
        X_list.append(multiscale_features(mA))
        y_list.append((mA != mB).astype(np.int32).reshape(-1))
    return np.concatenate(X_list), np.concatenate(y_list)


def train_ca_ann(X, y, seed=0):
    """Balances the heavily-skewed classes (most pixels never change) before
    training a small 2-hidden-layer MLP. Returns (model, beta), where beta
    is the fraction of the TRUE negative population actually kept -- needed
    afterward to correct the model's raw probabilities back to the real
    population rate (see calibrate_proba)."""
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


def calibrate_proba(p_resampled, beta):
    """
    Prior-correction for probabilities from a model trained on a
    negative-class-undersampled set (Dal Pozzolo et al., 2015 /
    Elkan, 2001 style correction). AUC-based ranking is invariant to this
    (monotonic transform), which is exactly why the ranking-based
    "markov count" variant looked fine while raw-probability-based
    variants ("ann_count", "threshold") collapsed -- the model's ordering
    of pixels was always correct, only the absolute probability values
    were miscalibrated to the artificial ~25% training rate instead of
    the true ~2% real-world rate.
    """
    p = np.clip(p_resampled, 1e-6, 1 - 1e-6)
    odds = p / (1 - p)
    corrected_odds = odds * beta
    return corrected_odds / (1 + corrected_odds)


def predict_calibrated(model, beta, X):
    raw = model.predict_proba(X)[:, 1]
    return calibrate_proba(raw, beta)


def baseline_edge_likeness(mask_bool):
    """The OLD heuristic's implied 'change likelihood': pixels near the
    density midpoint (0.5) are edge-like and more volatile; pixels deep
    inside a solid class (density near 0 or 1) are stable."""
    m = mask_bool.astype(np.float32)
    density = cv2.boxFilter(m, -1, (15, 15))
    return (1 - np.abs(density - 0.5) * 2).reshape(-1)


def evaluate_classifier(model, masks, test_transition):
    city, ya, yb = test_transition
    mA, mB = masks[(city, ya)], masks[(city, yb)]
    X_test = multiscale_features(mA)
    y_test = (mA != mB).astype(np.int32).reshape(-1)

    p_ann = model.predict_proba(X_test)[:, 1]
    auc_ann = roc_auc_score(y_test, p_ann)

    p_baseline = baseline_edge_likeness(mA)
    auc_baseline = roc_auc_score(y_test, p_baseline)

    return auc_ann, auc_baseline, y_test.mean()


def project_future_mask_ann_variant(model, beta, masks, city, target_year, count_mode="markov"):
    """
    WHICH pixels change is always ranked by the trained CA-ANN's predicted
    probability. HOW MANY change is decided differently per count_mode,
    to directly test whether the Markov-rate count is the bottleneck on
    the network's real benefit:

      "markov"    - production behaviour: compounded Markov transition
                    rate decides the count (network only affects ranking).
      "ann_count" - the network's OWN aggregate confidence (sum of its
                    CALIBRATED predicted probabilities over each candidate
                    pool) decides the count instead -- lets the trained
                    model determine "how much" too, not just "which".
      "threshold" - no external count at all: every pixel the model's
                    CALIBRATED probability rates above 0.5 counts as changing.

    beta: the negative-class subsampling rate from train_ca_ann, needed to
    correct raw probabilities back to the true population rate before they
    are used as absolute quantities (ranking alone doesn't need this, but
    we calibrate unconditionally since it's a monotonic transform and
    doesn't change the "markov" variant's ranking outcome).
    """
    city_years = sorted({y for (c, y) in masks if c == city}, key=int)
    latest_year = city_years[-1]
    latest_mask = masks[(city, latest_year)]

    non_forest_mask = ~latest_mask
    forest_mask = latest_mask

    X = multiscale_features(latest_mask)
    ann_score = predict_calibrated(model, beta, X).reshape(latest_mask.shape)

    if count_mode == "threshold":
        gain_mask = (ann_score > 0.5) & non_forest_mask
        loss_mask = (ann_score > 0.5) & forest_mask
        return (latest_mask | gain_mask) & ~loss_mask

    non_forest_count = int(np.sum(non_forest_mask))
    forest_count = int(np.sum(forest_mask))

    if count_mode == "markov":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import main as fw
        elapsed = int(target_year) - int(latest_year)
        rate_masks = {y: masks[(city, y)] for y in city_years}
        p_gain, p_loss = fw.compute_annual_transition_rates(rate_masks)
        p_gain_c = 1 - (1 - p_gain) ** elapsed
        p_loss_c = 1 - (1 - p_loss) ** elapsed
        expected_gain_px = min(non_forest_count, int(round(p_gain_c * non_forest_count)))
        expected_loss_px = min(forest_count, int(round(p_loss_c * forest_count)))
    elif count_mode == "ann_count":
        expected_gain_px = min(non_forest_count, int(round(np.sum(ann_score[non_forest_mask]))))
        expected_loss_px = min(forest_count, int(round(np.sum(ann_score[forest_mask]))))
    else:
        raise ValueError(f"unknown count_mode: {count_mode}")

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


def evaluate_full_pipeline(model, beta, masks, test_transition):
    """Head-to-head: old CA-Markov+density vs three CA-ANN count strategies."""
    city, ya, yb = test_transition
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import main as fw

    train_years = sorted({y for (c, y) in masks if c == city and y <= ya}, key=int)
    train_masks_bool = {y: masks[(city, y)] for y in train_years}
    # CRITICAL: only years up to and including ya -- passing the full masks
    # dict here would let the ANN see yb's real mask (this was a real bug
    # caught earlier: it produced a false 100% agreement / Kappa 1.0).
    masks_for_ann = {(city, y): masks[(city, y)] for y in train_years}

    old_pred, _ = fw.project_future_mask(train_masks_bool, int(yb))
    actual = masks[(city, yb)]

    def score(pred):
        agree = np.mean(pred == actual) * 100
        kappa = cohen_kappa_score(actual.ravel(), pred.ravel())
        return agree, kappa

    results = {"old": score(old_pred)}
    for mode in ["markov", "ann_count", "threshold"]:
        pred = project_future_mask_ann_variant(model, beta, masks_for_ann, city, yb, mode)
        results[mode] = score(pred)
    return results


def make_walkforward_splits(masks, min_train_transitions=2, n_test_points=4):
    """
    Builds chronological (city, year_a, year_b) transitions per city, then
    reserves the last `n_test_points` transitions per city as walk-forward
    test points. For each test point, the training pool is every transition
    (pooled across BOTH cities) that ends at or before that test point's
    start year -- i.e. train strictly on the past, test on the next step,
    then roll forward. This replaces a single train/test split with
    multiple independent evaluations, which a single held-out year cannot
    give you honest statistical confidence from.
    """
    city_transitions = {}
    for city in CITIES:
        city_years = sorted({y for (c, y) in masks if c == city}, key=int)
        pairs = list(zip(city_years[:-1], city_years[1:]))
        city_transitions[city] = [(city, a, b) for a, b in pairs]

    splits = []
    for city in CITIES:
        transitions = city_transitions[city]
        if len(transitions) <= min_train_transitions:
            continue
        n_test = min(n_test_points, len(transitions) - min_train_transitions)
        test_points = transitions[-n_test:] if n_test > 0 else []
        for test_t in test_points:
            ya = test_t[1]
            train_pool = [t2 for c2 in CITIES for t2 in city_transitions[c2] if t2[2] <= ya]
            splits.append((test_t, train_pool))
    return splits


def run():
    print("=== CA-ANN experimental test (walk-forward validation) ===\n")

    masks = {}
    for city in CITIES:
        for year in YEARS:
            try:
                masks[(city, year)] = load_mask(city, year)
            except Exception as e:
                print(f"  skip {city} {year}: {e}")
    print(f"Loaded {len(masks)} city-year masks.\n")

    splits = make_walkforward_splits(masks)
    if not splits:
        print("Not enough transitions yet for walk-forward validation. Need more years fetched.")
        return

    print(f"Running {len(splits)} walk-forward test points (train strictly on the past, test on the next step)\n")

    results = []
    for test_t, train_pool in splits:
        X_train, y_train = build_training_set(masks, train_pool)
        model, beta = train_ca_ann(X_train, y_train)

        auc_ann, auc_base, change_rate = evaluate_classifier(model, masks, test_t)
        pipeline_scores = evaluate_full_pipeline(model, beta, masks, test_t)

        results.append({
            "test": test_t, "auc_ann": auc_ann, "auc_base": auc_base,
            "change_rate": change_rate, "pipeline": pipeline_scores,
        })

        city, ya, yb = test_t
        print(f"{city} {ya}\u2192{yb}  (actual change rate {change_rate*100:.2f}%)")
        print(f"  AUC:  old {auc_base:.4f}  |  CA-ANN {auc_ann:.4f}  |  {(auc_ann-auc_base):+.4f}")
        old_agree, old_kappa = pipeline_scores["old"]
        print(f"  Kappa -- old (Markov count):        {old_kappa:.4f}  ({old_agree:.2f}% agreement)")
        for mode, label in [("markov", "CA-ANN + Markov count"), ("ann_count", "CA-ANN + own count"), ("threshold", "CA-ANN + 0.5 threshold")]:
            agree, kappa = pipeline_scores[mode]
            print(f"  Kappa -- {label:<24s} {kappa:.4f}  ({agree:.2f}% agreement)  [{(kappa-old_kappa):+.4f}]")
        print()

    print("=" * 70)
    print(f"SUMMARY across {len(results)} walk-forward test points")
    print("=" * 70)
    auc_deltas = [r["auc_ann"] - r["auc_base"] for r in results]
    print(f"AUC:   CA-ANN better in {sum(d>0 for d in auc_deltas)}/{len(results)}, mean delta {np.mean(auc_deltas):+.4f}, std {np.std(auc_deltas):.4f}\n")

    for mode, label in [("markov", "CA-ANN + Markov count (production-style)"),
                        ("ann_count", "CA-ANN + own aggregate count"),
                        ("threshold", "CA-ANN + fixed 0.5 threshold")]:
        deltas = [r["pipeline"][mode][1] - r["pipeline"]["old"][1] for r in results]
        wins = sum(d > 0 for d in deltas)
        mean_kappa = np.mean([r["pipeline"][mode][1] for r in results])
        print(f"{label}:")
        print(f"  Kappa better in {wins}/{len(results)}, mean delta {np.mean(deltas):+.4f}, std {np.std(deltas):.4f}, mean Kappa {mean_kappa:.4f}")

    old_mean_kappa = np.mean([r["pipeline"]["old"][1] for r in results])
    print(f"\nOld baseline mean Kappa: {old_mean_kappa:.4f}")


if __name__ == "__main__":
    run()