"""tests/validation/test_mt_composite.py
Tests 6-13: composite score, monotonicity, separation, calibration,
industry robustness, penalty tiers, composite dominance, summary.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import math
import numpy as np
import pytest
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from typing import Dict, List

from tests.validation.mt_shared import (
    CUTOFF_DATE, CACHE_DIR, MIN_PAIRED, MIN_BINARY_POS, MIN_INDUSTRY_N,
    SPEARMAN_MINIMUM, SPEARMAN_STRONG, SPEARMAN_HIGH,
    AUROC_MINIMUM, AUROC_STRONG, AUROC_HIGH,
    LIFT_MINIMUM, LIFT_STRONG, LIFT_HIGH,
    CAPTURE_MINIMUM, CAPTURE_STRONG,
    REGRESSION_SPEARMAN, MAX_CALIBRATION_VIOLATIONS,
    _get_scorer, _get_mt_scorer, _get_val_data,
    _read_csv, _load_validation_data,
)
import warnings
from src.scoring.penalty_percentiles import (
    load_percentiles,
    CACHE_FILENAME as PERCENTILE_CACHE,
)

def test_composite_spearman_minimum():
    """Composite score Spearman rho ≥ SPEARMAN_MINIMUM (minimum viable)."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, y_adv = _get_val_data()
    if len(rows) < MIN_PAIRED:
        pytest.skip("Insufficient data")

    preds     = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    y         = np.array(y_adv)

    rho, p_val = spearmanr(composites, y)
    print(
        f"\n  Composite Spearman rho={rho:.4f}  p={p_val:.4f}  "
        f"(minimum: {SPEARMAN_MINIMUM})"
    )
    assert rho >= SPEARMAN_MINIMUM, (
        f"Composite Spearman {rho:.4f} below minimum viable {SPEARMAN_MINIMUM}"
    )


def test_composite_spearman_strong():
    """Composite score Spearman rho ≥ SPEARMAN_STRONG (strong model)."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, y_adv = _get_val_data()
    if len(rows) < MIN_PAIRED:
        pytest.skip("Insufficient data")

    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    rho, _     = spearmanr(composites, np.array(y_adv))
    print(f"\n  Composite Spearman rho={rho:.4f}  (strong target: {SPEARMAN_STRONG})")
    assert rho >= SPEARMAN_STRONG, (
        f"Composite Spearman {rho:.4f} below strong target {SPEARMAN_STRONG}"
    )


# ====================================================================== #
#  Test 6b: Top-decile lift
# ====================================================================== #

def test_top_decile_lift_minimum():
    """Top-decile lift must be ≥ LIFT_MINIMUM."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, y_adv = _get_val_data()
    if len(rows) < MIN_PAIRED:
        pytest.skip("Insufficient data")

    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    y          = np.array(y_adv)
    lift       = _compute_decile_lift(composites, y, decile=10)
    print(f"\n  Top-decile lift: {lift:.3f}×  (minimum: {LIFT_MINIMUM})")
    assert lift >= LIFT_MINIMUM, (
        f"Top-decile lift {lift:.3f}× below minimum {LIFT_MINIMUM}"
    )


def test_top_decile_lift_strong():
    """Top-decile lift must be ≥ LIFT_STRONG (strong model)."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, y_adv = _get_val_data()
    if len(rows) < MIN_PAIRED:
        pytest.skip("Insufficient data")

    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    y          = np.array(y_adv)
    lift       = _compute_decile_lift(composites, y, decile=10)
    print(f"\n  Top-decile lift: {lift:.3f}×  (strong target: {LIFT_STRONG})")
    assert lift >= LIFT_STRONG, (
        f"Top-decile lift {lift:.3f}× below strong target {LIFT_STRONG}"
    )


def test_top_10pct_capture_minimum():
    """Top 10% of composite scores must capture ≥ CAPTURE_MINIMUM of injury/fatal events.

    We use any_injury_fatal (hospitalization or fatality) rather than any_wr_serious
    because the WR/serious base rate in inspected establishments typically exceeds 40%,
    making max achievable capture in the top 10% only ~25%.
    Injury/fatal events have a lower base rate, so capture ≥ CAPTURE_MINIMUM
    represents a meaningful lift over random.
    """
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    y_event    = np.array([r["any_injury_fatal"] for r in rows], dtype=int)
    base_rate  = float(y_event.mean())
    max_achievable = min(0.10, base_rate) / max(base_rate, 1e-9)
    if y_event.sum() < MIN_BINARY_POS:
        pytest.skip(f"Insufficient injury/fatal events ({y_event.sum()})")
    if base_rate > 0.35:
        pytest.skip(
            f"Injury/fatal base rate {base_rate:.1%} too high; "
            f"max achievable capture = {max_achievable:.1%} ≤ {CAPTURE_MINIMUM:.0%} threshold"
        )

    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])

    k           = max(1, int(len(composites) * 0.10))
    top_k_idx   = np.argsort(composites)[::-1][:k]
    capture     = y_event[top_k_idx].sum() / max(1, y_event.sum())
    print(
        f"\n  Top-10% injury/fatal capture: {capture:.1%}  "
        f"base_rate={base_rate:.1%}  max_achievable={max_achievable:.1%}  "
        f"(minimum: {CAPTURE_MINIMUM:.0%})"
    )
    assert capture >= CAPTURE_MINIMUM, (
        f"Top-10% injury/fatal capture {capture:.1%} below minimum {CAPTURE_MINIMUM:.0%}"
    )


def test_top_10pct_capture_strong():
    """Top 10% should capture ≥ CAPTURE_STRONG of injury/fatal events (strong model)."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    y_event    = np.array([r["any_injury_fatal"] for r in rows], dtype=int)
    base_rate  = float(y_event.mean())
    if y_event.sum() < MIN_BINARY_POS:
        pytest.skip(f"Insufficient injury/fatal events ({y_event.sum()})")
    if base_rate > 0.35:
        pytest.skip(
            f"Injury/fatal base rate {base_rate:.1%} too high for meaningful capture metric"
        )

    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    k           = max(1, int(len(composites) * 0.10))
    top_k_idx   = np.argsort(composites)[::-1][:k]
    capture     = y_event[top_k_idx].sum() / max(1, y_event.sum())
    print(f"\n  Top-10% injury/fatal capture: {capture:.1%}  (strong target: {CAPTURE_STRONG:.0%})")
    assert capture >= CAPTURE_STRONG, (
        f"Top-10% injury/fatal capture {capture:.1%} below strong target {CAPTURE_STRONG:.0%}"
    )


# ====================================================================== #
#  Test 7: Monotonicity — composite score bin means
# ====================================================================== #

def test_bin_mean_monotonicity():
    """Mean adverse outcome must be non-decreasing across composite score bins."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, y_adv = _get_val_data()
    if len(rows) < MIN_PAIRED:
        pytest.skip("Insufficient data")

    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    y          = np.array(y_adv)

    bin_means = _compute_bin_means(composites, y, n_bins=5)
    inversions = sum(
        1 for i in range(len(bin_means) - 1)
        if bin_means[i + 1] < bin_means[i] - 0.5       # allow 0.5-pt tolerance
    )
    print(f"\n  Bin means: {[f'{m:.1f}' for m in bin_means]}")
    print(f"  Monotonicity inversions: {inversions} (max allowed: {MAX_CALIBRATION_VIOLATIONS})")
    assert inversions <= MAX_CALIBRATION_VIOLATIONS, (
        f"{inversions} non-monotonic bin transitions; max allowed {MAX_CALIBRATION_VIOLATIONS}"
    )


def test_wr_prob_monotonic_bins():
    """WR/serious predicted probability must increase across composite score quintiles."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    p_wr       = np.array([p["p_serious_wr_event"] for p in preds])

    bin_means  = _compute_bin_means(composites, p_wr, n_bins=5)
    inversions = sum(
        1 for i in range(len(bin_means) - 1)
        if bin_means[i + 1] < bin_means[i] - 0.01
    )
    print(f"\n  WR prob quintile means: {[f'{m:.4f}' for m in bin_means]}")
    assert inversions <= MAX_CALIBRATION_VIOLATIONS, (
        f"WR probability bins not monotone: {[f'{m:.4f}' for m in bin_means]}"
    )


# ====================================================================== #
#  Test 8: Separation — high vs low risk
# ====================================================================== #

def test_high_vs_low_risk_separation():
    """Top quartile should have meaningfully higher adverse rates than bottom quartile."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, y_adv = _get_val_data()
    if len(rows) < MIN_PAIRED:
        pytest.skip("Insufficient data")

    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    y          = np.array(y_adv)

    q75  = np.percentile(composites, 75)
    q25  = np.percentile(composites, 25)
    top  = y[composites >= q75]
    bot  = y[composites <= q25]

    mean_top = np.mean(top) if len(top) else 0.0
    mean_bot = np.mean(bot) if len(bot) else 0.0
    ratio    = mean_top / max(mean_bot, 1e-9)
    print(
        f"\n  Top-quartile adverse mean: {mean_top:.2f}  "
        f"Bottom-quartile: {mean_bot:.2f}  Ratio: {ratio:.2f}×"
    )
    assert mean_top > mean_bot, (
        f"Top-quartile adverse mean ({mean_top:.2f}) not > bottom-quartile ({mean_bot:.2f})"
    )
    assert ratio >= 1.3, (
        f"Separation ratio {ratio:.2f}× too low (< 1.3×); high-risk group not distinct"
    )


# ====================================================================== #
#  Test 9: Calibration — p_wr vs actual event rate
# ====================================================================== #

def test_calibration_reliability():
    """Reliability curve: predicted p_wr must be positively correlated with actual rate."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    y_true = np.array([r["any_wr_serious"] for r in rows], dtype=int)
    if y_true.sum() < MIN_BINARY_POS:
        pytest.skip("Insufficient positives for calibration check")

    preds  = mt.predict_batch(X)
    y_pred = np.array([p["p_serious_wr_event"] for p in preds])

    # Compute reliability in 5 equal-size bins
    n_bins = 5
    sorted_idx = np.argsort(y_pred)
    bin_size   = len(sorted_idx) // n_bins
    pred_means = []
    actual_rates = []
    for b in range(n_bins):
        idx_b = sorted_idx[b * bin_size: (b + 1) * bin_size]
        pred_means.append(float(y_pred[idx_b].mean()))
        actual_rates.append(float(y_true[idx_b].mean()))

    rho, _ = spearmanr(pred_means, actual_rates)
    print(f"\n  Calibration reliability Spearman: {rho:.3f}")
    print(f"  Predicted: {[f'{m:.3f}' for m in pred_means]}")
    print(f"  Actual:    {[f'{r:.3f}' for r in actual_rates]}")
    assert rho > 0.5, (
        f"Calibration reliability Spearman {rho:.3f} too low; "
        "predicted probabilities not aligned with actual event rates"
    )


# ====================================================================== #
#  Test 10: Industry robustness
# ====================================================================== #

def test_industry_robustness():
    """Composite Spearman must be positive for major NAICS sectors with enough data."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, y_adv = _get_val_data()
    if len(rows) < MIN_PAIRED:
        pytest.skip("Insufficient data")

    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    y          = np.array(y_adv)

    # Group by 2-digit NAICS prefix (from features: columns 22-46 are one-hot NAICS)
    scorer = _get_scorer()
    naics_sectors = scorer.NAICS_SECTORS
    sector_composites: Dict[str, list] = defaultdict(list)
    sector_outcomes:   Dict[str, list] = defaultdict(list)

    for i, row in enumerate(rows):
        f46 = row["features_46"]
        naics = "unknown"
        for j, sector in enumerate(naics_sectors):
            if f46[22 + j] == 1:
                naics = sector
                break
        sector_composites[naics].append(composites[i])
        sector_outcomes[naics].append(y[i])

    results = {}
    for naics, s_comp in sector_composites.items():
        if len(s_comp) < MIN_INDUSTRY_N:
            continue
        rho, _ = spearmanr(s_comp, sector_outcomes[naics])
        results[naics] = rho

    if not results:
        pytest.skip(f"No NAICS sector has ≥ {MIN_INDUSTRY_N} samples")

    print(f"\n  Industry Spearman rho per sector:")
    for naics in sorted(results):
        n = len(sector_composites[naics])
        print(f"    NAICS {naics}: rho={results[naics]:.3f}  (n={n})")

    neg_count = sum(1 for rho in results.values() if rho < 0)
    total     = len(results)
    assert neg_count <= total // 3, (
        f"Negative Spearman in {neg_count}/{total} industries; model unstable across sectors"
    )


# ====================================================================== #
#  Test 11: Penalty tier ordering
# ====================================================================== #

def test_penalty_tier_threshold_monotonicity():
    """P75 < P90 < P95 for every NAICS sector in the thresholds file."""
    thresh_path = os.path.join(CACHE_DIR, PERCENTILE_CACHE)
    if not os.path.exists(thresh_path):
        pytest.skip("penalty_percentiles.json not found")

    thresholds = load_percentiles(thresh_path)
    violations = []
    for naics, t in thresholds.items():
        if t["p75"] >= t["p90"] or t["p90"] >= t["p95"]:
            violations.append(f"NAICS {naics}: P75={t['p75']:.0f} P90={t['p90']:.0f} P95={t['p95']:.0f}")

    assert not violations, (
        f"Non-monotone penalty thresholds:\n" + "\n".join(violations)
    )


def test_band_monotonicity_fatality_rate():
    """Injury/fatal rate must be non-decreasing across composite score quintiles."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    y_inj = np.array([r["any_injury_fatal"] for r in rows], dtype=float)
    if y_inj.sum() < MIN_BINARY_POS:
        pytest.skip("Insufficient injury/fatal positives for monotonicity check")

    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    bin_means  = _compute_bin_means(composites, y_inj, n_bins=5)

    inversions = sum(
        1 for i in range(len(bin_means) - 1)
        if bin_means[i + 1] < bin_means[i] - 0.005   # 0.5 pp tolerance
    )
    print(f"\n  Injury/fatal rate quintile means: {[f'{m:.4f}' for m in bin_means]}")
    print(f"  Inversions: {inversions} (max allowed: {MAX_CALIBRATION_VIOLATIONS})")
    assert inversions <= MAX_CALIBRATION_VIOLATIONS, (
        f"Injury/fatal rate not monotone across bands: {[f'{m:.4f}' for m in bin_means]}"
    )


def test_band_monotonicity_wr_rate():
    """WR/serious event rate must be non-decreasing across composite score quintiles."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    y_wr       = np.array([r["any_wr_serious"] for r in rows], dtype=float)
    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    bin_means  = _compute_bin_means(composites, y_wr, n_bins=5)

    inversions = sum(
        1 for i in range(len(bin_means) - 1)
        if bin_means[i + 1] < bin_means[i] - 0.01   # 1 pp tolerance
    )
    print(f"\n  WR/serious rate quintile means: {[f'{m:.4f}' for m in bin_means]}")
    print(f"  Inversions: {inversions} (max allowed: {MAX_CALIBRATION_VIOLATIONS})")
    assert inversions <= MAX_CALIBRATION_VIOLATIONS, (
        f"WR/serious rate not monotone across bands: {[f'{m:.4f}' for m in bin_means]}"
    )


# ====================================================================== #
#  Test 12: Summary report (diagnostic, always passes)
# ====================================================================== #

def test_summary_report():
    """Print comprehensive quality metrics. Always passes (diagnostic only)."""
    mt = _get_mt_scorer()
    X, rows, y_adv = _get_val_data()

    print("\n" + "=" * 70)
    print("  MULTI-TARGET MODEL QUALITY REPORT")
    print("=" * 70)

    if not rows:
        print("  No validation data available.")
        print("=" * 70)
        return
    if mt is None:
        print("  Multi-target model not loaded (run build_cache.py).")
        print("=" * 70)
        return

    n = len(rows)
    y = np.array(y_adv)
    y_wr  = np.array([r["any_wr_serious"]         for r in rows], dtype=int)
    y_inj = np.array([r["any_injury_fatal"]        for r in rows], dtype=int)
    y_pen = np.array([r["future_total_penalty"]    for r in rows])
    y_grav = np.array([r["gravity_weighted_score"] for r in rows], dtype=float)

    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    p_wr       = np.array([p["p_serious_wr_event"]  for p in preds])
    p_inj      = np.array([p["p_injury_event"]      for p in preds])
    p_pen      = np.array([p["expected_penalty_usd"] for p in preds])
    p_grav     = np.array([p["gravity_score"]       for p in preds])

    rho_comp,   _ = spearmanr(composites, y)
    lift           = _compute_decile_lift(composites, y, decile=10)
    k              = max(1, int(n * 0.10))
    top_k_idx      = np.argsort(composites)[::-1][:k]
    capture_inj    = y_inj[top_k_idx].sum() / max(1, y_inj.sum())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auroc_wr  = roc_auc_score(y_wr,  p_wr)  if y_wr.sum() >= MIN_BINARY_POS else float("nan")
        auroc_inj = roc_auc_score(y_inj, p_inj) if y_inj.sum() >= MIN_BINARY_POS else float("nan")
    rho_pen,  _ = spearmanr(p_pen,  y_pen)
    rho_grav, _ = spearmanr(p_grav, y_grav)

    print(f"  Validation cutoff:              {CUTOFF_DATE}")
    print(f"  N establishments:               {n:,}")
    print(f"  WR/Serious positive rate:       {y_wr.mean():.1%}")
    print(f"  Hospitalization/Fatal rate:     {y_inj.mean():.1%}")
    print()
    print(f"  === Ranking Power ===")
    print(f"  Composite Spearman rho:           {rho_comp:.4f}  "
          f"({'STRONG' if rho_comp >= SPEARMAN_STRONG else 'MINIMUM' if rho_comp >= SPEARMAN_MINIMUM else 'WEAK'})")

    print(f"  Top-decile lift:                {lift:.3f}×  "
          f"({'STRONG' if lift >= LIFT_STRONG else 'MINIMUM' if lift >= LIFT_MINIMUM else 'WEAK'}×)")
    print(f"  Top-10% injury/fatal capture:   {capture_inj:.1%}")
    print()
    print(f"  === Binary Heads ===")
    print(f"  WR/Serious AUROC (p_event):     {auroc_wr:.4f}  (target ≥ 0.78)")
    print(f"  Hospitalization AUROC (p_inj):  {auroc_inj:.4f}  (target ≥ 0.74)")
    print()
    print(f"  === Regression Heads ===")
    print(f"  Log-penalty Spearman:           {rho_pen:.4f}")
    print(f"  Gravity Spearman:               {rho_grav:.4f}")
    print()
    bin_means = _compute_bin_means(composites, y, n_bins=5)
    print(f"  === Monotonicity (5-bin composite vs adversity) ===")
    print(f"  {[f'{m:.1f}' for m in bin_means]}")
    wr_bands = _compute_bin_means(composites, y_wr.astype(float), n_bins=5)
    inj_bands = _compute_bin_means(composites, y_inj.astype(float), n_bins=5)
    print(f"  WR/Serious rate quinitles:      {[f'{m:.3f}' for m in wr_bands]}")
    print(f"  Injury/Fatal rate quintiles:    {[f'{m:.3f}' for m in inj_bands]}")
    print()
    print(f"  Composite weights: {[f'{w:.3f}' for w in mt._weights]}")
    print("=" * 70)


# ====================================================================== #
#  Internal helpers
# ====================================================================== #

def _compute_decile_lift(
    scores: np.ndarray, outcomes: np.ndarray, decile: int = 10
) -> float:
    """Compute top-decile lift: mean outcome in top decile / overall mean."""
    k         = max(1, int(len(scores) * decile / 100))
    top_idx   = np.argsort(scores)[::-1][:k]
    top_mean  = float(outcomes[top_idx].mean())
    overall   = float(outcomes.mean())
    return top_mean / max(overall, 1e-9)


def _compute_bin_means(
    scores: np.ndarray, outcomes: np.ndarray, n_bins: int = 5
) -> List[float]:
    """Compute mean outcome per equal-size score bin (sorted by score)."""
    sorted_idx = np.argsort(scores)
    bin_size   = max(1, len(sorted_idx) // n_bins)
    means = []
    for b in range(n_bins):
        idx = sorted_idx[b * bin_size: (b + 1) * bin_size]
        if len(idx) == 0:
            continue
        means.append(float(outcomes[idx].mean()))
    return means
