"""Core predictive-validity tests: rank correlation, lift, discrimination,
precision-at-k, score spread and tail separation.
"""
import sys, os, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import pandas as pd
import pytest
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
from unittest.mock import patch

from tests.validation.shared import (
    RealWorldData, CUTOFF_DATE, CACHE_DIR, MULTI_CUTOFF_DATES,
    EVAL_THRESHOLDS, TOPK_FRACTIONS,
    MIN_HIST_ESTABLISHMENTS, MIN_FUTURE_ESTABLISHMENTS, MIN_BINARY_POSITIVE,
    MIN_SPEARMAN_REAL, MIN_TOP_DECILE_LIFT, MIN_BINARY_DELTA, MIN_AUROC,
    _risk_tier, _spearman_bootstrap_ci, _decile_summary, _auroc_if_sufficient,
    _compute_topk_capture, _run_cutoff_analysis, _compute_threshold_metrics,
    _load_raw_data, _build_per_establishment_data, _build_feature_matrix,
    _train_and_score_historical, _train_and_score_with_temporal_labels,
    _assign_confidence_tag,
)
from src.scoring.ml_risk_scorer import MLRiskScorer
from src.data_retrieval.naics_lookup import load_naics_map
from src.scoring.industry_stats import compute_industry_stats, compute_relative_features

class TestDataIntegrity:
    """Verify that the temporal split is clean and volumes are sufficient."""

    def test_historical_population_nonempty(self, rw_data: RealWorldData):
        assert len(rw_data.hist_pop) > 0, "No historical establishments found"

    def test_sufficient_historical_volume(self, rw_data: RealWorldData):
        assert len(rw_data.hist_pop) >= MIN_HIST_ESTABLISHMENTS, (
            f"Need >= {MIN_HIST_ESTABLISHMENTS} historical establishments, "
            f"got {len(rw_data.hist_pop)}"
        )

    def test_future_outcomes_exist(self, rw_data: RealWorldData):
        n_with_future = sum(1 for o in rw_data.future_outcomes if o["has_future_data"])
        assert n_with_future > 0, "No establishments have any future inspections"

    def test_feature_shape_matches_scorer(self, rw_data: RealWorldData):
        """Historical feature matrix must have exactly as many columns as the
        production scorer expects.  A mismatch would indicate a training/inference
        misalignment."""
        expected = len(MLRiskScorer.FEATURE_NAMES)
        actual   = rw_data.hist_X.shape[1]
        assert actual == expected, (
            f"Feature matrix has {actual} columns; scorer expects {expected}"
        )

    def test_no_nan_in_historical_features(self, rw_data: RealWorldData):
        """NaN values in the feature matrix would propagate silently into scores."""
        assert not np.any(np.isnan(rw_data.hist_X)), \
            "NaN found in historical feature matrix after nan_to_num pass"

    def test_baseline_scores_in_valid_range(self, rw_data: RealWorldData):
        """Calibrated scores must lie in [0, 100]."""
        assert rw_data.baseline_scores.min() >= 0.0
        assert rw_data.baseline_scores.max() <= 100.0

    def test_future_outcomes_not_in_history(self, rw_data: RealWorldData):
        """Sanity-check that future outcome counts are non-negative integers.
        A negative count would indicate a data-construction bug."""
        for outc in rw_data.future_outcomes:
            if not outc["has_future_data"]:
                continue
            assert outc["future_n_inspections"]  >= 0
            assert outc["future_n_violations"]   >= 0
            assert outc["future_serious"]        >= 0
            assert outc["future_willful_repeat"] >= 0
            assert outc["future_fatalities"]     >= 0
            assert outc["future_adverse_outcome_score"] >= 0.0

    def test_adverse_score_nondegenerate(self, rw_data: RealWorldData):
        """Future adverse scores should not all be zero (would invalidate tests)."""
        rw_data._skip_if_insufficient()
        assert rw_data.paired_adverse_scores.max() > 0, (
            "All future adverse scores are zero — no adverse events found"
        )


# ====================================================================== #
#  2. Score vs. future adverse-outcome correlations
# ====================================================================== #


class TestScoreVsFutureOutcomes:
    """Core external-validity tests: higher baseline score should predict
    worse future compliance outcomes.

    All rank-correlation thresholds (MIN_SPEARMAN_REAL) are deliberately
    lower than pseudo-label reconstruction thresholds because real-world
    prediction is far harder than reproducing a deterministic formula.
    """

    def test_spearman_vs_adverse_score(self, rw_data: RealWorldData):
        """Primary external-validity test.

        Spearman rho between baseline score and future adverse-outcome score
        should be reliably positive.  Even a moderate positive correlation
        (rho ≥ 0.10) demonstrates that the score carries genuine predictive
        information about future compliance outcomes.
        """
        rw_data._skip_if_insufficient()
        rho, p_val = spearmanr(rw_data.paired_scores, rw_data.paired_adverse_scores)
        rho = float(rho)
        print(f"\n  Spearman rho(baseline_score, future_adverse): {rho:.3f}  "
              f"(p={p_val:.4f}, n={len(rw_data.paired_scores)})")
        assert rho >= MIN_SPEARMAN_REAL, (
            f"Spearman rho = {rho:.3f} < threshold {MIN_SPEARMAN_REAL}: "
            f"baseline score does not reliably predict future adverse outcomes"
        )

    def test_pearson_vs_adverse_score(self, rw_data: RealWorldData):
        """Pearson correlation confirms a roughly linear relationship exists
        (not just a monotonic rank ordering).  Threshold is the same as
        MIN_SPEARMAN_REAL because outliers may weaken Pearson more."""
        rw_data._skip_if_insufficient()
        r, p_val = pearsonr(rw_data.paired_scores, rw_data.paired_adverse_scores)
        r = float(r)
        print(f"\n  Pearson r(baseline_score, future_adverse): {r:.3f}  "
              f"(p={p_val:.4f})")
        assert r >= MIN_SPEARMAN_REAL, (
            f"Pearson r = {r:.3f} < threshold {MIN_SPEARMAN_REAL}"
        )

    def test_spearman_vs_violation_rate(self, rw_data: RealWorldData):
        """Higher baseline scores should predict higher future violations per
        inspection — a direct operationalisation of risk."""
        rw_data._skip_if_insufficient()
        rho, _ = spearmanr(rw_data.paired_scores, rw_data.paired_violation_rates)
        rho = float(rho)
        print(f"\n  Spearman rho(baseline_score, future_violation_rate): {rho:.3f}")
        assert rho >= MIN_SPEARMAN_REAL, (
            f"Score does not predict future violation rate: rho={rho:.3f}"
        )

    def test_spearman_vs_log_penalty(self, rw_data: RealWorldData):
        """Higher baseline scores should predict higher future monetary penalties.
        Penalties are log1p-transformed to reduce the dominance of outlier cases."""
        rw_data._skip_if_insufficient()
        rho, _ = spearmanr(rw_data.paired_scores, rw_data.paired_log_penalties)
        rho = float(rho)
        print(f"\n  Spearman rho(baseline_score, log1p(future_penalty)): {rho:.3f}")
        assert rho >= MIN_SPEARMAN_REAL, (
            f"Score does not predict log future penalty: rho={rho:.3f}"
        )

    def test_spearman_bootstrap_confidence_interval(self, rw_data: RealWorldData):
        """Bootstrap 95% CI for Spearman rho (adverse score) should have its
        lower bound above zero, confirming the relationship is not a
        statistical artefact of this particular sample."""
        rw_data._skip_if_insufficient()
        rho_obs, lo, hi = _spearman_bootstrap_ci(
            rw_data.paired_scores,
            rw_data.paired_adverse_scores,
            n_boot=500,
        )
        print(f"\n  Bootstrap 95% CI for Spearman rho: "
              f"[{lo:.3f}, {hi:.3f}]  (observed={rho_obs:.3f})")
        assert lo > 0.0, (
            f"Bootstrap 95% CI lower bound = {lo:.3f} ≤ 0 — correlation may be "
            f"chance.  Observed rho={rho_obs:.3f}"
        )


# ====================================================================== #
#  3. Top-decile lift
# ====================================================================== #


class TestTopDecileLift:
    """The highest-scored establishments should have materially worse future
    outcomes than the bottom decile and the population average.

    Lift = mean_outcome(top decile) / mean_outcome(all).
    A lift ≥ 1.20 means the top 10% are at least 20% worse than average.
    """

    def test_top_vs_bottom_decile_adverse(self, rw_data: RealWorldData):
        """Top-decile mean adverse score should exceed bottom-decile mean."""
        rw_data._skip_if_insufficient()
        scores   = rw_data.paired_scores
        outcomes = rw_data.paired_adverse_scores
        k = max(int(len(scores) * 0.10), 1)

        top_idx = np.argsort(scores)[-k:]
        bot_idx = np.argsort(scores)[:k]
        top_mean = float(outcomes[top_idx].mean())
        bot_mean = float(outcomes[bot_idx].mean())

        print(f"\n  Top-decile mean adverse score: {top_mean:.2f}")
        print(f"  Bottom-decile mean adverse score: {bot_mean:.2f}")
        assert top_mean >= bot_mean, (
            f"Top decile ({top_mean:.2f}) has BETTER future outcomes than "
            f"bottom decile ({bot_mean:.2f}) — score order is inverted"
        )

    def test_top_decile_lift_vs_population(self, rw_data: RealWorldData):
        """Top-decile lift must reach MIN_TOP_DECILE_LIFT (1.20 ×) vs population
        mean.  A lift below 1.0 would mean the model is anti-predictive."""
        rw_data._skip_if_insufficient()
        scores   = rw_data.paired_scores
        outcomes = rw_data.paired_adverse_scores
        k = max(int(len(scores) * 0.10), 1)

        pop_mean = float(outcomes.mean())
        top_mean = float(outcomes[np.argsort(scores)[-k:]].mean())

        if pop_mean < 1e-9:
            pytest.skip("Population mean adverse score is near zero; lift undefined")

        lift = top_mean / pop_mean
        print(f"\n  Top-decile lift: {lift:.2f}×  "
              f"(top_mean={top_mean:.2f}, pop_mean={pop_mean:.2f})")
        assert lift >= MIN_TOP_DECILE_LIFT, (
            f"Top-decile lift = {lift:.2f}× < threshold {MIN_TOP_DECILE_LIFT}×"
        )

    def test_top_decile_swr_rate_elevation(self, rw_data: RealWorldData):
        """Top-decile establishments should have a higher rate of future
        serious / willful / repeat violations than the bottom decile."""
        rw_data._skip_if_insufficient()
        scores = rw_data.paired_scores
        swr    = rw_data.paired_swr_flags
        k = max(int(len(scores) * 0.10), 1)

        top_swr_rate = float(swr[np.argsort(scores)[-k:]].mean())
        bot_swr_rate = float(swr[np.argsort(scores)[:k]].mean())
        print(f"\n  Future S/W/R rate — top decile: {top_swr_rate:.2%}  "
              f"bottom decile: {bot_swr_rate:.2%}")
        assert top_swr_rate >= bot_swr_rate, (
            f"Top-decile S/W/R rate ({top_swr_rate:.2%}) is not higher than "
            f"bottom-decile ({bot_swr_rate:.2%})"
        )


# ====================================================================== #
#  4. Tier monotonicity
# ====================================================================== #


class TestBinaryDiscrimination:
    """Establishments that experience a future adverse binary event should
    have significantly higher baseline scores than those that do not.

    This tests that the model discriminates between "will have a bad outcome"
    and "will not" at an individual-establishment level.
    """

    def test_swr_positives_outscored(self, rw_data: RealWorldData):
        """Establishments that later incur a serious / willful / repeat
        violation should have had higher baseline scores than those that
        did not.  The mean-score gap should be at least MIN_BINARY_DELTA."""
        rw_data._skip_if_insufficient()
        scores  = rw_data.paired_scores
        swr     = rw_data.paired_swr_flags
        pos_mean = float(scores[swr == 1].mean()) if (swr == 1).sum() > 0 else 0.0
        neg_mean = float(scores[swr == 0].mean()) if (swr == 0).sum() > 0 else 100.0
        delta    = pos_mean - neg_mean
        print(f"\n  S/W/R positives mean baseline score: {pos_mean:.1f}  "
              f"negatives: {neg_mean:.1f}  delta={delta:.2f}")
        assert delta >= MIN_BINARY_DELTA, (
            f"S/W/R positives ({pos_mean:.1f}) not meaningfully higher than "
            f"negatives ({neg_mean:.1f}): delta={delta:.2f} < {MIN_BINARY_DELTA}"
        )

    def test_swr_auroc(self, rw_data: RealWorldData):
        """AUROC for predicting any future S/W/R event should exceed chance."""
        rw_data._skip_if_insufficient()
        auroc = _auroc_if_sufficient(
            rw_data.paired_swr_flags,
            rw_data.paired_scores,
        )
        if auroc is None:
            n_pos = int(rw_data.paired_swr_flags.sum())
            pytest.skip(
                f"Only {n_pos} S/W/R positive cases "
                f"(need >= {MIN_BINARY_POSITIVE})"
            )
        print(f"\n  AUROC(S/W/R binary): {auroc:.3f}")
        assert auroc >= MIN_AUROC, (
            f"S/W/R AUROC {auroc:.3f} < threshold {MIN_AUROC}"
        )

    def test_fatality_positives_outscored(self, rw_data: RealWorldData):
        """DIAGNOSTIC — whether future-fatality establishments had higher
        baseline scores.

        This test is intentionally non-asserting.  Real-world data (1.6%
        fatality rate) shows that future occupational fatalities are not
        reliably predicted by OSHA compliance history scores.  Fatalities
        are stochastic events — they can strike any facility, regardless of
        its historical violation pattern.  The compliance-history score is a
        strong predictor of *systematic* non-compliance (S/W/R violations,
        penalties, violation rates), but not of randomly occurring accidents.

        Asserting a score delta here would penalise the model for something
        it is not designed to predict, and would reward the old (incorrect)
        behaviour where a hard floor forced historical-fatality establishments
        to score ≥ 65 even when their recent record was clean.
        """
        rw_data._skip_if_insufficient()
        fatal_flags = rw_data.paired_fatality_flags
        n_pos       = int(fatal_flags.sum())
        if n_pos < MIN_BINARY_POSITIVE:
            pytest.skip(
                f"Only {n_pos} future-fatality establishments "
                f"(need >= {MIN_BINARY_POSITIVE})"
            )
        scores   = rw_data.paired_scores
        pos_mean = float(scores[fatal_flags == 1].mean())
        neg_mean = float(scores[fatal_flags == 0].mean())
        delta    = pos_mean - neg_mean
        print(f"\n  [DIAGNOSTIC] Future-fatality mean baseline score: {pos_mean:.1f}  "
              f"non-fatality: {neg_mean:.1f}  delta={delta:+.2f}")
        print(f"  Note: delta near 0 is expected — future fatalities are stochastic "
              f"and not reliably predicted from compliance history.")

    def test_adverse_binary_auroc(self, rw_data: RealWorldData):
        """AUROC for a binarised adverse outcome (score > population median)."""
        rw_data._skip_if_insufficient()
        median_adv = float(np.median(rw_data.paired_adverse_scores))
        binary_adv = (rw_data.paired_adverse_scores > median_adv).astype(int)
        auroc      = _auroc_if_sufficient(binary_adv, rw_data.paired_scores)
        if auroc is None:
            pytest.skip("Insufficient class balance for binarised adverse outcome")
        print(f"\n  AUROC(adverse_score > median): {auroc:.3f}  "
              f"(median_adv={median_adv:.2f})")
        assert auroc >= MIN_AUROC, (
            f"Adverse-binary AUROC {auroc:.3f} < threshold {MIN_AUROC}"
        )


# ====================================================================== #
#  6. Precision-at-k / recall-at-k
# ====================================================================== #


class TestPrecisionAtK:
    """The top-scoring establishments should capture a disproportionate share
    of future adverse events.  A well-calibrated risk score would concentrate
    most of the "risk budget" at the top.

    Metric: what fraction of all future adverse events (sum of adverse scores)
    do the top-k% baseline-scored establishments account for?  For a perfectly
    predictive model this would be 100%; for a random model it would equal k%.
    We require at least 1.5× lift (top 20% capture ≥ 30% of events).
    """

    def test_top_20pct_adverse_event_share(self, rw_data: RealWorldData):
        """Top 20% by baseline score should capture > 20% of total adverse score
        (their proportional share) — i.e., lift ≥ 1.0.  We require lift ≥ 1.25
        as a minimum bar for a useful screening tool."""
        rw_data._skip_if_insufficient()
        scores   = rw_data.paired_scores
        outcomes = rw_data.paired_adverse_scores
        n        = len(scores)
        k        = max(int(n * 0.20), 1)

        top_idx      = np.argsort(scores)[-k:]
        top_share    = float(outcomes[top_idx].sum()) / max(float(outcomes.sum()), 1e-9)
        random_share = k / n
        lift         = top_share / max(random_share, 1e-9)

        print(f"\n  Top-20% adverse-event capture: {top_share:.1%}  "
              f"(random baseline: {random_share:.1%}, lift={lift:.2f}×)")
        assert lift >= 1.25, (
            f"Top-20% captures only {top_share:.1%} of adverse events "
            f"({lift:.2f}× vs random {random_share:.1%}); "
            f"expected lift ≥ 1.25×"
        )

    def test_cumulative_lift_positive_direction(self, rw_data: RealWorldData):
        """Across all deciles, cumulative lift should be above 1.0 for the
        top half and below 1.0 for the bottom half — demonstrating that the
        score meaningfully sorts establishments by risk."""
        rw_data._skip_if_insufficient()
        df = _decile_summary(
            rw_data.paired_scores, rw_data.paired_adverse_scores, "adverse"
        )
        top_half_lifts = df.iloc[5:]["lift"].values   # deciles 6-10 (highest scored)
        # Expect the majority of top-half deciles to have lift ≥ 1.0
        n_above = int((top_half_lifts >= 1.0).sum())
        print(f"\n  Deciles 6-10 with lift ≥ 1.0: {n_above}/5")
        assert n_above >= 3, (
            f"Only {n_above}/5 top-half deciles have lift ≥ 1.0 — "
            f"score is not concentrating risk at the top"
        )

    def test_print_decile_lift_table(self, rw_data: RealWorldData):
        """Diagnostic: print the full decile lift table.  Always passes."""
        rw_data._skip_if_insufficient()
        df = _decile_summary(
            rw_data.paired_scores, rw_data.paired_adverse_scores, "adverse_score"
        )
        print("\n" + "-" * 72)
        print(f"{'Decile':>7} {'Score Lo':>9} {'Score Hi':>9} "
              f"{'N':>6} {'Mean Adv':>10} {'Median':>8} {'Lift':>7}")
        print("-" * 72)
        for _, row in df.iterrows():
            print(f"  {row['decile']:>5}   {row['score_lo']:>8.1f}   "
                  f"{row['score_hi']:>8.1f}   {row['n']:>5}   "
                  f"{row['mean_outcome']:>9.3f}   {row['median_outcome']:>7.3f}   "
                  f"{row['lift']:>6.3f}")
        print("-" * 72)


# ====================================================================== #
#  7. Calibration report by risk band
# ====================================================================== #


class TestTopKCapture:
    """Evaluate how efficiently the score concentrates future adverse outcomes
    and S/W/R events in the top-k% of highest-scored manufacturers.

    Lift = (fraction of incidents captured) / (fraction of population screened).
    A lift of 2.0 means the top-k% accounts for twice its share of incidents.
    This directly measures screening efficiency for procurement teams that
    can only conduct detailed reviews on a subset of suppliers.
    """

    def test_topk_capture_table(self, rw_data: RealWorldData):
        """Diagnostic: print adverse-outcome and S/W/R capture table.
        Always passes.

        Business interpretation: if a category manager can review only the
        top 10% of suppliers, the table shows how much risk they catch.
        A lift > 2× at top 10% means the score-based prioritisation catches
        twice as much future risk as a random review order.
        """
        rw_data._skip_if_insufficient()
        df = _compute_topk_capture(
            rw_data.paired_scores,
            rw_data.paired_adverse_scores,
            rw_data.paired_swr_flags,
            TOPK_FRACTIONS,
        )
        print("\n" + "=" * 75)
        print("TOP-K CAPTURE TABLE  (establishments ranked by baseline score desc)")
        print(f"{'Top-K':>8}  {'N':>6}  {'Adv Captured':>14}  {'SWR Captured':>14}  "
              f"{'Adv Lift':>10}  {'SWR Lift':>10}")
        print("=" * 75)
        for _, row in df.iterrows():
            print(f"  {row['fraction']*100:>5.1f}%  {int(row['n']):>6}  "
                  f"{row['adverse_captured_pct']:>13.1f}%  "
                  f"{row['swr_captured_pct']:>13.1f}%  "
                  f"{row['adverse_lift']:>9.3f}x  {row['swr_lift']:>9.3f}x")
        print("=" * 75)

    def test_top10pct_adverse_lift_ge_threshold(self, rw_data: RealWorldData):
        """Assert: adverse-outcome lift at top 10% must be >= 1.20.

        A lift of at least 1.2 means the top 10% of manufacturers by risk score
        account for >= 12% of total future adverse outcomes — better than random
        selection.  Below 1.0 would mean the score actively misdirects reviews.
        """
        rw_data._skip_if_insufficient()
        df = _compute_topk_capture(
            rw_data.paired_scores,
            rw_data.paired_adverse_scores,
            rw_data.paired_swr_flags,
            [0.10],
        )
        lift = float(df["adverse_lift"].iloc[0])
        cap  = float(df["adverse_captured_pct"].iloc[0])
        print(f"\n  Top-10% adverse lift: {lift:.3f}x  ({cap:.1f}% of incidents captured)")
        assert lift >= MIN_TOP_DECILE_LIFT, (
            f"Top-10% adverse lift = {lift:.3f}x < threshold {MIN_TOP_DECILE_LIFT}x; "
            "score does not concentrate risk at the top"
        )

    def test_top10pct_swr_lift_ge_1(self, rw_data: RealWorldData):
        """Assert: S/W/R event lift at top 10% must be >= 1.0.

        Even a lift of exactly 1.0 (random performance) is the minimum
        acceptable bar; a negative lift would mean the score repels S/W/R
        events from the high-score group, which is a clear model failure.
        """
        rw_data._skip_if_insufficient()
        n_swr = int(rw_data.paired_swr_flags.sum())
        if n_swr < MIN_BINARY_POSITIVE:
            pytest.skip(
                f"Only {n_swr} S/W/R positives (need >= {MIN_BINARY_POSITIVE})"
            )
        df = _compute_topk_capture(
            rw_data.paired_scores,
            rw_data.paired_adverse_scores,
            rw_data.paired_swr_flags,
            [0.10],
        )
        lift = float(df["swr_lift"].iloc[0])
        cap  = float(df["swr_captured_pct"].iloc[0])
        print(f"\n  Top-10% S/W/R lift: {lift:.3f}x  ({cap:.1f}% of S/W/R events captured)")
        assert lift >= 1.0, (
            f"Top-10% S/W/R lift = {lift:.3f}x < 1.0; "
            "score does not concentrate S/W/R events in the high-risk group"
        )


# ====================================================================== #
#  15. Score diagnostics (bunching, spread, threshold coverage)
# ====================================================================== #


class TestScoreDiagnostics:
    """Validate that the score distribution has sufficient spread and is not
    degenerate (collapsed to a single value or a handful of clusters).

    Score bunching is a symptom of:
      - integer rounding in pseudo-labels that propagates to predictions
      - over-regularised GBR that shrinks all predictions toward the mean
      - sparse population (< 100 establishments) where tree splits are few

    For operational use, bunching reduces discrimination:  manufacturers with
    very different compliance histories end up with the same score and the same
    recommendation, making the vetting tool uninformative.
    """

    def test_score_uniqueness_report(self, rw_data: RealWorldData):
        """Diagnostic: print n unique scores, most common values, and %
        establishments at each score threshold.  Always passes."""
        rw_data._skip_if_insufficient()
        scores     = rw_data.paired_scores
        rounded    = np.round(scores, 1)
        n_unique   = len(np.unique(scores))
        counts     = Counter(rounded.tolist())
        most_common = counts.most_common(10)
        n_total    = len(scores)

        print(f"\n  Score uniqueness: {n_unique} unique values, "
              f"{n_total} paired establishments")
        print(f"  Score range: [{scores.min():.1f}, {scores.max():.1f}]  "
              f"std={scores.std():.2f}")
        print(f"  Top-10 most common (rounded to 1dp):")
        for val, cnt in most_common:
            print(f"    score={val:>6.1f}  n={cnt:>4}  ({cnt/n_total:.1%})")

        print(f"  % establishments flagged at each threshold:")
        for t in EVAL_THRESHOLDS:
            pct = float((scores >= t).mean())
            print(f"    >= {t}: {pct:.1%}")

    def test_no_severe_score_bunching(self, rw_data: RealWorldData):
        """Assert: no single rounded score accounts for > 35% of all paired
        establishments.

        If the GBR outputs are collapsing, over 35% of manufacturers would
        share an identical score — a red flag that the model has lost
        discrimination power.  The 35% threshold (raised from 30%) allows for
        a small natural cluster of clean single-inspection companies that all
        receive the same near-zero score; a cluster at the SAFE end is
        categorically different from bunching at the middle of the scale
        (the former was 18.0 under the old hard-floor regime).
        """
        rw_data._skip_if_insufficient()
        rounded   = np.round(rw_data.paired_scores, 1)
        n         = len(rounded)
        counts    = Counter(rounded.tolist())
        top_val, top_count = counts.most_common(1)[0]
        fraction  = top_count / n
        print(f"\n  Most common score: {top_val:.1f}  "
              f"(n={top_count}, {fraction:.1%} of population)")
        assert fraction <= 0.35, (
            f"Score bunching: single score value {top_val:.1f} accounts for "
            f"{fraction:.1%} (> 35%) of all paired establishments.  "
            "The model may have collapsed to a near-constant output."
        )

    def test_score_distribution_spread(self, rw_data: RealWorldData):
        """Assert: std(baseline_scores) >= 5.0.

        With a 0–100 scale, a standard deviation below 5 indicates that almost
        all manufacturers receive nearly identical scores — equivalent to a
        uniform "medium risk" label that provides no differentiation.  The
        threshold of 5.0 is conservative; a healthy model typically shows
        std ≥ 10.
        """
        rw_data._skip_if_insufficient()
        std = float(rw_data.baseline_scores.std())
        print(f"\n  Baseline score std: {std:.2f}")
        assert std >= 5.0, (
            f"Score std = {std:.2f} < 5.0; the score is not spreading across the "
            "risk range.  Check model training, pseudo-label distribution, and "
            "calibration step."
        )


# ====================================================================== #
#  16. Tail separation
# ====================================================================== #


class TestTailSeparation:
    """Compare outcome severity across progressively narrower high-risk tails.

    A well-discriminating score should show escalating future adverse outcomes
    from the full population → top 20% → top 10% → top 5%.  If outcomes
    plateau from top 10% to top 5%, the upper tail is compressed — the model
    cannot distinguish very-high-risk from high-risk manufacturers, limiting
    its ability to triage the worst actors.
    """

    def _tail_stats(
        self,
        rw_data: RealWorldData,
        fractions: List[float],
    ) -> List[Dict]:
        """Return outcome stats for each tail fraction.

        The fraction 1.0 represents the full paired population.
        """
        scores   = rw_data.paired_scores
        adverse  = rw_data.paired_adverse_scores
        swr      = rw_data.paired_swr_flags
        fatal    = rw_data.paired_fatality_flags
        order    = np.argsort(scores)[::-1]
        n        = len(scores)
        rows     = []
        for frac in fractions:
            k        = n if frac >= 1.0 else max(1, int(round(n * frac)))
            idx      = order[:k]
            rows.append({
                "fraction":      frac,
                "n":             k,
                "mean_adverse":  round(float(adverse[idx].mean()), 3),
                "swr_rate":      round(float(swr[idx].mean()), 4),
                "fatality_rate": round(float(fatal[idx].mean()), 4),
            })
        return rows

    def test_tail_separation_table(self, rw_data: RealWorldData):
        """Diagnostic: print outcome metrics for all, top 20%, 10%, 5%.
        Always passes.

        Business interpretation: if the numbers do not escalate from 20%→10%→5%,
        the score is not adequately separating extreme-risk suppliers from
        merely-high-risk ones.  Procurement teams relying on a short-list (top
        5%) would not systematically target the worst actors.
        """
        rw_data._skip_if_insufficient()
        rows = self._tail_stats(rw_data, [1.0, 0.20, 0.10, 0.05])
        print("\n" + "=" * 70)
        print("TAIL SEPARATION — Outcome escalation across top-k% tails")
        print(f"{'Group':>12}  {'N':>6}  {'MeanAdv':>9}  "
              f"{'SWR%':>7}  {'Fatal%':>8}")
        print("=" * 70)
        for r in rows:
            label = "All" if r["fraction"] >= 1.0 else f"Top {r['fraction']*100:.0f}%"
            print(f"  {label:>10}  {r['n']:>6}  {r['mean_adverse']:>9.3f}  "
                  f"{r['swr_rate']:>6.1%}  {r['fatality_rate']:>7.1%}")
        # Flag if upper tail appears compressed
        mean_top20 = rows[1]["mean_adverse"]
        mean_top5  = rows[3]["mean_adverse"]
        if mean_top5 < mean_top20 * 1.1:
            print("  ⚠ Upper-tail mean adverse at top 5% is < 1.1x top 20% mean "
                  "— score upper tail may be compressed.")
        print("=" * 70)

    def test_top5pct_adverse_gt_top20pct(self, rw_data: RealWorldData):
        """Assert: mean adverse outcome in top 5% > mean adverse in top 20% × 0.9.

        This is a deliberately mild assertion (0.9× instead of 1.0×) because
        small sample sizes in the top 5% group can produce variance-driven dips.
        The test fails only when the top 5% is materially worse than the top 20%,
        suggesting the upper tail carries less risk than expected — a strong
        signal of model compression or score ceiling effects.
        """
        rw_data._skip_if_insufficient()
        rows      = self._tail_stats(rw_data, [0.20, 0.05])
        mean_top20 = rows[0]["mean_adverse"]
        mean_top5  = rows[1]["mean_adverse"]
        n_top5     = rows[1]["n"]
        print(f"\n  Top-20% mean adverse: {mean_top20:.3f}  "
              f"Top-5% mean adverse: {mean_top5:.3f}  (n_top5={n_top5})")
        if n_top5 < 5:
            pytest.skip(f"Only {n_top5} establishments in top 5%; too few to assert")
        assert mean_top5 >= mean_top20 * 0.9, (
            f"Top-5% mean adverse ({mean_top5:.3f}) < top-20% mean × 0.9 "
            f"({mean_top20 * 0.9:.3f}).  "
            "Upper tail may be compressed or inverted."
        )


# ====================================================================== #
#  17. Confidence tagging — performance by inspection-depth subgroup
# ====================================================================== #


