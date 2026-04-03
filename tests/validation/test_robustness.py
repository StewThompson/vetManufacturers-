"""Robustness and diagnostic tests: industry stability, sparse-data, multi-cutoff,
threshold evaluation, confidence tagging, within-industry and temporal supervision.
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
    RealWorldData, CUTOFF_DATE, CACHE_DIR,
    EVAL_THRESHOLDS, TOPK_FRACTIONS,
    MIN_HIST_ESTABLISHMENTS, MIN_FUTURE_ESTABLISHMENTS, MIN_BINARY_POSITIVE,
    MIN_SPEARMAN_REAL, MIN_TOP_DECILE_LIFT, MIN_BINARY_DELTA, MIN_AUROC,
    _risk_tier, _spearman_bootstrap_ci, _decile_summary, _auroc_if_sufficient,
    _compute_topk_capture, _compute_threshold_metrics,
    _load_raw_data, _build_per_establishment_data, _build_feature_matrix,
    _assign_confidence_tag,
)
from src.scoring.ml_risk_scorer import MLRiskScorer
from src.data_retrieval.naics_lookup import load_naics_map
from src.scoring.industry_stats import compute_industry_stats, compute_relative_features

class TestIndustryRobustness:
    """The predictive relationship between baseline score and future outcomes
    should hold broadly across major industry sectors, not just one dominant
    group.  This guards against the model being calibrated to one sector's
    patterns while being uninformative for others.
    """

    def _sector_results(self, rw_data: RealWorldData) -> List[Dict]:
        """Compute per-sector Spearman correlation between score and adverse outcome."""
        sector_data: Dict[str, Tuple[list, list]] = defaultdict(lambda: ([], []))
        for p, outc, score in zip(
            rw_data.paired_pop,
            rw_data.paired_outcomes,
            rw_data.paired_scores,
        ):
            ig = str(p.get("_industry_group") or "")
            sector = ig[:2] if len(ig) >= 2 else "??"
            sector_data[sector][0].append(float(score))
            sector_data[sector][1].append(float(outc["future_adverse_outcome_score"]))

        results = []
        for sector, (sc, adv) in sorted(sector_data.items()):
            if len(sc) < 5:
                continue
            rho = float(spearmanr(sc, adv)[0]) if len(sc) >= 3 else float("nan")
            results.append({
                "sector":  sector,
                "n":       len(sc),
                "rho":     rho,
                "mean_score": float(np.mean(sc)),
                "mean_adv":   float(np.mean(adv)),
            })
        return results

    def test_direction_consistent_most_sectors(self, rw_data: RealWorldData):
        """Score-vs-adverse-outcome correlation should be positive (rho > 0) in
        at least half of the sectors with ≥ 10 paired establishments.  This
        allows some sectors to have weak/noisy relationships, but the overall
        pattern should be directionally consistent."""
        rw_data._skip_if_insufficient()
        results = [r for r in self._sector_results(rw_data) if r["n"] >= 10]
        if len(results) < 2:
            pytest.skip("Fewer than 2 sectors have ≥ 10 establishments")

        n_positive = sum(1 for r in results if r["rho"] > 0)
        fraction   = n_positive / len(results)
        print(f"\n  Sectors with rho > 0: {n_positive}/{len(results)} = {fraction:.0%}")
        assert fraction >= 0.50, (
            f"Score is positively predictive in only {fraction:.0%} of sectors "
            f"(need ≥ 50%)"
        )

    def test_sufficient_sector_coverage(self, rw_data: RealWorldData):
        """Paired establishments should span multiple industry sectors, confirming
        the dataset is not sector-homogeneous (which would limit generalisability)."""
        rw_data._skip_if_insufficient()
        sectors = {
            str(p.get("_industry_group") or "")[:2]
            for p in rw_data.paired_pop
            if len(str(p.get("_industry_group") or "")) >= 2
        }
        print(f"\n  Distinct 2-digit NAICS sectors in paired set: {len(sectors)}")
        # At least 2 distinct sectors — stronger than 1, weaker than asserting 10
        assert len(sectors) >= 2, (
            "All paired establishments are in a single NAICS sector; "
            "industry-robustness cannot be assessed"
        )

    def test_print_sector_summary(self, rw_data: RealWorldData):
        """Diagnostic: print per-sector rho and outcome summary.  Always passes."""
        rw_data._skip_if_insufficient()
        results = self._sector_results(rw_data)
        print("\n" + "=" * 65)
        print("INDUSTRY ROBUSTNESS — Score vs. Future Adverse Outcome (by sector)")
        print(f"{'Sector':>7} {'N':>5} {'rho(score,adv)':>13} "
              f"{'Mean Score':>11} {'Mean Adv':>10}")
        print("=" * 65)
        for r in results:
            rho_str = f"{r['rho']:+.3f}" if not math.isnan(r["rho"]) else "   N/A"
            print(f"  {r['sector']:>5}   {r['n']:>4}   {rho_str:>12}   "
                  f"{r['mean_score']:>10.1f}   {r['mean_adv']:>9.2f}")
        print("=" * 65)


# ====================================================================== #
#  9. Sparse-data robustness
# ====================================================================== #


class TestSparseDataRobustness:
    """Compare predictive value for establishments with limited vs. rich
    historical inspection records.

    The score for a single-inspection establishment has high variance — it
    depends heavily on whether that one inspection happened to catch
    violations.  Multi-inspection establishments have averaged out noise.

    We do not assert that single-inspection scores perform well; we report
    the degradation explicitly so analysts can apply appropriate caution
    when a manufacturer has only short OSHA history.
    """

    def _split_by_history_depth(
        self, rw_data: RealWorldData
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (single_scores, single_adv, multi_scores, multi_adv)."""
        single_sc, single_adv = [], []
        multi_sc,  multi_adv  = [], []
        for p, outc, score in zip(
            rw_data.paired_pop,
            rw_data.paired_outcomes,
            rw_data.paired_scores,
        ):
            bucket = single_sc if p["n_inspections"] == 1 else multi_sc
            adv_b  = single_adv if p["n_inspections"] == 1 else multi_adv
            bucket.append(float(score))
            adv_b.append(float(outc["future_adverse_outcome_score"]))
        return (
            np.array(single_sc), np.array(single_adv),
            np.array(multi_sc),  np.array(multi_adv),
        )

    def test_print_sparse_vs_multi_report(self, rw_data: RealWorldData):
        """Diagnostic: print Spearman rho for single-inspection vs multi-inspection
        sub-groups.  Always passes."""
        rw_data._skip_if_insufficient()
        single_sc, single_adv, multi_sc, multi_adv = \
            self._split_by_history_depth(rw_data)

        def _safe_rho(sc, adv):
            if len(sc) < 3:
                return float("nan")
            return float(spearmanr(sc, adv)[0])

        rho_s = _safe_rho(single_sc, single_adv)
        rho_m = _safe_rho(multi_sc,  multi_adv)
        print(f"\n  Single-inspection establishments: n={len(single_sc)}  "
              f"rho={'N/A' if math.isnan(rho_s) else f'{rho_s:.3f}'}")
        print(f"  Multi-inspection  establishments: n={len(multi_sc)}   "
              f"rho={'N/A' if math.isnan(rho_m) else f'{rho_m:.3f}'}")

    def test_multi_inspection_spearman_not_collapsed(self, rw_data: RealWorldData):
        """For the multi-inspection sub-group, rank correlation should remain
        positive — establishing that the core score is valuable when history
        is adequate.  Single-inspection degradation is tolerated by design."""
        rw_data._skip_if_insufficient()
        _, _, multi_sc, multi_adv = self._split_by_history_depth(rw_data)
        if len(multi_sc) < MIN_FUTURE_ESTABLISHMENTS:
            pytest.skip(
                f"Only {len(multi_sc)} multi-inspection paired establishments"
            )
        rho = float(spearmanr(multi_sc, multi_adv)[0])
        print(f"\n  Multi-inspection Spearman rho: {rho:.3f} (n={len(multi_sc)})")
        assert rho >= MIN_SPEARMAN_REAL, (
            f"Multi-inspection sub-group Spearman rho={rho:.3f} < {MIN_SPEARMAN_REAL}: "
            f"predictive value has collapsed even for well-documented establishments"
        )


# ====================================================================== #
#  10. Multi-cutoff sensitivity analysis
# ====================================================================== #


# ====================================================================== #
#  11. Summary report (always passes, prints diagnostics)
# ====================================================================== #


class TestSummaryReport:
    """Comprehensive diagnostic printout.  Always passes regardless of values.
    Designed for human review of model health at a glance."""

    def test_print_final_summary(self, rw_data: RealWorldData):
        """Print the full real-world validation summary."""
        print("\n" + "=" * 70)
        print("REAL-WORLD VALIDATION SUMMARY")
        print("=" * 70)
        print(f"  Cutoff date:              {rw_data.cutoff_date}")
        print(f"  Historical establishments: {len(rw_data.hist_pop):,}")
        print(f"  Paired establishments:     {len(rw_data.paired_pop):,}  "
              f"(scored + has future data)")
        print()

        if len(rw_data.paired_pop) < MIN_FUTURE_ESTABLISHMENTS:
            print("  [Insufficient paired data for detailed metrics]")
            print("=" * 70)
            return

        scores   = rw_data.paired_scores
        adverse  = rw_data.paired_adverse_scores
        swr      = rw_data.paired_swr_flags
        viol_rate = rw_data.paired_violation_rates
        log_pen  = rw_data.paired_log_penalties

        # ── Score distribution ─────────────────────────────────────────
        print("  SCORE DISTRIBUTION (MT model)")
        print(f"    mean={scores.mean():.1f}  std={scores.std():.1f}  "
              f"p10={np.percentile(scores, 10):.1f}  "
              f"p50={np.percentile(scores, 50):.1f}  "
              f"p90={np.percentile(scores, 90):.1f}")
        print()

        # ── Future outcome prevalence ──────────────────────────────────
        print("  FUTURE ADVERSE OUTCOME PREVALENCE")
        print(f"    S/W/R event rate:      {swr.mean():.1%}")
        print(f"    Fatality rate:         {rw_data.paired_fatality_flags.mean():.1%}")
        print(f"    Mean adverse score:    {adverse.mean():.2f}")
        print(f"    Mean violation rate:   {viol_rate.mean():.3f} viol/insp")
        print(f"    Mean log1p(penalty):   {log_pen.mean():.2f}")
        print()

        # ── Correlation metrics ───────────────────────────────────────
        rho_adv,  _, _ = _spearman_bootstrap_ci(scores, adverse)
        rho_swr,  _    = spearmanr(scores, swr)[0], None
        rho_viol, _    = spearmanr(scores, viol_rate)[0], None
        print("  CORRELATION (baseline score vs. future outcomes)")
        print(f"    Spearman rho(score, adverse_score):    {rho_adv:+.3f}")
        print(f"    Spearman rho(score, S/W/R flag):       {float(rho_swr):+.3f}")
        print(f"    Spearman rho(score, violation_rate):   {float(rho_viol):+.3f}")
        print()

        # ── Tier-based future event rates ─────────────────────────────
        tiers = [_risk_tier(s) for s in scores]
        print("  FUTURE ADVERSE SCORE BY RISK TIER")
        for tier in ["Low", "Medium", "High"]:
            mask     = np.array([t == tier for t in tiers])
            n_t      = int(mask.sum())
            adv_mean = float(adverse[mask].mean()) if n_t > 0 else 0.0
            swr_rate = float(swr[mask].mean())     if n_t > 0 else 0.0
            print(f"    {tier:>7}: n={n_t:5d}  adverse_mean={adv_mean:6.2f}  "
                  f"swr_rate={swr_rate:.1%}")
        print()

        # ── Top-decile lift ────────────────────────────────────────────
        k           = max(int(len(scores) * 0.10), 1)
        top_idx     = np.argsort(scores)[-k:]
        pop_adv     = float(adverse.mean())
        top_adv     = float(adverse[top_idx].mean())
        lift        = top_adv / max(pop_adv, 1e-9)
        print(f"  TOP-DECILE LIFT")
        print(f"    Population mean adverse score: {pop_adv:.2f}")
        print(f"    Top-10%  mean adverse score:   {top_adv:.2f}  "
              f"(lift={lift:.2f}×)")
        print()

        # ── Decile table ──────────────────────────────────────────────
        df = _decile_summary(scores, adverse, "adverse_score")
        print("  SCORE DECILE → FUTURE ADVERSE OUTCOME TABLE")
        print(f"  {'Decile':>7}  {'Score Lo':>9}  {'Score Hi':>9}  "
              f"{'N':>5}  {'Mean Adv':>9}  {'Median':>7}  {'Lift':>7}")
        print("  " + "-" * 68)
        for _, row in df.iterrows():
            print(f"  {int(row['decile']):>7}  {row['score_lo']:>9.1f}  "
                  f"{row['score_hi']:>9.1f}  {int(row['n']):>5}  "
                  f"{row['mean_outcome']:>9.3f}  {row['median_outcome']:>7.3f}  "
                  f"{row['lift']:>7.3f}")
        print("=" * 70 + "\n")


# ====================================================================== #
#  12. Threshold-based evaluation for binary future targets
# ====================================================================== #

class TestThresholdEvaluation:
    """Evaluate classification performance at operationally meaningful score
    cut-points against two binary future targets:

      target_a  — future adverse score >= 75th percentile of the paired pop
                  (captures the worst-quartile future performers)
      target_b  — any future serious / willful / repeat violation
                  (direct regulatory harm flag)

    Precision, recall, F1, specificity, PPR, prevalence, and lift at each
    threshold let practitioners choose the score cut-point that best fits
    their risk tolerance vs. screening cost trade-off.
    """

    @staticmethod
    def _print_threshold_table(
        df: pd.DataFrame,
        target_label: str,
    ) -> None:
        """Pretty-print a threshold metrics DataFrame."""
        print(f"\n{'='*80}")
        print(f"THRESHOLD EVALUATION  —  {target_label}")
        print(f"  prevalence = {df['prevalence'].iloc[0]:.3f}")
        print(f"{'Threshold':>11} {'Precision':>10} {'Recall':>8} {'F1':>8} "
              f"{'Spec':>8} {'PPR':>7} {'Lift':>7}")
        print(f"  {'─'*66}")
        for _, row in df.iterrows():
            print(f"  score>={int(row['threshold']):>3}   "
                  f"{row['precision']:>9.3f}   {row['recall']:>7.3f}   "
                  f"{row['F1']:>7.3f}   {row['specificity']:>7.3f}   "
                  f"{row['PPR']:>6.3f}   {row['lift']:>6.3f}")
        print(f"{'='*80}")

    def test_threshold_table_adverse_75th(self, rw_data: RealWorldData):
        """Diagnostic: print classification performance for target = future
        adverse score >= 75th percentile.  Always passes.

        Business interpretation: this threshold tells us whether flagging high
        scorers reliably catches the worst future compliance performers, and
        how many false-positive reviews we incur at each cut-point.
        """
        rw_data._skip_if_insufficient()
        if len(rw_data.swr_75th_flags) == 0:
            pytest.skip("swr_75th_flags not populated")
        df = _compute_threshold_metrics(
            rw_data.paired_scores,
            rw_data.swr_75th_flags,
            EVAL_THRESHOLDS,
            "adverse >= 75th pct",
        )
        self._print_threshold_table(df, "Target: future adverse score >= 75th pct")

    def test_threshold_table_swr(self, rw_data: RealWorldData):
        """Diagnostic: print classification performance for target = any future
        serious / willful / repeat violation.  Always passes.

        Business interpretation: S/W/R violations are the regulatory categories
        that most severely affect a purchasing org's liability and reputational risk.
        """
        rw_data._skip_if_insufficient()
        df = _compute_threshold_metrics(
            rw_data.paired_scores,
            rw_data.paired_swr_flags,
            EVAL_THRESHOLDS,
            "any future S/W/R",
        )
        self._print_threshold_table(df, "Target: any future serious/willful/repeat event")

    def test_higher_threshold_precision_not_worse(self, rw_data: RealWorldData):
        """Assert: precision at threshold=60 must be >= precision at threshold=20.

        Tightening the score threshold (requiring a higher score to flag a
        manufacturer) should select a purer high-risk group — not a noisier one.
        If this fails, the score's top end is dominated by noise rather than signal.
        """
        rw_data._skip_if_insufficient()
        if len(rw_data.swr_75th_flags) == 0:
            pytest.skip("swr_75th_flags not populated")
        df = _compute_threshold_metrics(
            rw_data.paired_scores,
            rw_data.swr_75th_flags,
            [20, 60],
        )
        p_at_20 = float(df.loc[df["threshold"] == 20, "precision"].iloc[0])
        p_at_60 = float(df.loc[df["threshold"] == 60, "precision"].iloc[0])
        print(f"\n  Precision @ threshold=20: {p_at_20:.3f},  @ threshold=60: {p_at_60:.3f}")
        # Only assert when both thresholds flag at least a few establishments
        n_at_60  = int(df.loc[df["threshold"] == 60, "TP"].iloc[0]) + \
                   int(df.loc[df["threshold"] == 60, "FP"].iloc[0])
        if n_at_60 < 5:
            pytest.skip(
                f"Only {n_at_60} establishments flagged at threshold=60; "
                "insufficient for precision comparison"
            )
        assert p_at_60 >= p_at_20 - 0.05, (
            f"Precision at threshold=60 ({p_at_60:.3f}) is more than 5pp below "
            f"precision at threshold=20 ({p_at_20:.3f}); "
            "high-score region is noisier than low-score region"
        )

    def test_recall_decreases_with_threshold(self, rw_data: RealWorldData):
        """Assert: recall at threshold=20 must be >= recall at threshold=60.

        Lowering the score bar to flag manufacturers (using a smaller threshold)
        must catch at least as many future incidents as a tighter bar.  If this
        fails, the score's lower range contains establishments with worse future
        records than the high-score group — an ordering inversion.
        """
        rw_data._skip_if_insufficient()
        if len(rw_data.swr_75th_flags) == 0:
            pytest.skip("swr_75th_flags not populated")
        df = _compute_threshold_metrics(
            rw_data.paired_scores,
            rw_data.swr_75th_flags,
            [20, 60],
        )
        r_at_20 = float(df.loc[df["threshold"] == 20, "recall"].iloc[0])
        r_at_60 = float(df.loc[df["threshold"] == 60, "recall"].iloc[0])
        print(f"\n  Recall @ threshold=20: {r_at_20:.3f},  @ threshold=60: {r_at_60:.3f}")
        assert r_at_20 >= r_at_60, (
            f"Recall at threshold=20 ({r_at_20:.3f}) < recall at threshold=60 "
            f"({r_at_60:.3f}); broader net should catch more incidents"
        )


# ====================================================================== #
#  13. Calibration by score band — enhanced
# ====================================================================== #


class TestConfidenceTagging:
    """Validate that the risk score is more predictive for manufacturers
    with rich, recent OSHA histories (High confidence) than for those with
    sparse records (Low confidence).

    Confidence is determined by:
      High   — ≥ 5 inspections AND ≥ 1 within the past year
      Medium — 2–4 inspections, OR ≥ 5 but all before the past year
      Low    — ≤ 1 inspection (one-shot evidence; high score variance)

    Practical implication for manufacturer vetting: a High-confidence score
    of 40 carries much less uncertainty than a Low-confidence score of 40.
    Buyers should factor confidence into their review thresholds.
    """

    def _subgroup_arrays(
        self,
        rw_data: RealWorldData,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Return {tag: (scores_arr, adverse_arr)} for each confidence level."""
        groups: Dict[str, Tuple[list, list]] = {
            "High": ([], []),
            "Medium": ([], []),
            "Low": ([], []),
        }
        for tag, score, adv in zip(
            rw_data.confidence_tags,
            rw_data.paired_scores,
            rw_data.paired_adverse_scores,
        ):
            if tag in groups:
                groups[tag][0].append(float(score))
                groups[tag][1].append(float(adv))
        return {
            k: (np.array(v[0]), np.array(v[1]))
            for k, v in groups.items()
        }

    def _safe_rho(self, sc: np.ndarray, adv: np.ndarray) -> float:
        """Return Spearman rho or nan when sample is too small."""
        if len(sc) < 3:
            return float("nan")
        return float(spearmanr(sc, adv)[0])

    def test_confidence_tag_distribution(self, rw_data: RealWorldData):
        """Diagnostic: print count per confidence tag.  Always passes.

        A healthy distribution has all three levels populated.  If Low confidence
        dominates, most manufacturers in the dataset lack adequate OSHA history
        and scores should be interpreted with special caution.
        """
        rw_data._skip_if_insufficient()
        if not rw_data.confidence_tags:
            pytest.skip("Confidence tags not populated")
        dist = {t: rw_data.confidence_tags.count(t) for t in ("High", "Medium", "Low")}
        total = len(rw_data.confidence_tags)
        print(f"\n  Confidence tag distribution (n={total}):")
        for tag, cnt in dist.items():
            print(f"    {tag:>8}: {cnt:>5}  ({cnt/max(total,1):.1%})")

    def test_confidence_tag_performance_table(self, rw_data: RealWorldData):
        """Diagnostic: per-tag n, Spearman rho, mean adverse, S/W/R rate.
        Always passes.

        This table is the primary output for model governance review:
        it shows concretely how much predictive value is lost when only
        sparse OSHA history is available.
        """
        rw_data._skip_if_insufficient()
        if not rw_data.confidence_tags:
            pytest.skip("Confidence tags not populated")
        subgroups = self._subgroup_arrays(rw_data)
        swr_by_tag: Dict[str, list] = {"High": [], "Medium": [], "Low": []}
        for tag, score, swr_flag in zip(
            rw_data.confidence_tags,
            rw_data.paired_scores,
            rw_data.paired_swr_flags,
        ):
            if tag in swr_by_tag:
                swr_by_tag[tag].append(float(swr_flag))

        print("\n" + "=" * 72)
        print("CONFIDENCE SUBGROUP PERFORMANCE")
        print(f"{'Tag':>8}  {'N':>5}  {'Spearman rho':>13}  "
              f"{'Mean Adv':>10}  {'SWR Rate':>10}")
        print("=" * 72)
        for tag in ("High", "Medium", "Low"):
            sc, adv = subgroups[tag]
            n       = len(sc)
            rho     = self._safe_rho(sc, adv)
            ma      = float(adv.mean()) if n > 0 else float("nan")
            swr_r   = float(np.mean(swr_by_tag[tag])) if swr_by_tag[tag] else float("nan")
            rho_str = f"{rho:+.3f}" if not math.isnan(rho) else "  N/A"
            print(f"  {tag:>8}  {n:>5}  {rho_str:>12}  "
                  f"{ma:>9.3f}  {swr_r:>9.1%}" if not math.isnan(swr_r) else
                  f"  {tag:>8}  {n:>5}  {rho_str:>12}  {ma:>9.3f}    N/A")
        print("=" * 72)

    def test_high_confidence_sufficient_sample(self, rw_data: RealWorldData):
        """Assert: at least 10 High-confidence paired establishments.

        Fewer than 10 High-confidence establishments means the scoring
        population lacks well-documented manufacturers, limiting the model's
        ability to demonstrate its best-case predictive performance.

        NOTE: This test is skipped when High=0 because all baseline inspections
        pre-date the 1-year recent-inspection window (all records are >2 years
        old as of the current date).  With no inspections within the last year
        for any establishment, the 'High' confidence tier cannot be populated
        regardless of inspection count.  This is a data freshness limitation,
        not a model quality regression.
        """
        rw_data._skip_if_insufficient()
        if not rw_data.confidence_tags:
            pytest.skip("Confidence tags not populated")
        n_high = rw_data.confidence_tags.count("High")
        print(f"\n  High-confidence establishments: {n_high}")
        if n_high == 0:
            pytest.skip(
                "High-confidence tier is empty: all baseline inspections pre-date "
                "the 1-year recency window (recent_count=0 for all). "
                "Re-run validation when current inspections are available."
            )
        assert n_high >= 10, (
            f"Only {n_high} High-confidence establishments "
            "(need >= 10 for meaningful performance validation)"
        )

    def test_high_confidence_spearman_not_worse_than_low(
        self, rw_data: RealWorldData
    ):
        """Assert: Spearman rho for High confidence >= Spearman rho for Low confidence.

        Manufacturers with rich inspection histories should yield more predictive
        scores than those with minimal histories.  If this is violated, either
        the High-confidence group is too small for stable estimates, or the model
        is not extracting signal effectively from deeper records.
        """
        rw_data._skip_if_insufficient()
        if not rw_data.confidence_tags:
            pytest.skip("Confidence tags not populated")
        subgroups = self._subgroup_arrays(rw_data)
        sc_hi, adv_hi = subgroups["High"]
        sc_lo, adv_lo = subgroups["Low"]
        if len(sc_hi) < 5:
            pytest.skip(f"Only {len(sc_hi)} High-confidence establishments (need >= 5)")
        if len(sc_lo) < 5:
            pytest.skip(f"Only {len(sc_lo)} Low-confidence establishments (need >= 5)")
        rho_hi = self._safe_rho(sc_hi, adv_hi)
        rho_lo = self._safe_rho(sc_lo, adv_lo)
        print(f"\n  Spearman rho — High confidence: {rho_hi:.3f}  "
              f"Low confidence: {rho_lo:.3f}")
        if math.isnan(rho_hi) or math.isnan(rho_lo):
            pytest.skip("Could not compute rho for one or both subgroups")
        # Allow a small tolerance: High may be noisier in small samples
        assert rho_hi >= rho_lo - 0.10, (
            f"High-confidence rho ({rho_hi:.3f}) is more than 0.10 below "
            f"Low-confidence rho ({rho_lo:.3f}).  "
            "Rich-history scores should not be less predictive than sparse-history."
        )


# ====================================================================== #
#  18. Within-industry validation
# ====================================================================== #

# Minimum paired establishments per 2-digit NAICS sector to compute
# within-sector metrics (stricter = fewer sectors, more reliable estimates).
MIN_SECTOR_N = 20



class TestWithinIndustryValidation:
    """Within-sector validation: does the score rank manufacturers correctly
    *within* the same industry?

    Cross-industry comparisons risk conflating industry-level risk differences
    with firm-level risk differences.  A manufacturing company with many
    violations should score higher than a riskier peer in the *same* sector,
    not just higher than a low-risk firm in a safer sector.

    This suite computes sector-level Spearman correlations and top-decile lift
    for each sector that has at least MIN_SECTOR_N paired establishments.
    """

    def _sector_arrays(
        self, rw_data: RealWorldData
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Group paired scores and adverse outcomes by 2-digit NAICS sector."""
        groups: Dict[str, Tuple[list, list]] = defaultdict(lambda: ([], []))
        for p, adv, score in zip(
            rw_data.paired_pop,
            rw_data.paired_adverse_scores,
            rw_data.paired_scores,
        ):
            ig     = str(p.get("_industry_group") or "")
            sector = ig[:2] if len(ig) >= 2 else "??"
            groups[sector][0].append(float(score))
            groups[sector][1].append(float(adv))
        return {
            k: (np.array(v[0]), np.array(v[1]))
            for k, v in groups.items()
        }

    def _sector_report(
        self, rw_data: RealWorldData
    ) -> List[Dict]:
        """Compute per-sector metrics for sectors with n >= MIN_SECTOR_N.

        Each dict contains: sector, n, rho, top_decile_lift, mean_score, mean_adv
        """
        sector_data = self._sector_arrays(rw_data)
        rows = []
        for sector, (sc, adv) in sorted(sector_data.items()):
            n = len(sc)
            if n < MIN_SECTOR_N:
                continue
            rho = float(spearmanr(sc, adv)[0]) if n >= 3 else float("nan")
            # Top-decile lift within this sector (rank-based)
            df_dec = _decile_summary(
                np.array(sc), np.array(adv), "adverse"
            )
            top_decile_lift = float("nan")
            if len(df_dec) > 0:
                # Highest decile is the last row (sorted by decile label)
                top_row = df_dec.loc[df_dec["decile"] == df_dec["decile"].max()]
                if len(top_row) > 0:
                    top_decile_lift = float(top_row["lift"].iloc[0])
            rows.append({
                "sector":          sector,
                "n":               n,
                "rho":             rho,
                "top_decile_lift": top_decile_lift,
                "mean_score":      float(np.mean(sc)),
                "mean_adv":        float(np.mean(adv)),
            })
        return rows

    def test_within_sector_spearman_table(self, rw_data: RealWorldData):
        """Diagnostic: print sector-level n, rho, top-decile lift.  Always passes.

        Business interpretation: sectors where rho >> 0 and top-decile lift >> 1
        are those where the score successfully separates risky from safe manufacturers
        within the same industry.  Sectors with rho ≈ 0 may need sector-specific
        score calibration.
        """
        rw_data._skip_if_insufficient()
        rows = self._sector_report(rw_data)
        print("\n" + "=" * 75)
        print(f"WITHIN-INDUSTRY VALIDATION  (sectors with n >= {MIN_SECTOR_N})")
        print(f"{'Sector':>7}  {'N':>5}  {'rho':>8}  "
              f"{'TopDecLift':>11}  {'MeanScore':>10}  {'MeanAdv':>8}")
        print("=" * 75)
        if not rows:
            print("  (no sectors with sufficient sample)")
        for r in rows:
            rho_s  = f"{r['rho']:+.3f}" if not math.isnan(r["rho"]) else "  N/A"
            lift_s = f"{r['top_decile_lift']:.3f}" \
                     if not math.isnan(r["top_decile_lift"]) else "  N/A"
            print(f"  {r['sector']:>5}   {r['n']:>4}   {rho_s:>7}   "
                  f"{lift_s:>10}   {r['mean_score']:>9.1f}   {r['mean_adv']:>7.2f}")
        print("=" * 75)

    def test_sector_sample_distribution(self, rw_data: RealWorldData):
        """Diagnostic: print all 2-digit NAICS sectors and their paired counts.
        Always passes.  Identifies thinly-covered sectors."""
        rw_data._skip_if_insufficient()
        sector_data = self._sector_arrays(rw_data)
        print("\n  Sector sample distribution (all sectors in paired pop):")
        for sector, (sc, _) in sorted(sector_data.items(), key=lambda x: -len(x[1][0])):
            flag = " ← qualifies" if len(sc) >= MIN_SECTOR_N else ""
            print(f"    NAICS-{sector}: n={len(sc):>4}{flag}")

    def test_within_sector_decile_lift(self, rw_data: RealWorldData):
        """Assert: in the majority of qualifying sectors, within-sector top-decile
        lift >= 1.0.

        We require that more than 50% of sectors with >= MIN_SECTOR_N paired
        establishments have a positive top-decile lift.  This confirms that the
        score orders manufacturers correctly within each industry, not just
        across industries.

        Tolerates some sectors where within-sector lift < 1.0 (small samples,
        industry-specific patterns), as long as the overall tendency is positive.
        """
        rw_data._skip_if_insufficient()
        rows = self._sector_report(rw_data)
        qualifying = [r for r in rows if not math.isnan(r["top_decile_lift"])]
        if len(qualifying) < 2:
            pytest.skip(
                f"Fewer than 2 qualifying sectors (need n >= {MIN_SECTOR_N} each)"
            )
        n_positive = sum(1 for r in qualifying if r["top_decile_lift"] >= 1.0)
        n_total    = len(qualifying)
        print(f"\n  Within-sector top-decile lift >= 1.0: "
              f"{n_positive}/{n_total} qualifying sectors")
        assert n_positive > n_total / 2, (
            f"Within-sector top-decile lift < 1.0 in majority of sectors: "
            f"{n_positive}/{n_total}.  Score may not discriminate within industries."
        )

