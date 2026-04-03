from typing import List, Optional
import math
import logging
from datetime import date, timedelta

from src.models.manufacturer import Manufacturer
from src.models.osha_record import OSHARecord
from src.models.assessment import RiskAssessment, ProbabilisticRiskTargets
from src.scoring.ml_risk_scorer import MLRiskScorer
from src.scoring.score_outlook import compute_12m_outlook

logger = logging.getLogger(__name__)

# Bühlmann credibility prior: sites with fewer than K inspections regress
# toward the portfolio mean; K=5 matches NCCI workers'-comp experience-rating.
_CREDIBILITY_K = 5

# Tail-exposure blend: final composite = (1-α)*credibility_mean + α*max_composite
# where α scales with risk_concentration so concentrated risk weighs the worst
# site more heavily (capped at 30% to avoid max dominating large clean portfolios).
_TAIL_BLEND_MAX_ALPHA = 0.30


class RiskAssessor:
    def __init__(self, osha_client=None):
        self.ml_scorer = MLRiskScorer(osha_client=osha_client)
        # Load multi-target scorer lazily (no exception if not yet built)
        from src.scoring.multi_target_scorer import MultiTargetRiskScorer
        self._mt_scorer: Optional[MultiTargetRiskScorer] = (
            MultiTargetRiskScorer.load_if_exists(self.ml_scorer.CACHE_DIR)
        )

    def assess(self, manufacturer: Manufacturer, records: List[OSHARecord]) -> RiskAssessment:
        """
        Assess risk using the ML-weighted scoring model, then build a
        human-readable explanation.
        """
        logger.info("Assessing risk for: %s based on %s records.", manufacturer.name, len(records))

        # --- ML scoring (legacy GBR — provides features, percentiles, site scores) ---
        ml_result = self.ml_scorer.score(records)
        risk_score = ml_result["risk_score"]
        percentile_rank = ml_result["percentile_rank"]
        feature_weights = ml_result["feature_weights"]
        features = ml_result["features"]
        industry_label = ml_result.get("industry_label", "Unknown Industry")
        industry_group = ml_result.get("industry_group")
        industry_percentile = ml_result.get("industry_percentile", 50.0)
        industry_comparison = ml_result.get("industry_comparison", [])
        missing_naics = ml_result.get("missing_naics", False)

        # Per-establishment fields
        establishment_count = ml_result.get("establishment_count", 1)
        site_scores = ml_result.get("site_scores", [])
        risk_concentration = ml_result.get("risk_concentration", 0.0)
        systemic_risk_flag = ml_result.get("systemic_risk_flag", False)
        aggregation_warning = ml_result.get("aggregation_warning", "")
        concentration_warning = ml_result.get("concentration_warning", "")

        # ── Multi-target probabilistic predictions ─────────────────────
        # Canonical approach: run the MT model **per establishment**, then
        # aggregate per-site composites using Bühlmann credibility theory.
        #
        # Rationale: pooling all records across N sites conflates per-inspection
        # rates — one low-risk site with 100 inspections can drown out a
        # high-risk site with 5 inspections.  Per-site scoring then credibility-
        # weighted aggregation (same math as NCCI workers'-comp experience
        # rating) preserves site-level signal throughout.
        #
        # Fallback: when per-site features are unavailable, falls back to the
        # legacy pooled approach so single-establishment companies are unaffected.
        risk_targets_obj: Optional[ProbabilisticRiskTargets] = None
        mt_predictions: Optional[dict] = None  # used by compute_12m_outlook

        if self._mt_scorer is not None and self._mt_scorer.is_fitted:
            try:
                from src.scoring.penalty_percentiles import load_percentiles, lookup_threshold
                import os
                from collections import Counter

                thresh_path = os.path.join(self.ml_scorer.CACHE_DIR, "penalty_percentiles.json")
                thresholds  = load_percentiles(thresh_path)
                naics_votes = Counter(r.naics_code for r in records if r.naics_code)
                top_naics   = naics_votes.most_common(1)[0][0] if naics_votes else None
                naics_2d    = str(top_naics)[:2] if top_naics else None
                p90_thresh  = lookup_threshold(thresholds, naics_2d, "p90")

                # _log_feats is stored inside each site_scores entry by
                # score_establishments() but is not a top-level key in the
                # score() return dict, so reconstruct it here.
                per_site_features = [s.get("_log_feats") for s in site_scores]
                has_per_site = (
                    len(per_site_features) == len(site_scores) > 0
                    and all(f is not None for f in per_site_features)
                )

                if has_per_site and establishment_count >= 2:
                    # ── Per-site MT scoring ───────────────────────────────
                    site_composites: list[float] = []
                    site_n_insps:    list[int]   = []
                    per_site_preds:  list[dict]  = []

                    for site, log_feats in zip(site_scores, per_site_features):
                        n_site = site["n_inspections"]
                        site_records = [r for r in records
                                        if (r.estab_name or "UNKNOWN").upper().strip()
                                        == site["name"]]
                        has_fat = any(a.fatality for r in site_records for a in r.accidents)
                        has_wil = any(v.is_willful for r in site_records for v in r.violations)

                        pred = self._mt_scorer.predict(log_feats[0])
                        comp = self._mt_scorer.composite_score(
                            pred,
                            n_inspections=n_site,
                            has_fatality=has_fat,
                            has_willful=has_wil,
                        )
                        site_composites.append(comp)
                        site_n_insps.append(n_site)
                        per_site_preds.append(pred)

                    # ── Bühlmann credibility aggregation ─────────────────
                    # Step 1: compute portfolio prior (simple mean — no self-
                    # referential dependence on the value we're computing).
                    portfolio_prior = sum(site_composites) / len(site_composites)

                    # Step 2: credibility-adjust each site score.
                    credible = []
                    for comp_i, n_i in zip(site_composites, site_n_insps):
                        Z = n_i / (n_i + _CREDIBILITY_K)
                        credible.append(Z * comp_i + (1.0 - Z) * portfolio_prior)

                    credibility_mean = sum(credible) / len(credible)
                    max_credible     = max(credible)

                    # Step 3: tail-exposure blend — weight worst site more
                    # heavily when risk is concentrated across many sites.
                    alpha = min(risk_concentration * _TAIL_BLEND_MAX_ALPHA, _TAIL_BLEND_MAX_ALPHA)
                    composite = (1.0 - alpha) * credibility_mean + alpha * max_credible

                    # For the outlook and RiskTargets, use the portfolio-level
                    # predictions from the pooled features (rates are already
                    # well-defined at portfolio level for forward projection).
                    agg_X = ml_result.get("_aggregate_features_raw")
                    if agg_X is None:
                        agg_X = self.ml_scorer.extract_features(records)
                    mt_predictions = self._mt_scorer.predict(agg_X[0])

                    # Weighted-average the per-head probabilities for RiskTargets display
                    total_n = sum(site_n_insps)
                    def _wavg(key: str) -> float:
                        return sum(p[key] * n for p, n in zip(per_site_preds, site_n_insps)) / total_n

                    mt_predictions = {
                        "p_serious_wr_event":  _wavg("p_serious_wr_event"),
                        "p_injury_event":      _wavg("p_injury_event"),
                        "p_penalty_ge_p75":    _wavg("p_penalty_ge_p75"),
                        "p_penalty_ge_p90":    _wavg("p_penalty_ge_p90"),
                        "p_penalty_ge_p95":    _wavg("p_penalty_ge_p95"),
                        "expected_penalty_usd": _wavg("expected_penalty_usd"),
                        "gravity_score":       _wavg("gravity_score"),
                    }

                else:
                    # Single establishment or missing per-site features — pooled path unchanged
                    agg_X = ml_result.get("_aggregate_features_raw")
                    if agg_X is None:
                        agg_X = self.ml_scorer.extract_features(records)
                    mt_predictions = self._mt_scorer.predict(agg_X[0])

                    n_total_insp = len(records)
                    has_fatality = any(a.fatality for r in records for a in r.accidents)
                    has_willful  = any(v.is_willful for r in records for v in r.violations)
                    composite = self._mt_scorer.composite_score(
                        mt_predictions,
                        n_inspections=n_total_insp,
                        has_fatality=has_fatality,
                        has_willful=has_willful,
                    )

                risk_targets_obj = ProbabilisticRiskTargets(
                    p_serious_wr_event=round(mt_predictions["p_serious_wr_event"], 4),
                    p_injury_event=round(mt_predictions["p_injury_event"], 4),
                    p_penalty_ge_p75=round(mt_predictions.get("p_penalty_ge_p75", 0.0), 4),
                    p_penalty_ge_p90=round(mt_predictions.get("p_penalty_ge_p90", 0.0), 4),
                    p_penalty_ge_p95=round(mt_predictions.get("p_penalty_ge_p95", 0.0), 4),
                    expected_penalty_usd_12m=round(mt_predictions.get("expected_penalty_usd", 0.0), 2),
                    gravity_score=round(mt_predictions.get("gravity_score", 0.0), 2),
                    composite_risk_score=round(composite, 1),
                )

                # Promote composite to canonical risk_score so score, recommendation,
                # and outlook band are all consistent.
                risk_score = round(composite, 1)

            except Exception as _mt_err:
                logger.warning("  [RiskAssessor] Multi-target prediction failed: %s", _mt_err)

        # --- Recommendation ---
        # Derived from the final risk_score (composite when MT model is available,
        # legacy GBR score otherwise).
        #
        # Multi-site rule: "Do Not Recommend" requires systemic risk spread across
        # sites — a single bad actor in a large portfolio should not trigger a
        # blanket rejection.  However, any site individually scoring ≥ 70 caps
        # the recommendation at "Proceed with Caution" (worst-site exposure floor).
        worst_site_score = max((s["score"] for s in site_scores), default=0.0)

        if risk_score >= 60 and (establishment_count <= 1 or systemic_risk_flag):
            recommendation = "Do Not Recommend"
        elif risk_score >= 60:
            recommendation = "Proceed with Caution"
        elif risk_score < 30:
            recommendation = "Recommend"
        else:
            recommendation = "Proceed with Caution"

        # Worst-site exposure: if any single establishment carries very high risk,
        # the portfolio cannot be recommended regardless of aggregate dilution.
        if recommendation == "Recommend" and worst_site_score >= 70:
            recommendation = "Proceed with Caution"

        # --- Explanation ---
        explanation_lines = self._build_explanation(
            records, risk_score, percentile_rank,
            feature_weights, features,
            industry_label=industry_label,
            industry_group=industry_group,
            industry_percentile=industry_percentile,
            industry_comparison=industry_comparison,
            missing_naics=missing_naics,
        )

        if not records:
            explanation_lines.insert(0, "No OSHA records found. Absence of records does not guarantee safety.")

        explanation = "\n".join(explanation_lines)

        confidence = 0.9 if records else 0.4
        risk_confidence, confidence_detail = self._compute_confidence(
            records, mt_predictions, risk_targets_obj,
        )

        outlook = compute_12m_outlook(
            risk_score=risk_score,
            features=features,
            n_sites=establishment_count,
            risk_targets=mt_predictions,
        )

        return RiskAssessment(
            manufacturer=manufacturer,
            records=records,
            risk_score=risk_score,
            recommendation=recommendation,
            explanation=explanation,
            confidence_score=confidence,
            risk_confidence=risk_confidence,
            confidence_detail=confidence_detail,
            feature_weights=feature_weights,
            percentile_rank=percentile_rank,
            industry_label=industry_label,
            industry_group=industry_group or "",
            industry_percentile=industry_percentile,
            industry_comparison=industry_comparison,
            missing_naics=missing_naics,
            establishment_count=establishment_count,
            site_scores=site_scores,
            risk_concentration=risk_concentration,
            systemic_risk_flag=systemic_risk_flag,
            aggregation_warning=aggregation_warning,
            concentration_warning=concentration_warning,
            outlook=outlook,
            risk_targets=risk_targets_obj,
        )

    # ------------------------------------------------------------------ #
    #  Structured confidence signal
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_confidence(
        records: list,
        mt_predictions: Optional[dict],
        risk_targets: Optional[ProbabilisticRiskTargets],
    ) -> tuple:
        """Return (risk_confidence: str, confidence_detail: dict).

        Factors:
        - n_inspections: more data -> higher confidence
        - recency_years: fresher data -> higher confidence
        - model_agreement: legacy GBR and MT model produce similar signal
        - has_mt_model: whether the probabilistic model ran at all
        """
        n_insp = len(records)
        detail: dict = {"n_inspections": n_insp}

        # Recency — years since most recent inspection
        if records:
            most_recent = max(r.date_opened for r in records)
            recency_years = (date.today() - most_recent).days / 365.25
        else:
            recency_years = 999.0
        detail["recency_years"] = round(recency_years, 1)

        detail["has_mt_model"] = mt_predictions is not None

        # Heuristic scoring
        score = 0.0
        # Inspection volume (0-40 points)
        score += min(n_insp / 10.0, 1.0) * 40.0
        # Recency (0-30 points)
        if recency_years <= 2:
            score += 30.0
        elif recency_years <= 5:
            score += 20.0
        elif recency_years <= 10:
            score += 10.0
        # MT model available (0-30 points)
        if mt_predictions is not None:
            score += 30.0

        detail["confidence_score_raw"] = round(score, 1)

        if score >= 70:
            band = "high"
        elif score >= 40:
            band = "medium"
        else:
            band = "low"

        return band, detail

    # ------------------------------------------------------------------ #
    #  Algorithmic executive summary builder
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_explanation(
        records, risk_score, percentile_rank,
        feature_weights, features,
        industry_label="Unknown Industry",
        industry_group=None,
        industry_percentile=50.0,
        industry_comparison=None,
        missing_naics=False,
    ) -> list:
        from collections import Counter
        industry_comparison = industry_comparison or []

        # ── Aggregate counts ──────────────────────────────────────────
        total_inspections = len(records)
        total_violations = sum(len(r.violations) for r in records)
        total_penalties = sum(v.penalty_amount for r in records for v in r.violations)
        willful_count = sum(1 for r in records for v in r.violations if v.is_willful)
        repeat_count = sum(1 for r in records for v in r.violations if v.is_repeat)
        serious_count = sum(
            1 for r in records for v in r.violations
            if (v.severity or "").lower() in ("serious", "willful", "repeat")
        )
        fatality_count = sum(1 for r in records for a in r.accidents if a.fatality)
        accident_count = sum(len(r.accidents) for r in records)

        three_years_ago = date.today() - timedelta(days=365 * 3)
        recent_count = sum(1 for r in records if r.date_opened >= three_years_ago)

        std_counter: Counter = Counter(
            v.category for r in records for v in r.violations if v.category
        )
        top_standards = std_counter.most_common(3)

        # ── Risk headline ─────────────────────────────────────────────
        if risk_score < 15:
            risk_phrase = "a low risk profile"
            outlook = "Inspection activity is minimal with no significant penalty or severity patterns."
        elif risk_score < 45:
            risk_phrase = "a moderate risk profile"
            outlook = "There is documented enforcement activity worth monitoring, though no systemic safety failures are evident."
        elif risk_score < 70:
            risk_phrase = "an elevated risk profile"
            outlook = "Significant violations, penalties, or repeated enforcement actions have been recorded."
        else:
            risk_phrase = "a high risk profile"
            outlook = "The OSHA record reflects serious, willful, or repeat violations and substantial penalties indicating systemic safety management problems."

        sentences = []

        if not records:
            sentences.append(
                f"This manufacturer has {risk_phrase} based on available OSHA history. "
                "No inspection records were found, though absence of records does not guarantee safety."
            )
            return sentences

        # ── Sentence 1: headline + enforcement snapshot ───────────────
        insp_str = f"{total_inspections} inspection{'s' if total_inspections != 1 else ''}"
        viol_str = f"{total_violations} violation{'s' if total_violations != 1 else ''}" if total_violations else "no violations"
        pen_str = f" and ${total_penalties:,.0f} in total penalties" if total_penalties else ""
        sentences.append(
            f"This manufacturer has {risk_phrase}, with {insp_str} on record producing {viol_str}{pen_str}. {outlook}"
        )

        # ── Sentence 2: severity + accidents ─────────────────────────
        detail_parts = []
        if serious_count:
            detail_parts.append(f"{serious_count} serious or high-severity")
        if willful_count:
            detail_parts.append(f"{willful_count} willful")
        if repeat_count:
            detail_parts.append(f"{repeat_count} repeat")
        if detail_parts:
            sev_str = ", ".join(detail_parts) + f" violation{'s' if sum([serious_count, willful_count, repeat_count]) != 1 else ''}"
        else:
            sev_str = ""

        if accident_count and fatality_count:
            acc_str = f"{accident_count} linked workplace incident{'s' if accident_count != 1 else ''}, including {fatality_count} {'fatality' if fatality_count == 1 else 'fatalities'}"
        elif accident_count:
            acc_str = f"{accident_count} linked workplace incident{'s' if accident_count != 1 else ''} with no fatalities"
        else:
            acc_str = ""

        if sev_str and acc_str:
            sentences.append(f"Of the violations cited, {sev_str} were recorded, alongside {acc_str}.")
        elif sev_str:
            sentences.append(f"Of the violations cited, {sev_str} were recorded.")
        elif acc_str:
            sentences.append(f"There were {acc_str} linked to inspections in this period.")

        # ── Sentence 3: recent activity ───────────────────────────────
        if recent_count:
            sentences.append(
                f"{recent_count} of those inspection{'s were' if recent_count != 1 else ' was'} conducted within the last three years."
            )

        # ── Sentence 4: industry context ─────────────────────────────
        if not missing_naics and industry_label and industry_label != "Unknown Industry":
            naics_str = f" (NAICS {industry_group})" if industry_group else ""
            pct_word = "above" if industry_percentile >= 50 else "below"
            sentences.append(
                f"Within the {industry_label}{naics_str} sector, this manufacturer ranks at the "
                f"{industry_percentile:.0f}th percentile, placing it {pct_word} most industry peers."
            )
        elif missing_naics:
            sentences.append("No NAICS code is on record so industry peer comparison is not available.")

        # ── Sentence 5: top risk drivers ─────────────────────────────
        driver_labels = {
            "log_inspections": "inspection frequency",
            "log_violations": "violation volume",
            "serious_violations": "serious violations",
            "willful_violations": "willful violations",
            "repeat_violations": "repeat violations",
            "log_penalties": "total penalties",
            "avg_penalty": "average penalty per violation",
            "max_penalty": "largest single penalty",
            "recent_ratio": "recent activity concentration",
            "severe_incidents": "severe workplace incidents",
            "violations_per_inspection": "violations per inspection",
            "accident_count": "linked accidents",
            "fatality_count": "fatalities",
            "injury_count": "reported injuries",
            "avg_gravity": "average violation gravity",
            "penalties_per_inspection": "penalties per inspection",
            "clean_ratio": "clean inspection ratio",
            "relative_violation_rate": "violation rate relative to industry peers",
            "relative_penalty": "penalty level relative to industry peers",
            "relative_serious_ratio": "serious violation ratio relative to peers",
            "relative_willful_repeat": "willful and repeat rate relative to peers",
        }
        top_drivers = sorted(feature_weights.items(), key=lambda kv: kv[1], reverse=True)[:3]
        if top_drivers:
            driver_names = [driver_labels.get(f, f.replace("_", " ")) for f, _ in top_drivers]
            if len(driver_names) == 1:
                drivers_str = driver_names[0]
            elif len(driver_names) == 2:
                drivers_str = f"{driver_names[0]} and {driver_names[1]}"
            else:
                drivers_str = f"{driver_names[0]}, {driver_names[1]}, and {driver_names[2]}"
            sentences.append(f"The primary model risk drivers are {drivers_str}.")

        # ── Most-cited standards (appended plain) ─────────────────────
        if top_standards:
            std_list = ", ".join(f"{std} (cited {n} times)" for std, n in top_standards)
            sentences.append(f"The most frequently cited OSHA standards were {std_list}.")

        return sentences


