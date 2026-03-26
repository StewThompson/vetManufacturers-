from typing import List
from datetime import date, timedelta

from src.models.manufacturer import Manufacturer
from src.models.osha_record import OSHARecord
from src.models.assessment import RiskAssessment
from src.scoring.ml_risk_scorer import MLRiskScorer


class RiskAssessor:
    def __init__(self, osha_client=None):
        self.ml_scorer = MLRiskScorer(osha_client=osha_client)

    def assess(self, manufacturer: Manufacturer, records: List[OSHARecord], reputation_data: List[dict] = None) -> RiskAssessment:
        """
        Assess risk using the ML-weighted scoring model, then build a
        human-readable explanation.
        """
        reputation_data = reputation_data or []
        print(f"Assessing risk for: {manufacturer.name} based on {len(records)} records and {len(reputation_data)} news items.")

        # --- ML scoring ---
        ml_result = self.ml_scorer.score(records, reputation_data)
        risk_score = ml_result["risk_score"]
        percentile_rank = ml_result["percentile_rank"]
        feature_weights = ml_result["feature_weights"]
        features = ml_result["features"]
        reputation_score = ml_result["reputation_score"]
        news_sentiment = ml_result["news_sentiment"]
        predicted_serious_prob = ml_result["predicted_serious_prob"]
        predicted_expected_violations = ml_result["predicted_expected_violations"]
        predictive_statement = ml_result["predictive_statement"]

        # --- Recommendation ---
        if risk_score < 30:
            recommendation = "Recommend"
        elif risk_score < 60:
            recommendation = "Proceed with Caution"
        else:
            recommendation = "Do Not Recommend"

        # --- Explanation ---
        explanation_lines = self._build_explanation(
            records, reputation_data, risk_score, percentile_rank,
            feature_weights, features, news_sentiment, reputation_score,
            predictive_statement,
        )

        if not records and not reputation_data:
            explanation_lines.insert(0, "No OSHA records or news found. Absence of records does not guarantee safety.")

        explanation = "\n".join(explanation_lines)

        confidence = 0.9 if (records or reputation_data) else 0.4

        return RiskAssessment(
            manufacturer=manufacturer,
            records=records,
            reputation_score=reputation_score,
            news_sentiment=news_sentiment,
            reputation_summary=f"Analyzed {len(reputation_data)} articles." if reputation_data else "No news found.",
            reputation_data=reputation_data,
            risk_score=risk_score,
            recommendation=recommendation,
            explanation=explanation,
            confidence_score=confidence,
            feature_weights=feature_weights,
            percentile_rank=percentile_rank,
            predicted_serious_prob=predicted_serious_prob,
            predicted_expected_violations=predicted_expected_violations,
            predictive_statement=predictive_statement,
        )

    # ------------------------------------------------------------------ #
    #  Explanation builder
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_explanation(
        records, reputation_data, risk_score, percentile_rank,
        feature_weights, features, news_sentiment, reputation_score,
        predictive_statement="",
    ) -> list:
        lines = []

        # Predictive statement (lead with the model's forward-looking prediction)
        if predictive_statement:
            lines.append(f"**Predictive Assessment:** {predictive_statement}")

        # Headline
        if risk_score < 15:
            lines.append("Low risk profile based on available history.")
        elif risk_score < 45:
            lines.append("Moderate risk detected.")
        else:
            lines.append("High risk! Significant enforcement activity, penalties, or negative reputation.")

        lines.append(f"\n**ML Risk Score: {risk_score}/100** (Percentile: {percentile_rank}% — higher means riskier than more peers)")

        # Top feature drivers
        sorted_weights = sorted(feature_weights.items(), key=lambda kv: kv[1], reverse=True)
        top_drivers = sorted_weights[:3]
        driver_labels = {
            "total_inspections": "Inspection Count",
            "total_violations": "Violation Count",
            "serious_violations": "Serious Violations",
            "willful_violations": "Willful Violations",
            "repeat_violations": "Repeat Violations",
            "total_penalties": "Total Penalties ($)",
            "avg_penalty": "Avg Penalty ($)",
            "max_penalty": "Max Single Penalty ($)",
            "recent_ratio": "Recent Activity Ratio",
            "severe_incidents": "Severe Incidents",
            "violations_per_inspection": "Violations / Inspection",
            "accident_count": "Linked Accidents",
            "fatality_count": "Fatalities",
            "injury_count": "Reported Injuries",
            "avg_gravity": "Avg Violation Gravity",
        }
        lines.append("\n**Top Risk Drivers (ML Feature Importance):**")
        for feat, weight in top_drivers:
            label = driver_labels.get(feat, feat)
            val = features.get(feat, 0)
            lines.append(f"  - {label}: {val:.2f}  (weight: {weight:.3f})")

        # OSHA record details
        if records:
            lines.append(f"\nFound {len(records)} OSHA inspection record(s).")
        else:
            lines.append("\nNo OSHA records found.")

        three_years_ago = date.today() - timedelta(days=365 * 3)
        total_penalties = 0.0

        for record in records:
            is_recent = record.date_opened >= three_years_ago
            tag = "[RECENT] " if is_recent else ""
            lines.append(f"\n{tag}Inspection {record.inspection_id} ({record.date_opened}):")

            if record.violations:
                for v in record.violations:
                    total_penalties += v.penalty_amount
                    lines.append(f"  - Standard Cited: {v.category}")
                    lines.append(f"    Description: {v.description}")
                    lines.append(f"    Penalty: ${v.penalty_amount:,.2f}")
                    if v.gravity:
                        lines.append(f"    Gravity: {v.gravity}")
                    if v.hazardous_substance:
                        lines.append(f"    Hazardous Substance: {v.hazardous_substance}")
                    if v.gen_duty_narrative:
                        snippet = v.gen_duty_narrative[:600]
                        if len(v.gen_duty_narrative) > 600:
                            snippet += "…"
                        lines.append(f"    Inspector Notes: {snippet}")
            else:
                lines.append("  - No violations cited. (Clean Inspection)")

            # Accident data for this inspection
            if record.accidents:
                for acc in record.accidents:
                    fat_tag = " ⚠️ FATALITY" if acc.fatality else ""
                    lines.append(f"  *** ACCIDENT{fat_tag} (ID: {acc.summary_nr}, {acc.event_date or 'unknown date'}):")
                    if acc.event_desc:
                        lines.append(f"      {acc.event_desc}")
                    for inj in acc.injuries:
                        lines.append(f"      - Injury: {inj.get('nature', 'Unknown')} to {inj.get('body_part', 'Unknown')} ({inj.get('degree', 'Unknown')})")

        if total_penalties > 0:
            lines.append(f"\nTotal Penalties: ${total_penalties:,.2f}")

        # Reputation
        if reputation_data:
            lines.append(f"\nReputation: {news_sentiment} sentiment (Score: {reputation_score:.1f})")

        return lines
