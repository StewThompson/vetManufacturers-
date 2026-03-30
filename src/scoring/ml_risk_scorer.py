import csv
import math
import os
import json
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from datetime import date, timedelta
from typing import List, Dict, Optional

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.models.osha_record import OSHARecord
from src.data_retrieval.naics_lookup import get_industry_name, load_naics_map
from src.scoring.industry_stats import compute_industry_stats, compute_relative_features
from src.scoring.tail_calibrator import TailCalibrator


# compute_industry_stats and compute_relative_features are re-exported
# from src.scoring.industry_stats for backward compatibility.


class MLRiskScorer:
    """
    Machine-learning risk scorer using scikit-learn.

    Trains a GradientBoostingRegressor on OSHA population data from the
    DOL API so that every manufacturer's risk score is *relative* to the
    broader population of inspected establishments.
    """
    INDUSTRY_MIN_SAMPLE = 10
    # Expand log transformation to more features with large ranges
    # Log-compress only volume counts and accident/fatality/injury rates —
    # NOT penalty amounts.  Large penalties directly signal severity;
    # compressing $10K → $1M to 5.5 → 13.8 destroys high-end differentiation.
    # time_adjusted_penalty (17) is also intentionally unlogged.
    LOG_FEATURE_INDICES = [0, 1, 11, 12, 13]  # inspection/violation counts, accident/fatality/injury rates

    # Canonical 2-digit NAICS sectors for one-hot encoding.
    NAICS_SECTORS = [
        "11", "21", "22", "23", "31", "32", "33",
        "42", "44", "45", "48", "49", "51", "52",
        "53", "54", "55", "56", "61", "62", "71",
        "72", "81", "92",
    ]

    FEATURE_NAMES = [
        # ── Absolute signals (18) ──────────────────────────────────────
        "log_inspections",
        "log_violations",
        "serious_violations",
        "willful_violations",
        "repeat_violations",
        "log_penalties",
        "avg_penalty",
        "max_penalty",
        "recent_ratio",
        "severe_incidents",
        "violations_per_inspection",
        "accident_count",
        "fatality_count",
        "injury_count",
        "avg_gravity",
        "penalties_per_inspection",
        "clean_ratio",
        "time_adjusted_penalty",
        # ── Industry-relative z-scores (4) ────────────────────────────
        # 0.0 sentinel when NAICS unavailable
        "relative_violation_rate",
        "relative_penalty",
        "relative_serious_ratio",
        "relative_willful_repeat",
        # ── 2-digit NAICS sector one-hot (25) ─────────────────────────
        "naics_11", "naics_21", "naics_22", "naics_23",
        "naics_31", "naics_32", "naics_33",
        "naics_42", "naics_44", "naics_45",
        "naics_48", "naics_49", "naics_51", "naics_52",
        "naics_53", "naics_54", "naics_55", "naics_56",
        "naics_61", "naics_62", "naics_71", "naics_72",
        "naics_81", "naics_92",
        "naics_unknown",
    ]

    FEATURE_DISPLAY = {
        "log_inspections": "Inspection Count (log)",
        "log_violations": "Violation Count (log)",
        "serious_violations": "Serious Violations",
        "willful_violations": "Willful Violations",
        "repeat_violations": "Repeat Violations",
        "log_penalties": "Total Penalties (log $)",
        "avg_penalty": "Avg Penalty ($)",
        "max_penalty": "Max Single Penalty ($)",
        "recent_ratio": "Recent Activity (1yr)",
        "severe_incidents": "Fat/Cat Inspections",
        "violations_per_inspection": "Violations / Inspection",
        "accident_count": "Linked Accidents",
        "fatality_count": "Fatalities",
        "injury_count": "Reported Injuries",
        "avg_gravity": "Avg Violation Gravity",
        "penalties_per_inspection": "Penalties / Inspection ($)",
        "clean_ratio": "Clean Inspection Ratio",
        "time_adjusted_penalty": "Time-Adjusted Penalty ($)",
        "relative_violation_rate": "Violation Rate vs. Industry (z)",
        "relative_penalty": "Avg Penalty vs. Industry (z)",
        "relative_serious_ratio": "Serious Ratio vs. Industry (z)",
        "relative_willful_repeat": "Willful+Repeat Rate vs. Industry (z)",
        "naics_11": "Agriculture/Forestry/Fishing",
        "naics_21": "Mining/Oil & Gas",
        "naics_22": "Utilities",
        "naics_23": "Construction",
        "naics_31": "Mfg (Food/Textile/Apparel)",
        "naics_32": "Mfg (Wood/Paper/Chemical/Plastics)",
        "naics_33": "Mfg (Metal/Machinery/Electronics)",
        "naics_42": "Wholesale Trade",
        "naics_44": "Retail Trade",
        "naics_45": "Retail Trade (Misc.)",
        "naics_48": "Transportation/Warehousing",
        "naics_49": "Warehousing/Storage",
        "naics_51": "Information",
        "naics_52": "Finance/Insurance",
        "naics_53": "Real Estate",
        "naics_54": "Professional/Scientific/Technical",
        "naics_55": "Management of Companies",
        "naics_56": "Admin/Support/Waste Mgmt",
        "naics_61": "Educational Services",
        "naics_62": "Health Care/Social Assistance",
        "naics_71": "Arts/Entertainment/Recreation",
        "naics_72": "Accommodation/Food Services",
        "naics_81": "Other Services",
        "naics_92": "Public Administration",
        "naics_unknown": "Industry Unknown",
    }

    CACHE_DIR = "ml_cache"
    MODEL_FILE = "risk_model.pkl"
    POP_FILE = "population_data.json"
    CALIBRATOR_FILE = "tail_calibrator.pkl"

    # ── Temporal supervision ──────────────────────────────────────────────
    # Cutoff used to build the real-label training sample.  Establishments
    # with inspections before this date get their features aggregated;
    # those same establishments' post-cutoff inspections become the
    # real outcome labels that replace circular pseudo-labels during training.
    #
    # Pushing this back (earlier than today-minus-1yr) gives a richer paired
    # sample because more establishments have had time to accumulate post-
    # cutoff inspections.  The current default (2020-01-01) gives ≥6 years of
    # outcome data within the 10-year bulk cache.
    #
    # NOTE: if TEMPORAL_CUTOFF falls outside the bulk cache window
    # (< today - 10yr) the labeler will find no pre-cutoff inspections and
    # fall back gracefully to pseudo-labels only.
    TEMPORAL_LABEL_CUTOFF = date(2020, 1, 1)

    # Maximum number of real-label rows to add to the training matrix.
    # 50_000 exceeds the ~30k paired pool so effectively "use all paired rows".
    TEMPORAL_SAMPLE_SIZE = 50_000

    # Weight multiplier applied to real-label training rows.
    # 8× gives real-label rows ~27% of the effective gradient signal even
    # when the pseudo-label population is 10× larger.  Real labels are direct
    # OSHA observations and should outweigh the heuristic pseudo-labels.
    TEMPORAL_LABEL_WEIGHT = 8.0

    def __init__(self, osha_client=None):
        self.osha_client = osha_client
        self.pipeline: Optional[Pipeline] = None
        self.population_features: Optional[np.ndarray] = None
        self._industry_stats: dict = {}
        self._calibrator: Optional[TailCalibrator] = None
        self._naics_map: dict = load_naics_map()
        self._n_temporal_labels: int = 0  # set by _train(); 0 when not yet trained
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self._load_or_build()

    # ------------------------------------------------------------------ #
    #  Feature transforms
    # ------------------------------------------------------------------ #
    @staticmethod
    def _log_transform_features(X: np.ndarray) -> np.ndarray:
        """Log1p-compress raw count features (inspections, violations, penalties).

        Prevents volume/aggregation bias: a 500-establishment company's
        raw counts become comparable to single-site training data.
        log1p(500)≈6.2 vs log1p(3)≈1.4 — not the 167× raw ratio.
        """
        X = X.copy()
        for i in MLRiskScorer.LOG_FEATURE_INDICES:
            X[..., i] = np.log1p(np.maximum(X[..., i], 0))
        return X

    def _encode_naics(self, naics_code) -> list:
        """One-hot encode 2-digit NAICS prefix. Returns list of len(NAICS_SECTORS)+1."""
        prefix = str(naics_code or "")[:2] if naics_code else ""
        vec = [0] * (len(self.NAICS_SECTORS) + 1)
        if prefix in self.NAICS_SECTORS:
            vec[self.NAICS_SECTORS.index(prefix)] = 1
        else:
            vec[-1] = 1  # naics_unknown
        return vec

    # ------------------------------------------------------------------ #
    #  Feature extraction
    # ------------------------------------------------------------------ #
    def _extract_establishment_features_raw(
        self, records: List[OSHARecord],
    ) -> tuple:
        """Extract the 17 absolute features for one establishment.

        Returns ``(feature_list_17, naics_code)`` where *feature_list_17*
        uses **simple per-inspection rates** — identical semantics to
        ``_fetch_population`` so that train and inference are aligned.

        No recency weighting is applied.  Recency is captured by the
        ``recent_ratio`` feature (inspections within the last year).
        """
        one_year_ago = date.today() - timedelta(days=1095)  # 3-year recency window
        n_insp = len(records)
        recent = 0
        severe = 0
        clean = 0
        n_viols = 0
        serious_raw = 0
        willful_raw = 0
        repeat_raw = 0
        total_pen = 0.0
        all_penalties: list = []
        acc_count = 0
        fat_count = 0
        inj_count = 0
        gravities: list = []
        time_adj_pen = 0.0

        naics_votes: Counter = Counter()

        for r in records:
            viols = r.violations
            n_viols += len(viols)

            serious_raw += sum(1 for v in viols if v.severity == "Serious")
            willful_raw += sum(1 for v in viols if v.is_willful)
            repeat_raw  += sum(1 for v in viols if v.is_repeat)

            pens = [v.penalty_amount for v in viols]
            insp_pen = sum(pens)
            total_pen += insp_pen
            all_penalties.extend(pens)
            age_years = max(0.0, (date.today() - r.date_opened).days / 365.25)
            time_adj_pen += insp_pen * math.exp(-age_years / 3.0)

            if r.date_opened >= one_year_ago:
                recent += 1

            if r.accidents:
                severe += 1
                for a in r.accidents:
                    acc_count += 1
                    if a.fatality:
                        fat_count += 1
                    inj_count += len(a.injuries)

            for v in viols:
                if v.gravity:
                    try:
                        gravities.append(float(v.gravity))
                    except (ValueError, TypeError):
                        pass

            if not viols:
                clean += 1

            if r.naics_code:
                naics_votes[r.naics_code] += 1

        naics_code = naics_votes.most_common(1)[0][0] if naics_votes else None

        # Per-inspection rates — same scale as _fetch_population training data
        avg_pen = float(np.mean(all_penalties)) if all_penalties else 0.0
        max_pen = max(all_penalties) if all_penalties else 0.0
        recent_ratio  = recent      / n_insp if n_insp else 0.0
        vpi           = n_viols     / n_insp if n_insp else 0.0
        avg_gravity   = float(np.mean(gravities)) if gravities else 0.0
        pen_per_insp  = total_pen   / n_insp if n_insp else 0.0
        clean_ratio   = clean       / n_insp if n_insp else 0.0
        serious_rate  = serious_raw / n_insp if n_insp else 0.0
        willful_rate  = willful_raw / n_insp if n_insp else 0.0
        repeat_rate   = repeat_raw  / n_insp if n_insp else 0.0
        severe_rate   = severe      / n_insp if n_insp else 0.0
        acc_rate      = acc_count   / n_insp if n_insp else 0.0
        fat_rate      = fat_count   / n_insp if n_insp else 0.0
        inj_rate      = inj_count   / n_insp if n_insp else 0.0

        # Fraction-of-violations metrics for industry comparison
        raw_serious_rate = serious_raw / max(n_viols, 1)
        raw_wr_rate      = (willful_raw + repeat_raw) / max(n_viols, 1)

        features_17 = [
            n_insp, n_viols, serious_rate, willful_rate, repeat_rate,
            total_pen, avg_pen, max_pen, recent_ratio, severe_rate, vpi,
            acc_rate, fat_rate, inj_rate, avg_gravity,
            pen_per_insp, clean_ratio,
            time_adj_pen,
        ]

        return features_17, naics_code, vpi, avg_pen, raw_serious_rate, raw_wr_rate

    def _complete_features(
        self,
        features_17: list,
        naics_code: str,
        vpi: float,
        avg_pen: float,
        raw_serious_rate: float,
        raw_wr_rate: float,
    ) -> np.ndarray:
        """Append industry z-scores + NAICS one-hot → 1×46 array (pre-log-transform)."""
        industry_group = (
            naics_code[:4]
            if naics_code and len(str(naics_code)) >= 4
            else None
        )

        rel = compute_relative_features(
            {
                "industry_group": industry_group,
                "raw_vpi": vpi,
                "raw_avg_pen": avg_pen,
                "raw_serious_rate": raw_serious_rate,
                "raw_wr_rate": raw_wr_rate,
            },
            self._industry_stats,
            self._naics_map,
            min_sample=self.INDUSTRY_MIN_SAMPLE,
        )

        def _safe(v: float) -> float:
            return 0.0 if (v != v) else v  # NaN → 0.0

        naics_vec = self._encode_naics(naics_code)

        row = features_17 + [
            _safe(rel["relative_violation_rate"]),
            _safe(rel["relative_penalty"]),
            _safe(rel["relative_serious_ratio"]),
            _safe(rel["relative_willful_repeat"]),
        ] + naics_vec

        return np.array([row])

    def extract_features(self, records: List[OSHARecord]) -> np.ndarray:
        """Convert a list of OSHARecords into a 1 × n_features array.

        Backward-compatible entry point.  Treats all *records* as a single
        establishment (aggregated).  Uses simple per-inspection rates —
        identical to the training path in ``_fetch_population``.
        """
        feat17, naics, vpi, avg_pen, sr, wr = (
            self._extract_establishment_features_raw(records)
        )
        raw = self._complete_features(feat17, naics, vpi, avg_pen, sr, wr)
        return self._log_transform_features(raw)

    # ------------------------------------------------------------------ #
    #  Heuristic label generation (domain knowledge) — internal only
    # ------------------------------------------------------------------ #
    @staticmethod
    def _heuristic_label(row: np.ndarray) -> float:
        """Map 46 raw OSHA features → 0-100 heuristic risk score used as
        GBR training target (rescaled to match temporal adverse distribution
        during training when real labels are available)."""
        (n_insp, n_viols, serious, willful, repeat,
         total_pen, avg_pen, max_pen, recent_ratio, severe, vpi,
         accident_count, fatality_count, injury_count, avg_gravity,
         pen_per_insp, clean_ratio,
         time_adj_pen) = row[:18]

        naics_unknown = float(row[-1])

        score = 0.0
        dormant_scale = 0.5 if recent_ratio < 0.01 else 1.0
        fat_dormant_scale = 0.5 if recent_ratio < 0.01 else 1.0
        insp_confidence = min(0.5 + n_insp / 10.0, 1.0)

        if fatality_count > 0:
            eff_fatalities = min(fatality_count * 5.0, 5.0)
            rate_contrib = min(12.0 + max(eff_fatalities - 1.0, 0.0) * 3.0, 24.0)
            abs_proxy = n_insp * fatality_count
            pattern_boost = min(np.log1p(abs_proxy) * 2.0, 8.0)
            score += (rate_contrib + pattern_boost) * fat_dormant_scale * insp_confidence
        score += min(severe * 25.0, 6)
        score += min(injury_count * 6.0, 4)
        score += min(accident_count * 10.0, 4)
        score = min(score, 34)

        score += min(willful * 60.0, 20) * dormant_scale
        score += min(repeat  * 50.0, 15) * dormant_scale
        score += min(serious * 25.0, 15)
        if avg_gravity > 0:
            score += min(avg_gravity * 0.6, 6)

        if recent_ratio > 0 and (willful + repeat) > 0:
            active_wr = recent_ratio * (willful * 3.0 + repeat * 2.0) * 25.0
            score += min(active_wr, 20.0)

        score += recent_ratio * 12.0
        score += min(vpi * 2.5, 10)

        if recent_ratio > 0.5 and vpi >= 1.5:
            trajectory_premium = min((recent_ratio - 0.5) * 2.0 * (vpi - 1.0), 8.0)
            score += trajectory_premium
        elif recent_ratio < 0.2 and vpi < 0.5 and n_insp >= 3:
            score -= min((0.2 - recent_ratio) * 10.0, 5.0)

        if time_adj_pen > 0:
            score += min(time_adj_pen / 40_000, 10.0)
        if pen_per_insp > 0:
            score += min(pen_per_insp / 20_000, 6.0)

        if naics_unknown > 0.5:
            score += 4.0

        is_dormant = recent_ratio < 0.01
        fat_blocks_clean = (
            (fatality_count > 0 or accident_count > 0 or severe > 0)
            and not is_dormant
        )
        if clean_ratio > 0 and not fat_blocks_clean:
            if n_insp >= 3:
                score -= clean_ratio * 10.0
            elif n_insp == 2:
                score -= clean_ratio * 4.0

        if n_insp <= 1:
            score += 2.5
        elif n_insp <= 3:
            score += 1.5
        elif n_insp <= 5:
            score += 0.5

        if fatality_count > 0 and (willful + repeat) > 0:
            score += 8.0 * insp_confidence * dormant_scale
        if willful > 0 and repeat > 0:
            score += 5.0 * insp_confidence * dormant_scale
        if recent_ratio > 0.5 and serious >= 0.25:
            score += 4.0

        return float(np.clip(score, 0, 100))

    _pseudo_label = _heuristic_label

    # ------------------------------------------------------------------ #
    #  Population data from bulk cache
    # ------------------------------------------------------------------ #
    def _fetch_population(self) -> List[Dict]:
        """
        Build population feature matrix from the OSHAClient's bulk cache.
        Zero additional API calls — the client handles caching.
        """
        if self.osha_client is None:
            from src.data_retrieval.osha_client import OSHAClient
            self.osha_client = OSHAClient()

        client = self.osha_client

        print("Building population data from bulk cache…")
        client.ensure_cache()

        insp_data = client.get_bulk_inspections()
        if not insp_data:
            print("  No inspection data in cache.")
            return []

        # Group inspections by establishment
        estab_inspections: Dict[str, list] = defaultdict(list)
        for insp in insp_data:
            name = (insp.get("estab_name") or "UNKNOWN").upper()
            estab_inspections[name].append(insp)

        one_year_ago = date.today() - timedelta(days=1095)  # 3-year recency window
        total_estabs = len(estab_inspections)
        population = []
        progress = 0

        for estab, inspections in estab_inspections.items():
            progress += 1
            if progress % 20000 == 0:
                print(f"    {progress:,}/{total_estabs:,} establishments…")
            n_insp = len(inspections)
            recent = 0
            severe = 0
            clean = 0
            viols = []
            acc_count = 0
            fat_count = 0
            inj_count = 0
            time_adj_pen = 0.0

            # Majority-vote NAICS code for this establishment
            naics_votes: dict = defaultdict(int)
            for insp in inspections:
                act = str(insp.get("activity_nr", ""))
                od = insp.get("open_date", "")
                try:
                    d = date.fromisoformat(od[:10])
                    if d >= one_year_ago:
                        recent += 1
                    age_years = max(0.0, (date.today() - d).days / 365.25)
                except (ValueError, TypeError):
                    age_years = 0.0
                insp_viols = client.get_violations_for_activity(act)
                insp_pen = sum(
                    float(v.get("current_penalty") or v.get("initial_penalty") or 0)
                    for v in insp_viols
                )
                time_adj_pen += insp_pen * math.exp(-age_years / 3.0)
                viols.extend(insp_viols)
                if not insp_viols:
                    clean += 1

                # Accident stats (uses cached indexes, no API calls)
                acc_stats = client.get_accident_count_for_activity(act)
                acc_count += acc_stats["accidents"]
                fat_count += acc_stats["fatalities"]
                inj_count += acc_stats["injuries"]
                if acc_stats["accidents"] > 0:
                    severe += 1

                # Track NAICS votes
                nc = str(insp.get("naics_code") or "").strip()
                if nc and nc.isdigit() and len(nc) >= 4:
                    naics_votes[nc[:4]] += 1

            naics_group = max(naics_votes, key=naics_votes.get) if naics_votes else None

            n_viols = len(viols)
            serious_raw = sum(1 for v in viols if v.get("viol_type") == "S")
            willful_raw = sum(1 for v in viols if v.get("viol_type") == "W")
            repeat_raw  = sum(1 for v in viols if v.get("viol_type") == "R")
            penalties = [
                float(v.get("current_penalty") or v.get("initial_penalty") or 0)
                for v in viols
            ]
            total_pen = sum(penalties)
            avg_pen = float(np.mean(penalties)) if penalties else 0.0
            max_pen = max(penalties) if penalties else 0.0
            recent_ratio = recent / n_insp if n_insp else 0.0
            vpi = n_viols / n_insp if n_insp else 0.0

            # Average violation gravity
            gravities = []
            for v in viols:
                g = v.get("gravity", "")
                if g:
                    try:
                        gravities.append(float(g))
                    except (ValueError, TypeError):
                        pass
            avg_gravity = float(np.mean(gravities)) if gravities else 0.0

            # Normalize count signals to per-inspection rates so that
            # multi-site companies train on the same scale as single sites.
            pen_per_insp  = total_pen    / n_insp if n_insp else 0.0
            clean_ratio   = clean        / n_insp if n_insp else 0.0
            serious_rate  = serious_raw  / n_insp if n_insp else 0.0
            willful_rate  = willful_raw  / n_insp if n_insp else 0.0
            repeat_rate   = repeat_raw   / n_insp if n_insp else 0.0
            severe_rate   = severe       / n_insp if n_insp else 0.0
            acc_rate      = acc_count    / n_insp if n_insp else 0.0
            fat_rate      = fat_count    / n_insp if n_insp else 0.0
            inj_rate      = inj_count    / n_insp if n_insp else 0.0

            # Fraction-of-violations metrics for industry comparison
            raw_serious_rate = serious_raw / max(n_viols, 1)
            raw_wr_rate      = (willful_raw + repeat_raw) / max(n_viols, 1)

            population.append({
                "name": estab,
                "features": [
                    n_insp, n_viols, serious_rate, willful_rate, repeat_rate,
                    total_pen, avg_pen, max_pen, recent_ratio, severe_rate, vpi,
                    acc_rate, fat_rate, inj_rate, avg_gravity,
                    pen_per_insp, clean_ratio,
                    time_adj_pen,
                    # Relative features will be appended below
                ],
                # Scratch fields for industry stats — removed before persisting
                "_industry_group": naics_group,
                "_raw_vpi": vpi,
                "_raw_avg_pen": avg_pen,
                "_raw_serious_rate": raw_serious_rate,
                "_raw_wr_rate": raw_wr_rate,
            })

        print(f"  Aggregated {len(population)} unique establishments.")

        # ── Compute industry stats from full population ────────────────────
        pop_df = pd.DataFrame([
            {
                "industry_group":  p["_industry_group"],
                "raw_vpi":         p["_raw_vpi"],
                "raw_avg_pen":     p["_raw_avg_pen"],
                "raw_serious_rate": p["_raw_serious_rate"],
                "raw_wr_rate":     p["_raw_wr_rate"],
            }
            for p in population
        ])
        self._industry_stats = compute_industry_stats(
            pop_df,
            min_sample=self.INDUSTRY_MIN_SAMPLE,
            naics_map=self._naics_map,
        )
        print(f"  Industry stats: {len(self._industry_stats)} industry groups.")

        # ── Append relative features + NAICS one-hot ──────────────────────
        for p in population:
            ig = p.pop("_industry_group")
            rel = compute_relative_features(
                {
                    "industry_group":  ig,
                    "raw_vpi":         p.pop("_raw_vpi"),
                    "raw_avg_pen":     p.pop("_raw_avg_pen"),
                    "raw_serious_rate": p.pop("_raw_serious_rate"),
                    "raw_wr_rate":     p.pop("_raw_wr_rate"),
                },
                self._industry_stats,
                self._naics_map,
                min_sample=self.INDUSTRY_MIN_SAMPLE,
            )
            # NaN z-scores for missing NAICS → nan_to_num(0.0) in _train()
            p["features"].extend([
                rel["relative_violation_rate"],
                rel["relative_penalty"],
                rel["relative_serious_ratio"],
                rel["relative_willful_repeat"],
            ])
            # NAICS sector one-hot
            naics_2digit = ig[:2] if ig else None
            p["features"].extend(self._encode_naics(naics_2digit))

        return population

    # ------------------------------------------------------------------ #
    #  Temporal label loading
    # ------------------------------------------------------------------ #

    def _load_temporal_labels(self) -> List[Dict]:
        """Load (or build) the real-label training sample from the bulk cache.

        Returns a list of row-dicts from ``temporal_labeler.load_or_build``.
        Returns an empty list when the bulk CSVs are unavailable or the build
        fails gracefully, so callers can treat it as an optional augmentation.
        """
        from src.scoring.labeling.temporal_labeler import load_or_build_temporal_labels

        insp_path = os.path.join(self.CACHE_DIR, "inspections_bulk.csv")
        viol_path = os.path.join(self.CACHE_DIR, "violations_bulk.csv")
        acc_path  = os.path.join(self.CACHE_DIR, "accidents_bulk.csv")
        inj_path  = os.path.join(self.CACHE_DIR, "accident_injuries_bulk.csv")

        outcome_end = date.today()  # use all available post-cutoff data

        try:
            rows = load_or_build_temporal_labels(
                scorer=self,
                cutoff_date=self.TEMPORAL_LABEL_CUTOFF,
                outcome_end_date=outcome_end,
                cache_dir=self.CACHE_DIR,
                inspections_path=insp_path,
                violations_path=viol_path,
                accidents_path=acc_path,
                injuries_path=inj_path,
                naics_map=self._naics_map,
                sample_size=self.TEMPORAL_SAMPLE_SIZE,
            )
        except Exception as e:
            print(f"  Temporal label build failed ({e}); using pseudo-labels only.")
            rows = []
        return rows

    # ------------------------------------------------------------------ #
    #  Model training
    # ------------------------------------------------------------------ #
    def _train(self, population: List[Dict]):
        """Build feature matrix, generate pseudo-labels, train pipeline.

        Training label strategy
        -----------------------
        Step 1 — Pseudo-label population (all ~370k establishments):
            Each row receives a heuristic pseudo-label computed from raw
            OSHA features.  These form the bulk of the training set and
            ensure the model covers all industries and risk levels.

        Step 2 — Real-label augmentation (≤ TEMPORAL_SAMPLE_SIZE rows):
            A stratified sample of paired establishments (pre-TEMPORAL_LABEL_
            CUTOFF features → post-cutoff real adverse outcomes) is appended
            to the training matrix.  These rows break the circular dependency
            for the labeled fraction.

            Real-label rows use features aggregated from pre-cutoff inspections
            only (LEAKAGE GUARD: no future information in features).  Their
            pre-cutoff feature distribution is compatible with the full-history
            population because all features are per-inspection RATES.

            A TEMPORAL_LABEL_WEIGHT × sample-weight multiplier ensures real
            labels exert proportionally more influence on the loss than
            pseudo-labels (which have higher uncertainty).

        Step 3 — 80/20 split and tail calibration:
            The augmented matrix is split 80/20.  The 20% calibration fold
            is drawn from the pseudo-label population only (real-label rows
            all go into training) so the calibrator still maps raw GBR scores
            onto the pseudo-label scale for backward compatibility.
        """
        # X_raw may contain NaN in z-score features for NAICS-missing rows.
        # Pseudo-labels use only absolute signals + naics_unknown flag;
        # the model trains on NaN→0.0 matrix for z-scores.
        X_raw = np.array([p["features"] for p in population], dtype=float)
        y_pseudo = np.array([self._pseudo_label(row) for row in X_raw])
        X_pop = np.nan_to_num(X_raw, nan=0.0)
        X_pop = self._log_transform_features(X_pop)

        # ── Step 2: load real-label augmentation rows ─────────────────────
        temporal_rows = self._load_temporal_labels()
        n_real = len(temporal_rows)

        if n_real > 0:
            X_real_raw = np.array(
                [r["features_46"] for r in temporal_rows], dtype=float
            )
            X_real_raw = np.nan_to_num(X_real_raw, nan=0.0)
            X_real = self._log_transform_features(X_real_raw)
            y_real = np.array([r["real_label"] for r in temporal_rows], dtype=float)
            print(
                f"  Temporal label augmentation: {n_real:,} real-label rows added "
                f"(cutoff={self.TEMPORAL_LABEL_CUTOFF}).  "
                f"Real label mean={y_real.mean():.1f}  "
                f"pseudo mean={y_pseudo.mean():.1f}"
            )
        else:
            print("  No temporal labels available; training on pseudo-labels only.")

        # ── Step 2b: rescale pseudo-labels to match real adverse distribution ──
        # Pseudo-labels are heuristic proxies that systematically over-predict
        # (mean≈29 vs real adverse mean≈12).  When real-label rows are present,
        # we z-score the pseudo-labels to share the same mean and std as the
        # real distribution.  This removes the contradictory gradient pull
        # (pseudo rows pushing toward 29, real rows pushing toward 12) while
        # preserving the rank structure (Spearman ρ≈0.25 with real adverses).
        if n_real >= 10:
            p_mean = float(y_pseudo.mean())
            p_std  = max(float(y_pseudo.std()), 1e-6)
            r_mean = float(y_real.mean())
            r_std  = max(float(y_real.std()), 1e-6)
            y_pseudo = np.clip(
                (y_pseudo - p_mean) * (r_std / p_std) + r_mean, 0.0, 100.0
            )
            print(
                f"  Pseudo-labels rescaled: mean {p_mean:.1f}->{y_pseudo.mean():.1f}  "
                f"std {p_std:.1f}->{y_pseudo.std():.1f}  "
                f"(matched to real adverse distribution)"
            )

        # ── Step 3: 80/20 split on the pseudo-label population ────────────
        # Real-label rows are always placed in the training fold so that the
        # calibration fold remains purely pseudo-label for backward compat.
        rng = np.random.default_rng(42)
        n_pop = len(X_pop)
        idx = rng.permutation(n_pop)
        split = int(n_pop * 0.8)
        train_idx, cal_idx = idx[:split], idx[split:]

        X_train_pop, y_train_pop = X_pop[train_idx], y_pseudo[train_idx]
        X_cal,        y_cal       = X_pop[cal_idx],   y_pseudo[cal_idx]

        # ── Combine pseudo-label train fold with real-label rows ──────────
        if n_real > 0:
            X_train = np.vstack([X_train_pop, X_real])
            y_train = np.concatenate([y_train_pop, y_real])
        else:
            X_train = X_train_pop
            y_train = y_train_pop

        # ── Tail sample weights ───────────────────────────────────────────
        # Ramp starts at the Recommend→Caution boundary (y=30) rather than 40
        # to give the model a stronger gradient signal throughout the Caution
        # band.  Steeper slope (1 unit per 10 score points) and an 8× cap
        # counteract the 79:1 class imbalance between Recommend and
        # Do-Not-Recommend establishments.
        sw_pseudo = np.clip(1.0 + np.maximum(0.0, y_train_pop - 30.0) / 10.0, 1.0, 8.0)
        if n_real > 0:
            # Flat weight for real rows — no tail ramp.
            # Real adverse labels are direct observations; their own magnitude
            # already encodes severity.  Applying the pseudo-label tail ramp
            # (calibrated for mean=29) to real labels (mean≈12) would suppress
            # most real rows to near-1× weight, negating the multiplier.
            sw_real   = np.full(len(y_real), self.TEMPORAL_LABEL_WEIGHT, dtype=float)
            sw_train  = np.concatenate([sw_pseudo, sw_real])
        else:
            sw_train = sw_pseudo

        # ── Model: Huber loss + higher capacity + min leaf ────────────────
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                loss="huber",
                alpha=0.95,
                min_samples_leaf=3,
                random_state=42,
            )),
        ])
        self.pipeline.fit(X_train, y_train, model__sample_weight=sw_train)
        self.population_features = X_pop
        self._n_temporal_labels = n_real  # stored for diagnostics / tests

        # ── Fit calibrator on real-label pairs (raw_pred → real adverse) ─────
        # TailCalibrator was designed to fit on (raw_score, future_adverse).
        # Using pseudo-labels here was feeding it circular targets (the same
        # heuristic the model was trained to reproduce).  Real adverse outcomes
        # are the correct calibration target: they tell the calibrator exactly
        # how each raw prediction level maps to observed future compliance harm.
        self._calibrator = TailCalibrator()
        if n_real >= 10:
            pred_real = self.pipeline.predict(X_real)
            self._calibrator.fit(pred_real, y_real)
            _cal_note = f"{n_real:,} real-label pairs (pred → real adverse outcome)"
        else:
            # Fallback when temporal labels unavailable
            cal_preds = self.pipeline.predict(X_cal)
            self._calibrator.fit(cal_preds, y_cal)
            _cal_note = f"{len(X_cal):,} pseudo-label holdout rows"
        print(
            f"ML Risk Model trained on {len(X_train):,} rows "
            f"({len(X_train_pop):,} pseudo-label + {n_real:,} real-label);  "
            f"calibrator fitted on {_cal_note}."
        )

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #
    #  Temporal calibration
    # ------------------------------------------------------------------ #
    def _refit_calibrator_from_temporal(self, population: List[Dict]) -> None:
        """Refit the tail calibrator using real future outcomes (2024+).

        After training, this method loads post-2024 inspections and violations
        from the bulk CSVs, computes real future adverse outcome scores for each
        training establishment that also received inspections after 2024-01-01,
        and re-fits the calibrator so that the raw-score→calibrated-score mapping
        reflects actual compliance outcomes rather than self-referential
        pseudo-labels.

        This corrects the circular calibration problem: fitting the calibrator
        on pseudo-label holdout only un-compresses back to already-faulty labels.
        Real temporal data reveals how raw scores actually predict future harm.
        """
        insp_path = os.path.join(self.CACHE_DIR, "inspections_bulk.csv")
        viol_path = os.path.join(self.CACHE_DIR, "violations_bulk.csv")

        if not os.path.exists(insp_path) or not os.path.exists(viol_path):
            print("  Temporal calibration skipped: bulk CSVs not found in ml_cache/.")
            return

        if self.pipeline is None or self.population_features is None:
            return

        CUTOFF = date(2024, 1, 1)
        print("  Refitting calibrator from real temporal outcomes (2024+ holdout)…")

        # Get raw model predictions for every population establishment.
        pop_raw_scores = self.pipeline.predict(self.population_features)
        name_to_raw: Dict[str, float] = {
            p["name"]: float(pop_raw_scores[i])
            for i, p in enumerate(population)
        }

        # Stream post-2024 inspections, keeping only names we trained on.
        post_estabs: Dict[str, list] = defaultdict(list)
        csv.field_size_limit(10 * 1024 * 1024)
        with open(insp_path, encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                name = (row.get("estab_name") or "UNKNOWN").upper()
                if name not in name_to_raw:
                    continue
                od = row.get("open_date", "")
                try:
                    d = date.fromisoformat(od[:10])
                except (ValueError, TypeError):
                    continue
                if d >= CUTOFF:
                    post_estabs[name].append(row)

        if not post_estabs:
            print("  No post-2024 inspections for known establishments; skipping.")
            return

        # Load violations for post-2024 activity numbers only (leakage guard).
        post_acts: set = {
            str(r.get("activity_nr", ""))
            for rows in post_estabs.values()
            for r in rows
        }
        viols_by_act: Dict[str, list] = defaultdict(list)
        with open(viol_path, encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("delete_flag") == "X":
                    continue
                act = str(row.get("activity_nr", ""))
                if act in post_acts:
                    viols_by_act[act].append(row)

        # Compute (raw_score, future_adverse) pairs.
        raw_scores_cal: list = []
        future_advs_cal: list = []

        for name, fut_inspections in post_estabs.items():
            raw_pred = name_to_raw[name]
            fut_viols: list = []
            fut_fat = 0
            for insp in fut_inspections:
                act = str(insp.get("activity_nr", ""))
                fut_viols.extend(viols_by_act.get(act, []))
                fut_fat += int(insp.get("fatalities", "0") or 0)

            fut_n   = len(fut_inspections)
            fut_wr  = sum(1 for v in fut_viols if v.get("viol_type") in ("W", "R"))
            fut_s   = sum(1 for v in fut_viols if v.get("viol_type") == "S")
            fut_pen = sum(
                float(v.get("current_penalty") or v.get("initial_penalty") or 0)
                for v in fut_viols
            )
            fut_vr    = len(fut_viols) / fut_n if fut_n > 0 else 0.0
            any_fatal = int(fut_fat > 0)

            # Mirror the adverse-outcome formula used in test_real_world_validation.
            adv = 0.0
            adv += 20.0 * any_fatal
            adv += min((fut_fat - 1) * 5.0, 15.0) if fut_fat > 1 else 0.0
            adv += 8.0 * int(fut_wr > 0)
            adv += min(fut_wr * 3.0, 15.0)
            adv += min(fut_s * 1.0, 10.0)
            adv += min(math.log1p(fut_pen) * 0.8, 10.0)
            adv += min(fut_vr * 2.0, 10.0)

            raw_scores_cal.append(raw_pred)
            future_advs_cal.append(adv)

        n_pairs = len(raw_scores_cal)
        if n_pairs < 100:
            print(
                f"  Only {n_pairs} temporal pairs found; "
                "keeping pseudo-label calibrator."
            )
            return

        print(
            f"  Fitting temporal calibrator on {n_pairs:,} "
            "(raw_score, future_adverse) pairs…"
        )
        new_cal = TailCalibrator()
        new_cal.fit(
            np.array(raw_scores_cal, dtype=float),
            np.array(future_advs_cal, dtype=float),
        )
        if new_cal.is_fitted:
            self._calibrator = new_cal
            print("  Temporal calibrator fitted and attached.")
        else:
            print("  Temporal calibrator fit produced too few bins; keeping pseudo-label version.")

    # ------------------------------------------------------------------ #
    def _save(self, population: List[Dict]):
        model_path = os.path.join(self.CACHE_DIR, self.MODEL_FILE)
        pop_path = os.path.join(self.CACHE_DIR, self.POP_FILE)
        with open(model_path, "wb") as f:
            pickle.dump(self.pipeline, f)
        # NaN is not valid JSON — replace with null before serialising.
        # On load we convert null (None) back to 0.0 (model was trained on 0.0).
        def _serialise_features(feats):
            return [None if isinstance(v, float) and math.isnan(v) else v for v in feats]

        safe_pop = [
            {"name": p["name"], "features": _serialise_features(p["features"])}
            for p in population
        ]
        with open(pop_path, "w") as f:
            json.dump(
                {
                    "date": str(date.today()),
                    "manufacturers": safe_pop,
                    "industry_stats": self._industry_stats,
                    "feature_names": self.FEATURE_NAMES,
                },
                f,
            )
        if self._calibrator is not None and self._calibrator.is_fitted:
            cal_path = os.path.join(self.CACHE_DIR, self.CALIBRATOR_FILE)
            self._calibrator.save(cal_path)

    def _load_or_build(self):
        model_path = os.path.join(self.CACHE_DIR, self.MODEL_FILE)
        pop_path = os.path.join(self.CACHE_DIR, self.POP_FILE)

        if os.path.exists(model_path) and os.path.exists(pop_path):
            try:
                with open(pop_path) as f:
                    meta = json.load(f)
                cache_date = meta.get("date", "")
                if cache_date and (date.today() - date.fromisoformat(cache_date)).days < 7:
                    with open(model_path, "rb") as f:
                        loaded_pipeline = pickle.load(f)
                    pop = meta["manufacturers"]
                    # Convert null → 0.0 (NaN was serialised as null)
                    feats = np.array(
                        [[0.0 if v is None else v for v in p["features"]] for p in pop],
                        dtype=float,
                    )
                    # Shape-mismatch guard: stale model after upgrade
                    expected_n = len(self.FEATURE_NAMES)
                    if feats.shape[1] != expected_n:
                        print(
                            f"  Model feature shape mismatch "
                            f"({feats.shape[1]} vs {expected_n} expected). "
                            "Deleting stale cache and retraining…"
                        )
                        try:
                            os.remove(model_path)
                        except OSError:
                            pass
                        raise ValueError("feature shape mismatch")

                    # Feature-name guard: catches renames (e.g. raw→log)
                    cached_names = meta.get("feature_names", [])
                    if cached_names and cached_names != self.FEATURE_NAMES:
                        print("  Feature names changed. Retraining…")
                        try:
                            os.remove(model_path)
                        except OSError:
                            pass
                        raise ValueError("feature names mismatch")

                    self.pipeline = loaded_pipeline
                    self.population_features = self._log_transform_features(feats)
                    self._industry_stats = meta.get("industry_stats", {})
                    # Load calibrator if available (graceful degradation if absent)
                    cal_path = os.path.join(self.CACHE_DIR, self.CALIBRATOR_FILE)
                    if os.path.exists(cal_path):
                        try:
                            self._calibrator = TailCalibrator.load(cal_path)
                        except Exception as cal_e:
                            print(f"  Calibrator load failed ({cal_e}); running uncalibrated.")
                    print(f"Loaded cached ML risk model (trained {cache_date}, {len(pop)} estabs).")
                    return
            except Exception as e:
                if "feature shape mismatch" not in str(e):
                    print(f"Cache load failed: {e}")
                # Fall through to rebuild

        print("Building ML risk model from DOL API data...")
        try:
            population = self._fetch_population()
            if len(population) < 5:
                print("Insufficient population data ")
                return
            self._train(population)
            self._refit_calibrator_from_temporal(population)
            self._save(population)
        except Exception as e:
            print(f"Population fetch failed ({e})")

    # ------------------------------------------------------------------ #
    #  Per-establishment scoring
    # ------------------------------------------------------------------ #
    def score_establishments(self, records: List[OSHARecord]) -> Dict:
        """Score each establishment independently and compute aggregate metrics.

        Groups *records* by ``estab_name`` (falls back to ``"UNKNOWN"``),
        extracts features per group using the same path as the training
        pipeline, and runs the GB model per establishment.

        Returns
        -------
        dict with keys:
            weighted_avg_score  – inspection-count-weighted mean of site scores
            max_score           – highest single-site score
            median_score        – median of site scores
            establishment_count – number of distinct establishments
            site_scores         – list of per-site dicts (name, score, n_inspections, naics_code)
            risk_concentration  – fraction of sites scoring ≥ 60
            systemic_risk_flag  – True when risk is systemic (>50% high-risk
                                  or willful/repeat violations across ≥ 2 sites)
            aggregate_features  – 1×46 log-transformed feature array (all records)
        """
        if self.pipeline is None:
            return {
                "weighted_avg_score": 50.0,
                "max_score": 50.0,
                "median_score": 50.0,
                "establishment_count": 0,
                "site_scores": [],
                "risk_concentration": 0.0,
                "systemic_risk_flag": False,
                "aggregate_features": None,
            }

        # ── Group records by establishment name ───────────────────────
        groups: Dict[str, List[OSHARecord]] = defaultdict(list)
        for r in records:
            key = (r.estab_name or "UNKNOWN").upper().strip()
            groups[key].append(r)

        # ── Score each establishment ──────────────────────────────────
        site_scores: list = []
        total_inspections = 0

        for estab_name, estab_records in groups.items():
            feat17, naics, vpi, avg_pen, sr, wr = (
                self._extract_establishment_features_raw(estab_records)
            )
            raw = self._complete_features(feat17, naics, vpi, avg_pen, sr, wr)
            log_feats = self._log_transform_features(raw)

            raw_pred = float(self.pipeline.predict(log_feats)[0])

            # ── Evidence-gated score ceiling ──────────────────────────
            # Single-event companies lack sufficient evidence for a high
            # "Do Not Recommend" classification.  Cap scores based on
            # inspection depth so sparse-evidence establishments cannot
            # land in the highest tier without sufficient historical data.
            # Exception: confirmed fatality + willful violations may
            # reach 70 regardless of inspection count.
            n_insp_this = len(estab_records)
            feat17_raw = feat17  # already extracted above
            # feat17 is a raw list; willful is index 3, fatalities index 12
            has_fatality = feat17_raw[12] > 0
            has_willful  = feat17_raw[3] > 0
            if has_fatality and has_willful:
                ceiling = 70.0
            elif n_insp_this <= 2:
                ceiling = 50.0
            elif n_insp_this <= 4:
                ceiling = 58.0
            else:
                ceiling = 100.0
            evidence_capped = raw_pred > ceiling
            clipped = float(np.clip(raw_pred, 0, ceiling))

            # ── Isotonic calibration ──────────────────────────────────
            if self._calibrator is not None and self._calibrator.is_fitted:
                score = self._calibrator.calibrate(clipped)
            else:
                score = clipped

            n = len(estab_records)
            # Resolve city/state from the most recent record in this group
            city = state = ""
            for _r in sorted(estab_records, key=lambda r: r.date_opened, reverse=True):
                if _r.site_city:
                    city = _r.site_city
                    state = _r.site_state or ""
                    break
            site_scores.append({
                "name": estab_name,
                "score": round(score, 1),
                "n_inspections": n,
                "naics_code": naics,
                "city": city,
                "state": state,
                "evidence_capped": evidence_capped,
                "_log_feats": log_feats,  # retained for MT per-site scoring; stripped before API serialisation
            })
            total_inspections += n

        # ── Aggregate metrics ─────────────────────────────────────────
        if site_scores:
            weighted_sum = sum(s["score"] * s["n_inspections"] for s in site_scores)
            weighted_avg = weighted_sum / total_inspections
            max_score = max(s["score"] for s in site_scores)
            median_score = float(np.median([s["score"] for s in site_scores]))
        else:
            weighted_avg = max_score = median_score = 50.0

        HIGH_RISK_THRESHOLD = 60.0
        high_risk_count = sum(1 for s in site_scores if s["score"] >= HIGH_RISK_THRESHOLD)
        risk_concentration = high_risk_count / len(site_scores) if site_scores else 0.0

        # Systemic flag: majority of sites are high-risk …
        systemic = risk_concentration > 0.5
        # … or willful/repeat violations span >= 2 distinct sites AND the
        # aggregate score is meaningfully elevated (>= 45). Without the
        # score gate, companies with many sites almost always trip this
        # condition from incidental repeat violations on a handful of sites.
        sites_with_wr = 0
        for estab_records in groups.values():
            if any(
                v.is_willful or v.is_repeat
                for r in estab_records
                for v in r.violations
            ):
                sites_with_wr += 1
        if len(groups) >= 2 and sites_with_wr >= 2 and weighted_avg >= 45:
            systemic = True

        # Aggregate features (all records as one blob) for display
        agg_feats = self.extract_features(records) if records else None

        # Per-site log-transformed feature arrays for MT model per-site scoring
        per_site_features = [s.get("_log_feats") for s in site_scores]

        return {
            "weighted_avg_score": round(weighted_avg, 1),
            "max_score": round(max_score, 1),
            "median_score": round(median_score, 1),
            "establishment_count": len(site_scores),
            "site_scores": sorted(site_scores, key=lambda s: s["score"], reverse=True),
            "risk_concentration": round(risk_concentration, 2),
            "systemic_risk_flag": systemic,
            "aggregate_features": agg_feats,
            "per_site_features": per_site_features,
        }

    # ------------------------------------------------------------------ #
    #  Public scoring API
    # ------------------------------------------------------------------ #
    def score(self, records: List[OSHARecord]) -> Dict:
        """
        Predict risk score for a manufacturer relative to the population.

        Scores each establishment independently, then produces a
        weighted-average company score.  The per-site breakdown is
        included in the return dict so callers can surface concentration.

        Returns:
            risk_score            – 0 to 100 (weighted avg of establishment scores)
            percentile_rank       – 0 to 100 (higher = riskier than more peers)
            feature_weights       – feature name → learned importance
            features              – feature name → raw value (aggregate)
            industry_label        – human-readable industry name (NAICS lookup)
            industry_group        – 4-digit (or coarser) NAICS group used
            industry_percentile   – company's violation-rate percentile within its industry
            industry_comparison   – list of comparison strings
            missing_naics         – True when no NAICS code was available
            establishment_count   – number of distinct scored establishments
            site_scores           – list of per-site dicts
            risk_concentration    – fraction of sites ≥ 60
            systemic_risk_flag    – True when risk is systemic
            aggregation_warning   – human-readable warning when multi-site
        """
        # ── Per-establishment scoring ─────────────────────────────────
        estab = self.score_establishments(records)
        risk_score = estab["weighted_avg_score"]

        # ── Percentile rank within population ─────────────────────────
        if self.population_features is not None and len(self.population_features) > 0:
            pop_scores = self.pipeline.predict(self.population_features)
            percentile = float(np.mean(pop_scores <= risk_score) * 100)
        else:
            percentile = 50.0

        # ── Feature importances from the GB model ─────────────────────
        gb = self.pipeline.named_steps["model"]
        importances = dict(zip(self.FEATURE_NAMES, gb.feature_importances_.tolist()))

        # Aggregate feature values for display
        agg_X = estab["aggregate_features"]
        if agg_X is not None:
            feature_vals = dict(zip(self.FEATURE_NAMES, agg_X[0].tolist()))
        else:
            feature_vals = {}

        # ── Industry context ──────────────────────────────────────────
        naics_votes = Counter(r.naics_code for r in records if r.naics_code)
        naics_code = naics_votes.most_common(1)[0][0] if naics_votes else None
        industry_group_raw = (naics_code[:4] if naics_code and len(str(naics_code)) >= 4 else None)
        missing_naics = naics_code is None

        # Resolve industry entry with 4→3→2-digit fallback
        industry_entry = None
        resolved_group = industry_group_raw
        if industry_group_raw and self._industry_stats:
            for grp_len in (4, 3, 2):
                key = industry_group_raw[:grp_len]
                entry = self._industry_stats.get(key)
                if entry and entry.get("count", 0) >= self.INDUSTRY_MIN_SAMPLE:
                    industry_entry = entry
                    resolved_group = key
                    break

        if industry_entry:
            industry_label = industry_entry.get("label", "Unknown Industry")
            if industry_label == "Unknown Industry" and resolved_group:
                fresh = get_industry_name(resolved_group, load_naics_map())
                if fresh != "Unknown Industry":
                    industry_label = fresh
        else:
            industry_label = get_industry_name(naics_code, load_naics_map())

        # Build comparison messages and percentile from z-scores
        industry_comparison: list = []
        industry_percentile = 50.0

        if industry_entry and not missing_naics:
            metric_defs = [
                ("relative_violation_rate",  "avg_violation_rate",  "std_violation_rate",  "violation rate"),
                ("relative_penalty",         "avg_penalty",         "std_penalty",         "average penalty"),
                ("relative_serious_ratio",   "avg_serious_ratio",   "std_serious_ratio",   "serious violation ratio"),
                ("relative_willful_repeat",  "avg_willful_repeat",  "std_willful_repeat",  "willful/repeat rate"),
            ]
            for feat_key, avg_key, std_key, label_str in metric_defs:
                z = feature_vals.get(feat_key, 0.0)
                avg = industry_entry.get(avg_key, 0.0)
                std = industry_entry.get(std_key, 1e-6)
                if abs(z) < 0.3 or avg == 0:
                    continue
                company_val = avg + z * std
                pct = abs((company_val - avg) / max(abs(avg), 1e-9)) * 100
                if pct < 10:
                    continue
                direction = "higher" if z > 0 else "lower"
                industry_comparison.append(
                    f"{pct:.0f}% {direction} {label_str} than "
                    f"{industry_label} average"
                )

            z_vr = feature_vals.get("relative_violation_rate", 0.0)
            if not math.isnan(z_vr):
                industry_percentile = round(
                    50.0 * (1.0 + math.erf(z_vr / math.sqrt(2.0))), 1
                )

        # ── Aggregation warning ───────────────────────────────────────
        n_estab = estab["establishment_count"]
        if n_estab > 1:
            scores = [s["score"] for s in estab["site_scores"]]
            aggregation_warning = (
                f"This score aggregates {n_estab} establishments. "
                f"Individual site scores range from {min(scores):.0f} to "
                f"{max(scores):.0f}."
            )
        else:
            aggregation_warning = ""

        # ── Concentration warning ─────────────────────────────────────
        concentration_warning = ""
        if estab["risk_concentration"] > 0 and n_estab > 1:
            high_ct = sum(1 for s in estab["site_scores"] if s["score"] >= 60)
            concentration_warning = (
                f"{high_ct} of {n_estab} establishment(s) scored as high-risk "
                f"(≥ 60). Review the per-site breakdown below."
            )

        return {
            "risk_score": round(risk_score, 1),
            "percentile_rank": round(percentile, 1),
            "feature_weights": importances,
            "features": feature_vals,
            "industry_label": industry_label,
            "industry_group": resolved_group,
            "industry_percentile": round(industry_percentile, 1),
            "industry_comparison": industry_comparison[:4],
            "missing_naics": missing_naics,
            # ── New per-establishment fields ──────────────────────────
            "establishment_count": n_estab,
            "site_scores": estab["site_scores"],
            "risk_concentration": estab["risk_concentration"],
            "systemic_risk_flag": estab["systemic_risk_flag"],
            "aggregation_warning": aggregation_warning,
            "concentration_warning": concentration_warning,
        }

    def retrain(self):
        """Force a full retrain from fresh API data."""
        population = self._fetch_population()
        if len(population) >= 5:
            self._train(population)
            self._refit_calibrator_from_temporal(population)
            self._save(population)
            return True
        return False
