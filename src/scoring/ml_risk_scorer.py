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


# ====================================================================== #
#  Module-level industry stats helpers
# ====================================================================== #

def compute_industry_stats(
    df: "pd.DataFrame",
    min_sample: int = 10,
    naics_map: Optional[dict] = None,
) -> Dict[str, dict]:
    """Compute industry-level risk benchmarks from the population DataFrame.

    Parameters
    ----------
    df : DataFrame with columns:
        ``industry_group`` (str|None), ``raw_vpi``, ``raw_avg_pen``,
        ``raw_serious_rate``, ``raw_wr_rate``
    min_sample : int
        Minimum establishments per group; groups below threshold are excluded.
    naics_map : dict
        NAICS code → title mapping.  Loaded from embedded table if ``None``.

    Returns
    -------
    dict mapping ``group_str`` → stats dict (4-digit preferred, 3/2 fallback).
    Both finer and coarser keys coexist so look-ups try 4→3→2-digit order.
    """
    if naics_map is None:
        naics_map = load_naics_map()

    df = df.copy()
    df["industry_group"] = df["industry_group"].fillna("").astype(str)

    stats: Dict[str, dict] = {}

    for digits in (4, 3, 2):
        df["_grp"] = df["industry_group"].apply(
            lambda g: g[:digits] if len(g) >= digits else ""
        )
        for grp, grp_df in df[df["_grp"] != ""].groupby("_grp"):
            if grp in stats or len(grp_df) < min_sample:
                continue
            vpi_series = grp_df["raw_vpi"].astype(float)
            pen_series = grp_df["raw_avg_pen"].astype(float)
            sr_series  = grp_df["raw_serious_rate"].astype(float)
            wr_series  = grp_df["raw_wr_rate"].astype(float)
            stats[grp] = {
                "label":               get_industry_name(grp, naics_map),
                "count":               int(len(grp_df)),
                "avg_violation_rate":  float(vpi_series.mean()),
                "std_violation_rate":  float(vpi_series.std(ddof=0)) or 1e-6,
                "avg_penalty":         float(pen_series.mean()),
                "std_penalty":         float(pen_series.std(ddof=0)) or 1e-6,
                "avg_serious_ratio":   float(sr_series.mean()),
                "std_serious_ratio":   float(sr_series.std(ddof=0)) or 1e-6,
                "avg_willful_repeat":  float(wr_series.mean()),
                "std_willful_repeat":  float(wr_series.std(ddof=0)) or 1e-6,
            }

    return stats


def compute_relative_features(
    company_row: dict,
    industry_stats: Dict[str, dict],
    naics_map: Optional[dict] = None,
    min_sample: int = 10,
) -> dict:
    """Compute industry-relative z-score features for a single establishment.

    Parameters
    ----------
    company_row : dict with keys
        ``industry_group`` (str|None), ``raw_vpi``, ``raw_avg_pen``,
        ``raw_serious_rate``, ``raw_wr_rate``
    industry_stats : dict
        Output of :func:`compute_industry_stats`.
    naics_map : dict
        NAICS lookup (loaded from embedded table if ``None``).
    min_sample : int
        Minimum group size to be considered viable.

    Returns
    -------
    dict with relative z-scores, ``industry_group``, ``industry_label``,
    ``industry_count``, and ``missing_naics`` flag.
    Relative z-score features are ``float('nan')`` when no viable industry
    group is found (NaN sentinel used in pseudo-label; 0.0 used in model).
    """
    if naics_map is None:
        naics_map = load_naics_map()

    raw_group = str(company_row.get("industry_group") or "").strip()
    missing_naics = not raw_group

    # 4→3→2-digit fallback to find a viable industry entry
    industry_entry = None
    resolved_group = raw_group or None
    for grp_len in (4, 3, 2):
        if not raw_group or len(raw_group) < grp_len:
            continue
        key = raw_group[:grp_len]
        entry = industry_stats.get(key)
        if entry and entry.get("count", 0) >= min_sample:
            industry_entry = entry
            resolved_group = key
            break

    if industry_entry is None:
        _nan = float("nan")
        return {
            "relative_violation_rate": _nan,
            "relative_penalty":        _nan,
            "relative_serious_ratio":  _nan,
            "relative_willful_repeat": _nan,
            "industry_group":          resolved_group,
            "industry_label":          get_industry_name(raw_group, naics_map),
            "industry_count":          0,
            "missing_naics":           missing_naics,
        }

    def _z(val: float, avg: float, std: float) -> float:
        return (float(val) - avg) / max(abs(std), 1e-6)

    return {
        "relative_violation_rate": _z(
            company_row.get("raw_vpi", 0.0),
            industry_entry["avg_violation_rate"],
            industry_entry["std_violation_rate"],
        ),
        "relative_penalty": _z(
            company_row.get("raw_avg_pen", 0.0),
            industry_entry["avg_penalty"],
            industry_entry["std_penalty"],
        ),
        "relative_serious_ratio": _z(
            company_row.get("raw_serious_rate", 0.0),
            industry_entry["avg_serious_ratio"],
            industry_entry["std_serious_ratio"],
        ),
        "relative_willful_repeat": _z(
            company_row.get("raw_wr_rate", 0.0),
            industry_entry["avg_willful_repeat"],
            industry_entry["std_willful_repeat"],
        ),
        "industry_group":  resolved_group,
        "industry_label":  industry_entry.get("label", "Unknown Industry"),
        "industry_count":  industry_entry.get("count", 0),
        "missing_naics":   missing_naics,
    }



class MLRiskScorer:
    """
    Machine-learning risk scorer using scikit-learn.

    Trains a GradientBoostingRegressor on OSHA population data from the
    DOL API so that every manufacturer's risk score is *relative* to the
    broader population of inspected establishments.
    """

    INDUSTRY_MIN_SAMPLE = 10

    FEATURE_NAMES = [
        # ── Absolute signals (17) ──────────────────────────────────────
        "total_inspections",
        "total_violations",
        "serious_violations",
        "willful_violations",
        "repeat_violations",
        "total_penalties",
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
        # ── Industry-relative z-scores (4) ────────────────────────────
        # 0.0 sentinel when NAICS unavailable; NaN used only in pseudo-label
        "relative_violation_rate",
        "relative_penalty",
        "relative_serious_ratio",
        "relative_willful_repeat",
    ]

    FEATURE_DISPLAY = {
        "total_inspections": "Inspection Count",
        "total_violations": "Violation Count",
        "serious_violations": "Serious Violations",
        "willful_violations": "Willful Violations",
        "repeat_violations": "Repeat Violations",
        "total_penalties": "Total Penalties ($)",
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
        "relative_violation_rate": "Violation Rate vs. Industry (z)",
        "relative_penalty": "Avg Penalty vs. Industry (z)",
        "relative_serious_ratio": "Serious Ratio vs. Industry (z)",
        "relative_willful_repeat": "Willful+Repeat Rate vs. Industry (z)",
    }

    CACHE_DIR = "ml_cache"
    MODEL_FILE = "risk_model.pkl"
    POP_FILE = "population_data.json"

    def __init__(self, osha_client=None):
        self.osha_client = osha_client
        self.pipeline: Optional[Pipeline] = None
        self.population_features: Optional[np.ndarray] = None
        self._industry_stats: dict = {}
        self._naics_map: dict = load_naics_map()
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self._load_or_build()

    # ------------------------------------------------------------------ #
    #  Feature extraction
    # ------------------------------------------------------------------ #
    def extract_features(self, records: List[OSHARecord]) -> np.ndarray:
        """Convert a list of OSHARecords into a 1 x n_features array.

        Count signals (serious, willful, repeat, accidents, etc.) are
        expressed as *rates per inspection* so that large multi-site
        companies are scored on their per-location safety rate rather
        than their aggregate totals.  This keeps training and inference
        on the same scale regardless of company size.

        More recent inspections are weighted higher via exponential decay
        (half-life = 3 years).
        """
        one_year_ago = date.today() - timedelta(days=365)
        today = date.today()
        half_life_days = 3 * 365  # 3-year half-life

        n_insp = len(records)

        # Per-inspection recency weight: exp(-ln2 * age / half_life)
        def recency_weight(d: date) -> float:
            age = max((today - d).days, 0)
            return float(np.exp(-np.log(2) * age / half_life_days))

        insp_weights = [recency_weight(r.date_opened) for r in records]
        total_w = sum(insp_weights) or 1.0

        # Weighted accumulation — will be divided by total_w to produce rates
        serious_w = 0.0
        willful_w = 0.0
        repeat_w = 0.0
        total_pen = 0.0
        all_penalties: list = []
        recent = 0
        severe_w = 0.0
        n_viols_w = 0.0
        accident_w = 0.0
        fatality_w = 0.0
        injury_w = 0.0
        gravities: list = []
        clean_w = 0.0

        for r, w in zip(records, insp_weights):
            viols = r.violations
            n_viols_w += len(viols) * w

            serious_w += sum(1 for v in viols if v.severity == "Serious") * w
            willful_w += sum(1 for v in viols if v.is_willful) * w
            repeat_w  += sum(1 for v in viols if v.is_repeat) * w

            pens = [v.penalty_amount for v in viols]
            total_pen += sum(pens) * w
            all_penalties.extend(pens)

            if r.date_opened >= one_year_ago:
                recent += 1

            if r.accidents:
                severe_w += w
                for a in r.accidents:
                    accident_w += w
                    if a.fatality:
                        fatality_w += w
                    injury_w += len(a.injuries) * w

            for v in viols:
                if v.gravity:
                    try:
                        gravities.append(float(v.gravity))
                    except (ValueError, TypeError):
                        pass

            if not viols:
                clean_w += w

        # --- Rates (per weighted inspection) ---
        # Dividing by total_w normalises for company size: a 1,000-location
        # chain and a single-site supplier both land on the same scale.
        serious       = serious_w  / total_w
        willful       = willful_w  / total_w
        repeat        = repeat_w   / total_w
        severe        = severe_w   / total_w
        accident_count = accident_w / total_w
        fatality_count = fatality_w / total_w
        injury_count   = injury_w   / total_w

        avg_pen = float(np.mean(all_penalties)) if all_penalties else 0.0
        max_pen = max(all_penalties) if all_penalties else 0.0
        recent_ratio = recent / n_insp if n_insp else 0.0
        vpi = n_viols_w / total_w if total_w else 0.0
        avg_gravity = float(np.mean(gravities)) if gravities else 0.0
        pen_per_insp = total_pen / total_w if total_w else 0.0
        clean_ratio = clean_w / total_w if total_w else 0.0

        # --- Industry-relative features ---
        # Majority-vote NAICS code across all records
        naics_votes = Counter(r.naics_code for r in records if r.naics_code)
        naics_code = naics_votes.most_common(1)[0][0] if naics_votes else None
        industry_group = (naics_code[:4] if naics_code and len(str(naics_code)) >= 4 else None)

        # Fraction-of-violations ratios for industry comparison
        serious_ratio_ind = serious_w / max(n_viols_w, 1.0)
        wr_ratio_ind = (willful_w + repeat_w) / max(n_viols_w, 1.0)

        rel = compute_relative_features(
            {
                "industry_group": industry_group,
                "raw_vpi": vpi,
                "raw_avg_pen": avg_pen,
                "raw_serious_rate": serious_ratio_ind,
                "raw_wr_rate": wr_ratio_ind,
            },
            self._industry_stats,
            self._naics_map,
            min_sample=self.INDUSTRY_MIN_SAMPLE,
        )

        # Use 0.0 sentinel when NAICS unavailable (consistent with model training)
        def _safe(v: float) -> float:
            return 0.0 if (v != v) else v  # NaN check via IEEE 754

        return np.array([[
            n_insp, n_viols_w, serious, willful, repeat,
            total_pen, avg_pen, max_pen, recent_ratio, severe, vpi,
            accident_count, fatality_count, injury_count, avg_gravity,
            pen_per_insp, clean_ratio,
            _safe(rel["relative_violation_rate"]),
            _safe(rel["relative_penalty"]),
            _safe(rel["relative_serious_ratio"]),
            _safe(rel["relative_willful_repeat"]),
        ]])

    # ------------------------------------------------------------------ #
    #  Pseudo-label generation (domain knowledge)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pseudo_label(row: np.ndarray) -> float:
        """
        Generate a training label (0-100 risk) from raw features using
        an explainable, credit-score-inspired heuristic.

        Scoring philosophy:
          direct harm signals         0-34   dominant, can create score floors
          violation type             0-24   willful/repeat matter heavily
          gravity / severity         0-14   serious + gravity corroborate risk
          recency                    0-12   recent bad behavior matters more
          violation rate             0-10   normalized by inspections
          penalties                  0-6    log-scaled corroboration only
          industry-relative signals  0-25   ~25% weight, cannot override fatality/willful
          clean inspection credit    0 to -10  reward repeated clean history
          uncertainty / sparse data  0-8    caution when evidence is thin

        Modifiers:
          + interaction penalties for compounding patterns
          + catastrophic-event floors
          + sparse-data floors
        """
        (n_insp, n_viols, serious, willful, repeat,
         total_pen, avg_pen, max_pen, recent_ratio, severe, vpi,
         accident_count, fatality_count, injury_count, avg_gravity,
         pen_per_insp, clean_ratio,
         rel_viol_rate, rel_penalty, rel_serious_ratio, rel_willful_repeat) = row

        # All count features (serious, willful, repeat, severe, accidents,
        # fatalities, injuries) are now *rates per inspection* — a value of 1.0
        # means every inspection had that event; 0.1 means 1-in-10 inspections.
        # This keeps Walmart (1,000 locations) on the same scale as a single
        # factory.  n_insp is retained raw so the scorer still knows how much
        # evidence exists.

        score = 0.0

        # --- Direct harm signals (up to 34) ---
        # Fatalities dominate: one death is categorically different from any number
        # of paper violations.  fatality_count is a *rate* (fatality inspections /
        # total inspections), so > 0 means at least one fatal inspection occurred;
        # the nonlinear jump is preserved — a company with a 10 % fatality rate
        # scores near the cap whereas a 0.1 % rate still triggers the floor.
        if fatality_count > 0:
            # Reconstruct a representative absolute count for the jump formula,
            # bounded so a 100 % fatality rate doesn't inflate beyond the cap.
            eff_fatalities = min(fatality_count * max(n_insp, 1), 5)
            score += min(22.0 + (eff_fatalities - 1) * 6.0, 34)
        score += min(severe * 25.0, 6)                      # rate × scale → up to 6
        score += min(injury_count * 6.0, 4)                 # rate × scale → up to 4
        score += min(accident_count * 10.0, 4)              # rate × scale → up to 4

        # Cap the full direct-harm block at 34
        score = min(score, 34)

        # --- Violation type (up to 24) ---
        # Willful violations = deliberate disregard for safety; treated more harshly
        # than repeat, which is a pattern signal but may reflect complexity, not malice.
        # Both are now rates: willful=1.0 means every inspection had a willful.
        score += min(willful * 14.0, 14)                    # rate × scale → up to 14
        score += min(repeat * 10.0, 10)                     # rate × scale → up to 10

        # --- Gravity / severity (up to 14) ---
        # Serious rate and avg_gravity corroborate real danger.
        score += min(serious * 8.0, 8)                      # rate × scale → up to 8
        if avg_gravity > 0:
            score += min(avg_gravity * 0.6, 6)              # avg is already scale-invariant

        # --- Recency — recent activity within 1yr (up to 12) ---
        # raw ratio is already scale-invariant
        score += recent_ratio * 12.0                        # up to 12

        # --- Violation rate per inspection (up to 10) ---
        # vpi is already a rate (violations / inspection); unchanged.
        score += min(vpi * 2.5, 10)                         # up to 10

        # --- Penalties — corroboration only (up to 6) ---
        # pen_per_insp is already normalized; total_pen is intentionally raw
        # (absolute fine size is a corroborating signal, not the primary driver).
        if total_pen > 0:
            score += min(np.log1p(total_pen) * 0.4, 4)     # up to 4
        if pen_per_insp > 0:
            score += min(np.log1p(pen_per_insp) * 0.3, 2)  # up to 2

        # --- Industry-relative signals (~25% total weight, max ±25 pts) ---
        # rel_X values are z-scores: (company - industry_avg) / industry_std.
        # NaN sentinel (v != v) means NAICScode was missing at training time.
        # Positive z = worse than industry average; negative = better.
        # RULE: Industry comparison CANNOT lower a score elevated by fatality or
        # willful violations — those absolute signals always dominate.
        _isnan = lambda v: v != v  # IEEE 754: NaN is not equal to itself
        missing_naics = _isnan(rel_viol_rate)

        if missing_naics:
            score += 4.0                                    # uncertainty penalty
        else:
            # Clamp z-scores to [-3, 3] to prevent outlier blow-up
            z_vr  = max(min(float(rel_viol_rate),     3.0), -3.0)
            z_pen = max(min(float(rel_penalty),        3.0), -3.0)
            z_sr  = max(min(float(rel_serious_ratio),  3.0), -3.0)
            z_wr  = max(min(float(rel_willful_repeat), 3.0), -3.0)

            ind = 0.0
            # Penalty for being above industry average (each capped to prevent dominance)
            if z_vr  > 0: ind += min(z_vr  * 4.0, 8.0)    # up to +8
            if z_pen > 0: ind += min(z_pen  * 3.0, 6.0)    # up to +6
            if z_sr  > 0: ind += min(z_sr   * 3.0, 6.0)    # up to +6
            if z_wr  > 0: ind += min(z_wr   * 4.0, 5.0)    # up to +5  → total ≤ 25

            # Credit for being well below industry average, ONLY when no absolute
            # danger signals are present (fatality/willful/repeat = none)
            if fatality_count == 0 and willful == 0 and repeat == 0:
                if z_vr  < -1.0: ind -= min(abs(z_vr)  * 1.5, 4.0)  # up to -4
                if z_pen < -1.0: ind -= min(abs(z_pen)  * 1.0, 2.0)  # up to -2

            # Absolute danger signals prevent industry comparison from reducing score
            if fatality_count > 0 or willful > 0:
                ind = max(ind, 0.0)

            score += max(min(ind, 25.0), -6.0)

        # --- Clean inspection credit (up to -10) ---
        # A pattern of clean inspections is genuinely positive, but it cannot
        # offset catastrophic events — a company with a fatality rate > 0 is
        # still high-risk even if most inspections were clean.
        if (clean_ratio > 0
                and fatality_count == 0
                and accident_count == 0
                and severe == 0):
            if n_insp >= 3:
                score -= clean_ratio * 10.0                 # full credit: sustained history
            elif n_insp == 2:
                score -= clean_ratio * 4.0                  # partial: only two data points
            # n_insp <= 1: no credit — one clean inspection proves nothing

        # --- Uncertainty / sparse data (up to 5) ---
        # Sparse data means we don't know enough to call this company safe.
        if n_insp <= 1:
            score += 5.0
        elif n_insp <= 3:
            score += 2.5
        elif n_insp <= 5:
            score += 1.0

        # --- Interaction effects ---
        # Rate-based: both conditions can fire even for a single-inspection
        # company if the rates are non-zero.
        if fatality_count > 0 and (willful + repeat) > 0:
            score += 8.0                                    # institutional disregard
        if willful > 0 and repeat > 0:
            score += 5.0                                    # known + ignored pattern
        if recent_ratio > 0.5 and serious >= 0.25:          # ≥25 % of inspections serious
            score += 4.0                                    # concentrated recent danger

        # --- Conservative floors ---
        if fatality_count > 0 and recent_ratio >= 0.25:
            score = max(score, 65.0)                        # recent fatality rate: never low-risk
        if n_insp <= 1 and score < 18:
            score = 18.0                                    # single-inspection: insufficient data

        return float(np.clip(score, 0, 100))

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

        one_year_ago = date.today() - timedelta(days=365)
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

            # Majority-vote NAICS code for this establishment
            naics_votes: dict = defaultdict(int)
            for insp in inspections:
                act = str(insp.get("activity_nr", ""))
                od = insp.get("open_date", "")
                try:
                    d = date.fromisoformat(od[:10])
                    if d >= one_year_ago:
                        recent += 1
                except (ValueError, TypeError):
                    pass
                insp_viols = client.get_violations_for_activity(act)
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

        # ── Append relative features (NaN sentinel for missing NAICS) ──────
        for p in population:
            rel = compute_relative_features(
                {
                    "industry_group":  p.pop("_industry_group"),
                    "raw_vpi":         p.pop("_raw_vpi"),
                    "raw_avg_pen":     p.pop("_raw_avg_pen"),
                    "raw_serious_rate": p.pop("_raw_serious_rate"),
                    "raw_wr_rate":     p.pop("_raw_wr_rate"),
                },
                self._industry_stats,
                self._naics_map,
                min_sample=self.INDUSTRY_MIN_SAMPLE,
            )
            # Keep NaN in features list — _train() will use NaN for pseudo-labels
            # and replace with 0.0 in the model's training matrix
            p["features"].extend([
                rel["relative_violation_rate"],
                rel["relative_penalty"],
                rel["relative_serious_ratio"],
                rel["relative_willful_repeat"],
            ])

        return population

    # ------------------------------------------------------------------ #
    #  Model training
    # ------------------------------------------------------------------ #
    def _train(self, population: List[Dict]):
        """Build feature matrix, generate pseudo-labels, train pipeline."""
        # X_raw may contain NaN in relative features for NAICS-missing rows;
        # pseudo-labels are generated with NaN so _pseudo_label() can apply the
        # uncertainty penalty, then the model trains on NaN→0.0 matrix.
        X_raw = np.array([p["features"] for p in population], dtype=float)
        y = np.array([self._pseudo_label(row) for row in X_raw])
        X = np.nan_to_num(X_raw, nan=0.0)  # 0.0 = "at industry average" for model

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
            )),
        ])
        self.pipeline.fit(X, y)
        self.population_features = X
        print(f"ML Risk Model trained on {len(X)} establishments.")

    # ------------------------------------------------------------------ #
    #  Persistence
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
                },
                f,
            )

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
                    # Shape-mismatch guard: stale 17-feature model after upgrade
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

                    self.pipeline = loaded_pipeline
                    self.population_features = feats
                    self._industry_stats = meta.get("industry_stats", {})
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
            self._save(population)
        except Exception as e:
            print(f"Population fetch failed ({e})")

    # ------------------------------------------------------------------ #
    #  Public scoring API
    # ------------------------------------------------------------------ #
    def score(self, records: List[OSHARecord], reputation_data: list = None) -> Dict:
        """
        Predict risk score for a manufacturer relative to the population.

        Returns:
            risk_score          – 0 to 100
            percentile_rank     – 0 to 100 (higher = riskier than more peers)
            feature_weights     – feature name → learned importance
            features            – feature name → raw value for this manufacturer
            industry_label      – human-readable industry name (NAICS lookup)
            industry_group      – 4-digit (or coarser) NAICS group used
            industry_percentile – company's violation-rate percentile within its industry
            industry_comparison – list of comparison strings, e.g. "67% higher than avg"
            missing_naics       – True when no NAICS code was available
        """
        X = self.extract_features(records)
        raw = float(self.pipeline.predict(X)[0])
        risk_score = float(np.clip(raw, 0, 100))

        # Reputation adjustment (outside the OSHA ML pipeline)
        reputation_score = 50.0
        news_sentiment = "Unknown"
        if reputation_data:
            neg_kw = {"violation", "fine", "penalty", "lawsuit", "injury",
                      "death", "accident", "unsafe", "danger", "settlement", "sued"}
            pos_kw = {"award", "safety", "recognized", "leader", "innovation"}
            neg = sum(1 for item in reputation_data
                      if any(k in (item.get("title", "") + " " + item.get("body", "")).lower() for k in neg_kw))
            pos = sum(1 for item in reputation_data
                      if any(k in (item.get("title", "") + " " + item.get("body", "")).lower() for k in pos_kw))
            total = neg + pos
            if total > 0:
                sentiment_adj = ((neg - pos) / total) * 15.0
                risk_score = float(np.clip(risk_score + sentiment_adj, 0, 100))
                ratio = (pos - neg) / (total + 1)
                reputation_score = 50.0 + (ratio * 30.0)
                if ratio > 0.2:
                    news_sentiment = "Positive"
                elif ratio < -0.2:
                    news_sentiment = "Negative"
                else:
                    news_sentiment = "Mixed"
            else:
                news_sentiment = "Neutral"

        # Percentile rank within population
        if self.population_features is not None and len(self.population_features) > 0:
            pop_scores = self.pipeline.predict(self.population_features)
            percentile = float(np.mean(pop_scores <= risk_score) * 100)
        else:
            percentile = 50.0

        # Feature importances from the GB model
        gb = self.pipeline.named_steps["model"]
        importances = dict(zip(self.FEATURE_NAMES, gb.feature_importances_.tolist()))

        # Raw feature values
        feature_vals = dict(zip(self.FEATURE_NAMES, X[0].tolist()))

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
        else:
            industry_label = get_industry_name(naics_code, self._naics_map)

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
                # Reconstruct company value from z-score and recompute % diff
                company_val = avg + z * std
                pct = abs((company_val - avg) / max(abs(avg), 1e-9)) * 100
                if pct < 10:
                    continue
                direction = "higher" if z > 0 else "lower"
                industry_comparison.append(
                    f"{pct:.0f}% {direction} {label_str} than "
                    f"{industry_label} average"
                )

            # Percentile within industry via normal approximation of violation-rate z
            z_vr = feature_vals.get("relative_violation_rate", 0.0)
            if not math.isnan(z_vr):
                industry_percentile = round(
                    50.0 * (1.0 + math.erf(z_vr / math.sqrt(2.0))), 1
                )

        return {
            "risk_score": round(risk_score, 1),
            "percentile_rank": round(percentile, 1),
            "feature_weights": importances,
            "features": feature_vals,
            "reputation_score": round(reputation_score, 1),
            "news_sentiment": news_sentiment,
            "industry_label": industry_label,
            "industry_group": resolved_group,
            "industry_percentile": round(industry_percentile, 1),
            "industry_comparison": industry_comparison[:4],
            "missing_naics": missing_naics,
        }

    def retrain(self):
        """Force a full retrain from fresh API data."""
        population = self._fetch_population()
        if len(population) >= 5:
            self._train(population)
            self._save(population)
            return True
        return False
