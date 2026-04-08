import csv
import logging
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
from src.scoring.features import extract_establishment_features_raw

logger = logging.getLogger(__name__)


class MLRiskScorer:
    """
    Machine-learning risk scorer using scikit-learn.

    Trains a GradientBoostingRegressor on OSHA population data from the
    DOL API so that every manufacturer's risk score is *relative* to the
    broader population of inspected establishments.
    """
    INDUSTRY_MIN_SAMPLE = 10
    # Log-compress only volume counts and accident/fatality/injury rates —
    # NOT penalty amounts.  Large penalties directly signal severity;
    # compressing $10K → $1M to 5.5 → 13.8 destroys high-end differentiation.
    # time_adjusted_penalty (index 17) is also intentionally unlogged.
    LOG_FEATURE_INDICES = [0, 1, 11, 12, 13]

    # Canonical 2-digit NAICS sectors for one-hot encoding.
    NAICS_SECTORS = [
        "11", "21", "22", "23", "31", "32", "33",
        "42", "44", "45", "48", "49", "51", "52",
        "53", "54", "55", "56", "61", "62", "71",
        "72", "81", "92",
    ]

    FEATURE_NAMES = [
        # ── Absolute signals (20) ──────────────────────────────────────
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
        "recent_wr_rate",
        "trend_delta",
        # ── High-signal penalty discriminators (4) ────────────────────
        "log_willful_raw",
        "log_repeat_raw",
        "has_any_fatality",
        "log_max_insp_penalty",
        "log_estab_size",
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
        "recent_wr_rate": "Recent W/R Violation Rate",
        "trend_delta": "Violation Trend (recent − all-time vpi)",
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
    # real outcome labels used to train the multi-target model.
    #
    # Pushing this back (earlier than today-minus-1yr) gives a richer paired
    # sample because more establishments have had time to accumulate post-
    # cutoff inspections.  Moved to 2022-01-01 to avoid the COVID-era inspection
    # suppression window (2020-2021) which creates false negatives for genuinely
    # risky companies that weren't reinspected during the pandemic.
    TEMPORAL_LABEL_CUTOFF = date(2022, 1, 1)

    # Maximum number of real-label rows to add to the training matrix.
    # 50_000 exceeds the ~30k paired pool so effectively "use all paired rows".
    TEMPORAL_SAMPLE_SIZE = 50_000

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
        """Delegate to the module-level feature extractor in features.py."""
        return extract_establishment_features_raw(records)

    def _complete_features(
        self,
        features_raw: list,
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

        row = features_raw + [
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
        feats_raw, naics, vpi, avg_pen, sr, wr = (
            self._extract_establishment_features_raw(records)
        )
        raw = self._complete_features(feats_raw, naics, vpi, avg_pen, sr, wr)
        return self._log_transform_features(raw)

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

        logger.info("Building population data from bulk cache…")
        client.ensure_cache()

        insp_data = client.get_bulk_inspections()
        if not insp_data:
            logger.warning("No inspection data in cache.")
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
                logger.info("    %s/%s establishments…", f"{progress:,}", f"{total_estabs:,}")
            n_insp = len(inspections)
            recent = 0
            severe = 0
            clean = 0
            viols = []
            acc_count = 0
            fat_count = 0
            inj_count = 0
            time_adj_pen = 0.0
            max_insp_pen = 0.0
            estab_sizes: list = []
            recent_viol_count = 0   # violation count from recent inspections only
            recent_wr_raw     = 0   # W/R violation count from recent inspections

            # Majority-vote NAICS code for this establishment
            naics_votes: dict = defaultdict(int)
            for insp in inspections:
                act = str(insp.get("activity_nr", ""))
                od = insp.get("open_date", "")
                try:
                    d = date.fromisoformat(od[:10])
                    is_recent = (d >= one_year_ago)
                    if is_recent:
                        recent += 1
                    age_years = max(0.0, (date.today() - d).days / 365.25)
                except (ValueError, TypeError):
                    is_recent = False
                    age_years = 0.0
                insp_viols = client.get_violations_for_activity(act)
                insp_pen = sum(
                    float(v.get("current_penalty") or v.get("initial_penalty") or 0)
                    for v in insp_viols
                )
                if insp_pen > max_insp_pen:
                    max_insp_pen = insp_pen
                time_adj_pen += insp_pen * math.exp(-age_years / 3.0)
                viols.extend(insp_viols)
                if not insp_viols:
                    clean += 1
                if is_recent:
                    recent_viol_count += len(insp_viols)
                    recent_wr_raw += sum(
                        1 for v in insp_viols
                        if v.get("viol_type") in ("W", "R")
                    )

                # Accident stats (uses cached indexes, no API calls)
                acc_stats = client.get_accident_count_for_activity(act)
                acc_count += acc_stats["accidents"]
                fat_count += acc_stats["fatalities"]
                inj_count += acc_stats["injuries"]
                if acc_stats["accidents"] > 0:
                    severe += 1

                # Track NAICS votes and establishment size
                nc = str(insp.get("naics_code") or "").strip()
                if nc and nc.isdigit() and len(nc) >= 4:
                    naics_votes[nc[:4]] += 1
                nr_raw = str(insp.get("nr_in_estab") or "").strip()
                if nr_raw:
                    try:
                        sz = float(nr_raw)
                        if sz > 0:
                            estab_sizes.append(sz)
                    except (ValueError, TypeError):
                        pass

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

            # Recent-window breakdown features
            total_wr = willful_raw + repeat_raw
            recent_wr_rate = recent_wr_raw / max(total_wr, 1)
            vpi_recent     = recent_viol_count / max(recent, 1)
            trend_delta    = vpi - vpi_recent

            median_estab_size = float(np.median(estab_sizes)) if estab_sizes else 0.0

            population.append({
                "name": estab,
                "features": [
                    n_insp, n_viols, serious_rate, willful_rate, repeat_rate,
                    total_pen, avg_pen, max_pen, recent_ratio, severe_rate, vpi,
                    acc_rate, fat_rate, inj_rate, avg_gravity,
                    pen_per_insp, clean_ratio,
                    time_adj_pen,
                    recent_wr_rate,
                    trend_delta,
                    # ── High-signal penalty discriminators (pre-log-transformed) ──
                    math.log1p(willful_raw),
                    math.log1p(repeat_raw),
                    1.0 if fat_count > 0 else 0.0,
                    math.log1p(max_insp_pen),
                    math.log1p(median_estab_size),
                    # Relative features will be appended below
                ],
                # Scratch fields for industry stats — removed before persisting
                "_industry_group": naics_group,
                "_raw_vpi": vpi,
                "_raw_avg_pen": avg_pen,
                "_raw_serious_rate": raw_serious_rate,
                "_raw_wr_rate": raw_wr_rate,
            })

        logger.info("  Aggregated %s unique establishments.", len(population))

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
        logger.info("  Industry stats: %s industry groups.", len(self._industry_stats))

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
            logger.warning("Temporal label build failed (%s); model will not be fitted.", e)
            rows = []
        return rows

    # ------------------------------------------------------------------ #
    #  Model training
    # ------------------------------------------------------------------ #
    def _train(self, population: List[Dict]):
        """Build population feature matrix and train GBR on real temporal labels.

        Training strategy
        -----------------
        The full population feature matrix is built from ~370k establishments
        for percentile ranking.  The GBR is trained solely on real adverse
        outcome labels from the temporal label builder (pre-cutoff features
        paired with post-cutoff adverse outcomes).

        If insufficient real-label rows are available the pipeline is left
        unfitted; score() falls back gracefully to 50.0.
        """
        X_raw = np.array([p["features"] for p in population], dtype=float)
        X_pop = np.nan_to_num(X_raw, nan=0.0)
        X_pop = self._log_transform_features(X_pop)
        self.population_features = X_pop

        # ── Load real temporal labels ─────────────────────────────────────
        temporal_rows = self._load_temporal_labels()
        n_real = len(temporal_rows)

        if n_real < 10:
            logger.warning("Insufficient real-label rows for training; model not fitted.")
            return

        X_real_raw = np.array(
            [r["features_46"] for r in temporal_rows], dtype=float
        )
        X_real_raw = np.nan_to_num(X_real_raw, nan=0.0)
        X_real = self._log_transform_features(X_real_raw)

        # Guard: population and training data must share the same feature width.
        # A mismatch means _fetch_population() and the temporal labeler were built
        # at different code versions.  Saving a model trained on X_real onto a
        # population_features matrix of a different width would cause the scaler
        # to crash on every score() call, so abort cleanly here instead.
        if X_real.shape[1] != X_pop.shape[1]:
            logger.error(
                "Feature-count mismatch: population has %d features but temporal "
                "labels have %d.  Delete ml_cache/population_data.json and "
                "ml_cache/temporal_labels.pkl to force a full rebuild.",
                X_pop.shape[1], X_real.shape[1],
            )
            self.population_features = None  # ensure score() uses graceful fallback
            return

        y_real = np.array([r["real_label"] for r in temporal_rows], dtype=float)
        logger.info(
            "Training on %s real-label rows (cutoff=%s)  label mean=%.1f  std=%.1f",
            f"{n_real:,}", self.TEMPORAL_LABEL_CUTOFF, y_real.mean(), y_real.std(),
        )

        # ── Tail sample weights (ramp on high-risk real rows) ─────────────
        sw = np.clip(1.0 + np.maximum(0.0, y_real - 30.0) / 10.0, 1.0, 8.0)

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
        self.pipeline.fit(X_real, y_real, model__sample_weight=sw)
        self._n_temporal_labels = n_real

        # ── Fit calibrator on real-label pairs (raw_pred → real adverse) ──
        self._calibrator = TailCalibrator()
        pred_real = self.pipeline.predict(X_real)
        self._calibrator.fit(pred_real, y_real)
        logger.info(
            "ML Risk Model trained on %s real-label rows; calibrator fitted.",
            f"{n_real:,}",
        )

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
                        logger.warning(
                            "Model feature shape mismatch (%s vs %s expected). "
                            "Deleting stale cache and retraining…",
                            feats.shape[1], expected_n,
                        )
                        for _stale in (model_path, pop_path):
                            try:
                                os.remove(_stale)
                            except OSError:
                                pass
                        raise ValueError("feature shape mismatch")

                    # Feature-name guard: catches renames (e.g. raw→log)
                    cached_names = meta.get("feature_names", [])
                    if cached_names and cached_names != self.FEATURE_NAMES:
                        logger.warning("Feature names changed. Retraining…")
                        for _stale in (model_path, pop_path):
                            try:
                                os.remove(_stale)
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
                            logger.warning("Calibrator load failed (%s); running uncalibrated.", cal_e)
                    logger.info("Loaded cached ML risk model (trained %s, %s estabs).", cache_date, len(pop))
                    return
            except Exception as e:
                if "feature shape mismatch" not in str(e):
                    logger.warning("Cache load failed: %s", e)
                # Fall through to rebuild

        logger.info("Building ML risk model from DOL API data...")
        try:
            population = self._fetch_population()
            if len(population) < 5:
                logger.warning("Insufficient population data.")
                return
            self._train(population)
            self._save(population)
        except Exception as e:
            logger.warning("Population fetch failed (%s)", e)

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
            feats_raw, naics, vpi, avg_pen, sr, wr = (
                self._extract_establishment_features_raw(estab_records)
            )
            raw = self._complete_features(feats_raw, naics, vpi, avg_pen, sr, wr)
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
            # feats_raw extracted above

            has_fatality = feats_raw[12] > 0
            has_willful  = feats_raw[3] > 0
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
        if self.pipeline is not None and self.population_features is not None and len(self.population_features) > 0:
            pop_scores = self.pipeline.predict(self.population_features)
            percentile = float(np.mean(pop_scores <= risk_score) * 100)
        else:
            percentile = 50.0

        # ── Feature importances from the GB model ─────────────────────
        if self.pipeline is not None:
            gb = self.pipeline.named_steps["model"]
            importances = dict(zip(self.FEATURE_NAMES, gb.feature_importances_.tolist()))
        else:
            importances = {}

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
            self._save(population)
            return True
        return False
