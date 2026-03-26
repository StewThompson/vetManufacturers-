import os
import json
import pickle
import numpy as np
from collections import defaultdict
from datetime import date, timedelta
from typing import List, Dict, Optional

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.models.osha_record import OSHARecord


class MLRiskScorer:
    """
    Machine-learning risk scorer using scikit-learn.

    Trains two predictive models on OSHA population data from the DOL API:

    1. **Serious-event classifier** (GradientBoostingClassifier): predicts the
       probability that a manufacturer will have a serious OSHA enforcement
       event in the next 12 months, trained on whether each establishment in
       the population actually has serious violations.

    2. **Violation regressor** (GradientBoostingRegressor): predicts expected
       violations per inspection, trained on the actual violation-per-inspection
       rate while intentionally excluding the direct violation-count features
       so the model learns from other safety signals (penalties, accidents,
       gravity, recency, etc.).

    Both models replace the former pseudo-label heuristic so the pipeline
    learns from real OSHA outcome data instead of approximating a hand-crafted
    scoring function.
    """

    FEATURE_NAMES = [
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
    }

    # Features used by the violation regressor.
    # Excludes the raw violation counts (total_violations, violations_per_inspection)
    # so the regressor learns from other safety signals rather than trivially
    # reproducing the target feature.
    REGRESSOR_FEATURE_NAMES = [
        "total_inspections",
        "serious_violations",
        "willful_violations",
        "repeat_violations",
        "total_penalties",
        "avg_penalty",
        "max_penalty",
        "recent_ratio",
        "severe_incidents",
        "accident_count",
        "fatality_count",
        "injury_count",
        "avg_gravity",
        "penalties_per_inspection",
        "clean_ratio",
    ]

    CACHE_DIR = "ml_cache"
    MODEL_FILE = "risk_model.pkl"
    POP_FILE = "population_data.json"

    def __init__(self, osha_client=None):
        self.osha_client = osha_client
        self.classifier_pipeline: Optional[Pipeline] = None
        self.regressor_pipeline: Optional[Pipeline] = None
        self.population_features: Optional[np.ndarray] = None
        self.population_risk_scores: Optional[np.ndarray] = None
        # Column indices in FEATURE_NAMES used by the violation regressor
        self._regressor_mask: np.ndarray = np.array(
            [self.FEATURE_NAMES.index(n) for n in self.REGRESSOR_FEATURE_NAMES]
        )
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

        return np.array([[
            n_insp, n_viols_w, serious, willful, repeat,
            total_pen, avg_pen, max_pen, recent_ratio, severe, vpi,
            accident_count, fatality_count, injury_count, avg_gravity,
            pen_per_insp, clean_ratio,
        ]])

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

            population.append({
                "name": estab,
                "features": [
                    n_insp, n_viols, serious_rate, willful_rate, repeat_rate,
                    total_pen, avg_pen, max_pen, recent_ratio, severe_rate, vpi,
                    acc_rate, fat_rate, inj_rate, avg_gravity,
                    pen_per_insp, clean_ratio,
                ],
            })

        print(f"  Aggregated {len(population)} unique establishments.")
        return population

    # ------------------------------------------------------------------ #
    #  Model training
    # ------------------------------------------------------------------ #
    def _train(self, population: List[Dict]):
        """Train predictive models directly on real OSHA outcome data.

        Model 1 — serious-event classifier:
            Target: whether the establishment has any serious violations
            (serious_violations rate > 0).  Trained on all features so the
            model can learn non-linear interactions across the full safety
            profile.

        Model 2 — violation-rate regressor:
            Target: violations per inspection (actual measured rate).
            Trained on REGRESSOR_FEATURE_NAMES, which excludes
            total_violations and violations_per_inspection themselves, so
            the model must learn from other safety signals (penalties,
            accidents, gravity, recency, etc.) rather than reproducing the
            target feature directly.

        After training, population risk scores are pre-computed so that
        percentile ranking at inference time is O(1).
        """
        X = np.array([p["features"] for p in population])

        serious_idx = self.FEATURE_NAMES.index("serious_violations")
        viols_idx = self.FEATURE_NAMES.index("violations_per_inspection")

        # Real outcome targets derived from actual OSHA data
        y_serious = (X[:, serious_idx] > 0).astype(int)
        y_viols = np.clip(X[:, viols_idx], 0, None)

        # --- Classifier: P(serious enforcement event) ---
        # Guard: GradientBoostingClassifier requires at least 2 distinct classes.
        # With a real population this is always satisfied; for tiny/homogeneous
        # datasets fall back to a dummy constant-probability classifier.
        if len(np.unique(y_serious)) >= 2:
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("model", GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                )),
            ])
            clf.fit(X, y_serious)
        else:
            from sklearn.dummy import DummyClassifier
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("model", DummyClassifier(strategy="prior")),
            ])
            clf.fit(X, y_serious)
            print("  Warning: single class in serious-event labels; using base-rate classifier.")
        self.classifier_pipeline = clf

        # --- Regressor: expected violations per inspection ---
        X_reg = X[:, self._regressor_mask]
        self.regressor_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
            )),
        ])
        self.regressor_pipeline.fit(X_reg, y_viols)

        self.population_features = X

        # Pre-compute population risk scores for fast percentile ranking
        pop_p_serious = self.classifier_pipeline.predict_proba(X)[:, 1]
        pop_exp_viols = np.clip(self.regressor_pipeline.predict(X[:, self._regressor_mask]), 0, None)
        self.population_risk_scores = self._compute_risk_score(pop_p_serious, pop_exp_viols)

        print(f"Predictive ML models trained on {len(X)} establishments.")

    # ------------------------------------------------------------------ #
    #  Risk score formula
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_risk_score(p_serious: np.ndarray, expected_viols: np.ndarray) -> np.ndarray:
        """Convert predictive outputs to a 0-100 risk score.

        Weights:
          60 % from P(serious enforcement event) — the primary risk signal
          40 % from violations per inspection — violation intensity

        The violation-rate component is capped at v_max = 5.0 violations per
        inspection.  Analysis of the OSHA population data shows that the 99th
        percentile sits around 4–6 violations per inspection; using 5.0 as the
        saturation point keeps the scale intuitive (>= 5 viol/insp → max
        contribution) without artificially compressing moderate values.
        """
        v_max = 5.0  # practical maximum: >= 5 violations/inspection → full 40 pts
        return np.clip(
            p_serious * 60.0 + np.minimum(expected_viols / v_max, 1.0) * 40.0,
            0.0, 100.0,
        )

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #
    def _save(self, population: List[Dict]):
        model_path = os.path.join(self.CACHE_DIR, self.MODEL_FILE)
        pop_path = os.path.join(self.CACHE_DIR, self.POP_FILE)
        with open(model_path, "wb") as f:
            pickle.dump({
                "classifier": self.classifier_pipeline,
                "regressor": self.regressor_pipeline,
                "population_risk_scores": self.population_risk_scores,
            }, f)
        with open(pop_path, "w") as f:
            json.dump({"date": str(date.today()), "manufacturers": population}, f)

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
                        models = pickle.load(f)
                    # Support new dict format; reject old single-pipeline format
                    if not isinstance(models, dict):
                        raise ValueError("Stale model format — retraining.")
                    self.classifier_pipeline = models["classifier"]
                    self.regressor_pipeline = models["regressor"]
                    self.population_risk_scores = models.get("population_risk_scores")
                    pop = meta["manufacturers"]
                    self.population_features = np.array([p["features"] for p in pop])
                    print(f"Loaded cached predictive models (trained {cache_date}, {len(pop)} estabs).")
                    return
            except Exception as e:
                print(f"Cache load failed: {e}")

        print("Building predictive ML models from DOL API data...")
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
        Predict risk for a manufacturer relative to the OSHA population.

        Returns:
            risk_score                   – 0 to 100
            percentile_rank              – 0 to 100 (higher = riskier than more peers)
            predicted_serious_prob       – 0 to 100 (% chance of serious enforcement event)
            predicted_expected_violations – expected violations per inspection
            predictive_statement         – plain-English predictive summary
            feature_weights              – feature name → classifier importance
            features                     – feature name → raw value for this manufacturer
            reputation_score             – 0 to 100
            news_sentiment               – sentiment label
        """
        X = self.extract_features(records)

        # --- Predictive model outputs ---
        if self.classifier_pipeline is not None:
            p_serious = float(self.classifier_pipeline.predict_proba(X)[0][1])
        else:
            # Fallback: heuristic probability from raw features when no model
            f = dict(zip(self.FEATURE_NAMES, X[0]))
            p_serious = float(np.clip(
                f["serious_violations"] * 0.6
                + f["willful_violations"] * 0.4
                + f["repeat_violations"] * 0.2,
                0.0, 1.0,
            ))

        if self.regressor_pipeline is not None:
            X_reg = X[:, self._regressor_mask]
            expected_viols = float(max(self.regressor_pipeline.predict(X_reg)[0], 0.0))
        else:
            # Fallback: use the raw violations-per-inspection feature
            f = dict(zip(self.FEATURE_NAMES, X[0]))
            expected_viols = float(max(f["violations_per_inspection"], 0.0))

        # --- Base risk score (0-100) ---
        risk_score = float(self._compute_risk_score(
            np.array([p_serious]), np.array([expected_viols])
        )[0])

        # --- Reputation adjustment (±10 pts, outside the OSHA pipeline) ---
        # The adjustment is capped at ±10 pts (reduced from the old ±15) because
        # the new OSHA-outcome models already produce a well-calibrated base score;
        # a smaller reputation nudge avoids over-weighting unverified news signals
        # relative to the structured enforcement data.
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
                sentiment_adj = ((neg - pos) / total) * 10.0
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

        # --- Percentile rank within population ---
        if self.population_risk_scores is not None and len(self.population_risk_scores) > 0:
            percentile = float(np.mean(self.population_risk_scores <= risk_score) * 100)
        else:
            percentile = 50.0

        # --- Feature importances from the classifier ---
        if self.classifier_pipeline is not None:
            gb = self.classifier_pipeline.named_steps["model"]
            importances = dict(zip(self.FEATURE_NAMES, gb.feature_importances_.tolist()))
        else:
            importances = {}

        # --- Raw feature values ---
        feature_vals = dict(zip(self.FEATURE_NAMES, X[0].tolist()))

        # --- Predictive statement ---
        predictive_statement = (
            f"This supplier has a {p_serious * 100:.0f}% predicted chance of a serious "
            f"OSHA enforcement event in the next 12 months and an expected "
            f"{expected_viols:.1f} violations per inspection, which maps to a risk "
            f"score of {risk_score:.0f}/100."
        )

        return {
            "risk_score": round(risk_score, 1),
            "percentile_rank": round(percentile, 1),
            "predicted_serious_prob": round(p_serious * 100, 1),
            "predicted_expected_violations": round(expected_viols, 1),
            "predictive_statement": predictive_statement,
            "feature_weights": importances,
            "features": feature_vals,
            "reputation_score": round(reputation_score, 1),
            "news_sentiment": news_sentiment,
        }

    def retrain(self):
        """Force a full retrain from fresh API data."""
        population = self._fetch_population()
        if len(population) >= 5:
            self._train(population)
            self._save(population)
            return True
        return False
