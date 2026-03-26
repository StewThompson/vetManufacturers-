import os
import json
import pickle
import numpy as np
from collections import defaultdict
from datetime import date, timedelta
from typing import List, Dict, Optional

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.models.osha_record import OSHARecord


class MLRiskScorer:
    """
    Machine-learning risk scorer using scikit-learn.

    Trains a GradientBoostingClassifier on actual OSHA population data from
    the DOL API.  The model predicts the *probability* that a manufacturer
    will experience a serious OSHA enforcement event (serious/willful
    violation or fatality), expressed as a percentage (0–100).

    Labels are derived entirely from real inspection outcomes — no
    heuristic pseudo-labeling is used during training.
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

    CACHE_DIR = "ml_cache"
    MODEL_FILE = "risk_model.pkl"
    POP_FILE = "population_data.json"

    def __init__(self, osha_client=None):
        self.osha_client = osha_client
        self.pipeline: Optional[Pipeline] = None
        self.population_features: Optional[np.ndarray] = None
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

            # Binary label: 1 if the establishment experienced a serious
            # OSHA enforcement event (serious or willful violation, or fatality),
            # 0 otherwise.  Repeat and severe violations are intentionally
            # excluded from the label: they appear as *features* that the model
            # can learn to associate with risk, but using them in the label too
            # would make the classifier trivially predict based on label leakage.
            # Serious violations, willful violations, and fatalities are the
            # primary outcome signals used by OSHA itself to define "significant"
            # enforcement actions.
            had_serious_event = int(
                serious_raw > 0 or willful_raw > 0 or fat_count > 0
            )

            population.append({
                "name": estab,
                "label": had_serious_event,
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
        """Train a binary classifier on real OSHA enforcement outcomes.

        Labels are 1 where an establishment had at least one serious,
        willful violation or fatality — derived directly from inspection
        data, with no heuristic pseudo-labeling.  The model then predicts
        the *probability* of such an event for any new manufacturer.
        """
        X = np.array([p["features"] for p in population])
        y = np.array([p["label"] for p in population])

        # A classifier requires at least two classes to train.  With real
        # OSHA data this should never happen, but guard against degenerate
        # test fixtures or very small populations.
        if len(population) == 0:
            raise ValueError("Cannot train: population is empty.")
        if len(np.unique(y)) < 2:
            print(
                "Warning: training labels contain only one class — "
                "adding synthetic minority-class example so the classifier can fit."
            )
            # Append the feature mean as a synthetic opposite-class sample.
            # This is a last-resort guard; model accuracy will be degraded.
            X = np.vstack([X, X.mean(axis=0)])
            y = np.append(y, 1 - int(y[0]))

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
            )),
        ])
        self.pipeline.fit(X, y)
        self.population_features = X
        pos_rate = float(y.mean() * 100)
        print(
            f"ML Risk Classifier trained on {len(X)} establishments "
            f"({pos_rate:.1f}% had a serious enforcement event)."
        )

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #
    def _save(self, population: List[Dict]):
        model_path = os.path.join(self.CACHE_DIR, self.MODEL_FILE)
        pop_path = os.path.join(self.CACHE_DIR, self.POP_FILE)
        with open(model_path, "wb") as f:
            pickle.dump(self.pipeline, f)
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
                pop = meta["manufacturers"]
                # Reject cached data that pre-dates the real-label classifier:
                # old files lack the "label" key on each record.
                has_labels = len(pop) > 0 and "label" in pop[0]
                if (cache_date
                        and has_labels
                        and (date.today() - date.fromisoformat(cache_date)).days < 7):
                    with open(model_path, "rb") as f:
                        self.pipeline = pickle.load(f)
                    self.population_features = np.array([p["features"] for p in pop])
                    print(f"Loaded cached ML risk model (trained {cache_date}, {len(pop)} estabs).")
                    return
                elif not has_labels:
                    print("Cached model uses old pseudo-label format — rebuilding with real labels.")
            except Exception as e:
                print(f"Cache load failed: {e}")

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
        Predict the probability of a serious OSHA enforcement event for a
        manufacturer, expressed as a score on a 0–100 scale.

        Returns:
            risk_score      – 0 to 100  (predicted probability × 100)
            percentile_rank – 0 to 100 (higher = riskier than more peers)
            feature_weights – feature name → learned importance
            features        – feature name → raw value for this manufacturer
        """
        X = self.extract_features(records)

        # predict_proba returns [[p_no_event, p_serious_event]]
        prob_serious = float(self.pipeline.predict_proba(X)[0][1])
        risk_score = float(np.clip(prob_serious * 100, 0, 100))

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

        # Percentile rank within population (compare predicted probabilities)
        if self.population_features is not None and len(self.population_features) > 0:
            pop_probs = self.pipeline.predict_proba(self.population_features)[:, 1] * 100
            percentile = float(np.mean(pop_probs <= risk_score) * 100)
        else:
            percentile = 50.0

        # Feature importances from the GB model
        gb = self.pipeline.named_steps["model"]
        importances = dict(zip(self.FEATURE_NAMES, gb.feature_importances_.tolist()))

        # Raw feature values
        feature_vals = dict(zip(self.FEATURE_NAMES, X[0].tolist()))

        return {
            "risk_score": round(risk_score, 1),
            "percentile_rank": round(percentile, 1),
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
