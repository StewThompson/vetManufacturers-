import os
import numpy as np
from src.scoring.multi_target_scorer import MultiTargetRiskScorer, MODEL_FILE
from src.scoring.ml_risk_scorer import MLRiskScorer
from src.scoring.multi_target_labeler import load_or_build, CACHE_FILENAME as LABEL_CACHE
from src.scoring.penalty_percentiles import load_percentiles, CACHE_FILENAME as PERC_CACHE
from datetime import date
from sklearn.model_selection import train_test_split

CACHE_DIR = "ml_cache"

mt = MultiTargetRiskScorer.load(os.path.join(CACHE_DIR, MODEL_FILE))

scorer = MLRiskScorer()
penalty_thresholds = load_percentiles(os.path.join(CACHE_DIR, PERC_CACHE))
rows = load_or_build(
    scorer=scorer,
    cutoff_date=scorer.TEMPORAL_LABEL_CUTOFF,
    outcome_end_date=date(2023, 12, 31),
    cache_dir=CACHE_DIR,
    inspections_path=os.path.join(CACHE_DIR, "inspections_bulk.csv"),
    violations_path=os.path.join(CACHE_DIR, "violations_bulk.csv"),
    accidents_path=os.path.join(CACHE_DIR, "accidents_bulk.csv"),
    injuries_path=os.path.join(CACHE_DIR, "accident_injuries_bulk.csv"),
    naics_map=scorer._naics_map,
    penalty_thresholds=penalty_thresholds,
    sample_size=scorer.TEMPORAL_SAMPLE_SIZE,
)

X_raw = np.array([r["features_46"] for r in rows], dtype=float)
X     = scorer._log_transform_features(np.nan_to_num(X_raw, nan=0.0))
_, X_val, _, _ = train_test_split(X, rows, test_size=0.20, random_state=42)

# Set the user weights explicitly:
mt._weights = [0.578, 0.256, 0.165]

# Rebuild CDF!
pred_val = mt.predict_batch(X_val)
raw_val = np.array([mt._raw_composite(p) for p in pred_val])
mt._score_cdf = np.sort(raw_val)

mt.save(os.path.join(CACHE_DIR, MODEL_FILE))
print("Saved fixed model with new CDF!")