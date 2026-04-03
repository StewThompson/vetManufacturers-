"""
reoptimize_weights.py — Re-optimize composite weights on an existing trained
MultiTargetRiskScorer using the improved Nelder-Mead optimizer.

This is much faster than full retraining because it only changes the 3 weights
in the composite formula.  The GBT heads are left untouched.

Usage:
    python scripts/reoptimize_weights.py

Prerequisites:
    ml_cache/multi_target_model.pkl  (trained model to update)
    ml_cache/multi_target_labels.pkl (cached training labels)
    OR
    ml_cache/inspections_bulk.csv + violations_bulk.csv etc. (to rebuild labels)
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

CACHE_DIR = "ml_cache"


def main():
    from src.scoring.ml_risk_scorer import MLRiskScorer
    from src.scoring.multi_target_scorer import MultiTargetRiskScorer, MODEL_FILE
    from src.scoring.multi_target_labeler import load_or_build, CACHE_FILENAME as LABEL_CACHE
    from src.scoring.penalty_percentiles import load_percentiles, CACHE_FILENAME as PERC_CACHE
    from datetime import date

    model_path  = os.path.join(CACHE_DIR, MODEL_FILE)
    label_cache = os.path.join(CACHE_DIR, LABEL_CACHE)
    thresh_path = os.path.join(CACHE_DIR, PERC_CACHE)

    print("=" * 60)
    print("  Weight Re-Optimizer (Nelder-Mead)")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"  ERROR: {model_path} not found; run train_multi_target.py first")
        return

    # Load model
    print(f"\nLoading model from {model_path} …")
    mt = MultiTargetRiskScorer.load(model_path)
    print(f"  Current weights: {[f'{w:.3f}' for w in mt._weights]}")
    print(f"  is_fitted: {mt.is_fitted}")

    # Load scorer for feature extraction
    print("\nLoading base MLRiskScorer …")
    scorer = MLRiskScorer()

    # Load or build labels
    print("\nLoading label cache …")
    penalty_thresholds = load_percentiles(thresh_path)
    cutoff_date = scorer.TEMPORAL_LABEL_CUTOFF
    # Match outcome_end used in training to avoid fingerprint mismatch
    outcome_end = date(2023, 12, 31)

    rows = load_or_build(
        scorer=scorer,
        cutoff_date=cutoff_date,
        outcome_end_date=outcome_end,
        cache_dir=CACHE_DIR,
        inspections_path=os.path.join(CACHE_DIR, "inspections_bulk.csv"),
        violations_path=os.path.join(CACHE_DIR, "violations_bulk.csv"),
        accidents_path=os.path.join(CACHE_DIR, "accidents_bulk.csv"),
        injuries_path=os.path.join(CACHE_DIR, "accident_injuries_bulk.csv"),
        naics_map=scorer._naics_map,
        penalty_thresholds=penalty_thresholds,
        sample_size=scorer.TEMPORAL_SAMPLE_SIZE,
    )
    print(f"  {len(rows):,} rows loaded")

    if len(rows) < 200:
        print("  ERROR: too few rows; aborting.")
        return

    # Build feature matrix
    X_raw = np.array([r["features_46"] for r in rows], dtype=float)
    X     = scorer._log_transform_features(np.nan_to_num(X_raw, nan=0.0))

    # Use 20% validation split (same as training)
    from sklearn.model_selection import train_test_split
    _, X_val, _, rows_val = train_test_split(X, rows, test_size=0.20, random_state=42)
    y_adv_val = np.array([r["real_label"] for r in rows_val])

    print(f"\nRunning weight optimization on {len(X_val):,} validation rows …")
    from scipy.stats import spearmanr

    pred_before = mt.predict_batch(X_val)
    comp_before = np.array([mt.composite_score(p) for p in pred_before])
    rho_before, _ = spearmanr(comp_before, y_adv_val)
    print(f"  Spearman (before):  ρ={rho_before:.4f}  weights={[f'{w:.3f}' for w in mt._weights]}")

    t0 = time.time()
    mt.fit_weights(X_val, y_adv_val)
    print(f"  Optimization done  ({time.time()-t0:.1f}s)")

    pred_after = mt.predict_batch(X_val)
    comp_after = np.array([mt.composite_score(p) for p in pred_after])
    rho_after, _ = spearmanr(comp_after, y_adv_val)
    print(f"  Spearman (after):   ρ={rho_after:.4f}  weights={[f'{w:.3f}' for w in mt._weights]}")

    if rho_after >= rho_before - 0.005:
        mt.save(model_path)
        print(f"\n  Saved updated model \u2192 {model_path}  ({rho_after-rho_before:+.4f} Spearman)")
    else:
        print(f"\n  New weights not better (Δ={rho_after-rho_before:+.4f}); NOT saving.")

    print("\n  Re-run tests:")
    print("  python -m pytest tests/test_multi_target_validation.py -v")


if __name__ == "__main__":
    main()
