"""
train_multi_target.py — Build only the penalty percentile thresholds
and multi-target probabilistic model without touching the base risk_model.pkl
or rebuilding the bulk CSV cache.

Usage:
    python scripts/train_multi_target.py
    python scripts/train_multi_target.py --sample-size 30000
    python scripts/train_multi_target.py --force   # ignore cached labels

Prerequisites:
    ml_cache/risk_model.pkl          (base MLRiskScorer — already built)
    ml_cache/inspections_bulk.csv    (OSHA inspection records)
    ml_cache/violations_bulk.csv     (OSHA violation records)
    ml_cache/accidents_bulk.csv      (optional — OSHA accident records)
    ml_cache/accident_injuries_bulk.csv  (optional)
"""
import argparse
import csv
import os
import sys
import time
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CACHE_DIR = "ml_cache"


# ------------------------------------------------------------------ #
#  Penalty percentile thresholds
# ------------------------------------------------------------------ #
def build_penalty_percentiles(scorer, *, force: bool = False):
    import pandas as pd
    from src.scoring.penalty_percentiles import (
        compute_penalty_percentiles,
        save_percentiles,
        CACHE_FILENAME,
    )

    out_path = os.path.join(CACHE_DIR, CACHE_FILENAME)
    if os.path.exists(out_path) and not force:
        print(f"  [skip] penalty_percentiles.json already exists ({out_path})")
        return

    viol_path = os.path.join(CACHE_DIR, "violations_bulk.csv")
    if not os.path.exists(viol_path):
        print("  ERROR: violations_bulk.csv not found")
        return

    cutoff_str = scorer.TEMPORAL_LABEL_CUTOFF.isoformat()
    insp_path  = os.path.join(CACHE_DIR, "inspections_bulk.csv")

    # Build activity_nr → naics_code mapping from inspections
    naics_by_act: dict = {}
    if os.path.exists(insp_path):
        print("  Building activity_nr → NAICS map from inspections …")
        csv.field_size_limit(10 * 1024 * 1024)
        with open(insp_path, "r", newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                act    = row.get("activity_nr", "").strip()
                naics  = row.get("naics_code",  "").strip()
                if act and naics:
                    naics_by_act[act] = naics[:2]
        print(f"  {len(naics_by_act):,} inspection records indexed")

    print(f"  Reading violations pre-{cutoff_str} …")
    rows = []
    csv.field_size_limit(10 * 1024 * 1024)
    with open(viol_path, "r", newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            iso = row.get("issuance_date", "")[:10]
            if iso and iso < cutoff_str:
                penalty = float(
                    row.get("current_penalty") or row.get("initial_penalty") or 0
                )
                act      = row.get("activity_nr", "").strip()
                naics_2d = naics_by_act.get(act, "__unknown__")
                if penalty > 0:
                    rows.append({"naics_2digit": naics_2d, "penalty_amount": penalty})

    if not rows:
        print("  No pre-cutoff violations found; check data.")
        return

    df = pd.DataFrame(rows)
    thresholds = compute_penalty_percentiles(df, min_group_n=50)
    save_percentiles(thresholds, out_path)
    g = thresholds["__global__"]
    print(
        f"  Saved {len(thresholds)-1} NAICS groups + global fallback → {out_path}"
    )
    print(
        f"  Global: P75=${g['p75']:,.0f}  P90=${g['p90']:,.0f}  P95=${g['p95']:,.0f}"
    )


# ------------------------------------------------------------------ #
#  Multi-target model
# ------------------------------------------------------------------ #
def build_multi_target_model(scorer, *, sample_size: int, force: bool = False):
    import numpy as np
    from src.scoring.multi_target_labeler import load_or_build
    from src.scoring.multi_target_scorer import MultiTargetRiskScorer, MODEL_FILE
    from src.scoring.penalty_percentiles import load_percentiles, CACHE_FILENAME

    model_path  = os.path.join(CACHE_DIR, MODEL_FILE)
    thresh_path = os.path.join(CACHE_DIR, CACHE_FILENAME)

    if os.path.exists(model_path) and not force:
        print(f"  [skip] multi_target_model.pkl already exists ({model_path})")
        return

    insp_path = os.path.join(CACHE_DIR, "inspections_bulk.csv")
    viol_path = os.path.join(CACHE_DIR, "violations_bulk.csv")
    acc_path  = os.path.join(CACHE_DIR, "accidents_bulk.csv")
    inj_path  = os.path.join(CACHE_DIR, "accident_injuries_bulk.csv")

    for p in (insp_path, viol_path, acc_path, inj_path):
        if not os.path.exists(p):
            print(f"  ERROR: {p} not found")
            return

    penalty_thresholds = load_percentiles(thresh_path)
    # Training feature cutoff uses the base-model cutoff (2022-01-01) which
    # excludes COVID-era data (2020-2021) from training features.  The
    # p_injury calibration handles the COVID-induced prevalence shift via the
    # combined temperature + logit-shift calibration using a target prevalence
    # of 12% (see _INJURY_CAL_TARGET_PREVALENCE in multi_target_scorer.py).
    cutoff_date = scorer.TEMPORAL_LABEL_CUTOFF
    # Outcome window runs from the training cutoff (2022-01-01) through the
    # most recent available data (2025-03-31), giving a 3+ year post-COVID
    # window.
    outcome_end = date(2025, 3, 31)

    # --force: delete label cache so load_or_build rebuilds from scratch
    if force:
        from src.scoring.multi_target_labeler import CACHE_FILENAME as LABEL_CACHE
        label_cache = os.path.join(CACHE_DIR, LABEL_CACHE)
        if os.path.exists(label_cache):
            os.remove(label_cache)
            print(f"  Deleted label cache: {label_cache}")

    print(
        f"  Building multi-target labels: cutoff={cutoff_date}  "
        f"outcome_end={outcome_end}  sample_size={sample_size:,}"
    )
    t0 = time.time()
    rows = load_or_build(
        scorer=scorer,
        cutoff_date=cutoff_date,
        outcome_end_date=outcome_end,
        cache_dir=CACHE_DIR,
        inspections_path=insp_path,
        violations_path=viol_path,
        accidents_path=acc_path,
        injuries_path=inj_path,
        naics_map=scorer._naics_map,
        penalty_thresholds=penalty_thresholds,
        sample_size=sample_size,
        min_hist_insp=2,
    )
    print(f"  Labels built: {len(rows):,} rows  ({time.time()-t0:.1f}s)")

    if len(rows) < 100:
        print(f"  ERROR: only {len(rows)} rows; cannot train.")
        return

    # Target distributions (staged pipeline)
    insp_rate = sum(r.get("future_has_inspection", 1) for r in rows) / len(rows)
    viol_rate_all = sum(r.get("future_has_violation", 0) for r in rows) / len(rows)
    wr_rate  = sum(r["any_wr_serious"]   for r in rows) / len(rows)
    inj_rate = sum(r["any_injury_fatal"] for r in rows) / len(rows)
    n_inspected = sum(1 for r in rows if r.get("future_has_inspection", 1))
    n_violated  = sum(1 for r in rows if r.get("future_has_violation", 0))
    print(f"  Stage 1 — Inspection rate:   {insp_rate:.1%}  ({n_inspected:,} inspected)")
    print(f"  Stage 2 — Violation rate:    {viol_rate_all:.1%}  ({n_violated:,} with violations)")
    print(f"  WR/Serious positive rate:    {wr_rate:.1%}")
    print(f"  Injury/Fatal positive rate:  {inj_rate:.1%}")

    X_raw = np.array([r["features_46"] for r in rows], dtype=float)
    X     = scorer._log_transform_features(np.nan_to_num(X_raw, nan=0.0))
    print(f"  Feature matrix: {X.shape}  (NaN replaced with 0)")

    t0 = time.time()
    print("  Training MultiTargetRiskScorer (patience ~2-5 min) …")
    mt = MultiTargetRiskScorer()
    mt.fit(X, rows, optimize_weights=True, val_fraction=0.20)
    print(f"  Training done  ({time.time()-t0:.1f}s)")
    print(f"  Composite weights: {[f'{w:.3f}' for w in mt._weights]}")

    mt.save(model_path)
    print(f"  Saved → {model_path}")


# ------------------------------------------------------------------ #
#  Entry point
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="Train multi-target OSHA risk model")
    parser.add_argument("--sample-size", type=int, default=50_000,
                        help="Training sample size (default 50,000)")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild even if cached files exist")
    args = parser.parse_args()

    print("=" * 60)
    print("  Multi-Target Model Trainer")
    print("=" * 60)

    # Load existing base scorer (loads risk_model.pkl, does NOT retrain)
    print("\nLoading base MLRiskScorer …")
    from src.scoring.ml_risk_scorer import MLRiskScorer
    t0 = time.time()
    scorer = MLRiskScorer()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Step 1: Penalty percentiles
    print("\n[1/2] Building penalty percentile thresholds …")
    build_penalty_percentiles(scorer, force=args.force)

    # Step 2: Multi-target model
    print("\n[2/2] Building multi-target probabilistic model …")
    build_multi_target_model(scorer, sample_size=args.sample_size, force=args.force)

    print("\n" + "=" * 60)
    print("  Done.  Re-run tests:")
    print("  python -m pytest tests/test_multi_target_validation.py -v")
    print("=" * 60)


if __name__ == "__main__":
    main()
