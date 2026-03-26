"""Diagnostic: analyze feature distribution in each score band."""
import sys; sys.path.insert(0, '.')
import numpy as np
import traceback

from tests.test_real_world_validation import RealWorldData

rw = RealWorldData.get()

# Feature indices (from pseudo_labeler tuple unpacking)
# (n_insp, n_viols, serious, willful, repeat, total_pen, avg_pen, max_pen,
#  recent_ratio, severe, vpi, accident_count, fatality_count, injury_count,
#  avg_gravity, pen_per_insp, clean_ratio)
FEAT = {
    'n_insp': 0, 'willful': 3, 'repeat': 4, 'recent_ratio': 8,
    'fat_rate': 12, 'clean_ratio': 16
}

try:
    bands = [(0,20),(20,40),(40,60),(60,80),(80,100)]
    print(f"\nBand analysis (paired n={len(rw.paired_pop):,})\n")
    header = f"{'Band':<10} {'n':>6} {'mean_adv':>10} {'zero_rec%':>10} {'mean_rr':>9} {'mean_wil':>9} {'mean_rep':>9} {'mean_fat':>9}"
    print(header)
    print("-" * len(header))
    for lo, hi in bands:
        idxs = [i for i,s in enumerate(rw.paired_scores) if lo <= s <= hi]
        n = len(idxs)
        if n == 0:
            print(f"[{lo}-{hi}]{'':>4} {'0':>6}")
            continue
        adv  = [rw.paired_adverse_scores[i] for i in idxs]
        feats = [rw.paired_pop[i]['features'] for i in idxs]
        rrs  = [f[FEAT['recent_ratio']] for f in feats]
        wils = [f[FEAT['willful']] for f in feats]
        reps = [f[FEAT['repeat']] for f in feats]
        fats = [f[FEAT['fat_rate']] for f in feats]
        zero_r = sum(1 for r in rrs if r < 0.01)
        print(f"[{lo}-{hi}]{'':<3} {n:>6,} {np.mean(adv):>10.2f} {100*zero_r/n:>9.0f}% "
              f"{np.mean(rrs):>9.3f} {np.mean(wils):>9.3f} {np.mean(reps):>9.3f} {np.mean(fats):>9.3f}")

    # Deeper look at 60-80 band
    print("\n--- 60-80 band: active (rr>0) vs dormant (rr=0) ---")
    idxs_60_80 = [i for i,s in enumerate(rw.paired_scores) if 60 <= s <= 80]
    feats_60_80 = [rw.paired_pop[i]['features'] for i in idxs_60_80]
    adv_60_80 = [rw.paired_adverse_scores[i] for i in idxs_60_80]
    active = [(a, f) for a,f in zip(adv_60_80, feats_60_80) if f[FEAT['recent_ratio']] >= 0.01]
    dormant = [(a, f) for a,f in zip(adv_60_80, feats_60_80) if f[FEAT['recent_ratio']] < 0.01]
    if active:
        print(f"  Active (rr>0): n={len(active)}, mean_adv={np.mean([a for a,_ in active]):.2f}, "
              f"mean_rr={np.mean([f[FEAT['recent_ratio']] for _,f in active]):.3f}")
    if dormant:
        print(f"  Dormant (rr=0): n={len(dormant)}, mean_adv={np.mean([a for a,_ in dormant]):.2f}")

    # 80-100 band breakdown
    print("\n--- 80-100 band: active vs dormant ---")
    idxs_80 = [i for i,s in enumerate(rw.paired_scores) if s > 80]
    for idx in idxs_80[:20]:
        f = rw.paired_pop[idx]['features']
        print(f"  score={rw.paired_scores[idx]:.1f}, adv={rw.paired_adverse_scores[idx]:.0f}, "
              f"rr={f[FEAT['recent_ratio']]:.2f}, wil={f[FEAT['willful']]:.2f}, "
              f"rep={f[FEAT['repeat']]:.2f}, fat={f[FEAT['fat_rate']]:.2f}")

except Exception:
    traceback.print_exc()
