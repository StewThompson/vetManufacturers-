"""Validation of NAICS one-hot + log-transform refactoring."""
import sys
sys.path.insert(0, ".")
import numpy as np
from src.scoring.pseudo_labeler import pseudo_label
from src.scoring.ml_risk_scorer import MLRiskScorer

# 1. Feature count
n_features = len(MLRiskScorer.FEATURE_NAMES)
n_sectors = len(MLRiskScorer.NAICS_SECTORS)
n_display = len(MLRiskScorer.FEATURE_DISPLAY)
print(f"Feature count:  {n_features} (expected 46)")
print(f"NAICS sectors:  {n_sectors} (expected 24)")
print(f"Display entries: {n_display} (expected {n_features})")
assert n_features == 46, f"Expected 46 features, got {n_features}"
assert n_sectors == 24, f"Expected 24 sectors, got {n_sectors}"
assert n_display == n_features, f"Display count {n_display} != feature count {n_features}"

# 2. _encode_naics
scorer = MLRiskScorer.__new__(MLRiskScorer)
vec = scorer._encode_naics("325199")
assert sum(vec) == 1 and vec[5] == 1, "NAICS 32 encoding failed"
print(f"Encode '325199': naics_32=1 at idx 5  OK")

vec_none = scorer._encode_naics(None)
assert vec_none[-1] == 1, "None should set naics_unknown"
print(f"Encode None:     naics_unknown=1      OK")

# 3. Log transform helper
X_raw = np.array([[500, 1000, 0, 0, 0, 5000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0] + [0]*25], dtype=float)
X_log = MLRiskScorer._log_transform_features(X_raw)
assert X_log[0, 0] == np.log1p(500), f"Index 0 should be log1p(500), got {X_log[0, 0]}"
assert X_log[0, 1] == np.log1p(1000), f"Index 1 should be log1p(1000), got {X_log[0, 1]}"
assert X_log[0, 5] == np.log1p(5000000), f"Index 5 should be log1p(5M), got {X_log[0, 5]}"
assert X_log[0, 2] == 0, "Non-log features should be unchanged"
assert X_log[0, 6] == 0, "Non-log features should be unchanged"
print(f"Log transform: log1p(500)={X_log[0,0]:.2f}, log1p(1000)={X_log[0,1]:.2f}, "
      f"log1p(5M)={X_log[0,5]:.2f}  OK")

# 4. Volume bias check: large vs small company with same rates
# The log compression should make these much closer than 100x raw difference
small_raw = np.array([[5, 3, 0, 0, 0, 10000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0] + [0]*25])
large_raw = np.array([[500, 300, 0, 0, 0, 1000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0] + [0]*25])
small_log = MLRiskScorer._log_transform_features(small_raw)
large_log = MLRiskScorer._log_transform_features(large_raw)
# Raw ratio: 500/5 = 100x; log ratio: ~6.2/1.8 = 3.4x
raw_ratio = large_raw[0, 0] / small_raw[0, 0]
log_ratio = large_log[0, 0] / small_log[0, 0]
print(f"Volume compression: raw_ratio={raw_ratio:.0f}x, log_ratio={log_ratio:.1f}x  OK")
assert log_ratio < 10, f"Log ratio should compress to <10x, got {log_ratio:.1f}x"

# 5. Pseudo-labeler still works on raw values (not modified)
row_danger = np.zeros(46)
row_danger[0] = 10     # raw n_insp (pseudo-labeler sees raw)
row_danger[1] = 20
row_danger[2] = 0.8
row_danger[3] = 0.3
row_danger[4] = 0.2
row_danger[5] = 500000
row_danger[8] = 0.6
row_danger[10] = 2.0
row_danger[12] = 0.2
row_danger[14] = 6.0
row_danger[21 + 6] = 1
score_danger = pseudo_label(row_danger)
print(f"Pseudo-label (dangerous, raw features): {score_danger}")
assert score_danger > 60, f"Should score >60, got {score_danger}"

# Large compliant company pseudo-label should NOT be inflated
row_large_clean = np.zeros(46)
row_large_clean[0] = 500   # lots of inspections
row_large_clean[1] = 50    # few violations overall
row_large_clean[10] = 0.1  # low vpi
row_large_clean[16] = 0.9  # mostly clean inspections
row_large_clean[21 + 6] = 1
score_large_clean = pseudo_label(row_large_clean)
print(f"Pseudo-label (500 inspections, low violation rate): {score_large_clean}")
assert score_large_clean < 30, f"Large compliant company should score <30, got {score_large_clean}"

# 6. Feature names alignment
assert MLRiskScorer.FEATURE_NAMES[0] == "log_inspections"
assert MLRiskScorer.FEATURE_NAMES[1] == "log_violations"
assert MLRiskScorer.FEATURE_NAMES[5] == "log_penalties"
assert MLRiskScorer.FEATURE_NAMES[16] == "clean_ratio"
assert MLRiskScorer.FEATURE_NAMES[17] == "relative_violation_rate"
assert MLRiskScorer.FEATURE_NAMES[45] == "naics_unknown"
print("Feature name alignment: OK")

# 7. LOG_FEATURE_INDICES matches renamed features
for i in MLRiskScorer.LOG_FEATURE_INDICES:
    assert "log_" in MLRiskScorer.FEATURE_NAMES[i], \
        f"Index {i} ({MLRiskScorer.FEATURE_NAMES[i]}) should be a log feature"
print("LOG_FEATURE_INDICES alignment: OK")

print("\nAll checks passed!")
