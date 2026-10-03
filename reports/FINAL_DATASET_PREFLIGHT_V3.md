# Final Dataset Pre-Flight Verification V3

**Date**: 2026-10-02 11:36:28Z  
**Dataset Path**: `datasets/processed_v3/`  
**Configuration**: `datasets/processed_v3/dataset.yaml`  

---

## 1. Pre-Flight Checklist

| Check Item | Requirement | Observed Status | Verdict |
| :--- | :--- | :--- | :---: |
| **No Nonexistent Classes** | Only classes with physical annotations in dataset | 5 classes: `['normal', 'head_down', 'turn_head', 'discuss', 'stand']` | **PASS** |
| **No needs_review Labels** | Excluded from training set | Zero `needs_review` labels entered | **PASS** |
| **No Rejected Labels** | Excluded from training set | Zero `rejected` labels entered | **PASS** |
| **No Invalid BBoxes** | BBox coords in [0, 1], positive finite dimensions | Strict geometry validated | **PASS** |
| **No Corrupt Images** | All images readable by PIL/OpenCV | All images validated | **PASS** |
| **No Exact Duplicate Leakage** | Identical image hashes never cross splits | 0 duplicate leakage across splits | **PASS** |
| **No Sequence Leakage** | Connected recording groups never cross splits | 0 sequence leakage (DSU partitioned) | **PASS** |
| **Class IDs inside Final Taxonomy** | IDs must be strictly in 0..4 | All IDs in 0..4 | **PASS** |
| **Real Train Instances** | All train classes have real instances | All 5 classes populated | **PASS** |
| **Val Class Representation** | Val includes all canonical classes | All 5 classes present in val | **PASS** |
| **Provenance Complete** | Traceable to source_dataset/subset/image | Complete in `manifest_v3.json` | **PASS** |
| **Annotation Conflicts Quarantined** | Contradictory duplicate labels quarantined | 2939 quarantined in `annotation_conflicts_v3.json` | **PASS** |

---

## 2. Dataset Split Summary

| Split | Images | Image Percentage | Total Bounding Boxes |
| :--- | :---: | :---: | :---: |
| **Train** | 6,666 | 71.5% | 40,406 |
| **Val** | 1,400 | 15.0% | 9,327 |
| **Test** | 1,255 | 13.5% | 9,466 |
| **Total** | **9,321** | **100.0%** | **59,199** |

---

## 3. Class Distribution Across Splits

| Canonical Class | Train Boxes | Val Boxes | Test Boxes | Total Boxes | Train Images | Val Images | Test Images |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `normal` | 20,871 | 5,137 | 3,386 | **29,394** | 2,881 | 581 | 493 |
| `head_down` | 1,630 | 253 | 638 | **2,521** | 541 | 132 | 138 |
| `turn_head` | 5,872 | 1,321 | 3,226 | **10,419** | 1,230 | 376 | 474 |
| `discuss` | 3,696 | 525 | 628 | **4,849** | 617 | 96 | 117 |
| `stand` | 8,337 | 2,091 | 1,588 | **12,016** | 5,225 | 1,164 | 966 |

---

## 4. Pre-Flight Conclusion

**PRE-FLIGHT STATUS: PASSED**  
The dataset `datasets/processed_v3/` satisfies all structural, geometric, ethical, and grouping criteria.
Ready for single-epoch sanity execution.
