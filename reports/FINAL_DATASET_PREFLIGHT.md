# Final Processed Dataset Preflight Integrity Report

**Preflight Result**: PASS — READY FOR SANITY TRAINING  
**Audit Timestamp**: 2026-10-02  
**Total Processed Images**: 8,116  
**Total Processed Labels**: 8,116  

## 1. Preflight Integrity Rules Check

| Rule Description | Requirement | Status | Observed Value / Details |
| :--- | :--- | :---: | :--- |
| **1. Valid Class IDs** | Every token in 0..6 | PASS | Class IDs seen: [0, 1, 3, 5] |
| **2. Image Pairing** | No missing image files | PASS | 100% paired (0 missing) |
| **3. Label Pairing** | No missing label files | PASS | 100% paired (0 missing) |
| **4. Bounding Box Geometry** | Normalized [0, 1], w>0, h>0 | PASS | Invalid boxes: 0 |
| **5. Canonical Vocabulary** | Only canonical class IDs | PASS | Max ID: 5 (limit: 6) |
| **6. Approved Mappings Only** | All annotations approved | PASS | All manifest records have status='approved' |
| **7. No Rejected Classes** | Zero 'cheating' labels | PASS | 0 rejected classes present |
| **8. No Ignored Classes** | Zero 'hand-raising' | PASS | 0 ignored classes present |
| **9. Zero Duplicate Leakage** | SHA256 clusters isolated | PASS | Shared frames across splits: 0 |
| **10. Zero Sequence Leakage** | Groups isolated to 1 split | PASS | Shared groups across splits: 0 |

## 2. Leakage Verification Details

- **Train ∩ Val Shared Frames**: 0
- **Train ∩ Test Shared Frames**: 0
- **Val ∩ Test Shared Frames**: 0
- **Train ∩ Val Shared Sequence Groups**: 0
- **Train ∩ Test Shared Sequence Groups**: 0
- **Val ∩ Test Shared Sequence Groups**: 0

## 3. Split Sizes & Groups

- **Train**: 5,665 unique frames across 136 sequence clips
- **Validation**: 1,218 unique frames across 27 sequence clips
- **Test**: 1,233 unique frames across 26 sequence clips

All 10 preflight rules are fully satisfied. The processed dataset is strictly verified for training.
