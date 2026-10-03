# V4C Master Report Claim-to-Artifact Traceability Register

**Document ID**: `reports/v4c/final_verification/MASTER_REPORT_CLAIM_TRACEABILITY.md`  
**Phase**: V4C Specialized Posture + Head-Pose Model Verification  
**Date**: 2026-10-03  
**Status**: 100% REGENERATED & VERIFIED AGAINST PHYSICAL RAW ARTIFACTS  
**Requirement**: V4C Specification Section 19 & Section 36 (No Orphan Numerical Claims)  

---

## 1. Traceability Standard

Every numerical figure published in `reports/V4C_SPECIALIZED_MODELS_FINAL.md` is strictly grounded in raw physical filesystem artifacts:
- Model Metrics JSON (`runs/v4c/*/metrics.json` or `headpose_metrics.json`)
- Raw Prediction Logs (`eval_predictions.json`)
- Checkpoint Metadata (`torch.load(*.pt)`)
- Hardware Benchmark Tables (`reports/v4c/MULTI_STUDENT_THROUGHPUT.md`)
- Dataset Manifests (`datasets/v4_crop/manifest.jsonl`, `datasets/v4_head_pose/manifest.jsonl`)
- Test Execution Log (`pytest tests/ -v`)

---

## 2. Complete Numerical Claim Traceability Matrix

| Index | Master Report Claim Description | Published Value | Physical Source File | Exact Source Key / Field | Verification Status |
| :---: | :--- | :---: | :--- | :--- | :---: |
| **01** | Total Normalized Crop Dataset Size | 20,490 | `datasets/v4_crop/manifest.jsonl` | Line count (physical JSON records) | **VERIFIED** |
| **02** | Supervised Posture Classes Count | 4 | `src/models/posture/posture_classifier.py` | `len(POSTURE_CLASSES)` | **VERIFIED** |
| **03** | Train Partition Crop Count | 14,821 | `datasets/v4_crop/splits/train.jsonl` | Line count | **VERIFIED** |
| **04** | Same-Domain Val Crop Count | 2,973 | `datasets/v4_crop/splits/same_domain_val.jsonl` | Line count | **VERIFIED** |
| **05** | High-Angle Holdout Crop Count | 1,618 | `datasets/v4_crop/splits/high_angle_holdout.jsonl` | Line count | **VERIFIED** |
| **06** | Cross-Source Holdout Crop Count | 543 | `datasets/v4_crop/splits/cross_source_holdout.jsonl` | Line count | **VERIFIED** |
| **07** | Temporal Holdout Crop Count | 535 | `datasets/v4_crop/splits/temporal_holdout.jsonl` | Line count | **VERIFIED** |
| **08** | Quarantined Ambiguous Crops Count | 4,476 | `datasets/v4_crop/manifest.jsonl` | Count where `quality_flags == ["QUARANTINED"]` | **VERIFIED** |
| **09** | Stage 1 Protected Baseline SHA-256 | `6d713808...3ce98a` | `models/trained/stage1_best.pt` | File SHA-256 Digest | **VERIFIED** |
| **10** | Stage 1.5 Protected Baseline SHA-256 | `68690cf8...e2c2c` | `models/trained/stage1_5_best.pt` | File SHA-256 Digest | **VERIFIED** |
| **11** | C1 224 Posture Winner SHA-256 | `529a23f9...ca180` | `models/trained/v4_posture_best.pt` | File SHA-256 Digest | **VERIFIED** |
| **12** | C1 320 Posture Alternative SHA-256 | `070a2e32...f0f4cf` | `runs/v4c/C1_..._320/best_model.pt` | File SHA-256 Digest | **VERIFIED** |
| **13** | HopeNet Head-Pose Winner SHA-256 | `5d15eec5...bca55` | `models/trained/v4_headpose_yaw_best.pt` | File SHA-256 Digest | **VERIFIED** |
| **14** | ResNet18 Head-Pose Alternative SHA-256 | `bc31d46c...a7d9` | `runs/v4c/headpose_resnet18_yaw/best_model.pt` | File SHA-256 Digest | **VERIFIED** |
| **15** | C1 224 Same-Domain Val Macro F1 | 0.8634 | `runs/v4c/C1_..._224/metrics.json` | `same_domain_val.macro_f1` | **VERIFIED** |
| **16** | C1 224 Same-Domain Accuracy | 0.8871 | `runs/v4c/C1_..._224/metrics.json` | `same_domain_val.accuracy` | **VERIFIED** |
| **17** | C1 224 Same-Domain Balanced Acc | 0.8795 | `runs/v4c/C1_..._224/metrics.json` | `same_domain_val.balanced_accuracy` | **VERIFIED** |
| **18** | C1 224 High-Angle Macro F1 | 0.8182 | `runs/v4c/C1_..._224/metrics.json` | `high_angle_holdout.macro_f1` | **VERIFIED** |
| **19** | C1 224 Cross-Source Macro F1 | 0.8660 | `runs/v4c/C1_..._224/metrics.json` | `cross_source_holdout.macro_f1` | **VERIFIED** |
| **20** | C1 224 Cross-Source Sleep Recall | 0.6579 | `runs/v4c/C1_..._224/metrics.json` | `cross_source_holdout.per_class.HEAD_REST_SLEEP.recall` (50/76) | **VERIFIED** |
| **21** | C1 224 Same-Domain Sleep Recall | 0.9890 | `runs/v4c/C1_..._224/metrics.json` | `same_domain_val.per_class.HEAD_REST_SLEEP.recall` (90/91) | **VERIFIED** |
| **22** | C1 224 Same-Domain Turn Recall | 0.7348 | `runs/v4c/C1_..._224/metrics.json` | `same_domain_val.per_class.TURN_HEAD_CLEAR.recall` (97/132) | **VERIFIED** |
| **23** | C1 224 Temporal Clip Majority Acc | 0.8889 | `runs/v4c/C1_..._224/metrics.json` | `temporal_holdout.clip_majority_accuracy` | **VERIFIED** |
| **24** | C1 224 Canonical Flicker Rate | 0.0467 | `runs/v4c/C1_..._224/metrics.json` | `temporal_holdout.prediction_flicker_rate` (10/214) | **VERIFIED** |
| **25** | C1 320 Same-Domain Val Macro F1 | 0.8976 | `runs/v4c/C1_..._320/metrics.json` | `same_domain_val.macro_f1` | **VERIFIED** |
| **26** | C1 320 High-Angle Macro F1 | 0.8188 | `runs/v4c/C1_..._320/metrics.json` | `high_angle_holdout.macro_f1` | **VERIFIED** |
| **27** | C1 320 Cross-Source Macro F1 | 0.8334 | `runs/v4c/C1_..._320/metrics.json` | `cross_source_holdout.macro_f1` | **VERIFIED** |
| **28** | C1 320 Cross-Source Sleep Recall | 0.6184 | `runs/v4c/C1_..._320/metrics.json` | `cross_source_holdout.per_class.HEAD_REST_SLEEP.recall` (47/76) | **VERIFIED** |
| **29** | C1 320 Temporal Clip Majority Acc | 0.9444 | `runs/v4c/C1_..._320/metrics.json` | `temporal_holdout.clip_majority_accuracy` | **VERIFIED** |
| **30** | C1 320 Temporal Flicker Rate | 0.0000 | `runs/v4c/C1_..._320/metrics.json` | `temporal_holdout.prediction_flicker_rate` (0/214) | **VERIFIED** |
| **31** | C1 320 Very-Small Student F1 | 0.8174 | `runs/v4c/C1_..._320/metrics.json` | `scale_slices.PERSON_VERY_SMALL.macro_f1` | **VERIFIED** |
| **32** | A1 Confirmed Same-Domain Val F1 | 0.8751 | `runs/v4c/A1_..._confirmed/metrics.json` | `same_domain_val.macro_f1` | **VERIFIED** |
| **33** | A1 Confirmed High-Angle Macro F1 | 0.8142 | `runs/v4c/A1_..._confirmed/metrics.json` | `high_angle_holdout.macro_f1` | **VERIFIED** |
| **34** | A1 Confirmed Cross-Source Macro F1 | 0.8033 | `runs/v4c/A1_..._confirmed/metrics.json` | `cross_source_holdout.macro_f1` | **VERIFIED** |
| **35** | A1 Recovered Same-Domain Val F1 | 0.8847 | `runs/v4c/A1_..._224/metrics.json` | `same_domain_val.macro_f1` | **VERIFIED** |
| **36** | A1 Recovered High-Angle Macro F1 | 0.8433 | `runs/v4c/A1_..._224/metrics.json` | `high_angle_holdout.macro_f1` | **VERIFIED** |
| **37** | A2 Context Same-Domain Val F1 | 0.8554 | `runs/v4c/A2_..._224/metrics.json` | `same_domain_val.macro_f1` | **VERIFIED** |
| **38** | B1 Retrained Same-Domain Val F1 | 0.8430 | `runs/v4c/B1_..._224/metrics.json` | `same_domain_val.macro_f1` | **VERIFIED** |
| **39** | B2 Retrained Same-Domain Val F1 | 0.8402 | `runs/v4c/B2_..._224/metrics.json` | `same_domain_val.macro_f1` | **VERIFIED** |
| **40** | C2 Context Same-Domain Val F1 | 0.8865 | `runs/v4c/C2_..._224/metrics.json` | `same_domain_val.macro_f1` | **VERIFIED** |
| **41** | Upper-Body 224 Same-Domain Val F1 | 0.8723 | `runs/v4c/UB_..._224/metrics.json` | `same_domain_val.macro_f1` | **VERIFIED** |
| **42** | Total Head-Pose Manifest Records | 23,080 | `datasets/v4_head_pose/manifest.jsonl` | Line count (total records) | **VERIFIED** |
| **43** | Active Primary Head-Pose Records | 21,080 | `datasets/v4_head_pose/splits/` | Sum of train (16,218) + val (2,862) + test (2,000) | **VERIFIED** |
| **44** | Quarantined Counterpart HP Records | 2,000 | `datasets/v4_head_pose/manifest.jsonl` | Records with `split == "test_counterpart"` | **VERIFIED** |
| **45** | HopeNet AFLW2000 Common-Support N | 1,995 | `runs/v4c/headpose_hopenet_yaw/headpose_metrics.json` | `comparison_a_common_support.n` | **VERIFIED** |
| **46** | HopeNet Common-Support Test MAE | 4.38° | `runs/v4c/headpose_hopenet_yaw/headpose_metrics.json` | `comparison_a_common_support.mae_deg` | **VERIFIED** |
| **47** | HopeNet Common-Support Median Error | 3.20° | `runs/v4c/headpose_hopenet_yaw/headpose_metrics.json` | `comparison_a_common_support.median_ae_deg` | **VERIFIED** |
| **48** | HopeNet Common-Support P75 Error | 5.70° | `runs/v4c/headpose_hopenet_yaw/headpose_metrics.json` | `comparison_a_common_support.p75_deg` | **VERIFIED** |
| **49** | HopeNet Common-Support P90 Error | 9.54° | `runs/v4c/headpose_hopenet_yaw/headpose_metrics.json` | `comparison_a_common_support.p90_deg` | **VERIFIED** |
| **50** | HopeNet Clear-Turn Slice MAE | 6.23° | `runs/v4c/headpose_hopenet_yaw/headpose_metrics.json` | `aflw2000_3d_test.slices.clear_turn_reference_slice.mae_deg` | **VERIFIED** |
| **51** | HopeNet Clear-Turn Slice N | 611 | `runs/v4c/headpose_hopenet_yaw/headpose_metrics.json` | `aflw2000_3d_test.slices.clear_turn_reference_slice.n` | **VERIFIED** |
| **52** | HopeNet Ontology-Turn Slice MAE | 6.15° | `runs/v4c/headpose_hopenet_yaw/headpose_metrics.json` | `aflw2000_3d_test.slices.ontology_turn_reference_slice.mae_deg` | **VERIFIED** |
| **53** | HopeNet AFLW2000-3D Forced Full MAE | 4.53° | `runs/v4c/headpose_hopenet_yaw/headpose_metrics.json` | `aflw2000_3d_test.mae_deg` ($N=2,000$) | **VERIFIED** |
| **54** | ResNet18 AFLW2000 Full-Domain N | 2,000 | `runs/v4c/headpose_resnet18_yaw/headpose_metrics.json` | `aflw2000_3d_test.count` | **VERIFIED** |
| **55** | ResNet18 AFLW2000 Full-Domain MAE | 4.83° | `runs/v4c/headpose_resnet18_yaw/headpose_metrics.json` | `aflw2000_3d_test.mae_deg` | **VERIFIED** |
| **56** | ResNet18 Common-Support N | 1,995 | `runs/v4c/headpose_resnet18_yaw/headpose_metrics.json` | `comparison_a_common_support.n` | **VERIFIED** |
| **57** | ResNet18 Common-Support MAE | 4.72° | `runs/v4c/headpose_resnet18_yaw/headpose_metrics.json` | `comparison_a_common_support.mae_deg` | **VERIFIED** |
| **58** | ResNet18 Clear-Turn Slice MAE | 6.62° | `runs/v4c/headpose_resnet18_yaw/headpose_metrics.json` | `aflw2000_3d_test.slices.clear_turn_reference_slice.mae_deg` | **VERIFIED** |
| **59** | ResNet18 Large 45-90 Slice MAE | 6.93° | `runs/v4c/headpose_resnet18_yaw/headpose_metrics.json` | `aflw2000_3d_test.slices.large_45_90.mae_deg` | **VERIFIED** |
| **60** | Posture MobileNet Batch 30 GPU Mean | 4.166 ms | `reports/v4c/MULTI_STUDENT_THROUGHPUT.md` | Table 2, Row 5, Col 4 | **VERIFIED** |
| **61** | Posture MobileNet Batch 30 CPU Preproc | 21.50 ms | `reports/v4c/MULTI_STUDENT_THROUGHPUT.md` | Table 2, Row 5, Col 2 | **VERIFIED** |
| **62** | Posture MobileNet Batch 30 H2D | 0.784 ms | `reports/v4c/MULTI_STUDENT_THROUGHPUT.md` | Table 2, Row 5, Col 3 | **VERIFIED** |
| **63** | Posture MobileNet Batch 30 Peak VRAM | 149.1 MB | `reports/v4c/MULTI_STUDENT_THROUGHPUT.md` | Table 2, Row 5, Col 8 | **VERIFIED** |
| **64** | Posture MobileNet Batch 30 Throughput | 7,201.2 img/s | `reports/v4c/MULTI_STUDENT_THROUGHPUT.md` | Table 2, Row 5, Col 9 | **VERIFIED** |
| **65** | HopeNet 10-Head GPU Inference Mean | 4.59 ms | `reports/v4c/MULTI_STUDENT_THROUGHPUT.md` | Table 4, Row 1, Col 4 | **VERIFIED** |
| **66** | ResNet18 10-Head GPU Inference Mean | 1.96 ms | `reports/v4c/MULTI_STUDENT_THROUGHPUT.md` | Table 4, Row 2, Col 4 | **VERIFIED** |
| **67** | Scheduled Peak Active Compute Estimate | ~16.8 ms | `reports/v4c/V4_RUNTIME_BUDGET.md` | Section 3, Timeline Sum | **VERIFIED** |
| **68** | Classroom Yaw Upright Population N | 4,465 | `reports/v4c/CLASSROOM_YAW_BRIDGE_ANALYSIS.md` | Table 1, Row 1 | **VERIFIED** |
| **69** | Classroom Yaw Turn-Head Population N | 1,001 | `reports/v4c/CLASSROOM_YAW_BRIDGE_ANALYSIS.md` | Table 1, Row 2 | **VERIFIED** |
| **70** | Classroom Yaw Bridge Shift | +1.27° | `reports/v4c/CLASSROOM_YAW_BRIDGE_ANALYSIS.md` | Section 3, Row 1 (18.73° - 17.46°) | **VERIFIED** |
| **71** | Classroom Yaw Bridge Cohen's d | 0.087 | `reports/v4c/CLASSROOM_YAW_BRIDGE_ANALYSIS.md` | Section 3, Row 2 | **VERIFIED** |
| **72** | Classroom Yaw Empirical Overlap | 87.64% | `reports/v4c/CLASSROOM_YAW_BRIDGE_ANALYSIS.md` | Section 3, Row 3 | **VERIFIED** |
| **73** | Regression Test Pass Rate | 93 / 93 (100%) | `tests/` Physical PyTest Execution | Test Runner Output (93 passed, 1 warning) | **VERIFIED** |

---

## 3. Audit Certification

All 73 numerical claims across the master report and specialized reports have been verified against raw physical JSON artifacts, manifests, benchmark tables, and live test executions. Zero orphan or un-traced numerical claims remain.
