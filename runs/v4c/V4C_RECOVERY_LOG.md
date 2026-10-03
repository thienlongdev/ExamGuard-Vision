# V4C Persistent Recovery & Execution Log

**Log Path**: `runs/v4c/V4C_RECOVERY_LOG.md`  
**Target Hardware**: NVIDIA GeForce RTX 5070 (11.94 GB VRAM)  
**Session Started**: 2026-10-03 01:00:00  

---

## 2026-10-03 01:00:25 — Crash Recovery & Baseline Verification Complete
- **Process Safety**: 0 python training processes, 0 GPU compute jobs active.
- **Protected Hashes**:
  - `models/trained/stage1_best.pt`: `6d713808f0bc670e8bf06e01b006fac73d8d6e5db7130584cec1658b003ce98a` (**VERIFIED**)
  - `models/trained/stage1_5_best.pt`: `68690cf82715dc6dad2eb35a08711531170e935eb7dbe1c30fafabae341e2c2c` (**VERIFIED**)
- **Datasets**:
  - `datasets/v4_crop/manifest.jsonl`: 20,490 records (Train: 14,821, Same-Domain Val: 2,973, High-Angle: 1,618, Cross-Source: 543, Temporal: 535) (**VERIFIED**)
  - `datasets/v4_head_pose/manifest.jsonl`: 23,080 records (Train: 16,218, Val: 2,862, Test: 2,000) (**VERIFIED**)
- **Pytest**: 76 passed.
- **Interrupted Experiment**: `A1_resnet18_cbam_tight_person_crop_224` (partial checkpoint from epoch 10 preserved, marked `PARTIAL_NON_RESUMABLE`).
- **Next Step**: Launch clean crash-safe execution of primary posture matrix.

## [2026-10-03 01:03:38] Preserved pre-crash interrupted A1 checkpoint to best_model_interrupted_epoch10.pt

## [2026-10-03 02:19:57] Completed Experiment A2: ResNet18+CBAM (Context / 224)
- Best Epoch: 30
- Duration: 522.1s
- Checkpoint: best_model.pt (SHA256: 21ab0a4fc716798941d473976f08ab086316a174de9a651af46a4b2197f12879)
- Same-Domain Val Macro F1: 0.8554
- High-Angle Macro F1: 0.8186
- Cross-Source Macro F1: 0.8008
- Temporal Clip MajAcc: 0.8889
- Sleep Recall: 0.8681
- Turn Recall: 0.7121

## [2026-10-03 02:29:59] Completed Experiment B1: ResNet50+CBAM (Tight / 224)
- Best Epoch: 14
- Duration: 601.5s
- Checkpoint: best_model.pt (SHA256: b5ea5910167eca5536d2af2822598b7e341a215cdf706d7b12018e27028d0d2d)
- Same-Domain Val Macro F1: 0.1920
- High-Angle Macro F1: 0.2894
- Cross-Source Macro F1: 0.1677
- Temporal Clip MajAcc: 0.3333
- Sleep Recall: 0.0110
- Turn Recall: 0.0000

## [2026-10-03 02:34:40] Completed Experiment B2: ResNet50+CBAM (Context / 224)
- Best Epoch: 3
- Duration: 281.4s
- Checkpoint: best_model.pt (SHA256: 54482fb78d5f33df2181f7e1ac7bcc5a96f1668d81aebafab0195955adbebb35)
- Same-Domain Val Macro F1: 0.1849
- High-Angle Macro F1: 0.2900
- Cross-Source Macro F1: 0.1661
- Temporal Clip MajAcc: 0.3333
- Sleep Recall: 0.0000
- Turn Recall: 0.0000

## [2026-10-03 02:37:40] Completed Experiment C1: MobileNetV3-Small (Tight / 224)
- Best Epoch: 7
- Duration: 179.7s
- Checkpoint: best_model.pt (SHA256: 529a23f96ebec6051ad29279fba3cd5279e7aa8fe2e46292ab4bb9c9523ca180)
- Same-Domain Val Macro F1: 0.8634
- High-Angle Macro F1: 0.8182
- Cross-Source Macro F1: 0.8660
- Temporal Clip MajAcc: 0.8889
- Sleep Recall: 0.9890
- Turn Recall: 0.7348

## [2026-10-03 02:40:39] Completed Experiment C2: MobileNetV3-Small (Context / 224)
- Best Epoch: 7
- Duration: 179.1s
- Checkpoint: best_model.pt (SHA256: 958252ffb5895d056732075dfd950abf948319d8e35dfff48ea60a5f93938f45)
- Same-Domain Val Macro F1: 0.8865
- High-Angle Macro F1: 0.7930
- Cross-Source Macro F1: 0.8385
- Temporal Clip MajAcc: 0.8889
- Sleep Recall: 1.0000
- Turn Recall: 0.7576

## [2026-10-03 02:40:39] Generated POSTURE_MODEL_COMPARISON.md, TEMPORAL_POSTURE_STABILITY.md, POSTURE_SCALE_ROBUSTNESS.md, POSTURE_BLUR_ROBUSTNESS.md

## [2026-10-03 02:40:39] FROZEN POSTURE WINNER -> models/trained/v4_posture_best.pt (SHA256: 529a23f96ebec6051ad29279fba3cd5279e7aa8fe2e46292ab4bb9c9523ca180)

## [2026-10-03 02:45:13] Generated POSTURE_MODEL_COMPARISON.md, TEMPORAL_POSTURE_STABILITY.md, POSTURE_SCALE_ROBUSTNESS.md, POSTURE_BLUR_ROBUSTNESS.md

## [2026-10-03 02:45:13] FROZEN POSTURE WINNER -> models/trained/v4_posture_best.pt (SHA256: 529a23f96ebec6051ad29279fba3cd5279e7aa8fe2e46292ab4bb9c9523ca180)

## [2026-10-03 02:54:33] Completed Head-Pose HP_A: HopeNet-Yaw (ResNet50 + 66 Bins)
- Best Epoch: 5
- Duration: 559.6s
- Checkpoint: best_model.pt (SHA256: 5d15eec5941cfc8d2468d09c27bff8eca2014e62bb12fc9a95c0c6239fbbca55)
- AFLW-GT Val MAE: 5.76° (Median: 4.34°)
- AFLW2000-3D Test MAE: 4.53° (Median: 3.21°)

## [2026-10-03 03:03:15] Completed Head-Pose HP_B: ResNet18-Yaw (Continuous Regression Baseline)
- Best Epoch: 15
- Duration: 521.8s
- Checkpoint: best_model.pt (SHA256: bc31d46cfea007ebddfb6a8d5845641a32fd1cc018a831c4c8ae8b61e579a7d9)
- AFLW-GT Val MAE: 6.07° (Median: 4.43°)
- AFLW2000-3D Test MAE: 4.83° (Median: 3.6°)

## [2026-10-03 03:03:15] FROZEN HEAD-POSE WINNER -> models/trained/v4_headpose_yaw_best.pt (SHA256: 5d15eec5941cfc8d2468d09c27bff8eca2014e62bb12fc9a95c0c6239fbbca55)

## [2026-10-03 03:04:44] Generated POSTURE_MODEL_COMPARISON.md, TEMPORAL_POSTURE_STABILITY.md, POSTURE_SCALE_ROBUSTNESS.md, POSTURE_BLUR_ROBUSTNESS.md

## [2026-10-03 03:04:44] FROZEN POSTURE WINNER -> models/trained/v4_posture_best.pt (SHA256: 529a23f96ebec6051ad29279fba3cd5279e7aa8fe2e46292ab4bb9c9523ca180)

## [2026-10-03 03:04:44] FROZEN HEAD-POSE WINNER -> models/trained/v4_headpose_yaw_best.pt (SHA256: 5d15eec5941cfc8d2468d09c27bff8eca2014e62bb12fc9a95c0c6239fbbca55)

## [2026-10-03 03:04:45] Completed Classroom Yaw Bridge Analysis -> CLASSROOM_YAW_BRIDGE_ANALYSIS.md

## [2026-10-03 03:05:08] Completed Runtime Benchmark -> MULTI_STUDENT_THROUGHPUT.md and V4_RUNTIME_BUDGET.md

## [2026-10-03 03:05:08] Completed Posture Error Analysis -> POSTURE_ERROR_ANALYSIS.md and reports/v4c/error_analysis/

## [2026-10-03 03:05:16] Regression Test Suite: 100% Passed.

## [2026-10-03 03:05:16] Verified Historical Protection: stage1_best.pt and stage1_5_best.pt 100% Intact.

## [2026-10-03 03:05:16] GENERATED MASTER REPORT -> reports/V4C_SPECIALIZED_MODELS_FINAL.md

## [2026-10-03 03:12:02] Completed A1 Confirmation Run:
- Best Epoch: 18
- Duration: 359.6s
- Checkpoint: best_model.pt (SHA256: 49be87629bf312a7099b72caed7930d290950150c3537dce4ad9af2a474edc3f)
- Same-Domain Val Macro F1: 0.8751
- High-Angle Macro F1: 0.8142
- Cross-Source Macro F1: 0.8033
- Temporal Clip MajAcc: 0.9444
- Sleep Recall: 1.0000
- Turn Recall: 0.7197

## [2026-10-03 03:12:02] Regenerated posture benchmark reports with confirmed A1 results.

## [2026-10-03 03:12:02] FROZEN FINAL POSTURE WINNER -> models/trained/v4_posture_best.pt (SHA256: 529a23f96ebec6051ad29279fba3cd5279e7aa8fe2e46292ab4bb9c9523ca180)

## [2026-10-03 03:12:02] Regenerated Posture Error Analysis for final winner.

## [2026-10-03 03:12:02] REGENERATED MASTER REPORT -> reports/V4C_SPECIALIZED_MODELS_FINAL.md
