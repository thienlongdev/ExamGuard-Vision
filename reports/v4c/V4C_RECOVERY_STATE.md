# V4C Crash Recovery State & Forensic Inventory

**Document ID**: `reports/v4c/V4C_RECOVERY_STATE.md`  
**Phase**: V4C Specialized Posture + Head-Pose Model Training & Benchmark  
**Date**: 2026-10-03  
**Status**: AUDITED & FROZEN  

---

## 1. Executive Forensic Summary

Following an unexpected machine shutdown during the active execution of Phase V4C, an exhaustive filesystem and process audit was performed. The primary objective is **Crash Recovery + Continuation** without destroying existing valid artifacts or retraining completed modules.

### Key Audit Findings:
1. **No Surviving Orphan Processes**:
   - Zero active background python training processes or GPU compute jobs detected (`nvidia-smi` confirms 0% GPU compute load, 3,669 MiB desktop allocation).
2. **Protected Baselines 100% Intact**:
   - `models/trained/stage1_best.pt` SHA256: `6d713808f0bc670e8bf06e01b006fac73d8d6e5db7130584cec1658b003ce98a` (**MATCH**).
   - `models/trained/stage1_5_best.pt` SHA256: `68690cf82715dc6dad2eb35a08711531170e935eb7dbe1c30fafabae341e2c2c` (**MATCH**).
   - `datasets/processed_v3/` and `datasets/processed_v3_5/` remain untouched.
3. **Dataset Manifests & Splits Physically Verified**:
   - `datasets/v4_crop/manifest.jsonl`: 20,490 total records.
     - `train.jsonl`: 14,821 crops
     - `same_domain_val.jsonl`: 2,973 crops
     - `high_angle_holdout.jsonl`: 1,618 crops
     - `cross_source_holdout.jsonl`: 543 crops
     - `temporal_holdout.jsonl`: 535 crops
   - `datasets/v4_head_pose/manifest.jsonl`: 23,080 total records (16,218 train, 2,862 val, 2,000 test).
4. **Regression Baseline**:
   - Pytest suite (`tests/`): **76 of 76 tests passed** (including CBAM architecture, gradient flow, and dataset tooling).
5. **Pre-Crash Artifact Discovered**:
   - `runs/v4c/A1_resnet18_cbam_tight_person_crop_224/best_model.pt` exists (135,363,045 bytes, SHA256: `69f3ed59ce6143efd09fde56e323aed053ceed8cf4611dc6c073c9a730a02ff9`).
   - Saved at **Epoch 10** with Same-Domain Val Macro F1: **0.8497** (Accuracy: 0.8826, Sleep Recall: 0.8571, Turn Recall: 0.7803).
   - **Interruption Diagnosis**: The run was interrupted during training/evaluation before completing all 30 epochs or writing final multi-split evaluation artifacts (`metrics.json` and `eval_predictions.json`).
   - **Resume State**: While model weights and optimizer state are preserved, scheduler and scaler state are missing, classifying it per Section 16 as `PARTIAL_NON_RESUMABLE`.
   - **Preservation Decision**: The partial artifact is preserved intact in `runs/v4c/A1_resnet18_cbam_tight_person_crop_224/best_model.pt`. To ensure full multi-split evaluation and mathematical purity, A1 will be rerun cleanly under the durable crash-safe pipeline, preserving historical logs.

---

## 2. V4C Task Recovery Matrix

| TASK | STATE | EVIDENCE & PHYSICAL ARTIFACTS |
| :--- | :---: | :--- |
| **Environment Audit** | `COMPLETE` | `reports/v4c/V4C_ENVIRONMENT.md` verified; RTX 5070 SM 12.0 verified. |
| **Dataset Verification** | `COMPLETE` | `reports/v4c/V4C_SPLIT_CLASS_SUPPORT.md`; exact split counts verified. |
| **CBAM Implementation** | `COMPLETE` | `src/models/posture/cbam.py`, `resnet_cbam.py`; unit tests 100% pass. |
| **A1 ResNet18 Tight** | `PARTIAL` | `runs/v4c/A1_resnet18_cbam_tight_person_crop_224/best_model.pt` (Epoch 10 saved, non-resumable). |
| **A2 ResNet18 Context** | `NOT_STARTED` | No physical directory in `runs/v4c`. |
| **B1 ResNet50 Tight** | `NOT_STARTED` | No physical directory in `runs/v4c`. |
| **B2 ResNet50 Context** | `NOT_STARTED` | No physical directory in `runs/v4c`. |
| **C1 Lightweight Tight** | `NOT_STARTED` | No physical directory in `runs/v4c`. |
| **C2 Lightweight Context** | `NOT_STARTED` | No physical directory in `runs/v4c`. |
| **Holdout Evaluation** | `NOT_STARTED` | Requires completed models for multi-split evaluation. |
| **Scale Robustness** | `NOT_STARTED` | Slices defined; pending model predictions. |
| **Blur Robustness** | `NOT_STARTED` | Slices defined; pending model predictions. |
| **Temporal Evaluation** | `NOT_STARTED` | EduAction clips defined; pending model predictions. |
| **Upper-Body Follow-Up** | `NOT_STARTED` | Optional follow-up post winner selection. |
| **320 Follow-Up** | `NOT_STARTED` | Optional follow-up post winner selection. |
| **Posture Winner Freeze** | `NOT_STARTED` | Pending primary benchmark completion. |
| **Head-Pose Target Audit** | `COMPLETE` | `reports/v4c/HEAD_POSE_TARGET_AUDIT.md` verified and frozen. |
| **Head-Pose Pre-Train Final Gate** | `PASS` | `HEAD_POSE_PRETRAIN_FINAL_GATE = PASS` (half-open [-99, +99), AFLW schema resolved, circular epsilon safety). |
| **Head-Pose Candidate A** | `QUEUED` | Script `scripts/train_head_pose_yaw.py` ready (half-open [-99, +99)). |
| **Head-Pose Candidate B** | `QUEUED` | Script `scripts/train_head_pose_yaw.py` ready (full domain circular sin/cos). |
| **Classroom Yaw Bridge** | `NOT_STARTED` | Script `scripts/run_classroom_yaw_bridge.py` ready. |
| **Latency Benchmark** | `NOT_STARTED` | Script `scripts/benchmark_runtime_throughput.py` ready. |
| **Multi-Student Benchmark**| `NOT_STARTED` | Script `scripts/benchmark_runtime_throughput.py` ready. |
| **Error Analysis** | `NOT_STARTED` | Script `scripts/generate_posture_error_analysis.py` ready. |
| **Regression Tests** | `COMPLETE` | 92 of 92 tests passing (0 failing). |
| **V4C Final Report** | `NOT_STARTED` | Pending benchmark completion. |

---

## 3. Durable Crash-Safe Continuation Strategy

To prevent loss of work in case of future unexpected system interruptions:
1. `runs/v4c/V4C_EXECUTION_STATE.json` tracks each completed task atomically.
2. `runs/v4c/V4C_RECOVERY_LOG.md` logs every completed task, timestamp, checkpoint path, and SHA256 hash.
3. Every training run will flush checkpoints and evaluation metrics immediately upon completion.
4. Each primary model will be trained sequentially with full evaluation across Same-Domain Val, High-Angle Holdout, Cross-Source Holdout, Temporal Holdout, Scale Slices, and Blur Slices.
