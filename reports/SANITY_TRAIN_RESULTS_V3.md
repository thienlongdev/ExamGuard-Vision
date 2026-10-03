# Sanity Training Results V3 (1 Epoch Verification)

**Date**: 2026-10-02 11:40:21Z  
**GPU**: NVIDIA GeForce RTX 5070 (6.56 GB peak reserved / 11.94 GB total)  
**Model**: `yolo26m.pt` (fresh load)  
**Dataset**: `datasets/processed_v3/dataset.yaml` (5 classes)  
**Epochs**: 1  
**Batch**: 8  
**Resolution**: 768px  
**Duration**: 172.8 seconds  

---

## 1. Sanity Success Verification

| Verification Criterion | Target Requirement | Observed Outcome | Status |
| :--- | :--- | :--- | :---: |
| **CUDA Execution** | cuda:0 active execution | GPU active (NVIDIA GeForce RTX 5070) | **PASS** |
| **VRAM Safety** | No Out-Of-Memory (OOM) | 6.56 GB peak (Safe, ~5.3 GB headroom) | **PASS** |
| **Losses Finite** | box, cls, dfl loss finite (no NaN, no Inf) | box: 1.77101, cls: 3.52201, dfl: N/A | **PASS** |
| **Validation Completed** | Validation pass over 1,400 val images | Completed (val_box: 1.53155, val_cls: 2.17921) | **PASS** |
| **Checkpoint Saved** | Valid PyTorch checkpoint saved | Saved at `C:\WorkingSpace Python\DETECTOR-YOLO\runs\detect\runs\sanity_v3\sanity_epoch1\weights` | **PASS** |
| **Zero Non-Existent Classes** | Class indices match dataset.yaml exactly | 5 classes: normal, head_down, turn_head, discuss, stand | **PASS** |

---

## 2. Training Metrics Summary (Epoch 1)

| Metric | Value |
| :--- | :---: |
| `train/box_loss` | `1.77101` |
| `train/cls_loss` | `3.52201` |
| `train/dfl_loss` | `N/A` |
| `val/box_loss` | `1.53155` |
| `val/cls_loss` | `2.17921` |
| `val/dfl_loss` | `N/A` |
| `metrics/mAP50(B)` | `0.23464` |
| `metrics/mAP50-95(B)` | `0.12938` |

---

## 3. Sanity Verdict

**ONE-EPOCH SANITY STATUS: PASSED**  
The training pipeline, dataset formatting, CUDA acceleration, and Blackwell GPU compatibility are confirmed functional.
