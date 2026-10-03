# V4C Environment Snapshot & Protected Baselines

**Document ID**: `reports/v4c/V4C_ENVIRONMENT.md`  
**Phase**: V4C Specialized Posture + Head-Pose Model Training & Benchmark  
**Date**: 2026-10-03  
**Status**: VERIFIED & FROZEN  

---

## 1. System & Compute Environment

| Parameter | Observed Physical Value |
| :--- | :--- |
| **Operating System** | Windows 11 Enterprise (AMD64) |
| **Python Runtime** | Python 3.13.9 (MSC v.1944 64-bit) |
| **PyTorch Version** | `2.14.1+cu130` |
| **Torchvision Version** | `0.29.1+cu130` |
| **CUDA Runtime** | CUDA 13.0 |
| **Target GPU** | NVIDIA GeForce RTX 5070 |
| **Compute Capability** | SM 12.0 (Blackwell architecture) |
| **Physical VRAM** | 11.94 GB (12,822,667,264 bytes) |
| **Ultralytics Version** | `8.4.171` |
| **OpenCV Version** | `5.0.0` |
| **`timm` Status** | Not installed; closest official lightweight backbone selected (`torchvision.models.mobilenet_v3_small` with official ImageNet weights) |

---

## 2. Protected Production Checkpoints Verification

The SHA-256 hashes of the protected production checkpoints were verified against original records prior to any V4C execution:

| Checkpoint Path | Expected SHA-256 Hash | Observed SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- |
| `models/trained/stage1_best.pt` | `6d713808f0bc670e8bf06e01b006fac73d8d6e5db7130584cec1658b003ce98a` | `6d713808f0bc670e8bf06e01b006fac73d8d6e5db7130584cec1658b003ce98a` | **MATCH (PROTECTED)** |
| `models/trained/stage1_5_best.pt` | `68690cf82715dc6dad2eb35a08711531170e935eb7dbe1c30fafabae341e2c2c` | `68690cf82715dc6dad2eb35a08711531170e935eb7dbe1c30fafabae341e2c2c` | **MATCH (PROTECTED)** |

Protected benchmarks:
- `datasets/processed_v3/` — Verified intact and unmodified.
- `datasets/processed_v3_5/` — Verified intact and unmodified.

---

## 3. Regression Baseline Verification

- **Pytest Suite (`tests/`)**: 69 of 69 tests passed in 4.60s.
- **Zero Cross-Split Leakage**: 0 crop ID collisions, 0 image ID collisions, 0 clip ID collisions across all 5 evaluation splits.
- **Quarantined Isolations**: All 4,476 context/ambiguous samples remain quarantined with zero contamination into supervised posture classes.
