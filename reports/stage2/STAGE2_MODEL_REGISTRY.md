# Stage 2 Model Registry & Provenance Audit
**Phase**: Stage 2 Full End-to-End Orchestration  
**Status**: Authoritative Model Checkpoint Audit  
**Date**: 2026-10-03  

---

## 1. Frozen Checkpoint Integrity & Verification

All perception models used in Stage 2 orchestration are strictly frozen baselines. No model retraining or weight mutation was executed. Checkpoints were verified bit-exact before and after Stage 2 orchestration benchmarking using cryptographic SHA-256 digests.

| Model Role | Checkpoint Path | Architecture | Input Resolution | Precision | Target Device | SHA-256 Digest | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **General Object Detector** | `yolo26m.pt` | YOLO26m (COCO 80 cls) | 640x640 | FP32 | `cuda:0` | `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7` | **VERIFIED** |
| **Macro Behavior Detector** | `models/trained/stage1_5_best.pt` | YOLO26m-Custom (5 cls) | 768x768 | FP32 | `cuda:0` | `68690cf82715dc6dad2eb35a08711531170e935eb7dbe1c30fafabae341e2c2c` | **VERIFIED** |
| **Primary Posture Classifier** | `models/trained/v4_posture_best.pt` | MobileNetV3-Small (4 cls) | 224x224 | FP32 | `cuda:0` | `529a23f96ebec6051ad29279fba3cd5279e7aa8fe2e46292ab4bb9c9523ca180` | **VERIFIED** |
| **Primary Head-Pose Estimator** | `models/trained/v4_headpose_yaw_best.pt` | HopeNet-Yaw (ResNet50, 66 bins, $[-99, +99)^\circ$) | 224x224 | FP32 | `cuda:0` | `5d15eec5941cfc8d2468d09c27bff8eca2014e62bb12fc9a95c0c6239fbbca55` | **VERIFIED** |
| **Historical Baseline (Protected)** | `models/trained/stage1_best.pt` | YOLO26m-Custom (5 cls) | 768x768 | FP32 | Offline Ref | `6d713808f0bc670e8bf06e01b006fac73d8d6e5db7130584cec1658b003ce98a` | **VERIFIED** |
| **High-Res Posture Candidate** | `runs/v4c/C1_mobilenet_v3_small_tight_person_crop_320/best_model.pt` | MobileNetV3-Small | 320x320 | FP32 | Gated Default OFF | `070a2e328a161b1c957576ec13647d65846a073f9dd35488c7c6edc847f0f4cf` | **VERIFIED** |
| **Full-Domain Head-Pose Fallback**| `runs/v4c/headpose_resnet18_yaw/best_model.pt` | ResNet18-Yaw (Circular) | 224x224 | FP32 | Gated Fallback | `bc31d46cfea007ebddfb6a8d5845641a32fd1cc018a831c4c8ae8b61e579a7d9` | **VERIFIED** |

---

## 2. Singleton Lifecycle & Runtime Warmup

1. **Zero Reload per Frame**: Model weights are loaded strictly once at pipeline initialization by `ModelRegistry.get_instance()`. Model reloading during frame loops is completely prohibited and verified by regression test `test_stage2_model_registry_singleton_initialization`.
2. **CUDA Kernel Warm-up**: Before ingesting real frames, `_warmup()` runs synthetic forward passes for posture and head-pose models with CUDA synchronization to eliminate initial kernel compilation latency spikes from runtime measurements.
3. **API Provenance Exposure**: Active model metadata and SHA-256 hashes are exposed via the `/api/system/models` endpoint for external auditability.
