# Model Checkpoint Integrity & Cryptographic Provenance

This document establishes the verified, canonical cryptographic identity and physical byte preservation of all 7 machine learning checkpoints in ExamGuard Vision.

## Authoritative Checkpoint Verification Table

| Model | Current Path | Storage | File Size (Bytes) | SHA-256 Digest / LFS OID | Previous Path (Commit 9d9e4b2) | Structural Move Preserved Bytes | Result |
|---|---|---|---|---|---|---|---|
| **General Object Detector** | `models/trained/yolo26m.pt` | Git | 44,255,705 | `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7` | `yolo26m.pt` | YES | PASS |
| **Macro Behavior Detector** | `models/trained/stage1_5_best.pt` | Git | 44,038,169 | `68690cf82715dc6dad2eb35a08711531170e935eb7dbe1c30fafabae341e2c2c` | `models/trained/stage1_5_best.pt` | YES | PASS |
| **Posture Classifier** | `models/trained/v4_posture_best.pt` | Git | 18,518,447 | `529a23f96ebec6051ad29279fba3cd5279e7aa8fe2e46292ab4bb9c9523ca180` | `models/trained/v4_posture_best.pt` | YES | PASS |
| **Headpose Yaw Estimator** | `models/trained/v4_headpose_yaw_best.pt` | Git LFS | 284,178,037 | `5d15eec5941cfc8d2468d09c27bff8eca2014e62bb12fc9a95c0c6239fbbca55` | `models/trained/v4_headpose_yaw_best.pt` | YES | PASS |
| **Headpose Circular Fallback** | `models/fallback/headpose_resnet18/best_model.pt` | Git LFS | 134,264,773 | `bc31d46cfea007ebddfb6a8d5845641a32fd1cc018a831c4c8ae8b61e579a7d9` | `runs/v4c/headpose_resnet18_yaw/best_model.pt` | YES | PASS |
| **High-Res Person Crop Fallback** | `models/fallback/posture_320/best_model.pt` | Git | 18,518,447 | `070a2e328a161b1c957576ec13647d65846a073f9dd35488c7c6edc847f0f4cf` | `runs/v4c/C1_mobilenet_v3_small_tight_person_crop_320/best_model.pt` | YES | PASS |
| **Stage 1 Baseline Checkpoint** | `models/trained/stage1_best.pt` | Git | 44,048,089 | `6d713808f0bc670e8bf06e01b006fac73d8d6e5db7130584cec1658b003ce98a` | `models/trained/stage1_best.pt` | YES | PASS |

## Audit Findings & Verification Summary

1. **Byte-for-Byte Preservation**:
   All 7 model binaries in the current tree are 100% bit-for-bit identical to the binaries at baseline commit `9d9e4b2` and initial commits. Not a single model binary byte changed during repository restructuring (`af3bfbb`).

2. **Resolution of Prior Hash Discrepancy**:
   An earlier structural migration summary table reported four anomalous hashes (`984407b5...`, `34d193d2...`, `ad24d622...`, `93e0b249...`). Independent verification confirmed this was strictly a **`REPORTING_ERROR_ONLY`** in the assistant summary text. The underlying file binaries, Git blobs, Git LFS OIDs, and PowerShell `Get-FileHash` executions on disk remained perfectly unaltered with their true certified digests.

3. **Storage Modes**:
   - `models/trained/v4_headpose_yaw_best.pt` (284 MB) and `models/fallback/headpose_resnet18/best_model.pt` (134 MB) are tracked via Git LFS in `.gitattributes`.
   - All other 5 models are under GitHub's 100 MB limit and tracked directly in Git.
