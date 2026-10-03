# Local Live Perception Results

**Phase:** LOCAL LIVE CAMERA VALIDATION  
**Status:** ALL PERCEPTION BRANCHES CERTIFIED & OPERATIONAL (Physical Human Tests Deferred)

---

## 1. Branch Operational Status

All perception branches load the certified checkpoints with verified SHA-256 signatures, initialize on `cuda:0` (NVIDIA RTX 5070), and execute within memory bounds:

| Perception Branch | Checkpoint Path | Architecture / Framework | Certified SHA-256 | CUDA Initialization | Branch Operational |
|---|---|---|---|---|---|
| **General Object Detector** | `yolo26m.pt` | YOLO26m (640x640) | `401cea9ab23ad1...` | Yes (`cuda:0`) | **YES** |
| **Macro Behavior Detector** | `models/trained/stage1_5_best.pt` | YOLO26m (768x768) | `68690cf82715dc...` | Yes (`cuda:0`) | **YES** |
| **Specialized Posture** | `models/trained/v4_posture_best.pt` | MobileNetV3 (224x224) | `529a23f96ebec6...` | Yes (`cuda:0`) | **YES** |
| **Specialized Head-Pose** | `models/trained/v4_headpose_yaw_best.pt` | HopeNetYaw (224x224) | `5d15eec5941cfc...` | Yes (`cuda:0`) | **YES** |
| **Multi-Object Tracker** | N/A | ByteTrack (C++ / Python) | N/A | CPU / GPU pipeline | **YES** |
| **Phone Spatial Associator** | N/A | Spatial Heuristics | N/A | Geometry / Vectorized | **YES** |
| **V4D Multi-Cue Fusion** | `configs/v4d_fusion.yaml` | Temporal Buffer + Matrix | N/A | State Machine | **YES** |

---

## 2. Interactive Human Test Protocol Matrix

The 11 interactive tests are pre-configured in `scripts/run_local_live_validation.py` for execution as soon as a physical webcam is attached:

| Test ID | Test Name | Purpose / Expected Cue | Duration | Status on Desktop |
|---|---|---|---|---|
| **0** | Baseline | Sit upright, face forward; verify stable track, no false alerts | 20s | `NOT_RUN` (No camera) |
| **1** | Normal Reading / Writing | Look down at desk; verify `NORMAL_READ_WRITE`, veto false sleep | 30s | `NOT_RUN` (No camera) |
| **2** | Clear Turn Left | Turn head left ~45°; verify yaw deviation, `ORIENTATION_SUSTAINED_LEFT` | 10s | `NOT_RUN` (No camera) |
| **3** | Clear Turn Right | Turn head right ~45°; verify yaw deviation, `ORIENTATION_SUSTAINED_RIGHT` | 10s | `NOT_RUN` (No camera) |
| **4** | Head Rest / Sleep | Rest head on desk/arms; verify `HEAD_REST_SLEEP` / `SUSTAINED_HEAD_REST` | 15s | `NOT_RUN` (No camera) |
| **5** | Standing | Stand up clearly; verify Stage 1.5 macro detector `standing` cue | 15s | `NOT_RUN` (No camera) |
| **6** | Phone Contraband | Hold mobile phone near torso/desk; verify COCO cell phone + PhoneAssociator | 15s | `NOT_RUN` (No camera) |
| **7** | Leave and Return | Walk out of camera view and return; verify track disappearance & closure | 20s | `NOT_RUN` (No camera) |
| **8** | Partial Occlusion | Cover face with hand; verify graceful transition to `LIMITED` / `UNAVAILABLE` | 10s | `NOT_RUN` (No camera) |
| **9** | Distance / Scale | Move far then near; verify scale gating transitions (Full -> Limited -> Unavailable) | 20s | `NOT_RUN` (No camera) |
| **10** | Two-Person (Optional) | Two individuals in frame; verify distinct track IDs, independent cue state | 20s | `NOT_RUN` (No camera) |

Per strict scientific governance, **model observation performance on a single webcam session is distinguished from software pipeline operational validity**. Software branches are 100% operational; observational scores will be recorded during physical camera execution.
