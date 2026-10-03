# Camera Calibration Report: pilot_cam_01

## 1. Camera Profile & Mounting
- **Camera ID**: `pilot_cam_01`
- **Camera Name**: Camera pilot_cam_01
- **Viewpoint Profile**: `FRONT_OBLIQUE`
- **Source Resolution**: 640x480
- **Nominal FPS**: 25.0 (Received: 14.0 FPS)
- **Room ID**: `hall_101`
- **Mounting Details**: Height UNKNOWNm, Pitch UNKNOWN°

---

## 2. Capability Evaluation
| Capability Branch | Level | Quantitative Physical Visibility Coverage |
| :--- | :--- | :--- |
| **Posture Capability** | **UNAVAILABLE** | ≥120px: 0.0%, 60-119px: 0.0%, <60px: 0.0% |
| **Head-Pose Capability** | **UNAVAILABLE** | Head crop ≥25x25px: 0.0% |
| **Phone Capability** | **UNAVAILABLE** | Median apparent size: NOT_OBSERVED |
| **Tracking Capability** | **SUPPORTED** | Mean track lifespan: 1.0 frames |

---

## 3. Camera Geometry Diagnostics
- `LOW_HEADPOSE_COVERAGE`
- `HEADPOSE_UNAVAILABLE`
- `PHONE_VISIBILITY_LIMITED`

---

## 4. Physical Runtime Telemetry
- **Frames Processed**: 90
- **Mean Whole-Loop Latency**: 20.97 ms (P95: 27.95 ms)
- **Decode Latency**: 50.02 ms
- **Frame Drop Count**: 0
- **Max Ingestion Queue Depth**: 0
- **GPU VRAM Allocated**: 289.3 MB
- **Host RSS Memory**: 2108.7 MB

*Notice: All capability indicators represent physical visibility and execution feasibility. Zero behavioral accuracy claims are made without labeled ground truth.*
