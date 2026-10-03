# Local Live Limitations & Scientific Governance Report

**Phase:** CAMERALESS SOFTWARE PREFLIGHT  
**Host Role:** `DESKTOP_ROLE = TRAINING_AND_BENCHMARK_WORKSTATION`  
**Target Demo Machine:** ASUS TUF Gaming A17 (`FA707RC`, Ryzen 7 6800H, RTX 3050 Laptop 4GB VRAM)  
**Document Purpose:** Explicitly document hardware boundaries, operational assumptions, and governance limitations.

---

## 1. Hardware Boundary & System Role Separation

- **Desktop Host:** The current machine is an ASUS Desktop PC workstation (`DESKTOP_ROLE = TRAINING_AND_BENCHMARK_WORKSTATION`) with no integrated camera and no physical USB webcam attached.
- **Strict Anti-Fabrication Rule:** Synthetic video feeds, virtual webcam drivers, or phone bridges were forbidden and were NOT used to simulate camera presence.
- **Role Boundary:** The desktop workstation is certified for training, 4K stress, dataset processing, and packaging. The ASUS TUF Gaming A17 laptop is the designated `PORTABLE_LIVE_DEMO_TARGET` for integrated webcam capture, real human validation, and live demo delivery.
- **Physical Webcam Testing Deferred:** The physical camera validation session (Tests 0 to 9) is strictly deferred to the ASUS TUF Gaming A17 (`LOCAL_LIVE_CAMERA_VALIDATED = DEFERRED_NO_CAMERA_HARDWARE`).

---

## 2. Distinction Between Software Operationality and Physical Live Camera Passes

Per authoritative governance rules:
- **Software Pipeline Operationality (VERIFIED ON DESKTOP):**
  - `FASTAPI_LOCALHOST_PASS = YES`
  - `WEBSOCKET_LOCALHOST_PASS = YES`
  - `DASHBOARD_SOFTWARE_INTEGRATION_PASS = YES`
  - `GENERAL_DETECTOR_OPERATIONAL = YES`
  - `TRACKER_OPERATIONAL = YES`
  - `POSTURE_BRANCH_OPERATIONAL = YES`
  - `HEADPOSE_BRANCH_OPERATIONAL = YES`
  - `MACRO_BRANCH_OPERATIONAL = YES`
  - `PHONE_BRANCH_OPERATIONAL = YES`
  - `V4D_FUSION_OPERATIONAL = YES`
  - `EVENT_LIFECYCLE_SOFTWARE_PASS = YES`
  - `EVIDENCE_PIPELINE_SOFTWARE_PASS = YES`
  - `RUNTIME_STATE_RESET_PASS = YES`
  - `CAMERALESS_SOFTWARE_MEMORY_STABLE = YES`
- **Physical Live Camera Flags (DEFERRED TO ASUS TUF A17):**
  - `GENERAL_DETECTOR_LIVE_CAMERA_PASS = DEFERRED_NO_CAMERA_HARDWARE`
  - `BYTETRACK_LIVE_CAMERA_PASS = DEFERRED_NO_CAMERA_HARDWARE`
  - `POSTURE_BRANCH_LIVE_CAMERA_PASS = DEFERRED_NO_CAMERA_HARDWARE`
  - `HEADPOSE_BRANCH_LIVE_CAMERA_PASS = DEFERRED_NO_CAMERA_HARDWARE`
  - `MACRO_BRANCH_LIVE_CAMERA_PASS = DEFERRED_NO_CAMERA_HARDWARE`
  - `PHONE_BRANCH_LIVE_CAMERA_PASS = DEFERRED_NO_CAMERA_HARDWARE`
  - `V4D_FUSION_LIVE_CAMERA_PASS = DEFERRED_NO_CAMERA_HARDWARE`
  - `EVENT_LIFECYCLE_PHYSICAL_CAMERA_PASS = DEFERRED_NO_CAMERA_HARDWARE`
  - `EVIDENCE_FROM_PHYSICAL_CAMERA_PASS = DEFERRED_NO_CAMERA_HARDWARE`
  - `CAMERA_RESTART_PASS = DEFERRED_NO_CAMERA_HARDWARE`
  - `PHYSICAL_CAMERA_MEMORY_STABLE = DEFERRED_NO_CAMERA_HARDWARE`
- **Scientific Parameter Governance:**
  - Single-user webcam perception does NOT represent universal classroom distribution.
  - No behavioral thresholds (e.g. posture confidence, yaw angle limits, sleep candidate duration) may be altered based on local webcam observations.

---

## 3. Production Readiness & Deployment Gates

- **Current Readiness:** `READY_FOR_GITHUB_PUSH = YES`, `READY_FOR_ASUS_A17_CLONE = YES`, `READY_TO_BEGIN_LAPTOP_LIVE_CAMERA_VALIDATION = YES`.
- **Target CCTV Pilot Status:** Governed separately by pilot package (`READY_FOR_CONTROLLED_TARGET_CCTV_PILOT = NO`).
- **Production Status:** `PRODUCTION_READY = NO`.
