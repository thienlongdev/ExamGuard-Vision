# Local Live Performance & Soak Report

**Phase:** CAMERALESS SOFTWARE PREFLIGHT  
**Host Environment:** Desktop Workstation (Ryzen 9 9950X, RTX 5070 12GB VRAM, 32GB RAM, Windows 11)  
**Host Role:** `DESKTOP_ROLE = TRAINING_AND_BENCHMARK_WORKSTATION`  
**Target Demo Machine:** ASUS TUF Gaming A17 (FA707RC, Ryzen 7 6800H, RTX 3050 Laptop 4GB VRAM)  
**Status:** `NOT_MEASURED_NO_PHYSICAL_CAMERA` (Physical Camera Soak & Frame Latency Deferred)

---

## 1. System Memory & CUDA VRAM Footprint

During initialization of the full multi-model stack (YOLO26m Object Detector, Stage 1.5 Macro Detector, MobileNetV3 Posture, HopeNetYaw Headpose, ByteTrack, and FastAPI/Uvicorn), resource allocation was instrumented:

| Resource Metric | Value Observed | Status / Bounds |
|---|---|---|
| **CUDA Allocated VRAM** | `130.3 MB` | Bounded (Within 12,226 MB device capacity) |
| **CUDA Reserved VRAM** | `256.0 MB` | Stable |
| **CUDA Peak VRAM** | `256.0 MB` | Highly bounded |
| **Host System RAM (Initial)** | `9,650 MB` | Normal baseline |
| **Host System RAM (Peak)** | `9,780 MB` | No abnormal surge |
| **Host System RAM (Final)** | `9,660 MB` | Clean deallocation on reset |
| **Cameraless Memory Stability** | **STABLE** | `CAMERALESS_SOFTWARE_MEMORY_STABLE = YES` |
| **Physical Camera Soak** | **DEFERRED** | `PHYSICAL_CAMERA_MEMORY_STABLE = DEFERRED_NO_CAMERA_HARDWARE` |

---

## 2. Ingestion Queue & Measured Drop Rate Policy

Per strict scientific integrity rules, **drop count and drop percentage cannot be represented as measured values (e.g. 0%) when zero physical camera frames were captured**:

| Ingestion Metric | Value | Status |
|---|---|---|
| **Queue Depth Setting** | `5` | Bounded queue configured |
| **Frames Captured from Physical Camera** | `0` | No physical webcam present |
| **Frames Processed from Physical Camera** | `0` | No physical webcam present |
| **Drop Count** | `null` | Cannot measure drops without an active stream |
| **Drop Percentage** | `null` | Cannot measure drops without an active stream |
| **Performance Status** | `NOT_MEASURED_NO_PHYSICAL_CAMERA` | Deferred to ASUS TUF A17 live test |

---

## 3. Physical Camera Metrics Deferral Policy

- Because no physical camera hardware is connected to this desktop workstation, **capture FPS, processed FPS, capture-to-result latency distributions (mean, P50, P90, P95, P99, max), and queue wait distributions are recorded as `null`**.
- A 10-minute continuous live webcam soak test will be performed on the target ASUS TUF Gaming A17 laptop (`steady_state_10min_run = DEFERRED_UNTIL_PHYSICAL_WEBCAM_CONNECTED`).
- All software instrumentation hooks (`source_read_ms`, `general_detector_ms`, `crop_extraction_ms`, `posture_ms`, `headpose_ms`, `fusion_ms`, `event_ms`, `post_decode_pipeline_ms`, `whole_loop_end_to_end_ms`) are implemented and ready to log real timings on the demo laptop.
