# ExamGuard Vision — Multi-Camera Validation & Provenance Report

## 1. Executive Summary & Hardware Classification

This document provides the authoritative, empirically measured multi-camera validation benchmarks for ExamGuard Vision on the reference pilot edge node (ASUS TUF Gaming A17).

Every metric reported herein is accompanied by its explicit **Validation Origin** (`PHYSICAL_USB`, `MIXED_PHYSICAL_REPLAY`, `REPLAY_VIDEO`, or `AUTOMATED_MOCK`). Synthetic or mock timings are never conflated with physical camera performance.

### 1.1 Physical Hardware Inventory
- **Node**: ASUS TUF Gaming A17 (`FA707RC`)
- **CPU**: AMD Ryzen 7 6800H (8 Cores / 16 Threads @ 3.20 GHz, up to 4.70 GHz)
- **RAM**: 16 GB DDR5 (4800 MT/s)
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MB VRAM, GA107, compute capability 8.6)
- **Host OS**: Microsoft Windows 11 Home 64-bit
- **Runtime**: Python 3.13.11 64-bit, PyTorch 2.14.1+cu126, CUDA 12.6, DirectShow (`CAP_DSHOW`)

### 1.2 Physical Camera Discovery
DirectShow device enumeration on the pilot hardware confirmed:
- **Device Index 0**: `USB2.0 HD UVC WebCam` (Hardware ID `USB\VID_322E&PID_202C&MI_00\7&359322B1&0&0000`, 1280x720 @ 30 FPS, DirectShow). **Functional & Opened Successfully**.
- **Device Index 1**: DirectShow probe returned `isOpened() == False`. No secondary physical camera is connected.

**Authoritative Status**:
- `AVAILABLE_PHYSICAL_CAMERAS = 1`
- `PHYSICAL_SINGLE_CAMERA_VALIDATED = YES`
- `PHYSICAL_DUAL_CAMERA_VALIDATED = NO` *(Not physically possible on single-camera testbed)*
- `MIXED_DUAL_CAMERA_VALIDATED = YES` *(1 physical webcam + 1 video replay source)*
- `REPLAY_MULTI_CAMERA_VALIDATED = YES` *(2-camera and 4-camera video replay)*
- `MAX_PHYSICALLY_VALIDATED_CAMERAS = 1`
- `MAX_REPLAY_VALIDATED_CAMERAS = 4`

---

## 2. Multi-Camera Benchmark Results

The following table presents empirical performance recorded during multi-camera stress runs on the reference platform.

| Scenario | Camera Count | Sources | Resolution | Duration | Per-camera AI FPS | p50 Latency | p95 Latency | Drop % | Max Queue | VRAM (Alloc / Reserved) | Host RAM (RSS) | Result | Validation Origin |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Scenario 1: Single Camera Physical Ingest** | 1 | `cam01_usb` (Physical USB) | 1280x720 | 15.0s | **7.12 FPS** | 66.0 ms | 88.2 ms | 0.0% | 1 | 321.3 MB / 544.0 MB | 1647.5 MB | **PASS** | `PHYSICAL_USB` |
| **Scenario 2: Dual Camera (1 USB + 1 Replay)** | 2 | `cam01_usb` (Physical)<br>`cam02_replay` (Replay) | 1280x720<br>640x480 | 20.0s | `cam01`: **9.12 FPS**<br>`cam02`: **13.86 FPS** | 63.5 ms | 108.3 ms | 0.0% | 1 | 353.3 MB / 662.0 MB | 1699.9 MB | **PASS** | `MIXED_PHYSICAL_REPLAY` |
| **Scenario 3: Dual Camera Video Replay** | 2 | `cam01_replay` (Replay)<br>`cam02_replay` (Replay) | 640x480<br>640x480 | 20.0s | `cam01`: **14.13 FPS**<br>`cam02`: **14.09 FPS** | 62.3 ms | 79.9 ms | 0.0% | 1 | 353.3 MB / 662.0 MB | 1716.0 MB | **PASS** | `REPLAY_VIDEO` |
| **Scenario 4: Quad Camera Concurrency Stress** | 4 | `cam01`–`cam04` (Replay) | 640x480 | 20.0s | `cam01`: **7.80 FPS**<br>`cam02`: **7.74 FPS**<br>`cam03`: **7.74 FPS**<br>`cam04`: **7.79 FPS** | 120.1 ms | 185.7 ms | 0.0% | 1 | 417.3 MB / 726.0 MB | 1714.3 MB | **PASS** | `REPLAY_VIDEO` |

*Note on Automated Mocks*: Unit and contract tests utilizing `AUTOMATED_MOCK` (e.g. `tests/test_multi_camera_orchestration.py`) test reconnect state machines, URL redaction, and ring buffer isolation only. They are **never** included in performance benchmark tables.

---

## 3. Pipeline Metric Definitions

To prevent ambiguous claims, performance metrics are defined with strict semantics:
- **Ingestion / Capture FPS**: The rate at which the capture backend (`WebcamSource`, `VideoFileSource`, `RTSPSource`) decodes and pushes raw video frames into the pipeline's ingestion buffer.
- **AI Effective FPS**: The rate at which full perception inference (YOLO detection + ByteTrack + MobileNetV3 posture + HopeNet yaw + V4D fusion) processes frames for a specific camera channel.
- **UI Stream FPS**: The delivery rate of downstream compressed MJPEG frames served to connected web browsers via `/api/cameras/stream`.
- **Latency (p50 / p95)**: The end-to-end elapsed time from frame acquisition timestamp to the completion of V4D state machine updates and snapshot persistence checks.

---

## 4. GPU Resource Footprint & Model Registry Lock

### 4.1 Real VRAM Allocation vs Exaggerated Claims
Previous rough estimates claimed ~1.8 GB VRAM footprint. Direct PyTorch memory instrumentation clarifies the exact memory hierarchy:
- **Baseline Model Weights (`torch.cuda.memory_allocated()`)**: **130.34 MB**
  - YOLOv8/v11 Object Detector (`yolo26m.pt`): 44.3 MB
  - Stage 1.5 Macro Behavior (`stage1_5_best.pt`): 44.0 MB
  - MobileNetV3 Posture Classifier (`v4_posture_best.pt`): 18.5 MB
  - HopeNet-Yaw Head Pose Estimator (`v4_headpose_yaw_best.pt`): 23.5 MB tensor weights
- **PyTorch CUDA Allocator Reserved (`torch.cuda.memory_reserved()`)**: **444.00 MB** (warm-up baseline), expanding up to **726.00 MB** under 4-camera concurrent batch processing.
- **Peak Dynamic Allocated Memory (`torch.cuda.max_memory_allocated()`)**: **504.35 MB** during maximum simultaneous batch activations.
- **Process Host Working Set (RAM RSS)**: **~1,650 MB – 1,720 MB**, which encompasses Python interpreter, PyTorch C++ runtime, OpenCV DirectShow buffers, and SQLite in-memory caches.

### 4.2 Shared ModelRegistry & Serialization Lock
All active camera pipelines share a single singleton `ModelRegistry` instance on GPU `cuda:0`.
- Inference calls (`predict_objects`, `predict_posture`, `predict_head_pose`) are protected by an internal re-entrant lock (`threading.RLock`).
- **Purpose**: Serializes access to shared PyTorch models across threads to prevent concurrent execution on shared CUDA streams and maintain deterministic execution.
- **VRAM Impact**: Prevents duplicate model weight instantiation across multiple camera streams, guaranteeing operation within 4 GB VRAM limits.
- **Throughput Impact**: As camera count scales from 1 to 4, inference time is fairly divided among active pipelines (evidenced by the equal ~7.75 FPS per camera in 4-channel replay).

---

## 5. Multi-Camera Fairness & Isolation

### 5.1 Fairness Distribution
During concurrent multi-stream execution, the dispatcher provides balanced throughput without starvation:
- **Dual Replay**: Camera 1 processed 283 frames; Camera 2 processed 283 frames (Ratio = **1.000**).
- **Quad Replay**: Camera 1 (156), Camera 2 (155), Camera 3 (155), Camera 4 (156) (Maximum variance < 0.7%).

### 5.2 Fault Isolation
- When one camera stream encounters disconnection, packet loss, or termination, its worker thread transitions independently (`MẤT KẾT NỐI -> ĐANG KẾT NỐI LẠI`) with exponential backoff.
- Neighboring camera pipelines continue undisturbed; active database connections and shared models remain intact.

### 5.3 Identity & Evidence Scoping
- **Track Scope**: Every tracked candidate is scoped strictly by the composite key `(camera_id, track_id)`. `cam01 track #1` and `cam02 track #1` are strictly independent entities.
- **No Cross-Camera Re-ID**: The system explicitly does not attempt appearance or biometric cross-camera re-identification.
- **Ring Buffer & Snapshot Isolation**: Frame ring buffers, snapshot directories, and event evidence files are segmented per camera channel (`storage/sessions/{session_id}/evidence/{camera_id}/`), eliminating cross-contamination.
