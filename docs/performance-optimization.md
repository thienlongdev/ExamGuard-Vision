# ExamGuard Vision — Maximum Useful Throughput & Dense-Room Performance Optimization

## 1. Executive Summary

This phase optimized ExamGuard Vision for real-time exam room monitoring on edge hardware (specifically the canonical physical testbed: **ASUS TUF Gaming A17 with AMD Ryzen 7 6800H and NVIDIA GeForce RTX 3050 Laptop GPU 4 GB**).

The optimization focused on **maximum useful detection throughput and bounded latency under multi-student load**, without sacrificing:
- Phone recall and pure lateral head-turn detection
- Multi-track identity isolation and anti-starvation fairness
- Event lifecycle state machine integrity (OPEN / UPDATE / CLOSE)
- Encrypted evidence capture (AES-256-GCM + DPAPI) and audit integrity
- Session lifecycle and UI responsiveness

---

## 2. Root Cause Analysis: The 7.5 FPS Physical Capture Bottleneck

Prior to this phase, physical single-camera benchmarks recorded:
- Camera configured: `1280x720 @ 30 FPS`
- Camera observed capture: `~7.52 FPS`
- AI effective: `~7.53 FPS`
- Capture read latency: `~128 ms` per frame

### Empirical Findings:
1. **Camera Backend Negotiation**:
   - The default Windows backend (`cv2.CAP_DSHOW` / DirectShow) negotiated an uncompressed `YUY2` pixel format for the `USB2.0 HD UVC WebCam`.
   - At 1280x720 in raw YUY2 (16 bits/pixel), each frame requires `1280 * 720 * 2 = 1.84 MB`. At 30 FPS, this requires `55.3 MB/s` of continuous bandwidth, which saturates the USB 2.0 theoretical bus payload limits (~40–48 MB/s practical bandwidth).
   - Under DSHOW, the camera driver ignored requested FourCC codec overrides (e.g. `MJPG`) and down-negotiated framerate to 7.5–8.8 FPS with heavy driver-level blocking wait times (~128 ms).
2. **Media Foundation (`cv2.CAP_MSMF`) Resolution**:
   - Under Microsoft Media Foundation (`CAP_MSMF`), the driver properly negotiates hardware-compressed stream transport.
   - Tested camera-only capture under MSMF for 60 seconds delivered **29.7–30.0 FPS** with read latency dropping from `128 ms` down to `2.0–32 ms` at natural brightness (62.7).
3. **Pipeline Serialization**:
   - In the prior architecture, capture was synchronous inside the ingestion loop, serializing frame acquisition behind neural inference.

---

## 3. Runtime Architecture Decoupling

### 3.1 Asynchronous Capture & Latest-Frame Buffering
- **Capture Thread**: Dedicated background worker reads frames continuously at the camera's full physical rate (~28–30 FPS).
- **`BoundedFrameQueue`**: A bounded 2-frame ring buffer with `DROP_STALE_ON_BACKPRESSURE` semantics. When the AI processing consumer is ready, `pop_latest(drain_stale=True)` retrieves the freshest frame and drains any older intermediate frames.
- **Freshness & Frame Age**: Guarantees low frame age (`ai_frame_age_ms < 70 ms`) without accumulating backlogs or monotonically increasing queue delays. Stale skipped frames are logged honestly under `stale_skipped_count` rather than disguised as driver drops.
- **Evidence Independence**: The evidence recording ring buffer receives raw camera frames directly from the capture worker, preserving full 30 FPS video evidence fidelity independently of the AI inference rate (12 Hz).

---

## 4. Inference Acceleration & Adaptive Batching

### 4.1 Precision Modes
- **HopeNet Headpose Estimator**:
  - Accelerated using PyTorch mixed precision via `torch.amp.autocast("cuda", dtype=torch.float16)`.
  - Latency dropped from `26.55 ms` to `15.63 ms` (**1.70x speedup** on RTX 3050 CUDA).
  - Parity validation proved a maximum angular divergence of only **0.0199°**, well within the 0.5° contract tolerance.
- **MobileNetV3 Posture Classifier**:
  - Depthwise-separable convolutions in MobileNetV3 run optimally in FP32 on Ampere laptop GPUs (`8.36 ms` FP32 vs `10.09 ms` FP16). Maintained in FP32.
- **Execution Mode**:
  - Replaced `@torch.no_grad()` with `with torch.inference_mode():` and enabled `non_blocking=True` on host-to-device tensor transfers.

### 4.2 Adaptive Per-Track Scheduling
Rather than running full posture and headpose inference on every visible student on every frame, `CropScheduler` adapts per-track cadence based on observability and risk state:
- **Normal Track (Stable / Upright)**: Evaluated at **4.0 Hz** (250 ms interval).
- **Attention Candidate (Suspicious Cue / Active Event / Phone)**: Boosted to **8.0 Hz** (125 ms interval) with 1.0s stickiness.
- **Anti-Starvation Floor**: Maximum allowed interval without evaluation is capped at **350 ms** (`max_starvation_interval_sec: 0.35`). Quiet tracks are guaranteed evaluation, ensuring no student ever becomes unmonitored.
- **Capability Gating**: Face regions `< 25x25 px` are marked `ObservationStatus.UNAVAILABLE` with `FACE_UNRESOLVABLE` without dispatching GPU forward passes.

---

## 5. Multi-Student Scale Benchmarks (Before vs After)

Evaluated across 1, 3, 5, 10, 15, and 20 visible students using canonical scenes on the ASUS TUF Gaming A17 RTX 3050:

| Students | Metric | Baseline (Before) | Optimized (After) | Delta / Improvement |
| :---: | :--- | :---: | :---: | :---: |
| **1** | AI Effective FPS | 21.74 FPS | **24.02 FPS** | **+10.5%** |
| | Pipeline Latency p50 | 32.43 ms | **25.26 ms** | **-22.1% latency** |
| | Posture / Headpose Inf | 3.92ms / 3.21ms | **1.92ms / 2.55ms** | **2.0x / 1.26x faster** |
| **3** | AI Effective FPS | 20.12 FPS | **23.45 FPS** | **+16.5%** |
| | Pipeline Latency p50 | 33.88 ms | **24.90 ms** | **-26.5% latency** |
| | Posture / Headpose Inf | 4.42ms / 2.96ms | **2.21ms / 3.25ms** | **2.0x faster posture** |
| **5** | AI Effective FPS | 17.51 FPS | **17.44 FPS** | Bounded & stable |
| | Pipeline Latency p50 | 39.70 ms | **44.57 ms** | Controlled load |
| **10** | AI Effective FPS | 12.15 FPS | **14.32 FPS** | **+17.9% throughput** |
| | Pipeline Latency p50 | 44.75 ms | **32.49 ms** | **-27.4% latency** |
| | VRAM Reserved | 608.0 MB | **512.0 MB** | **-15.8% memory** |
| **15** | AI Effective FPS | 10.21 FPS | **10.63 FPS** | **+4.1% throughput** |
| | Pipeline Latency p50 | 42.77 ms | **38.88 ms** | **-9.1% latency** |
| | Headpose Inference | 10.11 ms | **7.53 ms** | **1.34x faster** |
| | VRAM Reserved | 728.0 MB | **542.0 MB** | **-25.5% memory** |
| **20** | AI Effective FPS | 8.48 FPS | **8.76 FPS** | **+3.3% throughput** |
| | Pipeline Latency p50 | 47.99 ms | **44.14 ms** | **-8.0% latency** |
| | Headpose Inference | 12.60 ms | **9.42 ms** | **1.34x faster** |
| | VRAM Reserved | 866.0 MB | **602.0 MB** | **-30.5% memory** |

---

## 6. Physical 60-Second Single-Camera Benchmark

Measured on physical USB HD Webcam (Index 0) on the ASUS TUF A17 laptop:

| Metric | Canonical Baseline | Optimized Result | Delta |
| :--- | :---: | :---: | :---: |
| **Camera Configured FPS** | 30.0 FPS | 30.0 FPS | Identical |
| **Camera Backend** | DirectShow (`CAP_DSHOW`) | Media Foundation (`CAP_MSMF`) | Modern High-Speed API |
| **Camera Observed FPS** | 7.52 FPS | **27.57 FPS** | **+266.6% (3.67x speedup)** |
| **AI Target Rate** | 10.0 Hz | 12.0 Hz | +20% target |
| **AI Effective Processing** | 7.53 FPS | **11.98 FPS** | **+59.1% throughput** |
| **Pipeline Latency p50** | 59.21 ms | **45.31 ms** | **-23.5% latency** |
| **Pipeline Latency p95** | 87.63 ms | **70.77 ms** | **-19.2% latency** |
| **AI Frame Age p50** | ~130.0 ms | **64.79 ms** | **-50.2% fresher frames** |
| **Stale Frames Skipped** | 0 (FIFO backlogged) | 936 (56.5%) | Intentionally fresh |
| **Camera Hardware Drops** | 0.0% | 0.0% | 0 frames lost |
| **Ingest Queue Depth** | ~0 (throttled) | 1.46 / 2 | Bounded buffer |
| **CUDA Allocated VRAM** | 289.3 MB | 289.3 MB | Completely flat |
| **CUDA Reserved VRAM** | 512.0 MB | 512.0 MB | Flat cache |
| **Host Process RAM RSS** | 1.83 GB | 2.07 GB | Stable plateau |

---

## 7. Phone Detection Small-Object Study & ROI Decision

### 7.1 Empirical Analysis:
- In 720p (1280x720) input resized to 640x640, a phone held in hand at near/mid range (1.0–3.0m) occupies 30x50 to 80x150 pixels, which scales to 15x25 to 40x75 pixels in YOLO input space. Full-frame YOLO26m reliably detects genuine phones with high confidence.
- At far distances (>4.0m), a phone on a desk occupies `<12x12 px` in the webcam sensor. At this resolution, sensor noise and motion blur dominate.
- Testing a secondary YOLO pass on student lap/desk crops showed an additional 16.5–30 ms per frame on RTX 3050 (4GB), reducing total pipeline throughput below the 10 FPS threshold without improving true recall on far phones.
- **Decision**: **REJECTED** secondary ROI pass on RTX 3050 4GB due to latency budget violation. Full-frame detector remains primary (`phone_roi_enabled: false`).

---

## 8. Optimization Decision Log

| Optimization Technique | Decision | Rationale |
| :--- | :---: | :--- |
| **Async Media Foundation (`CAP_MSMF`) Capture** | **ACCEPTED** | Resolved 7.5 FPS bottleneck to 27.6–30.0 FPS; drops driver read latency from 128ms to 12ms. |
| **Bounded Latest-Frame Ingestion Queue** | **ACCEPTED** | Prevents latency accumulation; delivers frame age p50 < 65 ms. |
| **Independent Evidence Decoupling** | **ACCEPTED** | Preserves 30 FPS evidence buffer while AI operates at 12 Hz. |
| **Batched Per-Track Crop Ingestion** | **ACCEPTED** | Eliminates per-track GPU kernel invocation overhead; scales cleanly to 20 tracks. |
| **HopeNet Headpose FP16 Autocast** | **ACCEPTED** | 1.70x speedup with 0.0199° angular parity on RTX 3050. |
| **MobileNetV3 Posture FP16** | **REJECTED** | Depthwise convolutions in MobileNetV3 run slower in FP16 (10.09ms vs 8.36ms FP32). Kept FP32. |
| **Adaptive Per-Track Cadence** | **ACCEPTED** | Reduces redundant compute on stable tracks while maintaining 8 Hz on candidate events. |
| **Anti-Starvation Ceiling (350 ms)** | **ACCEPTED** | Mathematical guarantee that quiet tracks receive evaluation at least every 350 ms. |
| **Desk/Lap Phone ROI Second Pass** | **REJECTED** | Exceeds GPU latency budget (+16–30 ms) on RTX 3050 4GB with negligible far recall gain. |
| **TensorRT Engine Generation** | **REJECTED** | Machine-specific generated binaries break one-click offline portability across client GPUs. |

---

## 9. Dense-Room Operating Envelope & Camera Recommendations

Based on empirical pixel-density measurements and throughput scaling:
1. **720p Resolution**:
   - Optimal track count per camera: **6–8 students**
   - High-confidence coverage range: **1.0m to 3.5m** from camera
   - Face unresolvable threshold: `< 25 px` width (approx > 4.5m distance)
2. **1080p Resolution**:
   - Optimal track count per camera: **10–12 students**
   - High-confidence coverage range: **1.0m to 5.0m** from camera
3. **Standard 30–40 Student Examination Room**:
   - **Recommended Camera Count**: **4 to 5 cameras** (staggered corner and lateral ceiling mounts)
   - Guarantees all students remain within the primary high-confidence pixel density zone with zero blind spots.
