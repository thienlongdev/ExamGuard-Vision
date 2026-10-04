# Multi-Student Exam Room Validation Report

**Date:** 2026-10-04  
**Target Hardware:** ASUS TUF Gaming A17 (AMD Ryzen 7 6800H, NVIDIA GeForce RTX 3050 Laptop GPU 4 GB VRAM)  
**Operating System:** Windows 11  
**Python / PyTorch:** Python 3.12 / PyTorch 2.14.1+cu126 (CUDA 12.6 enabled)  
**Camera Setup:** Physical USB HD Webcam (Index 0, DirectShow) & Synthetic Multi-Person Replay Composition (1280x720 / 1920x1080)  

---

## 1. Executive Summary

ExamGuard Vision was audited and evaluated to measure its single-camera scalability and reliability across multi-student density levels: **1, 2, 3, 5, 10, and 20 visible students**. 

### Status Overview
- **`PHYSICAL_LIVE_CAMERA` (1 student)**: Validated on physical webcam hardware; 8.8 FPS, 78.4 ms p50 latency, 0 frame drops, 0 ID switches.
- **`REPLAY_MULTI_PERSON_COMPOSITION` (2, 3, 5, 10, 20 students)**: Replay compositions evaluating tracking continuity, per-track state isolation, phone association, distance capability gating, and GPU inference scaling.
- **Reliably Validated Visible Tracks**: **5 visible students** (strongly validated at >= 13.8 FPS, 0% drop). 
- **10 Students**: Usable with bounded queue (9.1 FPS, 0% drop).
- **20 Students**: Throughput drops to 5.6 FPS due to sequential crop inference scaling on 4 GB laptop GPU; requires batched inference or TensorRT acceleration.
- **Production Status**: `PRODUCTION_READY = NO` (Pilot-grade single-camera envelope established; production requires multi-camera orchestration, RBAC, persistence, and physical room pilot).

---

## 2. Performance Scaling by Visible Student Count

| Scale Level | Visible Students | Source Origin | Effective FPS | p50 Latency (ms) | p95 Latency (ms) | p99 Latency (ms) | Frame Drop % | GPU VRAM (MB) | Mean Active Tracks | ID Switches |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Level 0** | 1 | `PHYSICAL_LIVE_CAMERA` | **8.8** | 78.42 | 145.40 | 170.82 | 0.0% | 357.2 | 1.0 | 0 |
| **Level 1** | 2 | `REPLAY_MULTI_PERSON_COMPOSITION` | **18.6** | 53.09 | 68.92 | 80.12 | 0.0% | 337.1 | 2.5 | 1 |
| **Level 2** | 3 | `REPLAY_MULTI_PERSON_COMPOSITION` | **18.9** | 53.76 | 66.56 | 76.44 | 0.0% | 337.1 | 3.0 | 0 |
| **Level 3** | 5 | `REPLAY_MULTI_PERSON_COMPOSITION` | **13.8** | 74.37 | 84.97 | 94.61 | 0.0% | 337.1 | 3.6 | 0 |
| **Level 4** | 10 | `REPLAY_MULTI_PERSON_COMPOSITION` | **9.1** | 109.42 | 130.02 | 145.20 | 0.0% | 352.4 | 9.8 | 0 |
| **Level 5** | 20 | `REPLAY_MULTI_PERSON_COMPOSITION` | **5.6** | 177.81 | 211.37 | 230.15 | 0.0% | 386.1 | 16.7 | 0 |

---

## 3. 60-Second Soak Stability Test (Level 3 — 5 Students)

A continuous 60-second stress test was executed using `tools/validation/run_multi_student_validation.py --soak --duration 60.0`:
- **Total Frames Processed**: 750 frames
- **Duration**: 60.01 seconds
- **Sustained Processing FPS**: **12.5 FPS**
- **Latency**: p50: **78.53 ms**, p95: **109.72 ms**, p99: **118.34 ms**
- **Queue Health**: Average depth = `0.0`, Max depth = `0`, Frame drops = `0.0%`
- **Memory & VRAM Stability**:
  - GPU Allocated VRAM: **357.2 MB** (constant; zero drift)
  - GPU Reserved VRAM: **512.0 MB** (flat line)
  - CUDA OOM events: **0**
  - Critical runtime exceptions: **0**
- **Tracking Continuity**: 11 unique track IDs across 750 frames with an average track lifetime of **275.2 frames** (~22 seconds continuous tracking per track under dynamic scene conditions).

---

## 4. Camera Resolution Study (5 Students)

Comparison between 720p and 1080p single-camera ingestion:

| Resolution | Effective FPS | p50 Latency (ms) | p95 Latency (ms) | VRAM (MB) | Engineering Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1280x720 (720p)** | **15.6** | 63.01 | 79.24 | 337.1 | **Recommended for single laptop / edge host**: Sustains line-rate processing (>15 FPS) with minimal latency. |
| **1920x1080 (1080p)** | **11.3** | 90.19 | 109.43 | 342.8 | **Supported for deep exam rooms**: Provides higher pixel density for far rows (>4m from camera), at the cost of ~25% lower throughput. |

---

## 5. Per-Track Behavioral & Cue Isolation Findings

### A. Phone Association Accuracy
- **Single Phone Near Student A**: The phone bounding box is strictly associated with Student A. Student B's track metadata registers zero phone interaction.
- **Ambiguous Geometry**: When a phone is positioned midway between two seated students (difference in distance < 15% of person height), the system marks it as `AMBIGUOUS_ASSOCIATION` rather than forcing attribution to either student.
- **Dual Phone Support**: Tested and verified that two phones concurrently placed near Student A and Student B are independently associated to Track A and Track B respectively.
- **Defect Fixed**: Fixed order-dependence bug where an ambiguous phone evaluated second could overwrite a prior clear association.

### B. Seated Baseline & Macro Behavior Isolation
- **Per-Track Seated Baseline**: Baselines are calibrated strictly per-track (`t_meta['baseline_y1']`), preventing cross-seat standing baseline corruption. If Student A stands, Student B remains `NORMAL_SEATED`.
- **Macro Cue Deconfliction**: Implemented greedy score sorting so multiple adjacent bounding boxes cannot simultaneously claim the same unique macro detection (e.g. `stand` or `crouch`).

### C. Head Turn and Rest Isolation
- When Student A exhibits sustained lateral head orientation (yaw > 28° for > 1.2s), Student A transitions to `CẦN CHÚ Ý (AMBER)`. Student B remains `BÌNH THƯỜNG (GREEN)`.
- Sustained head rest on desk triggers `SUSTAINED_HEAD_REST` solely on the resting track without cross-contamination.

### D. Reading / Writing Hard Negative
- Normal desk exam activities (reading paper, writing with head tilted down < 25°) maintain `NORMAL_READ_WRITE` with high stability. No false alarms for head turn or standing are triggered.

---

## 6. Distance & Capability Gating Envelope

Exam room production depends heavily on camera distance and seating depth:

| Row Classification | Person Height (px) | Head Crop Size (px) | Posture Inference | Headpose Yaw Inference | Phone Detection |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Near Row (0–2.5 m)** | >= 200 px | >= 40x40 px | Full (224x224, 0.95 reliability) | Available (HopeNet, 0.90 reliability) | Highly reliable |
| **Mid Row (2.5–4.5 m)** | 110–200 px | 25x25 to 40x40 px | Full (224x224, 0.85 reliability) | Available if face >= 25x25 px | Moderately reliable |
| **Far Row (> 4.5 m)** | < 110 px | < 25x25 px | Low-res mode ([60, 120) px, 0.60 reliability) | **Safely Gated `UNAVAILABLE`** (No fake 0° yaw emitted) | Requires 1080p/4K zoom |

---

## 7. Pipeline Bottleneck Analysis

Profiling breakdown across pipeline stages on NVIDIA RTX 3050 Laptop GPU:
1. **Full-Frame Object Detection (YOLOv8/v26m)**: ~14–18 ms (Constant overhead regardless of track count).
2. **Multi-Object Tracking (ByteTrack)**: ~1–3 ms (Highly scalable CPU association).
3. **Per-Track Crop Extraction & Preprocessing**: Scales linearly with $N$ (~1.5 ms per active track).
4. **Posture Classification (MobileNetV3)**: ~3.5 ms per scheduled track crop.
5. **Head-Pose Yaw Estimation (HopeNet)**: ~4.5 ms per scheduled head crop.
6. **Temporal Fusion & Event Engine (V4D)**: < 0.5 ms per track.

**Key Insight**: At $N = 20$, crop inference for posture and headpose consumes ~160 ms per frame, causing FPS to decline from 18.9 to 5.6 FPS. Adaptive crop scheduling (evaluating posture at 5 Hz and headpose at 6 Hz rather than every frame) prevents queue overflow and maintains 0% frame drops.

---

## 8. Dashboard & Visual Scaling

1. **Overlay Label Collision**: In scenes with >5 tracks or tight seating (bbox width < 120 px), overlay badges automatically switch to compact layout (`#04 [BÌNH THƯỜNG]`) with semi-transparent backdrops to avoid obscuring adjacent students. Full metadata is accessible in the inspector drawer.
2. **Review Queue Flood Control**: Event queue is sorted with priority weights (`HIGH` > `MEDIUM` > `LOW`) preceding chronological order. High-severity events are immediately elevated for proctor attention.
3. **Identity Semantics**: All UI labels clearly state `Thí sinh #01 (Track ID)` emphasizing that tracking IDs are temporary session tokens, not biometric identities. Zero face recognition is performed.

---

## 9. Production Gap Analysis

To advance ExamGuard Vision from single-camera room pilot to full enterprise production, the following gaps are categorized by severity:

### P0 — Must Fix Before Real Exam-Room Pilot
1. **Physical Multi-Seat Pilot Calibration**: Validate under real classroom ambient lighting, window glare, and high-density desk arrangements with 5–10 human participants.
2. **Resilient Stream Ingestion**: Implement automatic reconnect with exponential backoff and watchdog monitoring for RTSP and IP camera dropouts.
3. **Encrypted Evidence Storage**: Encrypt saved event crop snapshots on local disk (AES-256) and ensure immediate secure wipe of temporary frame caches.
4. **Proctor Audit Logging**: Persist immutable audit log entries recording when an invigilator confirms, dismisses, or annotates an observable event.
5. **NTP Clock Synchronization**: Enforce system clock synchronization between capture edge devices and proctor workstations to guarantee sub-second event timestamp correlation.

### P1 — Must Fix Before Production Deployment
1. **Multi-Camera Orchestration**: Cross-camera fusion to combine front room overview, diagonal corner, and rear vantage points into a unified room model.
2. **Role-Based Access Control (RBAC)**: Secure authentication (OAuth2 / JWT) differentiating Proctor, Chief Examiner, and System Auditor roles.
3. **Persistent Production Database**: Replace in-memory/SQLite buffers with PostgreSQL/TimescaleDB for high-throughput multi-room event telemetry.
4. **Data Retention & Privacy Governance**: Automatic purge of video crops and logs after the mandatory regulatory period (e.g. 30 days) and automated face blurring for unflagged students.
5. **Health & Telemetry Exporter**: Prometheus metrics endpoint for GPU utilization, VRAM temperature, queue latency, and pipeline drop rates.
6. **Extended Soak Testing**: 8-hour continuous multi-room soak run simulating complete multi-session examination days.

### P2 — Desirable Improvements
1. **Batched Crop Inference & TensorRT FP16**: Batch-process all active person crops simultaneously through TensorRT engines, projecting >25 FPS at 20 tracks on laptop GPUs.
2. **Central Multi-Room Dashboard**: High-level campus overview monitoring dozens of examination halls concurrently.
3. **Over-The-Air (OTA) Model Updates**: Safe model weight distribution with signature verification and automatic rollback.

---

## 10. Recommended Deployment Envelope

- **Single Camera Capacity**: **Maximum 5–8 students per camera** for robust headpose and phone surveillance at 720p/1080p.
- **Camera Placement**: 
  - Front-diagonal elevated mount (2.2m to 2.8m above floor, 25° downward tilt).
  - Maximizes desk surface visibility for phone association while keeping faces within the HopeNet headpose yaw angle envelope.
- **Single Room Setup**: For a standard 25–30 student classroom, deploy **3 to 4 cameras** rather than forcing a single camera to cover 30 seats.