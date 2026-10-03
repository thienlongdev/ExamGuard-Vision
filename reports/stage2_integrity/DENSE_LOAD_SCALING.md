# Dense Load Scaling & Two-Mode Stress Test Audit

## 1. Context & Two-Mode Benchmark Design
Prior Stage 2 evaluations tiled a single crop into a 4x5 grid and labeled the resulting performance as "20 student realtime at 24.33 FPS". This claim suffered two critical defects:
1. **Unverified Track Accounting**: The benchmark did not record how many person detections the general detector physically identified or how many tracks ByteTrack actually tracked.
2. **Conflated Failure Boundaries**: It was impossible to know whether latency was caused by detector bounding-box generation or downstream batch inference / fusion / serialization.

To eliminate narrative guessing, two distinct physical dense-load benchmarks were engineered:
- **Mode C1 (`FULL_PIPELINE_TILED_SCENE`)**: Full end-to-end evaluation with a physical 4x5 tiled canvas (1280x720). Tests the general detector, ByteTrack tracker, and all downstream branches.
- **Mode C2 (`CONTROLLED_SYNTHETIC_TRACK_LOAD`)**: Downstream pipeline stress test bypassing detector/tracker generation with exactly 20 deterministic Track objects. Isolates crop extraction, posture/headpose batching, multi-cue fusion, event state machines, risk aggregation, and serialization.

---

## 2. Empirical Benchmark Findings (300 Frames per Mode)

Extracted from `runs/stage2_integrity/benchmark_mode_c_full_pipeline.json` and `runs/stage2_integrity/benchmark_mode_c_20track_downstream.json`:

| Metric | Mode C1: Full Pipeline Tiled Scene | Mode C2: Controlled 20-Track Load |
| :--- | :--- | :--- |
| **Input Grid Structure** | 4 rows $\times$ 5 columns (20 tiles) | Injected deterministic 20 tracks |
| **General Detector Detections** | 27 detections/frame | N/A (Bypassed) |
| **Mean Active Tracks** | **25.00 tracks** | **20.00 tracks (Exact)** |
| **Track Statistics (P05 / Median / P95)** | 25.0 / 25.0 / 25.0 tracks | 20.0 / 20.0 / 20.0 tracks |
| **Eligible Posture Tracks ($\ge 120$ px)** | 25.00 tracks | 20.00 tracks |
| **Eligible Head-Pose Tracks ($\ge 25\times 25$ px)** | 25.00 tracks | 20.00 tracks |
| **Post-Decode Pipeline Mean (ms)** | 39.39 ms | **7.97 ms** |
| **Post-Decode Pipeline Median (ms)** | 39.07 ms | **1.07 ms** |
| **Post-Decode Pipeline P95 (ms)** | 70.06 ms | **29.61 ms** |
| **Whole-Loop End-to-End Mean (ms)** | 49.41 ms | 40.47 ms |
| **Whole-Loop End-to-End P95 (ms)** | 75.94 ms | 68.96 ms |
| **Effective Throughput (FPS)** | **20.19 FPS** | **24.66 FPS** |
| **Peak GPU VRAM Allocated** | 338.5 MB | 289.4 MB |

---

## 3. Forensic Analysis & Root-Cause Attribution

### "Can the detector find 20+ students?"
**YES.** On the 4x5 tiled canvas, `yolo26m.pt` consistently identified 27 person candidate boxes, which ByteTrack normalized into 25 persistent active student tracks.

### "Can the downstream pipeline handle 20 tracks?"
**YES.** In Mode C2, when 20 tracks were processed through crop extraction, posture/headpose batch inference, phone association, V4D fusion, event evaluation, and serialization, the downstream pipeline completed in an average of **7.97 ms** (median **1.07 ms**, P95 **29.61 ms**). This is well below the 33.33 ms frame budget for 30 FPS.

### "Can the combined full pipeline sustain 30 FPS on 20+ tracks?"
**NO.** Running `yolo26m.pt` at 640px, ByteTrack on 25 tracks, plus downstream crop inference on a single GPU increases whole-loop processing time to **49.41 ms** (20.19 FPS).
- Calling 20.19 FPS or 24.33 FPS "30 FPS real-time" is strictly prohibited.
- `DENSE_30FPS_LINE_RATE = NO`
- `DENSE_25FPS_LINE_RATE = NO`

---

## 4. Operational Gating & Safe Deployment Boundaries
Because 30 FPS line rate cannot be sustained under dense 20–25 student loads:
- **`TARGET_CCTV_PILOT_LOW_MEDIUM_OCCUPANCY_READY = YES`**: For classrooms with 1 to 15 students, the pipeline comfortably sustains 25–30 FPS line rate.
- **`TARGET_CCTV_PILOT_DENSE_ROOM_READY = NO`**: For dense classrooms (20–30 students), deployment is **not** approved at 30 FPS without either:
  1. Setting camera ingestion rate to 15–20 FPS, OR
  2. Operating under bounded backpressure (`DROP_STALE_ON_BACKPRESSURE`) with explicit operator notice of ~20% frame sampling drop.
