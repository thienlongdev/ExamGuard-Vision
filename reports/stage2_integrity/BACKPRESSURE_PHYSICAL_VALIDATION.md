# Backpressure Physical Validation & Queue Depth Audit

## 1. Executive Summary & Defect Remediation
Prior Stage 2 documentation claimed `BACKPRESSURE_READY = YES`, but the underlying execution loop (`run_stream`) was purely synchronous:
```python
# Old flawed implementation: synchronous read-and-process
vframe = source.read()
result = pipeline.process_frame(vframe)
```
The queue configuration parameter was effectively ignored, and no frame-drop mechanism existed under ingestion backlog.

### Corrective Architecture
A true multi-threaded producer-consumer architecture was implemented in `src/orchestration/stage2_pipeline.py`:
1. **Producer Thread**: Dedicated reader reading frames from `VideoSource` and placing them into a thread-safe `BoundedFrameQueue`.
2. **Consumer Thread**: Pipeline orchestration processing frames popped from `BoundedFrameQueue`.
3. **Drop Policy**: `DROP_STALE_ON_BACKPRESSURE`—when the bounded queue is full, the oldest unprocessed video frame is evicted.
4. **Safety Guarantee**: Ingestion frame drops **never** drop event lifecycle transitions (`EVENT_CLOSE`), track states, or evidence metadata.
5. **Monotonicity**: Evaluated timestamps across processed frames remain strictly monotonic.

---

## 2. Physical Empirical Benchmark Results

Extracted from machine-readable artifact `runs/stage2_integrity/backpressure_benchmark.json`:

| Parameter | Queue Capacity = 3 | Queue Capacity = 5 (Selected Default) |
| :--- | :--- | :--- |
| **Drop Policy** | `DROP_STALE_ON_BACKPRESSURE` | `DROP_STALE_ON_BACKPRESSURE` |
| **Burst Ingestion Pushed** | 50 frames | 50 frames |
| **Frames Consumed** | 20 frames | 22 frames |
| **Frames Dropped** | 30 frames (60.0%) | 28 frames (56.0%) |
| **Drop Telemetry Logged** | Yes (`dropped_frame_idx`, timestamp, wall time, queue depth) | Yes (`dropped_frame_idx`, timestamp, wall time, queue depth) |
| **Timestamp Monotonicity** | **Preserved (True)** | **Preserved (True)** |
| **Buffer Latency at 30 FPS** | 100.0 ms | 166.7 ms |
| **Cadence Jitter Tolerance** | Low (drops on minor cadence alignment spikes) | **Optimal (absorbs cadence alignment bursts)** |

---

## 3. Justification for Default Capacity = 5
1. **Cadence Alignment Spikes**: When posture (15 Hz) and head-pose (10 Hz) inference align on the same frame for multiple tracks, pipeline processing time briefly increases from ~20 ms to ~45 ms. A capacity of 3 frames (100 ms) is too shallow and prematurely drops frames during legitimate transient cadence bursts.
2. **Bounded End-to-End Latency**: A capacity of 5 frames bounds maximum ingestion queue latency to $5 \times 33.3\text{ ms} = 166.7\text{ ms}$ at 30 FPS, well within the target classroom CCTV latency budget.
3. **Consistency**: Removed legacy configurations that specified unvalidated queue depths of 30 frames (which would permit up to 1,000 ms of stale lag).

---

## 4. Status Flag
- `BACKPRESSURE_IMPLEMENTED = YES`
