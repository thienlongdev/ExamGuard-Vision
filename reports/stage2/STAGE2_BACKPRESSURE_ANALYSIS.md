# Stage 2 Backpressure & Frame Drop Analysis

## 1. Executive Summary & Design Principle
In live physical video processing (RTSP streams, webcams, classroom CCTV), frame arrival is dictated by the camera sensor clock, while inference execution time varies due to cadence alignment, batch size, and scene complexity.
If execution falls behind the arrival rate, buffering unconsumed frames leads to unbounded latency accumulation (e.g., an event occurring at $T=10.0\text{ s}$ would only be processed at $T=25.0\text{ s}$).

**Core Rule**:
> Prefer dropping stale video frames rather than building unbounded processing latency.
> Never drop event lifecycle states, critical CLOSE transitions, or evidence metadata.

---

## 2. Queue Configuration & Boundaries
All internal queues within the Stage 2 orchestration pipeline are strictly bounded:

| Queue / Buffer | Max Capacity | Overflow Policy | Justification |
| :--- | :--- | :--- | :--- |
| **Input Frame Queue** (`input_queue_max_size`) | `3` frames | Drop oldest unconsumed frame (`popleft`) | Bounds input lag to $\le 100\text{ ms}$ at 30 FPS. |
| **Track Observation Buffer** (`max_buffer_sec`) | `10.0` seconds | Evict observations older than $t - 10.0\text{ s}$ | Prevents memory growth while preserving temporal debounce history. |
| **Evidence Snapshot Buffer** (`max_snapshots_per_event`) | `3` images | Reject additional snapshots beyond limit | Prevents disk filling and uncontrolled I/O latency. |
| **Rolling Video Clip Buffer** (`clip_buffer_seconds`) | `10.0` seconds | Evict oldest frames beyond sliding window | Bounded memory footprint per active event candidate. |

---

## 3. Timestamp-Driven Correctness Under Frame Drops
Traditional computer vision pipelines compute temporal durations using frame counts divided by a nominal frame rate ($N / \text{FPS}$). Under frame drops or irregular stream cadences, this assumption collapses.

Stage 2 enforces **monotonic timestamp-first processing**:
1. **Source Timestamps**: Every decoded frame receives a strictly monotonic `timestamp_sec`.
2. **Debounce Invariance**: The V4D Temporal Buffer computes elapsed duration as:
   $$\Delta t = t_{\text{current}} - t_{\text{first\_candidate}}$$
3. **Drop Robustness Test Validation**:
   - In `tests/test_stage2_backpressure.py::test_backpressure_drop_preserves_timestamp_continuity`, a simulated 30 FPS stream experiencing 50% bursty frame drops maintained exact temporal event triggering:
     - 45 frames submitted across $1.5\text{ s}$ real-time timeline.
     - Frame drop rate: `66.7%` under deliberate processing delays.
     - Result: `SUSTAINED_HEAD_REST` event successfully transitioned to `OPEN` at $t = 1.50\text{ s}$ without temporal corruption.

---

## 4. Benchmark Drop Rate Under Physical Classroom Streams

In `runs/stage2/integrated_runtime_benchmark.json`:
- **Mode A (Single Student Physical, 24.79 FPS)**:
  - Input Rate: 25.0 FPS
  - Processed Rate: 24.79 FPS
  - Dropped Frames: 0 (pipeline operates faster than video presentation rate).
- **Mode B (Medium Classroom Physical, 29.59 FPS)**:
  - Input Rate: 30.0 FPS
  - Processed Rate: 29.59 FPS
  - Dropped Frames: 0 across 150 test frames.
- **Mode C (Synthetic 20-Student Workload Scaling)**:
  - Peak Cadence Alignment Latency: $57.50\text{ ms}$ (at aligned 10 Hz / 6 Hz ticks).
  - Effective Processing Rate: 24.33 FPS.
  - Frame Drop Handling: Safe frame skip ensures real-time lock without unbounded queue buildup.

---

## 5. Verification Status
- **Backpressure Unit Tests**: Passed (`tests/test_stage2_backpressure.py`).
- **Timestamp Flow Tests**: Passed (`tests/test_stage2_timestamp_flow.py`).
- **BACKPRESSURE_READY**: **YES**.
