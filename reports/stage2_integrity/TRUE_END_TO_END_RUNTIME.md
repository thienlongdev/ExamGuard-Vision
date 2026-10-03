# True End-to-End Frame Loop Runtime & Latency Audit

## 1. Canonical Definition of End-to-End Latency
Prior documentation claimed an "end-to-end latency of ~23 ms". Forensic code audit revealed that this measurement:
- Commenced **after** video decoding had already finished.
- Excluded frame acquisition, image decompression, and buffer movement from the camera/source.
- Assumed a synthetic serialization constant (0.05 ms).

### Corrective Measurement Standard
In accordance with Section 24, the canonical end-to-end frame loop latency is defined as:
$$t_0 = \text{immediately BEFORE } \texttt{source.read()}$$
$$\text{Frame decode} \longrightarrow \text{Detection} \longrightarrow \text{Tracking} \longrightarrow \text{Crop Scheduling} \longrightarrow \text{Inference} \longrightarrow \text{Fusion} \longrightarrow \text{Event State} \longrightarrow \text{Risk} \longrightarrow \text{Serialization}$$
$$t_1 = \text{AFTER frame result is ready for downstream WebSocket / API publication}$$
$$\text{whole\_loop\_end\_to\_end\_ms} = t_1 - t_0$$

Only this comprehensive loop latency may be legitimately designated `END_TO_END_FRAME_LOOP_LATENCY`.

---

## 2. Integrated Physical Results Across Workload Modes

Extracted from machine-readable artifact `runs/stage2_integrity/whole_loop_runtime.json`:

| Workload Mode | Active Tracks | Effective FPS | Whole-Loop Mean (ms) | Whole-Loop Median (ms) | Whole-Loop P95 (ms) | Post-Decode Mean (ms) | Frame Decode Mean (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mode A: Single Student Physical** | 1.0 | 24.14 | 41.35 | 40.33 | 71.74 | 26.28 | 15.01 |
| **Mode B: Medium Classroom Physical** | 1.0–2.0 | 28.36 | 35.19 | 33.81 | 60.75 | 22.94 | 12.20 |
| **Mode C1: Dense Full Pipeline Tiled** | 25.0 | 20.19 | 49.41 | 49.47 | 75.94 | 39.39 | 5.21 |
| **Mode C2: Controlled 20-Track Load** | 20.0 | 24.66 | 40.47 | 40.49 | 68.96 | 7.97 | 32.44 |

---

## 3. Analysis & Operational Verdicts

### Mode A & B Operational Line-Rate
- Mode A operates at **24.14 FPS** on a 25.0 FPS physical clip (96.6% line rate).
- Mode B operates at **28.36 FPS** on a 30.0 FPS physical clip (94.5% line rate).
- Decode time contributes $12.2\text{ ms} - 15.0\text{ ms}$ of wall-clock overhead per frame. The pure neural + algorithmic pipeline executes in $22.9\text{ ms} - 26.3\text{ ms}$.

### Mode C1 Dense Full Pipeline Behavior
- On a 1280x720 canvas with 20 tiled students, the general detector found an average of 27 candidate boxes, resulting in **25.0 active ByteTrack tracks**.
- Post-decode pipeline processing averaged **39.39 ms**, with whole-loop averaging **49.41 ms** (20.19 FPS).
- Because 20.19 FPS < 30.0 FPS and P95 latency is 75.94 ms, the pipeline **cannot** sustain a 30 FPS line rate without dropping frames under full 25-track simultaneous detector load.
- Under the implemented `DROP_STALE_ON_BACKPRESSURE` policy with queue depth 5, the pipeline safely drops oldest frames during burst spikes without corrupting event states.

### Mode C2 Downstream Scaling
- When isolating downstream branches (crop extraction, posture/headpose batching, fusion, event, risk, serialization) with an exact 20-track workload, downstream processing executes in an average of **7.97 ms** (median 1.07 ms, P95 29.61 ms).
- This conclusively proves that the downstream multi-cue fusion and event engine scales to 20 tracks well within a 30 FPS (33.3 ms) budget. The primary latency bottleneck under dense load is the full-frame high-resolution object detector.

---

## 4. Status Flags
- `WHOLE_LOOP_END_TO_END_MEASURED = YES`
- `MODE_A_RUNTIME_ACCEPTABLE = YES`
- `MODE_B_RUNTIME_ACCEPTABLE = YES`
- `DENSE_30FPS_LINE_RATE = NO`
- `DENSE_25FPS_LINE_RATE = NO`
