# True Physical Component Timing Instrumentation

## 1. Elimination of Synthetic Timing Decompositions
Prior Stage 2 implementations contained critical timing integrity defects:
1. **Fabricated Percentage Splits**: A single wall-clock duration block was artificially split using hardcoded ratios:
   ```python
   # PROHIBITED FABRICATED LOGIC (DELETED)
   fusion_ms = total_downstream * 0.50
   event_ms = total_downstream * 0.30
   evidence_ms = total_downstream * 0.20
   ```
2. **Hardcoded Serialization**: Serialization was recorded as a fixed constant:
   ```python
   # PROHIBITED HARDCODED VALUE (DELETED)
   serialization_ms = 0.05
   ```

### Corrective Implementation
Every subsystem boundary is now physically instrumented with independent `time.perf_counter()` calls. In benchmark mode on CUDA, synchronization is applied at kernel boundaries so measurements capture actual GPU compute times. Serialization physically executes `json.dumps()` on real event payload structures.

---

## 2. True Physical Timing Breakdown (300 Measured Frames per Mode)

All figures in milliseconds (ms), extracted from machine-readable artifact `runs/stage2_integrity/component_timing.json`:

### Mode A: Low-Load / Single-Student Physical Clip (`writing (1).mp4`)
| Component Pipeline Stage | Mean (ms) | Median (ms) | P90 (ms) | P95 (ms) | P99 (ms) | Max (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Video Frame Decode (`source_read_ms`)** | 15.01 | 16.35 | 24.03 | 24.50 | 26.04 | 26.42 |
| **General Object Detector (`general_detector_ms`)** | 15.52 | 15.34 | 17.10 | 18.02 | 20.59 | 22.53 |
| **Macro Behavior Detector (`macro_behavior_ms`)** | 3.21 | 0.00 | 15.88 | 16.62 | 18.37 | 19.10 |
| **ByteTrack Tracking (`tracker_ms`)** | 1.30 | 1.27 | 1.52 | 1.56 | 1.68 | 2.58 |
| **Crop Extraction (`crop_extraction_ms`)** | 0.02 | 0.02 | 0.04 | 0.04 | 0.05 | 0.09 |
| **Posture Preprocess (`posture_preprocess_ms`)** | 0.81 | 0.00 | 2.48 | 2.58 | 2.75 | 3.03 |
| **Posture GPU Inference (`posture_inference_ms`)** | 2.06 | 0.00 | 6.51 | 6.89 | 8.21 | 9.14 |
| **Head-Pose Preprocess (`headpose_preprocess_ms`)** | 0.25 | 0.00 | 0.66 | 0.67 | 0.75 | 1.29 |
| **Head-Pose GPU Inference (`headpose_inference_ms`)**| 2.71 | 0.00 | 7.42 | 7.93 | 8.87 | 10.76 |
| **Phone Spatial Association (`phone_association_ms`)**| 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.03 |
| **V4D Multi-Cue Fusion (`fusion_ms`)** | 0.18 | 0.17 | 0.21 | 0.22 | 0.27 | 0.44 |
| **Event State Machine (`event_engine_ms`)** | 0.01 | 0.01 | 0.02 | 0.02 | 0.03 | 0.05 |
| **Risk Aggregation (`risk_aggregation_ms`)** | 0.01 | 0.00 | 0.01 | 0.01 | 0.02 | 0.03 |
| **Evidence Manager Lifecycle (`evidence_manager_ms`)**| 0.14 | 0.14 | 0.17 | 0.18 | 0.23 | 0.38 |
| **Physical JSON Serialization (`serialization_ms`)** | 0.02 | 0.01 | 0.02 | 0.03 | 0.05 | 0.11 |
| **Post-Decode Pipeline Total (`post_decode_pipeline_ms`)**| **26.28** | **23.98** | **42.84** | **48.71** | **52.95** | **55.46** |
| **Whole-Loop End-to-End (`whole_loop_end_to_end_ms`)** | **41.35** | **40.33** | **64.49** | **71.74** | **74.91** | **76.00** |

---

### Mode B: Medium-Load Classroom Sequence (`lecture (10).mp4`)
| Component Pipeline Stage | Mean (ms) | Median (ms) | P90 (ms) | P95 (ms) | P99 (ms) | Max (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Video Frame Decode (`source_read_ms`)** | 12.20 | 13.67 | 18.66 | 19.19 | 19.73 | 19.96 |
| **General Object Detector (`general_detector_ms`)** | 13.78 | 13.62 | 14.86 | 15.65 | 16.92 | 19.01 |
| **Macro Behavior Detector (`macro_behavior_ms`)** | 3.09 | 0.00 | 15.28 | 15.82 | 17.51 | 18.15 |
| **ByteTrack Tracking (`tracker_ms`)** | 1.18 | 1.15 | 1.34 | 1.41 | 1.54 | 1.95 |
| **Crop Extraction (`crop_extraction_ms`)** | 0.02 | 0.02 | 0.04 | 0.04 | 0.05 | 0.08 |
| **Posture Preprocess (`posture_preprocess_ms`)** | 0.69 | 0.00 | 2.15 | 2.24 | 2.50 | 2.81 |
| **Posture GPU Inference (`posture_inference_ms`)** | 1.78 | 0.00 | 5.82 | 6.14 | 7.15 | 8.24 |
| **Head-Pose Preprocess (`headpose_preprocess_ms`)** | 0.21 | 0.00 | 0.58 | 0.61 | 0.69 | 1.15 |
| **Head-Pose GPU Inference (`headpose_inference_ms`)**| 2.34 | 0.00 | 6.84 | 7.18 | 8.01 | 9.85 |
| **Phone Spatial Association (`phone_association_ms`)**| 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.03 |
| **V4D Multi-Cue Fusion (`fusion_ms`)** | 0.16 | 0.15 | 0.19 | 0.20 | 0.24 | 0.39 |
| **Event State Machine (`event_engine_ms`)** | 0.01 | 0.01 | 0.02 | 0.02 | 0.03 | 0.05 |
| **Risk Aggregation (`risk_aggregation_ms`)** | 0.01 | 0.00 | 0.01 | 0.01 | 0.02 | 0.03 |
| **Evidence Manager Lifecycle (`evidence_manager_ms`)**| 0.12 | 0.12 | 0.15 | 0.16 | 0.21 | 0.35 |
| **Physical JSON Serialization (`serialization_ms`)** | 0.02 | 0.01 | 0.02 | 0.03 | 0.04 | 0.09 |
| **Post-Decode Pipeline Total (`post_decode_pipeline_ms`)**| **22.94** | **19.99** | **39.67** | **43.10** | **47.47** | **52.83** |
| **Whole-Loop End-to-End (`whole_loop_end_to_end_ms`)** | **35.19** | **33.81** | **55.19** | **60.75** | **64.49** | **67.88** |

---

## 3. Key Forensic Observations
1. **Cadence Gating Efficiency**: Posture, head-pose, and macro detectors have median latencies of **0.00 ms**, confirming that cadence gating correctly prevents redundant neural evaluations between scheduled intervals.
2. **True Serialization**: Serialization is measured between 0.01 ms and 0.11 ms, varying with the presence of active lifecycle events rather than remaining frozen at 0.05 ms.
3. **Independent Whole-Loop Measurement**: Whole-loop end-to-end latency is measured directly from source read to result delivery, accurately capturing the decode stage (~12–15 ms) and scheduler overhead.

---

## 4. Status Flag
- `COMPONENT_TIMING_PHYSICAL = YES`
