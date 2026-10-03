# Stage 2 End-to-End Orchestration & Integration Master Report

---

## 1. STAGE 2 FINAL STATUS
- **Stage Execution Mode**: Autonomous Integration & Verification
- **Perception Retraining**: None (0 new training runs, all 6 model checkpoints frozen and bit-exact)
- **Dataset Acquisition**: None (0 new dataset downloads, evaluated exclusively on physical repository assets)
- **Integration Scope**: End-to-end ingestion, full-frame detection, ByteTrack tracking, cadence-driven crop scheduling, batched inference, phone spatial association, unified track updating, V4D multi-cue temporal fusion, event lifecycle management, bounded evidence collection, FastAPI REST and WebSocket streaming.
- **Overall Pipeline Status**: **FULLY OPERATIONAL & VERIFIED**
- **Test Suite Status**: **146 passed, 0 failed, 1 warning** (10.95 s total runtime)
- **Physical Latency (Classroom Mode B)**: **23.084 ms mean, 28.964 ms P95 (29.59 FPS real-time)**

---

## 2. V4D RAW-METRIC RECONCILIATION STATUS
All historical contradictions identified in earlier draft documentation have been systematically resolved by recomputing canonical metrics directly from isolated raw JSON evaluation logs (`runs/v4d/V4D_EVALUATION_RESULTS.json`):

| Evaluation Dimension | Historic Narrative Discrepancy | Canonical Reconciled Truth | Source Artifact | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Debounce Recall vs Full Engine** | Reports conflated concatenated 60-clip timeline (20/20) with isolated 20-clip evaluation (17/20). | Isolated evaluation achieves **17/20 (85.0%) recall**; 3 missed clips are bounded by static perception misses. | `runs/v4d/V4D_EVALUATION_RESULTS.json` (`isolated_evaluation.positive_recall`) | **RESOLVED** |
| **Negative False Alarms** | One sweep reported 1 false alarm under unvetoed raw debounce; full engine reported 0. | Raw debounce produces 1 false alarm on writing clips. **Full V4D Engine with Read/Write Veto produces exactly 0 / 40 false alarms (0.0% FA rate)**. | `runs/v4d/V4D_EVALUATION_RESULTS.json` (`isolated_evaluation.total_negative_false_alarms`) | **RESOLVED** |
| **Threshold Optimality Claim** | 1.50 s was historically asserted as "optimal" without mathematical proof. | Reclassified to **RECOMMENDED_CANDIDATE_DURATION = 1.50 s** based on multi-objective tie-breaking (zero writing false alarms, max recall, min activation latency). | `runs/v4d/V4D_EVALUATION_RESULTS.json` (`recommended_operating_point`) | **RESOLVED** |
| **Risk Terminology** | "Probability of Cheating" and "Calibrated Risk Score" improperly implied posterior cheating probability. | Replaced across all schemas and reports with **CONFIGURED_RISK_SCORE** and **PROVISIONAL_ENGINEERING_THRESHOLDS**. | `reports/v4d/V4D_EVENT_RISK_POLICY.md` | **RESOLVED** |

---

## 3. END-TO-END ARCHITECTURE
The production orchestration pipeline integrates 11 discrete stages into a deterministic execution graph:

```
+-----------------------------------------------------------------------------------+
|                                STAGE 2 ORCHESTRATOR                               |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |   VideoSource (RTSP/Webcam/   |
                         |   File) + Decode + Timestamp  |
                         +-------------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |  Full-Frame Detector (Stage1  |
                         |    YOLOv8x 640x640 Person)    |
                         +-------------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |   ByteTrack Multi-Object      |
                         |   Tracker (Preserves Track ID)|
                         +-------------------------------+
                                         |
               +-------------------------+-------------------------+
               |                                                   |
               v                                                   v
+-------------------------------+                 +-------------------------------+
| CropScheduler (Cadence Gated) |                 | PhoneAssociator (Spatial IoU, |
| - Posture: 10 Hz (MobileNet)  |                 | Distance & Ambiguity Checks)  |
| - Headpose: 6 Hz (HopeNet)    |                 +-------------------------------+
+-------------------------------+                                  |
               |                                                   |
               +-------------------------+-------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |      UnifiedTrackUpdate       |
                         |  (Canonical State per Track)  |
                         +-------------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |      V4D Temporal Fusion      |
                         |  (MultiCueFusionEngine +      |
                         |   TrackObservationBuffer)     |
                         +-------------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |    EventEngine (OPEN /        |
                         |    UPDATE / CLOSE Episodes)   |
                         +-------------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |      EvidenceManager &        |
                         |   Bounded Snapshot Ring       |
                         +-------------------------------+
                                         |
                     +-------------------+-------------------+
                     |                                       |
                     v                                       v
      +-----------------------------+         +-----------------------------+
      |      FastAPI REST API       |         |     WebSocket Live Bus      |
      | (/api/events, /api/status)  |         |        (/ws/events)         |
      +-----------------------------+         +-----------------------------+
```

---

## 4. VIDEO SOURCE STATUS
The video abstraction subsystem (`src/video/`) was validated across three operational modes:
- **`WebcamSource`**: Captures from local DirectShow/V4L2 devices using monotonic wall-clock arrival timestamps (`time.monotonic()`).
- **`VideoFileSource`**: Reads offline MP4/AVI containers extracting embedded container timestamps; validated against `datasets/raw_v4/other_candidates/eduaction/writing/writing (1).mp4` and `lecture (10).mp4`.
- **`RTSPSource`**: Decodes network camera streams with safe reconnection timeouts and frame drop tolerance.
- **Timestamp Guarantee**: Pipeline durations depend strictly on `timestamp_sec`, rendering event logic immune to variable frame rate, decode stalls, or dropped packets.
- **Status**: **VIDEO_SOURCE_READY = YES**

---

## 5. MODEL REGISTRY
All 6 frozen checkpoints are managed via the singleton `ModelRegistry` (`src/orchestration/model_registry.py`). Models are loaded once at startup, verified cryptographically, and warmed up with dummy inference:

| Role | Checkpoint Path | Architecture | Input Res | Device | Precision | SHA-256 Checksum Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Detector Baseline** | `models/trained/stage1_best.pt` | YOLOv8x | 640x640 | CUDA:0 | FP16 | `6d713808f0bc...` **MATCH** |
| **Detector Secondary**| `models/trained/stage1_5_best.pt` | YOLOv8x | 640x640 | CUDA:0 | FP16 | `68690cf82715...` **MATCH** |
| **Primary Posture** | `models/trained/v4_posture_best.pt` | MobileNetV3-Small | 224x224 | CUDA:0 | FP16 | `529a23f96ebe...` **MATCH** |
| **Primary Head-Pose** | `models/trained/v4_headpose_yaw_best.pt` | HopeNet-Yaw | 224x224 | CUDA:0 | FP32 (Safe)| `5d15eec5941c...` **MATCH** |
| **High-Res Posture** | `runs/v4c/C1_mobilenet_v3_small_tight_person_crop_320/best_model.pt` | MobileNetV3-Small | 320x320 | N/A | Cached | `070a2e328a16...` **MATCH** |
| **Full-Domain Yaw** | `runs/v4c/headpose_resnet18_yaw/best_model.pt` | ResNet18-Circular | 224x224 | N/A | Cached | `bc31d46cfea0...` **MATCH** |

---

## 6. TRACKING STATUS
- **Tracking Integration**: ByteTrack assigns persistent `track_id` integers across consecutive frames.
- **Track-to-Batch Mapping**: `CropScheduler` guarantees exact index preservation between dynamic batch predictions and track IDs via `track_id_map`.
- **Unit Verification**: `tests/test_stage2_batch_track_mapping.py` proves zero cross-track prediction contamination across randomized permutation schedules.
- **Track Lifecycle**: Disconnected or expired tracks automatically trigger safe closure of open events.
- **Status**: **TRACKING_INTEGRATION_READY = YES**

---

## 7. FUSION STATUS
- **Core Engine**: Unified track updates feed directly into `MultiCueFusionEngine` (`src/fusion/`).
- **Temporal Debouncing**: `TrackObservationBuffer` buffers observations in a monotonic time-sorted window ($10.0\text{ s}$ max capacity).
- **Read/Write Veto**: Active `NORMAL_READ_WRITE` posture vetoes spurious head-rest detections, eliminating false alarms on exam writing tasks.
- **Missing Cue Robustness**: Missing headpose or rear-view faces map to `UNAVAILABLE`; the engine relies safely on posture cues without crashing or zero-filling yaw.
- **Status**: **V4D_FUSION_INTEGRATION_READY = YES**

---

## 8. EVENT / EVIDENCE STATUS
- **Event Lifecycle**: Events undergo clean `EVENT_OPEN`, `EVENT_UPDATE`, and `EVENT_CLOSE` transitions with zero duplicate row spam.
- **Bounded Snapshots**: `EvidenceManager` captures at most 3 images per event episode (Open snapshot, High-risk escalation snapshot, Close snapshot).
- **Metadata Traceability**: Closing an event writes an evidentiary JSON record documenting camera ID, track ID, event duration, cue reliability, and model provenance.
- **Status**: **EVENT_LIFECYCLE_READY = YES**, **EVIDENCE_MANAGER_READY = YES**

---

## 9. API / WEBSOCKET STATUS
- **FastAPI Endpoints**:
  - `GET /health`: System operational status and camera list.
  - `GET /api/system/status`: Real-time queue depths, dropped frames, active tracks, and model registry versions.
  - `GET /api/system/models`: Cryptographic hash provenance for all loaded models.
  - `GET /api/events`: Filterable event history with bounding box and evidence links.
  - `GET /api/cameras`: Camera registry and active stream status.
- **WebSocket Streaming**: `/ws/events` broadcasts serialized JSON events on lifecycle transitions (`EVENT_OPEN`, `EVENT_UPDATE`, `EVENT_CLOSE`).
- **Status**: **API_READY = YES**, **WEBSOCKET_READY = YES**

---

## 10. PHYSICAL INTEGRATED RUNTIME
Measured on evaluation hardware (**NVIDIA GeForce RTX 5070 12GB**, 32-core CPU, 32GB RAM) and recorded in `runs/stage2/integrated_runtime_benchmark.json`:

| Operating Benchmark Mode | Video Input Stream | Frames Processed | Effective Throughput | Total Latency (Mean) | Total Latency (P95) | Peak Cadence Latency (Mean) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mode A: Single Student Physical** | `writing (1).mp4` (1080p, 25 FPS) | 150 frames | **24.79 FPS** | 24.758 ms | 31.090 ms | 31.573 ms |
| **Mode B: Medium Classroom Physical**| `lecture (10).mp4` (1080p, 30 FPS) | 150 frames | **29.59 FPS** | 23.084 ms | 28.964 ms | 30.514 ms |
| **Mode C: Synthetic 20-Student Scaling**| 20-Student Tiled Real Crops | 150 frames | **24.33 FPS** | 19.358 ms | 37.619 ms | 57.496 ms |

- **Real-Time Assessment**: On physical classroom streams (Mode B), **P95 latency is 28.964 ms** (below the 33.33 ms frame budget), operating at a steady **29.59 FPS** with zero frame drops.
- **Status**: **INTEGRATED_RUNTIME_MEASURED = YES**

---

## 11. BACKPRESSURE / FRAME DROP STATUS
- **Queue Limits**: Input frame queue is strictly capped at `max_size = 3`.
- **Drop Policy**: If inference falls behind, the oldest unconsumed frame is dropped immediately, bounding latency to $< 100\text{ ms}$.
- **Lifecycle Protection**: Critical event lifecycle transitions (`CLOSE`) and evidence disk writes are never dropped.
- **Temporal Invariance**: Tested under 50% simulated bursty drops; events triggered at identical real-world timestamps.
- **Status**: **BACKPRESSURE_READY = YES**

---

## 12. MEMORY STABILITY
- **Peak VRAM Allocated**: **415.34 MB** (Peak Reserved: **616.00 MB** out of 12,226 MB total).
- **GPU Headroom**: $> 95.0\%$ unallocated GPU memory headroom.
- **Dynamic VRAM Leaks**: **0.00 MB** (verified via `torch.cuda.memory_allocated()` between Frame 10 and 150).
- **Host RSS Memory Delta**: **+15.95 MB** across hundreds of frames on physical streams.
- **Buffer Pruning**: Confirmed automated eviction of observations older than 10.0 seconds.
- **Status**: **MEMORY_STABILITY_READY = YES**

---

## 13. FAILURE RECOVERY
- **Corrupted Frame Injection**: Safely bypassed; pipeline logs warning and continues without crashing.
- **Sub-Resolution Head Crops ($< 20\text{ px}$)**: Automatically maps to `YAW_UNAVAILABLE`; yaw is never substituted as zero.
- **Ambiguous Phones**: Bounded ambiguity logic marks phones equidistant to multiple students as `AMBIGUOUS_ASSOCIATION`, suppressing false alarms.
- **Stream EOF**: Automatically closes all open events and flushes evidence metadata.
- **Status**: **FAILURE_RECOVERY_READY = YES**

---

## 14. TEST RESULT
Full automated test regression executed via Pytest:
- **Total Tests Executed**: 146
- **Passed**: **146**
- **Failed**: **0**
- **Warnings**: 1 (Pydantic deprecation notice for `parse_obj`, non-breaking)
- **Execution Time**: **10.95 seconds**
- **Targeted Test Suites**:
  - `tests/test_stage2_model_registry.py` (PASSED)
  - `tests/test_stage2_batch_track_mapping.py` (PASSED)
  - `tests/test_stage2_pipeline_orchestration.py` (PASSED)
  - `tests/test_stage2_backpressure.py` (PASSED)
  - `tests/test_stage2_event_integration.py` (PASSED)
  - `tests/test_stage2_api_contract.py` (PASSED)
  - `tests/test_stage2_evidence_lifecycle.py` (PASSED)
  - `tests/test_stage2_failure_recovery.py` (PASSED)
  - `tests/test_stage2_timestamp_flow.py` (PASSED)
  - `tests/test_stage2_memory_bounds.py` (PASSED)
  - All existing V4D / V4C / tracking suites (PASSED)

---

## 15. CHECKPOINT INTEGRITY
All 6 frozen checkpoints were cryptographically re-verified via SHA-256 after the complete Stage 2 implementation, benchmark runs, and test regression:

```
stage1_best.pt:          6d713808f0bc670e8bf06e01b006fac73d8d6e5db7130584cec1658b003ce98a [BIT-EXACT]
stage1_5_best.pt:        68690cf82715dc6dad2eb35a08711531170e935eb7dbe1c30fafabae341e2c2c [BIT-EXACT]
v4_posture_best.pt:      529a23f96ebec6051ad29279fba3cd5279e7aa8fe2e46292ab4bb9c9523ca180 [BIT-EXACT]
v4_headpose_yaw_best.pt: 5d15eec5941cfc8d2468d09c27bff8eca2014e62bb12fc9a95c0c6239fbbca55 [BIT-EXACT]
C1 320 best_model.pt:    070a2e328a161b1c957576ec13647d65846a073f9dd35488c7c6edc847f0f4cf [BIT-EXACT]
resnet18_yaw best_model: bc31d46cfea007ebddfb6a8d5845641a32fd1cc018a831c4c8ae8b61e579a7d9 [BIT-EXACT]
```

Zero checkpoint mutations occurred.

---

## 16. SCIENTIFIC LIMITATIONS
1. **Observable Facts Only**: The system observes and records behavioral cues (`SUSTAINED_HEAD_REST`, `SUSTAINED_LATERAL_HEAD_ORIENTATION`, `PHONE_ASSOCIATED`). It NEVER infers guilt, intent, or cheating.
2. **Yaw Invariance**: Yaw measures horizontal head rotation only; it cannot infer reading, writing, or sleeping.
3. **Unlabeled Video Protocols**: Physical classroom videos were used exclusively for runtime and stability profiling; zero accuracy metrics were fabricated from unlabeled media.
4. **Unsupported Temporal Metrics**: While sleep rest was formally evaluated on 60 ground-truth video clips, temporal turn and temporal phone association have not yet undergone large-scale multi-annotator temporal ROC benchmarking.

---

## 17. READINESS VERDICTS

```
+-------------------------------------------------------------+
|                     STAGE 2 READINESS MATRIX                |
+-------------------------------------------------------------+
| VIDEO_SOURCE_READY              | YES                       |
| DETECTOR_INTEGRATION_READY      | YES                       |
| TRACKING_INTEGRATION_READY      | YES                       |
| POSTURE_INTEGRATION_READY       | YES                       |
| HEADPOSE_INTEGRATION_READY      | YES                       |
| PHONE_BRANCH_READY              | YES                       |
| V4D_FUSION_INTEGRATION_READY    | YES                       |
| EVENT_LIFECYCLE_READY           | YES                       |
| EVIDENCE_MANAGER_READY          | YES                       |
| API_READY                       | YES                       |
| WEBSOCKET_READY                 | YES                       |
| BACKPRESSURE_READY              | YES                       |
| FAILURE_RECOVERY_READY          | YES                       |
| MEMORY_STABILITY_READY          | YES                       |
| INTEGRATED_RUNTIME_MEASURED     | YES                       |
+-------------------------------------------------------------+
| READY_FOR_TARGET_CCTV_PILOT     | YES                       |
| PRODUCTION_READY                | NO                        |
+-------------------------------------------------------------+
```

### Readiness Justification:
- **`READY_FOR_TARGET_CCTV_PILOT = YES`**: The entire end-to-end software pipeline is structurally complete, mathematically and cryptographically validated, real-time capable (29.59 FPS, 28.96 ms P95), memory-bounded, and fully passing all 146 regression tests. It is ready for controlled deployment in target-school pilot CCTV trials.
- **`PRODUCTION_READY = NO`**: Formal production readiness requires on-site camera calibration, multi-room pilot trials, long-duration (24-hour) soak testing, user experience evaluation with actual exam invigilators, and institutional privacy compliance clearance.
