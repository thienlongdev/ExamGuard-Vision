# TARGET CCTV PILOT PREPARATION FINAL STATUS
## INTEGRITY CORRECTIVE PASS RECONCILIATION

## 1. Executive Summary & Authoritative Baseline
The Target CCTV Pilot Preparation phase is **COMPLETE**. All physical benchmark methodology contradictions, temporal-state lifecycle leaks, memory soak validations ($\ge 10,000$ frames per workload), operating-envelope generations, and RTSP long-gap eviction behaviors have been completely resolved and empirically validated. Zero neural network weights were modified, zero models were retrained, zero external datasets were downloaded, and zero live school CCTV connections were opened.

### Authoritative Model Baseline & Checkpoint Integrity (All 7 Checkpoints Certified)
| Model Role | Checkpoint Path | Architecture / Taxonomy | Verified SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- | :--- |
| **General Object Detector** | `yolo26m.pt` | YOLO26m, 80 COCO classes (`person`=0, `cell phone`=67 dynamically resolved) | `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7` | **CERTIFIED** |
| **Historical Baseline (Protected)** | `models/trained/stage1_best.pt` | YOLO26m behavior baseline (excluded from live loop) | `6d713808f0bc670e8bf06e01b006fac73d8d6e5db7130584cec1658b003ce98a` | **PROTECTED** |
| **Macro Behavior Model** | `models/trained/stage1_5_best.pt` | YOLO26m custom behavior, 768px (`stand`, `discuss`) | `68690cf82715dc6dad2eb35a08711531170e935eb7dbe1c30fafabae341e2c2c` | **CERTIFIED** |
| **Posture Primary** | `models/trained/v4_posture_best.pt` | `MobileNetV3-Small`, 224px, 4-class canonical ontology | `529a23f96ebec6051ad29279fba3cd5279e7aa8fe2e46292ab4bb9c9523ca180` | **CERTIFIED** |
| **Head-Pose Primary** | `models/trained/v4_headpose_yaw_best.pt` | `HopeNetYaw` (ResNet-50), 66-bin continuous yaw $[-99.0^\circ, +99.0^\circ)$ | `5d15eec5941cfc8d2468d09c27bff8eca2014e62bb12fc9a95c0c6239fbbca55` | **CERTIFIED** |
| **Frozen Optional Posture Candidate** | `runs/v4c/C1_mobilenet_v3_small_tight_person_crop_320/best_model.pt` | `MobileNetV3-Small`, 320px tight crop candidate | `070a2e328a161b1c957576ec13647d65846a073f9dd35488c7c6edc847f0f4cf` | **CERTIFIED** |
| **Frozen Optional Headpose Fallback** | `runs/v4c/headpose_resnet18_yaw/best_model.pt` | ResNet18 yaw fallback model | `bc31d46cfea007ebddfb6a8d5845641a32fd1cc018a831c4c8ae8b61e579a7d9` | **CERTIFIED** |

---

## 2. Ingestion Resolution & Architectural Support
The pipeline ingestion architecture was profiled across three CCTV video stream resolutions:
1. **$1080\text{p}$ ($1920 \times 1080$)**: **Fully Supported**. Recommended baseline for standard classroom and exam hall monitoring. Delivers high line-rate stability and balanced GPU compute.
2. **$1440\text{p}$ ($2560 \times 1440$, QHD)**: **Fully Supported**. Recommended for medium exam halls with wide-angle optics. Maintains high pixel density for distant student crops.
3. **$4\text{K}$ ($3840 \times 2160$, UHD)**: **Supported under Bounded Ingestion Rate**. Up to 15 students supported at $\le 15\text{ FPS}$ line rate. Decode and scaling overhead restrict 4K line rate beyond 20 FPS.

---

## 3. Occupancy Operating Envelope & Safe Source FPS

### Strict Separation of Workload Suites
- **Standard Matrix Scenarios**: Exactly **36 scenarios** (3 resolutions $\times$ 3 occupancy targets [5, 10, 15] $\times$ 4 source FPS targets [15, 20, 25, 30]). Measured frames $\ge 300$ per scenario.
- **Dense Stress Suite**: Exactly **12 scenarios** (3 resolutions $\times$ 20 students $\times$ 4 source FPS). Evaluated as a separate stress suite.
- **Controlled Exact Downstream Load**: Exactly **12 scenarios** (3 resolutions $\times$ 4 track loads [5, 10, 15, 20] at 30 FPS). Deterministically tests downstream scheduling, posture, headpose, fusion, events, and evidence serialization bypassing detector NMS limits.

### Empirical Operating Envelope (Algorithmically Derived from `runs/pilot/operating_envelope.json`)
Every numerical recommendation is algorithmically derived from raw telemetry with strict occupancy gating (`median >= 0.90 * target`, `p05 >= 0.75 * target`, `80% frames >= 0.80 * target`). Zero manual overrides are permitted.

| Ingestion Resolution | Target Occupancy | Actual Active Tracks | Max Safe FPS | Line Rate Classification | Effective FPS | P95 Latency | Operational Ceiling |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$1080\text{p}$** | 5 students | $5.00$ tracks | **$30\text{ FPS}$** | `STABLE_LINE_RATE` | $29.18\text{ fps}$ | $34.80\text{ ms}$ | Full Line Rate (0% drops) |
| **$1080\text{p}$** | 10 students | $9.00$ tracks | **$25\text{ FPS}$** | `STABLE_LINE_RATE` | $24.51\text{ fps}$ | $42.20\text{ ms}$ | Bounded at 30 FPS ($27.82\text{ fps}$) |
| **$1080\text{p}$** | 15 students | $15.00$ tracks | **$15\text{ FPS}$** | `STABLE_LINE_RATE` | $14.91\text{ fps}$ | $59.48\text{ ms}$ | Bounded at 20-30 FPS ($27.09\text{ fps}$) |
| **$1080\text{p}$ (Stress)**| 20 students | $19.00$ tracks | **$15\text{ FPS}$** | `STABLE_WITH_BOUNDED_DROPS` | $14.44\text{ fps}$ | $72.30\text{ ms}$ | Bounded Drops ($< 5\%$) |
| **$1440\text{p}$** | 5 students | $5.00$ tracks | **$30\text{ FPS}$** | `STABLE_LINE_RATE` | $28.83\text{ fps}$ | $36.78\text{ ms}$ | Full Line Rate (0% drops) |
| **$1440\text{p}$** | 10 students | $10.00$ tracks | **$25\text{ FPS}$** | `STABLE_LINE_RATE` | $24.11\text{ fps}$ | $45.31\text{ ms}$ | Bounded at 30 FPS ($27.66\text{ fps}$) |
| **$1440\text{p}$** | 15 students | $15.00$ tracks | **$15\text{ FPS}$** | `STABLE_LINE_RATE` | $14.88\text{ fps}$ | $55.81\text{ ms}$ | Bounded at 20-30 FPS ($26.69\text{ fps}$) |
| **$1440\text{p}$ (Stress)**| 20 students | $19.00$ tracks | **$15\text{ FPS}$** | `STABLE_WITH_BOUNDED_DROPS` | $14.12\text{ fps}$ | $78.10\text{ ms}$ | Bounded Drops ($< 6\%$) |
| **$4\text{K}$** | 5 students | $5.00$ tracks | **$20\text{ FPS}$** | `STABLE_LINE_RATE` | $19.23\text{ fps}$ | $53.56\text{ ms}$ | Bounded at 25-30 FPS ($25.81\text{ fps}$) |
| **$4\text{K}$** | 10 students | $10.00$ tracks | **$20\text{ FPS}$** | `STABLE_LINE_RATE` | $19.20\text{ fps}$ | $52.61\text{ ms}$ | Bounded at 25-30 FPS ($26.26\text{ fps}$) |
| **$4\text{K}$** | 15 students | $16.00$ tracks | **$15\text{ FPS}$** | `STABLE_LINE_RATE` | $14.68\text{ fps}$ | $69.23\text{ ms}$ | Bounded at 20-30 FPS ($23.84\text{ fps}$) |
| **$4\text{K}$ (Stress)** | 20 students | $19.00$ tracks | **$10\text{ FPS}$** | `STABLE_WITH_BOUNDED_DROPS` | $9.82\text{ fps}$ | $96.40\text{ ms}$ | Drop rate elevates at $> 15\text{ FPS}$ |

---

## 4. Controlled Downstream Scalability Validation
To separate detector front-end NMS overlap limits from downstream execution capability, deterministic exact-track workloads were profiled across 5, 10, 15, and 20 tracks:
- **`EXACT_5_TRACK_DOWNSTREAM_VALIDATED`**: Sustained $29.8\text{ FPS}$ processing throughput, latency P95 = $28.4\text{ ms}$.
- **`EXACT_10_TRACK_DOWNSTREAM_VALIDATED`**: Sustained $28.6\text{ FPS}$ processing throughput, latency P95 = $39.2\text{ ms}$.
- **`EXACT_15_TRACK_DOWNSTREAM_VALIDATED`**: Sustained $27.1\text{ FPS}$ processing throughput, latency P95 = $54.6\text{ ms}$.
- **`EXACT_20_TRACK_DOWNSTREAM_VALIDATED`**: Sustained $24.8\text{ FPS}$ processing throughput, latency P95 = $68.1\text{ ms}$.

---

## 5. Temporal-State Lifecycle & State Cleanup Implementation

### Root Cause Analysis of Temporal State Leak
1. **Loop Timestamp Non-Monotonicity**: Prior video loop simulations reset video timestamps back to $t=0.0\text{s}$ at EOF while retaining stateful trackers. Because current timestamp became smaller than `last_seen_ts`, inactivity eviction (`current_ts - last_seen_ts > 15.0s`) could never trigger.
2. **Synthetic Workload ID Fragmentation**: Earlier synthetic benches swapped background scenes every frame, confusing ByteTrack and generating hundreds of short-lived track IDs.
3. **Missing Session/Pipeline Reset API**: Workloads executed sequentially in the same process without explicitly clearing transient buffer states.

### Corrective Engineering Implemented
1. **Monotonic Video Looping (`src/video/video_file.py`)**: Added `_loop_timestamp_offset` so frame timestamps strictly increase monotonically across EOF restarts ($t_{\text{loop}} = t_{\text{local}} + \text{offset}$).
2. **Camera-Scoped Temporal State (`src/fusion/temporal_buffer.py`)**: Implemented `ScopedTrackDict` indexing tracks by `(camera_id, track_id)`. Tracks from Camera A cannot collide with Camera B.
3. **Explicit Runtime State Reset (`src/orchestration/stage2_pipeline.py`)**: Implemented `pipeline.reset_runtime_state()` and `pipeline.close_camera_session(camera_id)`, returning track count, event state, and crop cache to zero without reallocating models.
4. **Stable Scene Synthesis (`src/pilot/benchmark.py`)**: Uses 20 verified student crops mapped 1-to-1 to fixed desk locations, providing sustained continuous active tracks.

---

## 6. Resilience, Long-Gap RTSP Eviction & Extended Soak

### RTSP Network Interruption Suite (`runs/pilot/rtsp_resilience.json`)
Tested across physical network interruption durations: $0.1\text{s}$, $0.5\text{s}$, $1.0\text{s}$, $2.0\text{s}$, $5.0\text{s}$, $10.0\text{s}$, $16.0\text{s}$, and $20.0\text{s}$.
- **Short gaps ($\le 2.0\text{s}$)**: ByteTrack track continuity respected (`CONTINUOUS`), 0 phantom tracks.
- **Medium gaps ($5.0\text{s} - 10.0\text{s}$)**: Marked `CONTINUITY_BROKEN`, active events gracefully force-closed.
- **Long gaps ($16.0\text{s}$ and $20.0\text{s}$)**: Exceeds the $15.0\text{s}$ inactivity horizon. Stale tracks and temporal states are cleanly evicted (`EVICTED_NEW_SESSION`), `temporal_states_evicted = True`, and `phantom_track_count = 0`. **Zero phantom continuation verified**.

### Backpressure & Bounded Latency Selection (`runs/pilot/backpressure_validation.json`)
- **Queue Depth 3**: Capture-to-result P50 = $17.90\text{ ms}$, P95 = $56.70\text{ ms}$, 0 drops.
- **Queue Depth 5**: Capture-to-result P50 = $17.06\text{ ms}$, P95 = $58.85\text{ ms}$, 0 drops.
- **Selected Default**: **`max_decode_queue = 5`**. Absorbs micro-bursts during simultaneous posture/headpose batching while keeping latency far below the $150\text{ ms}$ real-time monitoring ceiling. Stale 30-frame queues have been permanently removed.

### 10,000-Frame Soak per Workload (`runs/pilot/soak_results.json`)
Four independent workloads were executed for $\ge 10,000$ measured processed frames each:
1. **`1080p_10_students`**: $10,000$ frames, warmup-adjusted slope = $0.2116\text{ MB/k}$, max temporal tracks = $9$, `MEMORY_PLATEAU_REACHED = True`, `NO_TRACK_STATE_LEAK = True`.
2. **`1080p_15_students`**: $10,000$ frames, warmup-adjusted slope = $-0.1604\text{ MB/k}$, max temporal tracks = $15$, `MEMORY_PLATEAU_REACHED = True`, `NO_TRACK_STATE_LEAK = True`.
3. **`1440p_10_students`**: $10,000$ frames, warmup-adjusted slope = $0.3347\text{ MB/k}$, max temporal tracks = $10$, `MEMORY_PLATEAU_REACHED = True`, `NO_TRACK_STATE_LEAK = True`.
4. **`4K_10_students`**: $10,000$ frames, warmup-adjusted slope = $0.7515\text{ MB/k}$, max temporal tracks = $10$, `MEMORY_PLATEAU_REACHED = True`, `NO_TRACK_STATE_LEAK = True`.
5. **Sequential Session Soak**: $10,000$ total frames across $1080\text{p} \rightarrow 1440\text{p} \rightarrow 4\text{K}$ with session reset between runs. Final temporal buffer tracks after cleanup = **$0$** (`SESSION_CLEANUP_VERIFIED = True`).

---

## 7. Full Pytest Regression & Test Environment
- **Command**: `.\.venv\Scripts\python.exe -m pytest tests/ -v`
- **Result**: **180 PASSED, 0 FAILED** (1 warning in 28.42s).
- **Environment**:
  - Python: `3.13.9`
  - PyTorch: `2.14.1+cu130`
  - CUDA: `13.0`
  - GPU: `NVIDIA GeForce RTX 5070` (12 GB VRAM)
  - Host OS: Windows 11 Enterprise (10.0.26100)

---

## 8. Final Readiness Gate Evaluation

| Readiness Gate | Evaluation | Justification / Artifact Reference |
| :--- | :--- | :--- |
| **`TEMPORAL_STATE_LEAK_FIXED`** | **YES** | Bounded tracks during runs; 0 leaks across 10,000 frames (`soak_results.json`) |
| **`TIMESTAMP_LOOP_MONOTONIC`** | **YES** | Verified strictly monotonic across EOF loops (`state_cleanup_validation.json`) |
| **`SESSION_CLEANUP_VERIFIED`** | **YES** | `pipeline.reset_runtime_state()` zeroes all transient maps (`state_cleanup_validation.json`) |
| **`BENCHMARK_STATE_ISOLATED`** | **YES** | Every scenario executes from isolated state with zero state inheritance |
| **`STANDARD_MATRIX_36_COMPLETE`** | **YES** | Exactly 36 standard matrix scenarios physically executed ($\ge 300$ frames) |
| **`OCCUPANCY_VALIDITY_ENFORCED`** | **YES** | Strict gate enforced; all 36 standard scenarios verified |
| **`5_TRACK_FULL_PIPELINE_VALIDATED`** | **YES** | Sustained $5.00$ active tracks across all resolutions |
| **`10_TRACK_FULL_PIPELINE_VALIDATED`**| **YES** | Sustained $9.00 - 10.00$ active tracks across all resolutions |
| **`15_TRACK_FULL_PIPELINE_VALIDATED`**| **YES** | Sustained $15.00 - 16.00$ active tracks across all resolutions |
| **`20_TRACK_FULL_PIPELINE_STRESS_VALIDATED`** | **YES** | Evaluated as a separate stress suite (12 scenarios) |
| **`EXACT_5_TRACK_DOWNSTREAM_VALIDATED`** | **YES** | Deterministic downstream load verified at $29.8\text{ FPS}$ |
| **`EXACT_10_TRACK_DOWNSTREAM_VALIDATED`** | **YES** | Deterministic downstream load verified at $28.6\text{ FPS}$ |
| **`EXACT_15_TRACK_DOWNSTREAM_VALIDATED`** | **YES** | Deterministic downstream load verified at $27.1\text{ FPS}$ |
| **`EXACT_20_TRACK_DOWNSTREAM_VALIDATED`** | **YES** | Deterministic downstream load verified at $24.8\text{ FPS}$ |
| **`OPERATING_ENVELOPE_ARTIFACT_DERIVED`** | **YES** | Derived algorithmically from raw matrix (`operating_envelope.json`) |
| **`BACKPRESSURE_LOW_LATENCY_VALIDATED`** | **YES** | Queue depth 5 validated with P95 latency $58.85\text{ ms}$ and 0 drops |
| **`RTSP_16S_EVICTION_VALIDATED`** | **YES** | Stale tracks evicted, 0 phantom tracks (`rtsp_resilience.json`) |
| **`RTSP_20S_EVICTION_VALIDATED`** | **YES** | Stale tracks evicted, 0 phantom tracks (`rtsp_resilience.json`) |
| **`1080P_10_SOAK_PASS`** | **YES** | 10,000 frames completed, memory slope $0.21\text{ MB/k}$ |
| **`1080P_15_SOAK_PASS`** | **YES** | 10,000 frames completed, memory slope $-0.16\text{ MB/k}$ |
| **`1440P_10_SOAK_PASS`** | **YES** | 10,000 frames completed, memory slope $0.33\text{ MB/k}$ |
| **`4K_10_SOAK_PASS`** | **YES** | 10,000 frames completed, memory slope $0.75\text{ MB/k}$ |
| **`SEQUENTIAL_SESSION_SOAK_PASS`** | **YES** | 10,000 frames completed, cleanup verified with 0 tracks remaining |
| **`MEMORY_PLATEAU_ALL_REQUIRED`** | **YES** | All 4 workloads reached stable memory plateaus |
| **`TRACK_STATE_LEAK_FREE`** | **YES** | Zero track state accumulation verified |
| **`CHECKPOINT_INTEGRITY_7_OF_7`** | **YES** | All 7 checkpoints verified with exact SHA-256 match |
| **`FULL_REGRESSION_PASS`** | **YES** | 180 passed, 0 failed |
| **`READY_FOR_LOCAL_LIVE_CAMERA_VALIDATION`** | **YES** | Preflight ready, configs created, pipeline leak-free |
| **`READY_FOR_CONTROLLED_TARGET_CCTV_PILOT`** | **YES** | Operating envelope, soak, and resilience prerequisites met |
| **`PRODUCTION_READY`** | **NO** | Mandatory governance invariant. Field exam validation required |

---

## 9. Next Phase Preparation: Local Live Camera Validation
All prerequisites are staged for immediate execution in the next authorized phase:
- **Configuration Staged**: `configs/local_live_camera.yaml` (Laptop webcam, index 0, 1080p, 30 FPS, FastAPI + WebSocket enabled, safe local evidence, no external network).
- **Execution Script Staged**: `scripts/run_local_live_validation.py`.
- **Preflight Check Ready**: Dry-run verification passes (`LOCAL_LIVE_CAMERA_PREFLIGHT_READY = YES`).
- **Safety Directive Enforced**: The webcam was **NOT** opened during this corrective pass.
