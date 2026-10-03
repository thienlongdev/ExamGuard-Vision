# PILOT CLAIM TRACEABILITY MATRIX

## 1. Traceability Standard
Every operational assertion, architectural guarantee, and capability boundary reported in this pilot preparation phase is mapped to verifiable source code, physical configuration files, and raw benchmark artifacts in `runs/pilot/`. Zero ungrounded or narrative-only claims are permitted.

---

## 2. Claim Verification Table

| Claim ID | Formal Statement | Evidence Source | JSON Key / Path | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **`CLM-001`** | System models, APIs, and dashboards never declare `CHEATING` or `GUILTY` verdicts. | `src/fusion/types.py`, `src/fusion/risk_aggregator.py`, `configs/v4d_fusion.yaml` | `RiskLevel`, `EventFamily` definitions | **VERIFIED_INVARIANT** |
| **`CLM-002`** | All 7 certified checkpoints verified against certified SHA-256 hashes without modification. | `runs/pilot/checkpoint_hashes.json`, `src/pilot/preflight.py` | `*.verified == true` (all 7 checkpoints) | **VERIFIED_PHYSICAL_WEIGHTS** |
| **`CLM-003`** | Operating envelope derived algorithmically from raw matrix with strict occupancy gating. | `runs/pilot/operating_envelope.json`, `scripts/generate_operating_envelope.py` | `metadata.methodology == "STRICT_ARTIFACT_DERIVED_MECHANICAL_SELECTION"` | **VERIFIED_EMPIRICAL_BENCHMARK** |
| **`CLM-004`** | Ceiling high viewpoints disable facial head-pose due to steep pitch angle. | `runs/pilot/viewpoint_capability_matrix.json`, `src/pilot/profile.py` | `viewpoints.CEILING_HIGH.headpose_allowed == false` | **VERIFIED_GEOMETRY_POLICY** |
| **`CLM-005`** | RTSP stream long interruptions (>15s) evict stale tracks without phantom continuation. | `runs/pilot/rtsp_resilience.json`, `src/pilot/resilience.py` | `interruption_16s.track_continuity_status == "EVICTED_NEW_SESSION"`, `phantom_track_count == 0` | **VERIFIED_SIMULATED_HARNESS** |
| **`CLM-006`** | Memory reaches plateau and zero temporal track leaks across 10,000 frames per workload. | `runs/pilot/soak_results.json`, `src/pilot/benchmark.py` | `*.MEMORY_PLATEAU_REACHED == true`, `NO_TRACK_STATE_LEAK == true` (4 workloads) | **VERIFIED_SOAK_TELEMETRY** |
| **`CLM-007`** | Zero model retraining or dataset modification conducted in pilot preparation. | `git diff`, `runs/pilot/checkpoint_hashes.json` | Clean working tree for models and datasets | **VERIFIED_GOVERNANCE_INVARIANT** |
| **`CLM-008`** | Dense 20+ student rooms do not sustain 25-30 FPS line rate on current pipeline. | `runs/pilot/resolution_occupancy_matrix.json` | `dense_stress_results[*].effective_processed_fps < 20.0` | **VERIFIED_LIMITATION_DENSE_ROOM_NOT_READY** |
| **`CLM-009`** | Temporal buffer is camera-scoped `(camera_id, track_id)` preventing multi-camera collisions. | `runs/pilot/state_cleanup_validation.json`, `src/fusion/temporal_buffer.py` | `camera_scoped_temporal_state.verified == true` | **VERIFIED_ARCHITECTURE** |
| **`CLM-010`** | Backpressure low-latency default queue depth 5 selected for burst absorption. | `runs/pilot/backpressure_validation.json`, `configs/stage2_pipeline.yaml` | `selected_default_queue_depth == 5`, `capture_to_result_p95_ms == 58.85` | **VERIFIED_EMPIRICAL_BENCHMARK** |

---

## 3. Operational Integrity Audit Trail
- **Stale Artifact Archive**: `runs/pilot/archive/pre_corrective_20261003_144848/`
- **Execution State Artifact**: `runs/pilot/PILOT_CORRECTIVE_EXECUTION_STATE.json`
- **Regression Test Coverage**: `tests/test_pilot_integrity_corrective.py` (9/9 pass, 180/180 suite pass)
