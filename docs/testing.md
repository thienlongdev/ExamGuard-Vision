# ExamGuard Vision — Testing & Quality Assurance Guide

This document describes the testing architecture, test categorization, execution commands, and expected outcomes across all automated validation suites in ExamGuard Vision.

---

## 1. Test Architecture Overview

The repository utilizes **pytest** as its automated test runner, configured at the root via `pytest.ini`. The test suite consists of **219 collected tests** designed to safeguard code correctness, cryptographic integrity, API contracts, and real-time inference safety:

```
tests/
├── fixtures/                           # Lightweight test fixtures and metadata
│   ├── scb_dataset5_full/              # Dataset statistics & identity reports
│   ├── annotation_conflicts_v3.json    # Quarantined conflict index
│   ├── generate_sample_video.py        # Synthetic test video generator
│   └── sample_exam.mp4                 # Synthetic offline exam hall clip
│
├── test_stage2_pipeline_orchestration.py # End-to-end single frame & stream pipeline tests
├── test_stage2_integrity.py            # Checkpoint identity, taxonomy, and cadence tests
├── test_stage2_model_registry.py       # Singleton model lifecycle & SHA-256 validation
├── test_stage2_api_contract.py         # FastAPI REST endpoints and WebSocket lifecycle
├── test_stage2_event_integration.py    # Multi-cue event state progression & cooldown
├── test_stage2_batch_track_mapping.py  # ByteTrack association & crop scheduler gating
├── test_stage2_backpressure.py         # Bounded ingestion queue and frame dropping
├── test_stage2_memory_bounds.py        # RSS and VRAM memory leak prevention
├── test_stage2_evidence_lifecycle.py   # Snapshot capture, evidence linking & cleanup
├── test_stage2_timestamp_flow.py       # Monotonic clock validation, NaN/Inf rejection
├── test_dashboard_ui.py                # Dashboard HTML, components & offline assets
├── test_laptop_preflight.py            # Hardware probing, GPU capability, port tests
├── test_desktop_semantic_integrity.py  # Decoupled software validation & origin tagging
├── test_pilot_preparation.py           # Camera profile schemas, seat zones, retention
├── test_pilot_integrity_corrective.py  # Operating envelope, RTSP recovery, 7/7 hashes
├── test_rules_scorer.py                # Risk scoring, thresholds & debounce rules
├── test_temporal_buffer.py             # Sliding window history & track eviction
├── test_v4d_capability.py              # Dynamic headpose & posture dispatch tests
├── test_v4d_event_engine.py            # Observable event state machine transitions
├── test_v4d_fusion_engine.py           # Multi-cue temporal fusion & veto rules
├── test_v4d_missing_cues.py            # Graceful degradation under missing cue inputs
├── test_v4d_replay.py                  # Telemetry JSONL replay determinism
├── test_v4d_risk_aggregator.py         # Risk level recurrence escalation
├── test_v4d_temporal_buffer.py         # Duration-in-state calculation & timestamp order
├── test_v4d_timestamp_robustness.py    # FPS invariance (15, 25, 30 FPS)
├── test_video_source.py                # VideoSource abstractions (Webcam, File, RTSP)
├── test_pipeline_smoke.py              # Rapid end-to-end software smoke test
├── test_acquisition_v2.py              # Preflight bounding box geometry checks
├── test_acquisition_v3.py              # SCB5 identity handling & duplicate resolution
├── test_stage1_5_pipeline.py           # Stage 1.5 holdout & duplicate hash checks
└── test_v4b_crop_dataset.py            # Crop dataset manifest & ontology checks
```

---

## 2. Test Categorization & Expected Results

When executing the complete test suite (`pytest -ra`), tests fall into two distinct categories:

### 2.1 Runtime & Operational Test Suite (100% PASS)
- **Scope**: Covers all software logic, model inference, tracking, fusion engine, event state machines, risk scoring, FastAPI REST routes, WebSocket broadcasting, and Dashboard UI rendering.
- **Result**: **196 PASSED**, **2 SKIPPED**, **0 FAILURES**.
- **Execution Command**:
  ```powershell
  .venv\Scripts\pytest.exe -k "not dataset and not v4b and not stage1_5_holdout and not processed_v3"
  ```

### 2.2 Offline Dataset-Dependent Tests (Expected Failures on Deployment)
- **Scope**: Tests located in `test_v4b_crop_dataset.py`, `test_acquisition_v3.py`, and `test_stage1_5_pipeline.py` that verify the raw offline training directory trees (`datasets/processed_v3`, `datasets/stage1_5_holdout`, `datasets/v4_crop`).
- **Why they fail on clean clone**: In accordance with privacy and repository cleanliness standards, **raw training datasets are excluded from git tracking** via `.gitignore`. The repository distributes certified frozen model checkpoints, not gigabytes of student training images.
- **Expected Failures**: ~21 dataset manifest tests fail because `datasets/` is intentionally absent on deployment workstations. This is the **correct, expected behavior** and does not impact software runtime operations.

---

## 3. Recommended Test Commands

### Run Rapid Runtime Smoke Suite (< 15 seconds)
```powershell
.venv\Scripts\pytest.exe tests/test_pipeline_smoke.py tests/test_stage2_api_contract.py tests/test_stage2_model_registry.py
```

### Run Full Stage 2 & V4D Core Suite
```powershell
.venv\Scripts\pytest.exe tests/test_stage2_*.py tests/test_v4d_*.py
```

### Run Laptop Hardware & Checkpoint Certification
```powershell
.venv\Scripts\python.exe scripts/laptop_preflight.py
```

### Run Headless Software Stack Verification
```powershell
.venv\Scripts\python.exe scripts/run_local_live_validation.py --validate-software-stack
```

---

## 4. Test Fixtures & Synthetic Data

- **Synthetic Exam Video (`tests/fixtures/sample_exam.mp4`)**: Automatically generated via `tools/dev/generate_sample_video.py`. Renders synthetic geometric student avatars without using real human imagery.
- **Conflict Quarantine Fixture (`tests/fixtures/annotation_conflicts_v3.json`)**: Contains the 2,939 quarantined multi-label conflict records used to assert negative supervision exclusion.
- **SCB-Dataset5 Metadata (`tests/fixtures/scb_dataset5_full/`)**: Contains dataset identity and statistics JSON files verifying data provenance.
