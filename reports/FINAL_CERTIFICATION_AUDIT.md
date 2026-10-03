# ExamGuard Vision — Final Certification Audit Report
**Certification Target:** ASUS TUF Gaming A17 (FA707RC) — School Demo Build  
**Timestamp:** 2026-10-03T23:38:00+07:00  
**Audit Status:** AUDIT COMPLETE — EVIDENCE-BACKED CERTIFICATION

---

## 1. Repository State Before Audit

- **Branch:** `main` (up to date with `origin/main`)
- **Remote:** `https://github.com/thienlongdev/ExamGuard-Vision.git`
- **Head Commit:** `0851261 Initial commit - ExamGuard AI`
- **Python Runtime:** Python 3.13.11 (`.\.venv\Scripts\python.exe`)
- **PyTorch Environment:** `2.14.1+cu126`
- **CUDA Runtime:** CUDA 12.6, `torch.cuda.is_available() == True`
- **Target Hardware Accelerator:** `NVIDIA GeForce RTX 3050 Laptop GPU` (4,096 MiB VRAM)
- **Git LFS Tracked Models:**
  - `5d15eec594 * models/trained/v4_headpose_yaw_best.pt`
  - `bc31d46cfe * runs/v4c/headpose_resnet18_yaw/best_model.pt`

---

## 2. Exact Pytest Collection Result

Command executed:
```powershell
.\.venv\Scripts\pytest.exe --collect-only -q
```

- **Collected:** 219 tests (208 baseline + 11 newly authored final certification audit tests)
- **Deselected:** 0 tests
- **Collection Errors:** 0
- **Collection Warnings:** 1 (StarletteDeprecationWarning: httpx with testclient)

---

## 3. Exact Pytest Execution Result

Command executed:
```powershell
.\.venv\Scripts\pytest.exe -ra
```

### Raw Terminal Summary:
```text
=========================== short test summary info ===========================
SKIPPED [1] tests\test_headpose_angle_canonicalization.py:314: Raw AFLW configs directory not found
SKIPPED [1] tests\test_headpose_angle_canonicalization.py:357: v4_head_pose manifest not found
FAILED tests/test_acquisition_v3.py::test_duplicate_source_preference - AssertionError
FAILED tests/test_acquisition_v3.py::test_viewpoint_aware_grouping - AssertionError
FAILED tests/test_acquisition_v3.py::test_no_needs_review_mapping_enters_dataset - AssertionError
FAILED tests/test_acquisition_v3.py::test_provenance_after_fusion - FileNotFoundError
FAILED tests/test_acquisition_v3.py::test_dynamic_taxonomy_ids - AssertionError
FAILED tests/test_acquisition_v3.py::test_omitted_class_not_appearing_in_dataset_yaml - AssertionError
FAILED tests/test_stage1_5_pipeline.py::test_stage1_5_holdout_integrity - AssertionError
FAILED tests/test_stage1_5_pipeline.py::test_stage1_5_zero_group_leakage - FileNotFoundError
FAILED tests/test_stage1_5_pipeline.py::test_stage1_5_zero_duplicate_hash_leakage - FileNotFoundError
FAILED tests/test_stage1_5_pipeline.py::test_stage1_5_dataset_class_ids_and_geometry - FileNotFoundError
FAILED tests/test_v4b_crop_dataset.py::test_v4b_manifest_files_exist_and_readable - AssertionError
FAILED tests/test_v4b_crop_dataset.py::test_v4b_crop_files_physical_existence_and_normalization - AssertionError
FAILED tests/test_v4b_crop_dataset.py::test_v4b_ontology_values_and_no_cheating_labels - AssertionError
FAILED tests/test_v4b_crop_dataset.py::test_v4b_split_leakage_zero_image_and_clip_overlap - AssertionError
FAILED tests/test_v4b_crop_dataset.py::test_v4b_split_leakage_zero_duplicate_hash_overlap - FileNotFoundError
FAILED tests/test_v4b_crop_dataset.py::test_v4b_head_pose_split_safety_and_provenance - FileNotFoundError
FAILED tests/test_v4b_crop_dataset.py::test_v4b_head_pose_angle_validity_no_nan_inf - FileNotFoundError
FAILED tests/test_v4b_crop_dataset.py::test_v4b_quarantine_exclusion_from_supervised_classes - FileNotFoundError
FAILED tests/test_v4b_crop_dataset.py::test_v4b_license_field_and_provenance - FileNotFoundError
FAILED tests/test_v4b_crop_dataset.py::test_v4b_crop_geometry_validity - FileNotFoundError
FAILED tests/test_v4b_crop_dataset.py::test_v4b_v3_benchmark_and_checkpoints_protection - AssertionError
====== 21 failed, 196 passed, 2 skipped, 1 warning in 105.65s (0:01:45) =======
```

- **Passed:** 196
- **Failed:** 21
- **Skipped:** 2
- **XFailed:** 0
- **XPassed:** 0
- **Total Executed:** 219

---

## 4. Dataset-Dependent Test Audit & Resolution of Inconsistency

### Resolution of Prior Report Contradiction:
The previous report stated `"185 passed, 2 skipped"` alongside `"21 skipped/failed tests relate solely to external offline dataset directories"`.
This was factually imprecise and contradictory. In reality, the full canonical test suite on this repository runs 208 items (now 219 items with audit additions):
- **196 passed**: 100% of runtime orchestration, physical video ingestion, YOLO detector, ByteTrack, posture classification, HopeNet head-pose, V4D fusion, event state machines, risk scoring, evidence capture, FastAPI HTTP endpoints, WebSocket broadcasting, and dashboard contracts passed.
- **2 skipped**: Explicitly skipped using `pytest.skip` due to absent AFLW/v4_head_pose offline directories.
- **21 failed**: Failed assertions/FileNotFoundError because offline research dataset paths in `.gitignore` are not checked into the deployment repo.

None of the 21 tests were hidden, weakened, or deleted.

### Classification Table:

| Test Name / File | Collected? | Executed? | Result | Classification & Exact Cause |
|---|---|---|---|---|
| `test_acquisition_v3.py::test_duplicate_source_preference` | Yes | Yes | FAILED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: Offline dataset file `datasets/processed_v3/manifest_v3.json` absent (`.gitignore` line 38) |
| `test_acquisition_v3.py::test_viewpoint_aware_grouping` | Yes | Yes | FAILED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: `datasets/processed_v3/manifest_v3.json` absent |
| `test_acquisition_v3.py::test_no_needs_review_mapping_enters_dataset` | Yes | Yes | FAILED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: `datasets/processed_v3/manifest_v3.json` absent |
| `test_acquisition_v3.py::test_provenance_after_fusion` | Yes | Yes | FAILED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: `datasets/processed_v3/dataset.yaml` absent |
| `test_acquisition_v3.py::test_dynamic_taxonomy_ids` | Yes | Yes | FAILED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: `datasets/processed_v3/dataset.yaml` absent |
| `test_acquisition_v3.py::test_omitted_class_not_appearing_in_dataset_yaml` | Yes | Yes | FAILED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: `datasets/processed_v3/dataset.yaml` absent |
| `test_stage1_5_pipeline.py::test_stage1_5_holdout_integrity` | Yes | Yes | FAILED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: `datasets/stage1_5/manifest.json` absent |
| `test_stage1_5_pipeline.py::test_stage1_5_zero_group_leakage` | Yes | Yes | FAILED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: `datasets/stage1_5/manifest.json` absent |
| `test_stage1_5_pipeline.py::test_stage1_5_zero_duplicate_hash_leakage` | Yes | Yes | FAILED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: `datasets/stage1_5/manifest.json` absent |
| `test_stage1_5_pipeline.py::test_stage1_5_dataset_class_ids_and_geometry` | Yes | Yes | FAILED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: `datasets/stage1_5/manifest.json` absent |
| `test_v4b_crop_dataset.py::test_v4b_manifest_files_exist_and_readable` | Yes | Yes | FAILED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: `datasets/v4_crop/manifest.jsonl` absent |
| `test_v4b_crop_dataset.py::test_v4b_crop_files_physical_existence_and_normalization` | Yes | Yes | FAILED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: `datasets/v4_crop/manifest.jsonl` absent |
| `test_v4b_crop_dataset.py::test_v4b_ontology_values_and_no_cheating_labels` | Yes | Yes | FAILED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: `datasets/v4_crop/manifest.jsonl` absent |
| `test_v4b_crop_dataset.py::test_v4b_split_leakage_zero_image_and_clip_overlap` | Yes | Yes | FAILED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: `datasets/v4_crop/splits` absent |
| `test_v4b_crop_dataset.py::test_v4b_split_leakage_zero_duplicate_hash_overlap` | Yes | Yes | FAILED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: `datasets/v4_crop/splits/train.jsonl` absent |
| `test_v4b_crop_dataset.py::test_v4b_head_pose_split_safety_and_provenance` | Yes | Yes | FAILED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: `datasets/v4_head_pose/splits/train.jsonl` absent |
| `test_v4b_crop_dataset.py::test_v4b_head_pose_angle_validity_no_nan_inf` | Yes | Yes | FAILED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: `datasets/v4_head_pose/manifest.jsonl` absent |
| `test_v4b_crop_dataset.py::test_v4b_quarantine_exclusion_from_supervised_classes` | Yes | Yes | FAILED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: `datasets/v4_crop/manifest.jsonl` absent |
| `test_v4b_crop_dataset.py::test_v4b_license_field_and_provenance` | Yes | Yes | FAILED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: `datasets/v4_crop/manifest.jsonl` absent |
| `test_v4b_crop_dataset.py::test_v4b_crop_geometry_validity` | Yes | Yes | FAILED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: `datasets/v4_crop/manifest.jsonl` absent |
| `test_v4b_crop_dataset.py::test_v4b_v3_benchmark_and_checkpoints_protection` | Yes | Yes | FAILED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: `datasets/processed_v3` absent |
| `test_headpose_angle_canonicalization.py::test_aflw_pose_array_schema` | Yes | Skipped | SKIPPED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: Explicit skip on absent `datasets/raw_v4/head_pose_aflw2000/configs` |
| `test_headpose_angle_canonicalization.py::test_yaw_array_length_consistency` | Yes | Skipped | SKIPPED | `ENVIRONMENT_DEPENDENT_NOT_AVAILABLE`: Explicit skip on absent `datasets/v4_head_pose/manifest.jsonl` |

**Suite Assessment:** All 21 failures and 2 skips belong strictly to **offline research dataset preprocessing and historical split validation**. The entire **runtime perception and school-demo delivery test surface (196 tests)** passes unconditionally.

---

## 5. Source Origin Semantics Audit & Fix

### Audit Finding:
- Previously, `FusedEvent.event_origin` defaulted to `"LIVE_OBSERVATION"`, and certain validation scripts manually tagged `"origin": "PHYSICAL_LIVE_CAMERA"`.
- `EventEngine` and `TrackEventStateMachine` did not dynamically resolve origin from the incoming `UnifiedTrackUpdate` or video source context.

### Hardened Architecture:
1. Introduced canonical `SourceOrigin` Enum in `src.fusion.types`:
   - `PHYSICAL_LIVE_CAMERA`
   - `SOFTWARE_VALIDATION_FIXTURE`
   - `REPLAY_STREAM`
   - `VIDEO_FILE`
   - `RTSP_STREAM`
   - `SYNTHETIC_TEST`
   - `UNKNOWN`
2. Propagated origin through the canonical chain:
   `VideoSource` -> `Stage2Pipeline` -> `UnifiedTrackUpdate.source_origin` -> `MultiCueFusionEngine.update_track` -> `PerTrackCueState.source_origin` -> `TrackEventStateMachine.process_frame` -> `FusedEvent.event_origin` / `obs_snapshot["origin"]` -> `FastAPI / API Response`.
3. Fallback compatibility: If origin is omitted in legacy tests or synthetic records, it defaults to `"UNKNOWN"` and is **never falsely elevated** to `PHYSICAL_LIVE_CAMERA`.
4. Dedicated tests added in `tests/test_final_certification_audit.py` proving:
   - Physical webcam -> `PHYSICAL_LIVE_CAMERA`
   - Software validation fixture -> `SOFTWARE_VALIDATION_FIXTURE`
   - Synthetic/missing origin -> `UNKNOWN`
   - REST API serialization preserves origin truth.

---

## 6. Evidence Route Ambiguity & Hardening Audit

### Audit Finding:
- The legacy evidence endpoint `/api/evidence/{file_path:path}` previously used a naive glob fallback `**/{filename}` that would pick the first file found across multiple session directories.
- If multiple sessions had `snapshots/event_0001.jpg`, the endpoint could silently serve evidence from the wrong session.

### Hardened Implementation:
1. **Canonical Subpath Addressing:**
   - Evidence paths preserve session subfolders (e.g., `/api/evidence/asus_a17_demo/snapshots/{id}.jpg`).
   - Direct resolution resolves against `allowed_roots` using `.resolve().relative_to(root)` in constant time without search.
2. **Ambiguity Prevention (HTTP 409 Conflict):**
   - If a legacy filename-only query matches `> 1` files across different session directories, the server returns `HTTP 409 Conflict` with detail `Ambiguous evidence reference '<filename>'. Multiple conflicting evidence records exist.`
   - No arbitrary selection occurs.
   - Internal filesystem paths are never leaked to the client.
3. **Traversal Security & Path Hardening:**
   - Multi-pass URL decoding (3 passes) defeats single and double URL encodings (`%2e%2e`, `%252e%252e`).
   - Explicit rejection of `..`, `%2e`, drive letters (`C:`), UNC paths (`\\`, `//`), and leading slashes.
   - Whitelist of permitted extensions: `.jpg`, `.jpeg`, `.png`, `.mp4`, `.json`.
4. **Physical Event to JPEG Content Correctness:**
   - Verified that `event_id` in metadata links deterministically to the relative snapshot file, which serves the exact matching binary JPEG payload over HTTP.

---

## 7. Screenshot Visual Inspection Audit

All 7 certification screenshots exist physically in `reports/` and were visually inspected:

| Screenshot | Resolution | Size | Visual QA Result | Key Observations |
|---|---|---|---|---|
| `reports/final_monitor_normal.png` | 1920x1080 | 161 KB | **PASS** | Clean 16:9 stream; no burned-in HUD; WS LIVE green indicator; CAM 01 (PHYSICAL) badge; 4 KPI counts aligned; clean typography. |
| `reports/final_monitor_phone_event.png` | 1920x1080 | 119 KB | **PASS** | Drawer open; PHONE_ASSOCIATED; HIGH Evidence Risk (88/100); thumbnail loaded; event cues displayed; human review actions present. |
| `reports/final_monitor_head_rest.png` | 1920x1080 | 121 KB | **PASS** | Drawer open; SUSTAINED_HEAD_REST; MEDIUM Evidence Risk (65/100); HEAD_REST_SLEEP (96%); human review status awaiting. |
| `reports/final_event_drawer.png` | 1920x1080 | 121 KB | **PASS** | Close-up of drawer panel; clean metadata blocks; cues table; notes input field; Confirm/Dismiss buttons. |
| `reports/final_review_view.png` | 1920x1080 | 436 KB | **PASS** | Full audit log table; thumbnails rendering; human confirmed vs dismissed badges; 4 summary KPIs; zero dead space. |
| `reports/final_system_view.png` | 1920x1080 | 365 KB | **PASS** | System view; 4 status cards; model registry info (YOLO, ByteTrack, Posture, HopeNet, Stage1.5); real performance charts; zero dead space. |
| `reports/final_1366x768.png` | 1366x768 | 134 KB | **PASS** | Scaled laptop layout; zero horizontal overflow or clipping; 16:9 camera preserved; review queue and timeline fully visible. |

---

## 8. Physical Live Smoke Validation (ASUS TUF Gaming A17)

- **Execution Command:** `& ".\.venv\Scripts\python.exe" scripts/run_asus_a17_demo.py --headless --duration 65`
- **Elapsed Time:** 65.0 seconds
- **Webcam Ingestion:** Index 0 via `CAP_DSHOW` (1280x720 @ 30.0 FPS)
- **Frames Captured:** 646 frames
- **Frames Processed:** 646 frames
- **Frames Dropped:** 0 (0.0% drop rate, bounded queue strictly respected)
- **Cadence / Latency:** 10.0 FPS scheduled inference, p50 latency 93.0–96.8 ms
- **Target Hardware Accelerator:** `NVIDIA GeForce RTX 3050 Laptop GPU` (`cuda:0`)
- **VRAM Utilization:** 289.3 MB allocated out of 4,096 MB capacity (> 3,800 MB safe headroom)
- **Host RAM:** 1,745.8 MB RSS (zero leak observed)
- **FastAPI Backend:** Healthy on `http://127.0.0.1:8000`
- **WebSocket Health:** `ws://127.0.0.1:8000/ws/events` connected, tested reconnect scenario cleanly without event duplication.
- **Hardware Shutdown:** OpenCV VideoCapture released cleanly, all background thread workers shut down gracefully.

---

## 9. Machine-Specific Source Hardcoding Audit

Grep search for machine-specific path strings (`E:\WorkingSpace`, `C:\Users`) across `src/`, `configs/`, and `scripts/`:
- **Result:** ZERO occurrences.
- All file paths use repository-relative paths, `pathlib.Path`, or runtime configuration resolution.

---

## 10. Git Diff & Working Tree State

Modified files are strictly scoped to certification hardening and verification:
- `src/fusion/types.py`: `SourceOrigin` enum, `source_origin` on `UnifiedTrackUpdate`, safe default on `FusedEvent`
- `src/fusion/__init__.py`: Export `SourceOrigin`
- `src/fusion/cue_state.py`: `source_origin` on `PerTrackCueState`
- `src/fusion/fusion_engine.py`: Propagate origin through temporal fusion
- `src/fusion/event_engine.py`: Propagate origin to snapshots and events
- `src/fusion/replay.py`: Parse origin from replay records
- `src/video/webcam.py`, `video_file.py`, `rtsp.py`: Explicit provenance tagging
- `src/orchestration/stage2_pipeline.py`: VideoSource-to-pipeline origin derivation
- `src/behavior/event_manager.py`: Safe default origin `UNKNOWN`
- `src/api/main.py`: Path traversal protection, 409 ambiguity handling, session subpath preservation
- `scripts/run_asus_a17_demo.py`: Optional `--duration` argument for automated validation
- `tests/test_final_certification_audit.py`: 11 certification tests

---

## 11. Final Readiness Flags

| Readiness Flag | Value | Verification Evidence |
|---|---|---|
| `WEBSOCKET_REALTIME_PASS` | **PASS** | Live connection established and verified on `ws://127.0.0.1:8000/ws/events` |
| `WEBSOCKET_RECONNECT_PASS` | **PASS** | Disconnect -> reconnect cycle executed during live physical run without errors |
| `EVIDENCE_CAPTURE_PASS` | **PASS** | Physical JPEG snapshot generated and stored under session folder |
| `EVIDENCE_HTTP_SERVING_PASS` | **PASS** | Canonical direct route verified (`200 OK`, verified JPEG header bytes) |
| `EVENT_SOURCE_ORIGIN_PASS` | **PASS** | Full provenance propagation verified; fixtures cannot masquerade as physical |
| `EVENT_TIME_CUE_SNAPSHOT_PASS` | **PASS** | Immutable observation snapshot preserved across event updates and closures |
| `EVENT_LIFECYCLE_SEMANTICS_PASS` | **PASS** | Automated event closure leaves human review status in `awaiting` |
| `HUMAN_REVIEW_SEMANTICS_PASS` | **PASS** | `PATCH /api/events/{id}` persists status and notes |
| `KPI_CONSISTENCY_PASS` | **PASS** | Header and review table counts match active event store |
| `EVENT_DEDUPLICATION_PASS` | **PASS** | Temporal state machine debounces sustained behaviors into single open/close cycles |
| `CLEAN_PRODUCT_CAMERA_STREAM_PASS` | **PASS** | Browser stream clean of burned-in debug text; overlay decoupled |
| `RUNTIME_REGRESSION_SUITE_PASS` | **PASS** | 196 runtime/demo-relevant tests passed |
| `OFFLINE_DATASET_VALIDATION` | **NOT_AVAILABLE_ON_DEPLOYMENT_MACHINE** | Required research datasets are intentionally absent from deployment machine |
| `FULL_PYTEST_SUITE_ALL_GREEN` | **NO** | 196 passed, 21 failed, 2 skipped |
| `PHYSICAL_LIVE_REGRESSION_PASS` | **PASS** | 65.0s physical webcam smoke completed on RTX 3050 (0 dropped frames, 289 MB VRAM) |
| `READY_FOR_SCHOOL_DEMO` | **YES** | Certified for ASUS A17 live laptop school demonstration |
| `READY_FOR_PRODUCTION` | **NO** | Single-camera local demo build; enterprise multi-room auth/audit out of scope |
