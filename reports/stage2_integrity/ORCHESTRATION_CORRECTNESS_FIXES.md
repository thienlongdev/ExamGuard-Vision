# Stage 2 Orchestration Correctness & Integrity Repairs

## 1. Overview
A direct source-code forensic audit identified 14 critical integrity defects in the initial Stage 2 implementation. All 14 defects have been physically repaired in code without retraining any perception models and without downloading external assets.

---

## 2. Summary of 14 Integrity Repairs

| Defect # | Area | Defect Identified | Corrective Implementation | Verification Method |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Taxonomy Assumptions** | Assumed `stage1_best.pt` has COCO person (0) and phone (67). | Physically verified `stage1_best.pt` has 5 behavior classes (`normal`, `head_down`, `turn_head`, `discuss`, `stand`). Dynamically resolved person/phone from physical COCO model `yolo26m.pt`. | `tests/test_stage2_integrity.py::test_detector_taxonomy_introspection_stage1_and_stage1_5` |
| **2** | **Architecture Metadata** | Synthetic architecture labeling (e.g. YOLO11n / YOLO26m) without proof. | Registry extracts exact model graph descriptors, parameter counts, strides, and versions directly from checkpoints. | `src/orchestration/model_registry.py` |
| **3** | **Resolution Provenance** | Ran both detectors at 640px despite `stage1_5_best.pt` being certified at 768px. | Decoupled configuration: `general_object_detector.image_size: 640` vs `macro_behavior_detector.image_size: 768`. | `configs/stage2_pipeline.yaml` |
| **4** | **Latency Definition** | Post-decode processing was incorrectly labeled as "end-to-end". | Canonical wall-clock $t_0$ immediately before `source.read()` to $t_1$ after result publication preparation (`whole_loop_end_to_end_ms`). | `scripts/benchmark_stage2_end_to_end.py` |
| **5** | **Fabricated Splits** | Synthetic 50% fusion / 30% event / 20% evidence decomposition and constant 0.05 ms serialization. | Deleted synthetic percentage logic. Physically instrumented every module boundary with high-resolution timers. | `src/orchestration/stage2_pipeline.py` |
| **6** | **Macro-Detector Timing** | Macro timing was unmeasured or conflated with general detection. | Isolated macro detector execution timer (`macro_behavior_ms`). | `runs/stage2_integrity/component_timing.json` |
| **7** | **Cadence Mismatch** | `macro_hz: 6.0` was configured but executed every single frame. | Cadence gating implemented with cue caching and explicit `ObservationStatus` tracking. | `tests/test_stage2_integrity.py::test_macro_cadence_gating` |
| **8** | **Backpressure Defect** | Execution was synchronous; queue was unused. | Implemented `BoundedFrameQueue` with `DROP_STALE_ON_BACKPRESSURE`, drop count, drop log, and queue depth telemetry. | `tests/test_stage2_integrity.py::test_bounded_queue_drop_oldest` |
| **9** | **Risk Aggregation Defect** | Hardcoded `active_cues_count = 1`, `independent_cues_count = 1`. | Derived real supporting cues and clustered correlated cues (`ORIENTATION_CLUSTER` for turn+yaw). | `tests/test_stage2_integrity.py::test_multi_cue_risk_independent_clustering` |
| **10** | **Track State Defect** | Hardcoded `track_age_frames = 1`, `time_since_seen_sec = 0.0`. | Persistent per-track metadata dictionary maintains real age, last-seen timestamps, and missing durations. | `tests/test_stage2_integrity.py::test_real_track_age_and_time_since_seen` |
| **11** | **Scale Gating Divergence** | Permissive scale thresholds diverged from V4C evidence. | Enforced conservative scale gating: $\ge 120$ px primary 224; $60 \le h < 120$ px reduced reliability (0.60); $< 60$ px sub-resolution. Head $< 25\times 25$ px unresolvable. | `src/orchestration/crop_scheduler.py` |
| **12** | **Dense Track Accounting** | Synthetically tiled scenes claimed "20 tracks" regardless of detector detections. | Accurate accounting: records actual detector person detections and active tracks (mean, median, P05, P95, min, max). | `runs/stage2_integrity/benchmark_mode_c_full_pipeline.json` |
| **13** | **Memory Stability Defect** | 150-frame test showed RSS growth without proving stability. | Conducted 10,000-frame long-duration soak test to verify plateau and calculate slope in second half and last 25%. | `scripts/run_memory_soak.py` |
| **14** | **Stale V4D Traceability** | Referenced non-existent JSON keys (`isolated_evaluation.*`). | Reconciled against physical keys in `runs/v4d/V4D_EVALUATION_RESULTS.json` (`sleep_temporal_calibration`, `temporal_threshold_sweep`). | `reports/stage2_integrity/V4D_TRACEABILITY_RECONCILIATION.md` |

---

## 3. Governance Compliance
All fixes strictly preserve:
- Zero model retraining.
- Zero weight modification.
- Zero downloading of external models.
- Objective, observable evidence language (prohibiting `CHEATING`, `CHEATER`, `GUILTY`, `FRAUD`).
