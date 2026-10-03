# Stage 2 Integrity Repair Claim Traceability Matrix

## 1. Traceability Standard
Every metric, claim, and status flag asserted in the Stage 2 Integrity Repair documentation is anchored to a physical, machine-readable artifact located on disk. No metric is derived from estimation or undocumented assertions.

---

## 2. Complete Traceability Mapping

| Claim / Metric | Value | Primary Source Artifact | Exact JSON Key Path / Evidence |
| :--- | :--- | :--- | :--- |
| **`stage1_best.pt` SHA-256** | `6d713808f0bc670e...` | `runs/stage2_integrity/detector_checkpoint_introspection.json` | `checkpoints["stage1_best.pt"].sha256` |
| **`stage1_best.pt` Taxonomy** | `{0: 'normal', 1: 'head_down', 2: 'turn_head', 3: 'discuss', 4: 'stand'}` | `runs/stage2_integrity/detector_checkpoint_introspection.json` | `checkpoints["stage1_best.pt"].model_names` |
| **`stage1_5_best.pt` SHA-256** | `68690cf82715dc6d...` | `runs/stage2_integrity/detector_checkpoint_introspection.json` | `checkpoints["stage1_5_best.pt"].sha256` |
| **`stage1_5_best.pt` Taxonomy** | `{0: 'normal', 1: 'head_down', 2: 'turn_head', 3: 'discuss', 4: 'stand'}` | `runs/stage2_integrity/detector_checkpoint_introspection.json` | `checkpoints["stage1_5_best.pt"].model_names` |
| **`v4_posture_best.pt` SHA-256** | `529a23f96ebec605...` | `runs/stage2_integrity/v4_checkpoint_identity.json` | `posture.sha256` |
| **`v4_posture_best.pt` Architecture** | `mobilenet_v3_small` (`MobileNetV3`, 1.52M params) | `runs/stage2_integrity/v4_checkpoint_identity.json` | `posture.model_name`, `posture.actual_loaded_class` |
| **`v4_posture_best.pt` Taxonomy** | `{0: 'NORMAL_UPRIGHT', 1: 'NORMAL_READ_WRITE', 2: 'HEAD_REST_SLEEP', 3: 'TURN_HEAD_CLEAR'}` | `runs/stage2_integrity/v4_checkpoint_identity.json` | `posture.class_names` |
| **`v4_headpose_yaw_best.pt` SHA-256** | `5d15eec5941cfc8d...` | `runs/stage2_integrity/v4_checkpoint_identity.json` | `headpose.sha256` |
| **`v4_headpose_yaw_best.pt` Architecture** | `hopenet_yaw` (`HopeNetYaw`, ResNet50, 23.64M params) | `runs/stage2_integrity/v4_checkpoint_identity.json` | `headpose.model_name`, `headpose.actual_loaded_class` |
| **`v4_headpose_yaw_best.pt` Native Support** | `[-99.0, 99.0]` (66 bins + continuous expectation) | `runs/stage2_integrity/v4_checkpoint_identity.json` | `headpose.native_support`, `headpose.num_bins` |
| **General Object Detector Model** | `yolo26m.pt` | `runs/stage2_integrity/runtime_role_mapping.json` | `GENERAL_OBJECT_DETECTOR.model_path` |
| **Person Class Resolution** | Class 0 (`person`) | `runs/stage2_integrity/runtime_role_mapping.json` | `GENERAL_OBJECT_DETECTOR.target_classes["person"]` |
| **Phone Class Resolution** | Class 67 (`cell phone`) | `runs/stage2_integrity/runtime_role_mapping.json` | `GENERAL_OBJECT_DETECTOR.target_classes["phone"]` |
| **Phone Object Availability** | `YES` | `runs/stage2_integrity/runtime_role_mapping.json` | `PHONE_OBJECT_DETECTION_AVAILABLE` |
| **Macro Behavior Role Model** | `models/trained/stage1_5_best.pt` | `runs/stage2_integrity/runtime_role_mapping.json` | `MACRO_BEHAVIOR_DETECTOR.model_path` |
| **Certified Macro Resolution** | 768 px | `runs/stage2_integrity/runtime_role_mapping.json` | `MACRO_BEHAVIOR_DETECTOR.certified_resolution` |
| **Mode A Whole-Loop P95** | 71.74 ms | `runs/stage2_integrity/benchmark_mode_a.json` | `whole_loop_end_to_end_ms.p95` |
| **Mode A Effective FPS** | 24.14 FPS | `runs/stage2_integrity/benchmark_mode_a.json` | `effective_fps` |
| **Mode B Whole-Loop P95** | 60.75 ms | `runs/stage2_integrity/benchmark_mode_b.json` | `whole_loop_end_to_end_ms.p95` |
| **Mode B Effective FPS** | 28.36 FPS | `runs/stage2_integrity/benchmark_mode_b.json` | `effective_fps` |
| **Mode C1 Active Tracks (Mean)** | 25.00 tracks | `runs/stage2_integrity/benchmark_mode_c_full_pipeline.json` | `track_accounting.mean_active_tracks` |
| **Mode C1 Post-Decode Mean** | 39.39 ms | `runs/stage2_integrity/benchmark_mode_c_full_pipeline.json` | `post_decode_pipeline_ms.mean` |
| **Mode C1 Whole-Loop P95** | 75.94 ms | `runs/stage2_integrity/benchmark_mode_c_full_pipeline.json` | `whole_loop_end_to_end_ms.p95` |
| **Mode C1 Effective FPS** | 20.19 FPS | `runs/stage2_integrity/benchmark_mode_c_full_pipeline.json` | `effective_fps` |
| **Mode C2 Downstream Mean** | 7.97 ms | `runs/stage2_integrity/benchmark_mode_c_20track_downstream.json` | `post_decode_pipeline_ms.mean` |
| **Mode C2 Downstream P95** | 29.61 ms | `runs/stage2_integrity/benchmark_mode_c_20track_downstream.json` | `post_decode_pipeline_ms.p95` |
| **Mode C2 Exact Track Count** | 20.00 tracks | `runs/stage2_integrity/benchmark_mode_c_20track_downstream.json` | `track_accounting.mean_active_tracks` |
| **Dense 30 FPS Line Rate Verdict** | `NO` | `runs/stage2_integrity/whole_loop_runtime.json` | `modes.mode_c1_dense_full.line_rate_verdict_30fps` |
| **Backpressure Default Depth** | 5 frames (166.7 ms) | `runs/stage2_integrity/backpressure_benchmark.json` | `selected_default_capacity` |
| **Backpressure Monotonicity** | `true` | `runs/stage2_integrity/backpressure_benchmark.json` | `queue_capacity_5.timestamp_monotonicity_preserved` |
| **Sleep Recall (1.5s window)** | 17 / 20 = 85.0% | `runs/v4d/V4D_EVALUATION_RESULTS.json` | `temporal_threshold_sweep["1.5"].clip_recall` |
| **Sleep FP Negative Clips** | 0 / 40 = 0.0% | `runs/v4d/V4D_EVALUATION_RESULTS.json` | `temporal_threshold_sweep["1.5"].total_negative_false_alarm_clips` |
| **Read/Write FP Suppression** | 100.0% | `runs/v4d/V4D_EVALUATION_RESULTS.json` | `read_write_false_positive_audit.false_positive_suppression_pct` |
| **Test Environment GPU** | NVIDIA GeForce RTX 5070 | `runs/stage2_integrity/test_environment.json` | `hardware.gpu_name` |
| **PyTorch / CUDA Version** | 2.14.1+cu130 / CUDA 13.0 | `runs/stage2_integrity/test_environment.json` | `frameworks.torch`, `frameworks.cuda_runtime` |
| **Python Version** | 3.13.9 | `runs/stage2_integrity/test_environment.json` | `python.version` |
