# Stage 2 Major Claim Traceability Matrix

## 1. Traceability Standard
Every numerical and architectural claim presented in the Stage 2 documentation must directly trace to:
- A machine-readable benchmark JSON artifact (`runs/stage2/integrated_runtime_benchmark.json`),
- A canonical V4D calibration artifact (`runs/v4d/V4D_EVALUATION_RESULTS.json`),
- A cryptographic hash computation,
- An automated unit/integration test execution log.

No narrative-only or ungrounded claims are permitted.

---

## 2. Claim Traceability Matrix

| # | Claim Description | Value | Source Artifact | Exact JSON Key / Source Reference | Support Type | Status |
| :- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Mode A Throughput | `24.79 FPS` | `runs/stage2/integrated_runtime_benchmark.json` | `modes.mode_a_single_student.effective_fps` | Physical Video Execution Trace | **VERIFIED** |
| **2** | Mode A Pipeline Latency (Mean) | `24.758 ms` | `runs/stage2/integrated_runtime_benchmark.json` | `modes.mode_a_single_student.component_latencies_ms.total_pipeline_ms.mean` | Physical Video Execution Trace | **VERIFIED** |
| **3** | Mode A Pipeline Latency (P95) | `31.090 ms` | `runs/stage2/integrated_runtime_benchmark.json` | `modes.mode_a_single_student.component_latencies_ms.total_pipeline_ms.p95` | Physical Video Execution Trace | **VERIFIED** |
| **4** | Mode A Peak Cadence Latency (Mean)| `31.573 ms` | `runs/stage2/integrated_runtime_benchmark.json` | `modes.mode_a_single_student.cadence_breakdown.peak_cadence_alignment_mean_ms` | Aligned 10Hz/6Hz Sub-sample | **VERIFIED** |
| **5** | Mode B Throughput | `29.59 FPS` | `runs/stage2/integrated_runtime_benchmark.json` | `modes.mode_b_medium_student.effective_fps` | Physical Video Execution Trace | **VERIFIED** |
| **6** | Mode B Pipeline Latency (Mean) | `23.084 ms` | `runs/stage2/integrated_runtime_benchmark.json` | `modes.mode_b_medium_student.component_latencies_ms.total_pipeline_ms.mean` | Physical Video Execution Trace | **VERIFIED** |
| **7** | Mode B Pipeline Latency (P95) | `28.964 ms` | `runs/stage2/integrated_runtime_benchmark.json` | `modes.mode_b_medium_student.component_latencies_ms.total_pipeline_ms.p95` | Physical Video Execution Trace | **VERIFIED** |
| **8** | Mode B Peak Cadence Latency (Mean)| `30.514 ms` | `runs/stage2/integrated_runtime_benchmark.json` | `modes.mode_b_medium_student.cadence_breakdown.peak_cadence_alignment_mean_ms` | Aligned 10Hz/6Hz Sub-sample | **VERIFIED** |
| **9** | Mode C Synthetic Throughput | `24.33 FPS` | `runs/stage2/integrated_runtime_benchmark.json` | `modes.mode_c_controlled_scaling.effective_fps` | Synthetic 20-Track Scaling Trace | **VERIFIED** |
| **10**| Mode C Pipeline Latency (P95) | `37.619 ms` | `runs/stage2/integrated_runtime_benchmark.json` | `modes.mode_c_controlled_scaling.component_latencies_ms.total_pipeline_ms.p95` | Synthetic 20-Track Scaling Trace | **VERIFIED** |
| **11**| Mode C Peak Cadence Latency (Mean)| `57.496 ms` | `runs/stage2/integrated_runtime_benchmark.json` | `modes.mode_c_controlled_scaling.cadence_breakdown.peak_cadence_alignment_mean_ms` | Aligned 10Hz/6Hz Sub-sample | **VERIFIED** |
| **12**| Peak Allocated VRAM | `415.34 MB` | `runs/stage2/integrated_runtime_benchmark.json` | `summary.peak_vram_allocated_mb` | PyTorch CUDA Telemetry | **VERIFIED** |
| **13**| Peak Reserved VRAM | `616.00 MB` | `runs/stage2/integrated_runtime_benchmark.json` | `modes.mode_c_controlled_scaling.memory.peak_gpu_reserved_vram_mb` | PyTorch CUDA Telemetry | **VERIFIED** |
| **14**| Physical Replay RAM Stability | `+15.95 MB` | `runs/stage2/integrated_runtime_benchmark.json` | `summary.ram_stability_delta_mb` | System Memory Telemetry | **VERIFIED** |
| **15**| Test Suite Pass Count | `146 passed` | Pytest Execution Log | Command: `pytest tests/ -v` | Automated Unit/Integration Suite | **VERIFIED** |
| **16**| Test Suite Failure Count | `0 failed` | Pytest Execution Log | Command: `pytest tests/ -v` | Automated Unit/Integration Suite | **VERIFIED** |
| **17**| Checkpoint Cryptographic Integrity | `6 / 6 match`| `src/orchestration/model_registry.py` | `ModelRegistry.EXPECTED_HASHES` vs file SHA-256 | SHA-256 Cryptographic Check | **VERIFIED** |
| **18**| Canonical Sleep Temporal Recall | `85.0% (17/20)`| `runs/v4d/V4D_EVALUATION_RESULTS.json` | `isolated_evaluation.positive_recall` | Raw Episodic Video Evaluation | **VERIFIED** |
| **19**| Canonical Sleep False Alarms | `0 / 40 (0.0%)`| `runs/v4d/V4D_EVALUATION_RESULTS.json` | `isolated_evaluation.total_negative_false_alarms` | Raw Episodic Video Evaluation | **VERIFIED** |
| **20**| Recommended Candidate Duration | `1.50 s` | `runs/v4d/V4D_EVALUATION_RESULTS.json` | `recommended_operating_point.candidate_duration_sec` | Multi-Objective Tie-Break Criterion | **VERIFIED** |

---

## 3. Verification Summary
- **Total Major Numeric Claims Traced**: 20
- **Unverified / Narrative Claims**: 0
- **Integrity Status**: **100% COMPLETE & VERIFIED**.
