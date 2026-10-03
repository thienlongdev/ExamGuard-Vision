# V4C Realtime Surveillance System Computational Budget & Cadence Specification

**Document ID**: `reports/v4c/V4_RUNTIME_BUDGET.md`  
**Phase**: V4C Specialized Posture + Head-Pose Model Verification (Audited & Patched)  
**Hardware**: NVIDIA GeForce RTX 5070 (12,226.56 MiB GDDR7 VRAM, 11.94 GiB, SM 12.0)  
**Target Camera Feed**: 1080p / 4K surveillance video @ 25-30 FPS (33.3 ms to 40.0 ms frame budget)  
**Primary Posture Model**: `MobileNetV3-Small` (`models/trained/v4_posture_best.pt`)  
**Primary Head-Pose Model**: `HopeNet-Yaw` (`models/trained/v4_headpose_yaw_best.pt`)  
**Alternative Head-Pose Model**: `ResNet18-Yaw-Circular` (`runs/v4c/headpose_resnet18_yaw/best_model.pt`)  
**Date**: 2026-10-03  
**Status**: AUTHORITATIVE FEASIBILITY SPECIFICATION (RESOLVED TERMINOLOGY & SEPARATE VRAM ACCOUNTING)  

---

## 1. Executive Summary

This report defines the computational budget, execution cadences, and hardware feasibility allocation for the V4 surveillance pipeline. All component latency figures are derived from physical measurements of the frozen winner checkpoints on the NVIDIA GeForce RTX 5070.

> [!IMPORTANT]
> **Strict Runtime Terminology Boundaries**:
> The active compute metrics in this report represent the **`CORE_PERCEPTION_COMPONENT_ACTIVE_COMPUTE_ESTIMATE`** (~16.8 ms) and the theoretical **`SCHEDULED_COMPONENT_ACTIVE_COMPUTE_ESTIMATE`** (~19.5 ms under secondary phone check concurrence).
> 
> They establish component-level GPU computational feasibility under staggered cadences.
> They are **NOT**:
> - `END_TO_END_LATENCY`
> - `FULL_PIPELINE_MEASURED_LATENCY`
> - `PEAK_WALL_CLOCK_LATENCY`
> 
> These estimates do NOT include CPU preprocessing (e.g., 14.36 ms for 20 posture crops, 21.50 ms for 30 crops), Host-to-Device (H2D) PCIe memory transfers (0.53 ms - 0.78 ms), video decoding, temporal fusion, risk/event processing, evidence persistence, or WebSocket/API serialization.
> Full empirical end-to-end wall-clock pipeline measurement is reserved for Phase V4D / Stage 2:
> **`INTEGRATED_END_TO_END_RUNTIME_NOT_YET_MEASURED = TRUE`**.

---

## 2. Component Cadence & Active Compute Allocation

| Pipeline Subsystem | Execution Cadence | Measured Active Compute on RTX 5070 | Computational Role |
| :--- | :--- | :---: | :--- |
| **YOLO Full-Frame Detector** | Every 2 frames (15 Hz) | ~7.2 ms | Global student & object detection |
| **ByteTrack Associator** | Every frame (30 Hz) | ~0.8 ms | Real-time temporal track maintenance |
| **Posture Classifier (`MobileNetV3-Small`)** | Sampled: every 3 frames per student (10 Hz) | **~4.21 ms** (batch 20 GPU) | Discrete posture classification (Primary V4D Baseline) |
| **Head-Pose Estimator (`HopeNet-Yaw`)** | Sampled: every 5 frames for `HEAD_POSE_ELIGIBLE` | **~4.59 ms** (batch 10 GPU) | Continuous orientation verification |
| **Secondary Contraband/Phone Association** | Periodic: every 5 frames (optional) | ~2.5 ms | Desk-area phone & contraband association |
| **Sliding Window Temporal State Machine** | Continuous (every frame) | < 0.2 ms | Debounce, cooldown, and risk scoring |

---

## 3. Active Compute Feasibility Estimates (30 FPS, 33.3 ms Deadline)

### 3.1. Core Perception Component Active Compute Estimate
The core 4-subsystem perception cycle consists of:
```
[ ByteTrack: ~0.8 ms ] + [ Posture Batch 20 GPU: ~4.21 ms ] + [ Head-Pose 10 Heads GPU: ~4.59 ms ] + [ Full-Frame YOLO GPU: ~7.2 ms ]
CORE_PERCEPTION_COMPONENT_ACTIVE_COMPUTE_ESTIMATE = ~16.8 ms
Frame Budget Headroom: ~16.5 ms (49.5% headroom relative to 33.3 ms deadline)
```

### 3.2. Scheduled Component Active Compute Estimate (Theoretical Concurrence)
When secondary periodic phone association (~2.5 ms) and continuous temporal state updates (<0.2 ms) are scheduled concurrently with the core modules:
```
[ CORE_PERCEPTION: ~16.8 ms ] + [ Phone Association: ~2.5 ms ] + [ Temporal State Machine: < 0.2 ms ]
SCHEDULED_COMPONENT_ACTIVE_COMPUTE_ESTIMATE = ~19.5 ms
Frame Budget Headroom: ~13.8 ms (41.4% headroom relative to 33.3 ms deadline)
```

### 3.3. Tracking & Light State Update Cycle (Interleaved Frame t+1)
```
[ ByteTrack: ~0.8 ms ] + [ Temporal State Machine: < 0.2 ms ]
SCHEDULED_COMPONENT_ACTIVE_COMPUTE_ESTIMATE = ~1.0 ms
Frame Budget Headroom: ~32.3 ms (97.0% headroom relative to 33.3 ms deadline)
```

---

## 4. Crucial Engineering Distinctions & Wall-Clock Caveats

1. **Active Compute vs Empirical Wall-Clock Runtime**:
   - The ~16.8 ms sum represents isolated component GPU execution times under peak concurrence.
   - In physical execution, CPU preprocessing (14.36 ms for 20 crops, 21.50 ms for 30 crops), Host-to-Device (H2D) PCIe memory transfers (0.53 ms - 0.78 ms), video decoding, temporal state management, and IPC transfers contribute additional wall-clock latency.
   - Pipelined execution and multi-threaded asynchronous workers will overlap CPU crop preparation with GPU detection in Stage 2.

2. **Feasibility Verdict**:
   - Because the core perception active compute of ~16.8 ms occupies ~50% of the 33.3 ms frame deadline, real-time 30 FPS operation is computationally **FEASIBLE**.
   - Formal verification of end-to-end latency will be established upon full integration:
     **`INTEGRATED_END_TO_END_RUNTIME_NOT_YET_MEASURED = TRUE`**.

---

## 5. VRAM Accounting on NVIDIA GeForce RTX 5070

All VRAM quantities are reported in consistent **MiB** units based on physical measurement on the target hardware:
- **Physical Total VRAM**: **12,226.56 MiB** (11.94 GiB GDDR7)

### 5.1. Component VRAM Footprint Breakdown

| Subsystem | Peak Allocated VRAM (MiB) | Peak Reserved VRAM (MiB) | Computational Context |
| :--- | :---: | :---: | :--- |
| **Operating System & Display Server** | ~600.0 MiB | ~800.0 MiB | OS desktop & driver baseline |
| **YOLO Full-Frame Model** | ~1,200.0 MiB | ~1,600.0 MiB | Global detection tensor workspace |
| **Posture Classifier (`MobileNetV3-Small`)** | ~149.1 MiB | ~182.0 MiB | Batch 30 crop classification |
| **Head-Pose Estimator (`HopeNet-Yaw`)** | ~416.1 MiB | ~1,200.0 MiB | Batch 10 head crop orientation |
| **ByteTrack & Video Frame Buffers** | ~400.0 MiB | ~600.0 MiB | Circular frame and track memory |
| **Total Pipeline Footprint** | **~2,765.2 MiB** | **~4,382.0 MiB** | Peak pipeline footprint |

### 5.2. Separate Headroom Calculation

$$ \text{Headroom} = \text{Physical Total VRAM} - \text{Measured Footprint} $$

1. **Allocated VRAM Accounting**:
   - **Total Allocated VRAM**: **2,765.20 MiB** (22.62% of physical total)
   - **ALLOCATED HEADROOM**: $12,226.56\text{ MiB} - 2,765.20\text{ MiB} =$ **9,461.36 MiB** (**77.38%** unallocated headroom)

2. **Reserved VRAM Accounting**:
   - **Total Reserved VRAM**: **4,382.00 MiB** (35.84% of physical total)
   - **RESERVED HEADROOM**: $12,226.56\text{ MiB} - 4,382.00\text{ MiB} =$ **7,844.56 MiB** (**64.16%** unreserved headroom)

> [!NOTE]
> Previously, documentation cited "7.56 GB (63.3%)" next to an allocated footprint label; this was an accounting discrepancy caused by calculating headroom from the reserved footprint ($12,226.56 - 4,382 = 7,844.56\text{ MiB} \approx 7.66\text{ GiB}$) while presenting it adjacent to allocated figures.
> The separation above strictly resolves this:
> - Allocated headroom is **9,461.36 MiB (77.38%)**.
> - Reserved headroom is **7,844.56 MiB (64.16%)**.

---

## 6. Accounting Patch Certification

- **`V4C_RUNTIME_ACCOUNTING_PATCH = COMPLETE`**
- Terminology updated: Canonical labels `CORE_PERCEPTION_COMPONENT_ACTIVE_COMPUTE_ESTIMATE` and `SCHEDULED_COMPONENT_ACTIVE_COMPUTE_ESTIMATE` established.
- VRAM accounting unified to MiB with separate allocated and reserved headroom calculations.
