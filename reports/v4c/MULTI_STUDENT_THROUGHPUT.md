# V4C Multi-Student Posture Inference Throughput Benchmark

**Document ID**: `reports/v4c/MULTI_STUDENT_THROUGHPUT.md`  
**Phase**: V4C Specialized Posture + Head-Pose Model Verification  
**Hardware**: NVIDIA GeForce RTX 5070 (11.94 GB GDDR7 VRAM, SM 12.0)  
**Posture Model Tested**: `models/trained/v4_posture_best.pt` (**MobileNetV3-Small**, 1.52M params, 224x224)  
**Head-Pose Model Tested**: `models/trained/v4_headpose_yaw_best.pt` (**HopeNet-Yaw**, 23.6M params) & `ResNet18-Yaw-Circular` (11.2M params)  
**Date**: 2026-10-03  
**Status**: AUTHORITATIVE BENCHMARK WITH STRICT CUDA SYNCHRONIZATION  

---

## 1. Executive Summary

Per the V4C Specification, runtime benchmarks use the **ACTUAL final frozen winners** (`models/trained/v4_posture_best.pt` = `MobileNetV3-Small` @ 224x224), completely resolving previous report contradictions that referenced ResNet-18 figures. All GPU benchmarks were executed on the NVIDIA GeForce RTX 5070 with strict `torch.cuda.synchronize()` before and after execution across 25 warmup iterations and 100 timed repetitions.

---

## 2. Actual Posture Winner: MobileNetV3-Small (224x224)

| Batch Size | CPU Preprocess (ms) | H2D Transfer (ms) | GPU Inference Mean (ms) | GPU Inference Median (ms) | GPU Inference P95 (ms) | GPU Postprocess (ms) | Peak VRAM (MB) | Effective Throughput (img/s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.70 ms | 0.066 ms | **4.008 ms** | 4.040 ms | 4.373 ms | 0.124 ms | 102.9 MB | **249.5 img/s** |
| **5** | 3.26 ms | 0.160 ms | **4.037 ms** | 3.995 ms | 4.619 ms | 0.041 ms | 108.7 MB | **1,238.5 img/s** |
| **10** | 6.72 ms | 0.267 ms | **3.991 ms** | 3.935 ms | 4.548 ms | 0.043 ms | 115.6 MB | **2,505.6 img/s** |
| **20** | 14.36 ms | 0.534 ms | **4.212 ms** | 4.126 ms | 5.004 ms | 0.042 ms | 131.7 MB | **4,748.3 img/s** |
| **30** | 21.50 ms | 0.784 ms | **4.166 ms** | 4.110 ms | 4.531 ms | 0.037 ms | 149.1 MB | **7,201.2 img/s** |

---

## 3. Posture Winner vs ResNet18+CBAM Comparison

| Model | Batch 1 GPU (ms) | Batch 10 GPU (ms) | Batch 20 GPU (ms) | Batch 30 GPU (ms) | Batch 30 Throughput | Parameters | Checkpoint Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`MobileNetV3-Small` (Winner)** | **4.01 ms** | **3.99 ms** | **4.21 ms** | **4.17 ms** | **7,201.2 img/s** | **1.52M** | **17.66 MB** (Weights: 5.8 MB) |
| `ResNet18+CBAM` (Reference) | 4.50 ms | 4.81 ms | 5.12 ms | 6.70 ms | 4,476.9 img/s | 11.2M | 129.09 MB (Weights: 43.1 MB) |

---

## 4. Head-Pose Load Scaling

| Model | 1 Head (ms) | 5 Heads (ms) | 10 Heads (ms) | 20 Heads (ms) | 10 Heads Throughput | Peak VRAM | Native Domain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`HopeNet-Yaw` (ResNet50)** | 3.82 ms | 3.62 ms | **4.59 ms** | 8.88 ms | 2,179.2 heads/s | 416.1 MB | Common $[-99^\circ, +99^\circ)$ |
| **`ResNet18-Yaw-Circular`** | 1.64 ms | 1.64 ms | **1.96 ms** | 3.20 ms | 5,108.7 heads/s | 391.0 MB | Full $360^\circ$ Continuous |

---

## 5. Architectural Findings & Physical Boundary Interpretation

1. **Measured Posture GPU Latency**:
   - The authoritative posture winner `MobileNetV3-Small` (224x224) achieves a mean GPU inference latency of **4.166 ms** (median: **4.110 ms**, P95: **4.531 ms**) for an entire classroom batch of 30 students.
   - GPU inference throughput reaches **7,201.2 img/s** under batch 30 execution with a compact peak VRAM footprint of **149.1 MB**.
   *(Note: Obsolete preliminary drafts claiming "0.95 ms" or "> 30,000 crops/sec" are formally struck as un-synchronized or un-batched artifacts).*

2. **CPU Preprocessing & Host-to-Device Bottlenecks**:
   - For a full 30-student classroom, single-threaded CPU preprocessing requires **21.50 ms** and Host-to-Device (H2D) PCIe transfer requires **0.784 ms**.
   - Total sequential pipeline time (CPU + H2D + GPU + Post) is **26.49 ms**.
   - This proves that in multi-student scenarios, CPU preprocessing dominates wall-clock time if unpipelined. In Phase V4D / Stage 2, CPU preprocessing must be parallelized across worker threads or overlapped via CUDA streams.

3. **Head-Pose Inference Efficiency**:
   - For a typical classroom load of 10 candidate turning heads:
     - `ResNet18-Yaw-Circular` executes in **1.96 ms** GPU inference (yielding **5,108.7 heads/s**).
     - `HopeNet-Yaw` executes in **4.59 ms** GPU inference (yielding **2,179.2 heads/s**).
   - Both models fit comfortably within real-time budgets, but ResNet18 offers a **2.3x latency advantage** and saves 25 MB of VRAM.

4. **Component vs Integrated Pipeline Distinction**:
   - The figures in this benchmark represent **isolated component-level active compute times** measured with strict synchronization.
   - They confirm algorithmic and hardware **feasibility**, but do **NOT** represent an integrated end-to-end wall-clock latency measurement. Full pipeline wall-clock measurement is an explicit Stage 2 responsibility.
