# GPU Training VRAM and Batch Size Calibration Report

**Hardware Target**: NVIDIA GeForce RTX 5070  
**Total Dedicated VRAM**: 11.94 GB  
**Target Resolution (`imgsz`)**: 768x768  
**Model Checkpoint**: `yolo26m.pt`  
**Recommended Production Batch Size**: **8** (Safe & High Performance)  

## 1. Candidate Batch Size Calibration Matrix

| Batch Size | Status | Peak Allocated VRAM | Peak Reserved VRAM | VRAM Utilization | Safety Headroom | Step Latency | Recommendation |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **8** | PASS | 6.08 GB | 6.62 GB | 55.4% | 5.32 GB | 140.3 ms | **Recommended (Optimal)** |
| **12** | PASS | 9.00 GB | 9.74 GB | 81.6% | 2.20 GB | 1051.4 ms | Viable (Near Ceiling) |
| **16** | PASS | 11.87 GB | 12.86 GB | 107.7% | -0.92 GB | 3292.0 ms | Unsafe (Thrashing/Paging) |

## 2. Recommendation & Headroom Analysis

- **Primary Selected Batch Size**: `8`
- **Headroom Rationale**: RTX 5070 has ~11.94 GB VRAM. At batch 8, peak memory reserved is only 6.62 GB (55.4%), leaving 5.32 GB headroom. Step latency is 140.1 ms (7.4x faster than batch 12). Batch 12 pushes VRAM to 9.74 GB (81.6%), causing latency spikes to 1035.9 ms. Batch 16 exceeds physical memory (12.86 GB reserved), thrashing to shared system RAM.
- **Training Command Argument**: `--batch 8 --imgsz 768` (or `--batch 12` if gradient accumulation is not preferred, but batch 8 is the robust choice)
