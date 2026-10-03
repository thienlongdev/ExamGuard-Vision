# Future V4C Specialized Model Benchmark Specification

**Document ID**: `reports/v4b/V4C_BENCHMARK_PLAN.md`  
**Phase**: V4B Benchmark Specification for Future V4C  
**Author**: Machine Learning Engineering & Benchmark Architecture  
**Date**: 2026-10-02  
**Status**: APPROVED SPECIFICATION (No Training / Architecture Design Only)  

---

## 1. Executive Summary

This document establishes the official experimental benchmark design for the upcoming **V4C Specialized Posture & Head-Pose Modeling Phase**.  
In strict compliance with V4B constraints, **no model training is performed in this phase**. This specification formalizes the candidate backbones, evaluation protocols, hardware budgets, and decision criteria to ensure that future model selection is empirical, reproducible, and unbiased.

---

## 2. Hardware Target & Budget (RTX 5070 / Edge Classroom Deployment)

- **Target GPU**: NVIDIA GeForce RTX 5070 (12 GB GDDR7 VRAM, Ada/Blackwell Tensor Cores)
- **Classroom Workload**: 15 to 30 simultaneous student tracks processed in real-time
- **Inference Latency Budget**: $\le 15\text{ ms}$ total per batch of 20 person crops
- **Target Real-Time Framerate**: $\ge 60\text{ FPS}$ sustained on multi-crop pipeline
- **VRAM Budget for Posture Module**: $\le 1.5\text{ GB}$ (leaving remainder for YOLO full-frame detector, ByteTrack buffers, and OS)

---

## 3. Candidate Posture Classifiers (Crop-Level)

Three distinct architecture families will be trained and evaluated head-to-head under identical data and optimization regimens in V4C:

| Candidate Model | Architecture Family | Parameter Count | Computational Cost (GFLOPs @ 224x224) | Key Architectural Mechanism | Target Operational Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A: ResNet18 + CBAM** | Residual Network with Attention | ~11.7 M | ~1.82 GFLOPs | Convolutional Block Attention Module (channel + spatial attention on feature maps) | Fast, robust medium baseline |
| **Model B: ResNet50 + CBAM** | Deep Residual Bottleneck + Attention | ~28.1 M | ~4.12 GFLOPs | 3-layer bottleneck blocks with CBAM refinement across stages 2–4 | High-capacity reference model for subtle posture distinctions |
| **Model C: MobileNetV4-Small / ConvNeXt-Femto** | Modern Ultra-Lightweight Edge Backbone | ~3.8 M | ~0.45 GFLOPs | Universal Inverted Bottleneck (UIB) with Mobile MQA attention | High-throughput real-time deployment (targets >200 FPS on batch 30) |

---

## 4. Candidate Head-Pose Estimators (Face/Head Regression)

Two specialized regression architectures will be benchmarked on continuous Euler angles:

| Model Identity | Architecture Baseline | Target Outputs | Loss Formulation | Benchmark Dataset |
| :--- | :--- | :--- | :--- | :--- |
| **Branch A: 6DRepNet** | RepVGG backbone with continuous 6D rotation representation | $\theta_{yaw}, \theta_{pitch}, \theta_{roll}$ | Geodesic Distance Loss in $SO(3)$ | AFLW / AFLW2000-3D |
| **Branch B: HopeNet (ResNet50)** | Multi-loss ResNet50 with binned classification + MSE regression | $\theta_{yaw}, \theta_{pitch}, \theta_{roll}$ | Combined Cross-Entropy + MSE Loss | AFLW / AFLW2000-3D |

---

## 5. Evaluation Partitions & Performance Metrics

To eliminate selection bias and ensure out-of-domain robustness, no candidate will be evaluated on a single mixed validation score. All models must report across 5 distinct partitions:

### 5.1 Evaluation Partitions
1. **Same-Domain Validation (`same_domain_val.jsonl`)**: Evaluates convergence and in-distribution generalization on independent images/clips from the same source families.
2. **Cross-Source Holdout (`cross_source_holdout.jsonl`)**: Evaluates transfer from SCBehavior to unseen EduAction classes and vice-versa.
3. **High-Angle Holdout (`high_angle_holdout.jsonl`)**: Tests performance degradation on steep ceiling-mounted 4K cameras ($35^\circ \text{--} 45^\circ$ downward pitch).
4. **Small-Person Slice (`PERSON_SMALL` & `PERSON_VERY_SMALL`)**: Measures accuracy on distant students ($H < 180$ px).
5. **Temporal Holdout (`temporal_holdout.jsonl`)**: Evaluates stability across completely unseen continuous video sequences.

### 5.2 Mandatory Metrics Table

| Metric Category | Specific Metrics Required | Target Threshold |
| :--- | :--- | :--- |
| **Accuracy & Discrimination** | Multi-class Macro F1, Per-class Recall (`HEAD_DOWN_DEEP`, `TURN_HEAD_CLEAR`, `NORMAL_READ_WRITE`), Confusion Matrix | Macro F1 $\ge 0.88$ |
| **Hard Negative Rejection** | False Positive Rate of `head_down` on `NORMAL_READ_WRITE` samples | $\text{FPR} \le 3.0\%$ |
| **Cross-Source Generalization**| F1 score degradation from same-domain to cross-source | $\Delta\text{F1} \le 12.0\%$ |
| **High-Angle Robustness** | F1 score on `high_angle_holdout` | F1 $\ge 0.82$ |
| **Inference Speed** | Batch-1 latency, Batch-20 latency, Native FPS (FP16 & TensorRT) | FPS $\ge 80$ |
| **Resource Footprint** | Peak GPU VRAM, Model Disk Size (ONNX / TensorRT engine) | VRAM $\le 1.5\text{ GB}$ |

---

## 6. Selection Protocol

The winning model will **NOT** be selected by accuracy alone. The decision matrix will compute a weighted utility score:

$$\text{Utility} = 0.35 \cdot \text{F1}_{\text{same}} + 0.25 \cdot \text{F1}_{\text{cross}} + 0.15 \cdot \text{F1}_{\text{high\_angle}} + 0.15 \cdot \text{SpeedScore} + 0.10 \cdot \text{SmallPersonScore}$$

Where $\text{SpeedScore} = \min\left(1.0, \frac{\text{FPS}}{100.0}\right)$.
