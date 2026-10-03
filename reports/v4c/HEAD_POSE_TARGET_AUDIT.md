# V4C Head-Pose Target Normalization & Angle Audit

**Document ID**: `reports/v4c/HEAD_POSE_TARGET_AUDIT.md`  
**Phase**: V4C Specialized Posture + Head-Pose Model Training & Benchmark  
**Date**: 2026-10-03  
**Status**: AUDITED & VERIFIED  

---

## 1. Executive Summary & Problem Formulation

Per Section 23 and 25 of the V4C Specification, the primary head-pose perception task is strictly **YAW REGRESSION** $(\theta_{yaw} \in [-90^\circ, +90^\circ])$.

In physical surveillance and classroom environments:
- **Yaw** represents horizontal head turning (looking away from desk, looking at neighbor).
- Yaw is directly physically available across **21,080 AFLW-GT images** (partitioned into 16,218 train and 2,862 val) and **2,000 AFLW2000-3D images** (held out strictly as the external test benchmark).
- Pitch and Roll are **physically absent** in AFLW-GT (count = 0). Pitch and roll exist only in AFLW2000-3D.

---

## 2. Source Dataset Physical Target Comparison

| Dataset | Physical Samples | Yaw Availability | Pitch Availability | Roll Availability | Yaw Range (deg) | Yaw Mean $\pm$ Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `AFLW2000-3D` | 2,000 | 2,000 (100%) | 2000 (100%) | 2000 (100%) | [-351.2$^\circ$, 187.8$^\circ$] | 0.81$^\circ \pm 37.47^\circ$ |
| `AFLW-GT` | 21,080 | 21,080 (100%) | **0 (ABSENT)** | **0 (ABSENT)** | [-125.1$^\circ$, 167.9$^\circ$] | 1.00$^\circ \pm 42.84^\circ$ |

---

## 3. Split Angle Distribution (Yaw in Degrees)

| Split | Source | N Samples | Min Yaw | Max Yaw | Mean Yaw | Frontal ($<15^\circ$) | Moderate ($15^\circ\text{--}45^\circ$) | Large Lateral ($\ge 45^\circ$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train` | `AFLW-GT` | 16,218 | -125.1$^\circ$ | 167.9$^\circ$ | 1.21$^\circ$ | 32.0% | 37.7% | 30.3% |
| `val` | `AFLW-GT` | 2,862 | -110.7$^\circ$ | 116.1$^\circ$ | 0.40$^\circ$ | 33.1% | 37.8% | 29.2% |
| `test` | `AFLW2000-3D` | 2,000 | -351.2$^\circ$ | 187.8$^\circ$ | 0.81$^\circ$ | 44.6% | 30.8% | 24.6% |

---

## 4. Normalization Rules & Transformation Formulation

1. **Angle Unit**: Degrees ($^\circ$). Both datasets provide yaw in degrees ($[-90^\circ, +90^\circ]$), with occasional extreme facial profile orientations up to $\pm 99^\circ$.
2. **No Blind Clamping**: Values in $[-99^\circ, +99^\circ]$ correspond to physically plausible steep profile views. No truncation is applied during training.
3. **Loss Function**: Smooth L1 (Huber) Loss or MSE on continuous angle error in degrees: $L(\theta, \hat{\theta}) = \text{SmoothL1}(\hat{\theta}_{yaw} - \theta_{yaw})$.
4. **No Horizontal Flip Augmentation for Yaw**: Horizontal flipping would invert the sign of the yaw angle $(\theta \to -\theta)$. Unless target labels are explicitly negated, horizontal flipping is strictly prohibited during head-pose training.
5. **Secondary Pitch / Roll Status**: Per Section 27, pitch and roll are physically absent in AFLW-GT and cannot be trained jointly across the dataset. The primary model is strictly dedicated to **Yaw Regression**.
