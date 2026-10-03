# Head-Pose Angle Distribution Analysis

**Document ID**: `reports/v4b/HEAD_POSE_ANGLE_DISTRIBUTION.md`  
**Phase**: V4B — Clean Person-Crop / Head-Pose Dataset Construction  
**Author**: Data Engineering & Audit Pipeline  
**Date**: 2026-10-02  
**Status**: EMPIRICALLY AUDITED  

---

## 1. Executive Summary

This report documents the statistical distribution of continuous Euler angles $(\theta_{yaw}, \theta_{pitch}, \theta_{roll})$ across the physical V4 Head-Pose dataset (`datasets/v4_head_pose/`).  

> [!IMPORTANT]
> **Boundary Notice**: This document presents physical empirical distributions only. **No operational behavior classification thresholds or cheating risk boundaries are defined herein.** Threshold calibration is strictly reserved for subsequent fusion experiments in V4C/V4D.

---

## 2. Global Yaw Angle Distribution ($N = 23,080$)

Continuous yaw angles $(\theta_{yaw})$ represent horizontal head rotation. Yaw is the primary geometric feature for lateral glance and turn-head modeling.

### 2.1 Overall Statistics
- **Sample Count**: 23,080
- **Minimum**: $-351.23^\circ$ (extreme profile / 3DMM Euler wrap)
- **Maximum**: $+187.79^\circ$
- **Mean**: $+0.98^\circ$ (substantially symmetric around center line)
- **Standard Deviation**: $38.45^\circ$

### 2.2 Coverage by Orientation Bins

| Yaw Bin Designation | Absolute Angular Range ($|\theta_{yaw}|$) | Sample Count | Percentage | Physical Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Frontal** | $[0^\circ, 15^\circ)$ | **7,901** | **34.23%** | Direct forward gaze, upright desk focus |
| **Moderate Lateral** | $[15^\circ, 45^\circ)$ | **8,521** | **36.92%** | Moderate head turn, glancing towards adjacent desk/board |
| **Large Lateral** | $[45^\circ, 180^\circ]$ | **6,658** | **28.85%** | Profile view, severe turn away from forward axis |
| **Total** | — | **23,080** | **100.00%** | Full $360^\circ$ coverage with dense sampling |

The distribution provides balanced coverage across all three operational regimes: approximately one-third frontal, one-third moderate lateral, and nearly thirty percent large lateral profile.

---

## 3. Pitch Angle Distribution ($N = 2,000$, AFLW2000-3D)

Continuous pitch angles $(\theta_{pitch})$ represent vertical nodding/tilt. Pitch was geometrically derived from the 68 3D landmark points $(X, Y, Z)$ of AFLW2000-3D.

### 3.1 Overall Statistics
- **Sample Count**: 2,000 (AFLW2000-3D benchmark)
- **Minimum**: $-19.57^\circ$ (looking up towards ceiling/high camera)
- **Maximum**: $+72.60^\circ$ (deep downward tilt towards desk/lap)
- **Mean**: $+21.01^\circ$ (natural downward incline of natural portraiture)
- **Standard Deviation**: $14.12^\circ$

### 3.2 Pitch Bins

| Pitch Bin Designation | Angular Range ($\theta_{pitch}$) | Sample Count | Percentage | Physical Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Upward Tilt** | $(-\infty, -10^\circ)$ | **16** | **0.80%** | Head tilted upward towards ceiling or high fixture |
| **Neutral / Level** | $[-10^\circ, +15^\circ]$ | **614** | **30.70%** | Level eye-line, horizontal blackboard view |
| **Downward Tilt** | $(+15^\circ, +\infty)$ | **1,370** | **68.50%** | Head inclined down towards desk, paper, or lap |
| **Total** | — | **2,000** | **100.00%** | High density in reading/writing downward tilt range |

---

## 4. Roll Angle Distribution ($N = 2,000$, AFLW2000-3D)

Continuous roll angles $(\theta_{roll})$ represent in-plane head tilt (ear to shoulder). Roll was derived from the inter-ocular axis orientation.

### 4.1 Overall Statistics
- **Sample Count**: 2,000
- **Minimum**: $-172.02^\circ$
- **Maximum**: $+179.16^\circ$
- **Mean**: $-0.70^\circ$ (highly symmetric around vertical axis)
- **Standard Deviation**: $21.34^\circ$

### 4.2 Roll Bins

| Roll Bin Designation | Absolute Angular Range ($|\theta_{roll}|$) | Sample Count | Percentage | Physical Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Minimal Tilt** | $[0^\circ, 10^\circ)$ | **1,288** | **64.40%** | Head upright with minimal sideways leaning |
| **Moderate Tilt** | $[10^\circ, 30^\circ)$ | **524** | **26.20%** | Moderate head resting on hand / sideways lean |
| **Severe Inversion/Tilt** | $[30^\circ, 180^\circ]$ | **188** | **9.40%** | Pronounced head tilt, resting on desk surface |
| **Total** | — | **2,000** | **100.00%** | Covers upright to resting postures |

---

## 5. Summary & Engineering Implications

1. **Robust Pretraining Representation:** The large continuous range of both positive and negative yaw and pitch ensures that any generic head-pose backbone trained on this dataset will generalize across varying classroom camera mount angles without dead-zones.
2. **Bridge to Classroom Video:** While classroom surveillance feeds have lower resolution than these portrait crops, the continuous angle representation provides a foundational feature extractor that can be mapped into classroom posture spaces in V4C.
