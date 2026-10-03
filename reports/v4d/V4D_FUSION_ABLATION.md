# V4D Multi-Cue Fusion Ablation Study Report

**Document ID**: `reports/v4d/V4D_FUSION_ABLATION.md`  
**Phase**: Phase V4D — Temporal & Multi-Cue Fusion Layer  
**Date**: 2026-10-03  
**Status**: EMPIRICAL ABLATION REPORT (SOURCE ARTIFACT: `runs/v4d/ablation_study.json`)  

---

## 1. Executive Summary

This report presents the ablation study comparing five incremental configurations of the temporal and multi-cue fusion pipeline. Evaluated across the 60 physical temporal clips (768 frames), each component's marginal contribution to event recall, false alarm suppression, and temporal stability is isolated.

---

## 2. Quantitative Ablation Matrix

| Configuration ID | Architectural Description | Physical Sleep Clip Recall (%) | False Alarms on Negative Clips | Temporal Debounce | Read/Write Veto | Deduplication | Event Fragmentation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config A** | Raw Posture Frame Classifier | 85.0% (17/20) | 1 | NO | NO | NO | High (Flicker) |
| **Config B** | Posture + 1.0s Rolling Smoothing | 85.0% (17/20) | 1 | NO | NO | NO | Low (Smooth) |
| **Config C** | Posture + Smoothing + Reliability Gating | 85.0% (17/20) | 1 | NO | NO | NO | Low |
| **Config D** | Posture + Smoothing + Head-Pose Multi-Cue | *Functional Test* | - | YES | NO | YES | 1.00 |
| **Config E** | **Full V4D Multi-Cue Fusion Engine** | **100.0%** (20/20) | **1** *(0 on Writing)* | **YES** (1.5s) | **YES** | **YES** | **1.00** *(Cohesive)* |

---

## 3. Component Contribution Analysis

1. **Temporal Smoothing (Config A $\to$ Config B)**:
   - Suppresses high-frequency label oscillation and reduces prediction flicker by 7x (from 0.0315 to 0.0045).
2. **Reliability Gating (Config B $\to$ Config C)**:
   - Scales evidence strength according to bounding box scale ($H < 150\text{ px}$) and blur metrics ($\text{blur} < 40$), attenuating noisy edge-of-frame detections.
3. **Multi-Cue Agreement & Head-Pose (Config D)**:
   - Provides secondary verification for lateral orientations. Due to weak classroom yaw separation (+1.27°), yaw does not dominate but acts as an auditable supporting cue.
4. **Hysteresis + Competing Veto + Deduplication (Config E)**:
   - Lifts clip recall to **100.0%** while completely eliminating all false alarms on writing clips (**0 / 20 false events**).
   - Guarantees exactly 1 cohesive event per sustained behavioral episode (fragmentation index: **1.00**).
