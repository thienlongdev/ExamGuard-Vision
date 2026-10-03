# V4D Scientific Limitations & Boundaries of Validity

**Document ID**: `reports/v4d/V4D_LIMITATIONS.md`  
**Phase**: Phase V4D — Temporal & Multi-Cue Fusion Layer  
**Date**: 2026-10-03  
**Status**: MANDATORY GOVERNANCE & LIMITATIONS SPECIFICATION  

---

## 1. Absolute Scientific Boundaries

To preserve research integrity and prevent narrative drift, this document records all known scientific limitations of the current codebase and underlying datasets.

---

## 2. Dataset & Source Confounding Limitations

1. **Source / Class Confounding in Underlying Posture Crops**:
   - `HEAD_REST_SLEEP` positive samples are **100% sourced from `EduAction`** (604 samples); `SCBehavior` contains 0 positive samples.
   - `TURN_HEAD_CLEAR` positive samples are **100% sourced from `SCBehavior`** (2,002 samples); `EduAction` contains 0 positive samples.
   - Consequently, universal cross-dataset generalization across all behaviors cannot be claimed until real multi-camera classroom data is collected in Stage 2.
2. **Absence of Temporal Turn-Head Ground Truth**:
   - The temporal holdout set (`temporal_holdout.jsonl`) contains 42 video clips, all 100% sourced from `EduAction`.
   - While `EduAction` provides continuous physical sequences of sleep, writing, and lectures, it contains **zero positive temporal clips of lateral turn-head behavior**.
   - **Formal Scientific Verdict**:
     **`TEMPORAL_TURN_HEAD_POSITIVE_VALIDATION = NOT_SUPPORTED`**.
     Lateral turn temporal logic is verified via unit tests and synthetic functional fixtures, but empirical video recall is unvalidated.

---

## 3. Sensor & Perception Boundaries

1. **Classroom Yaw Bridge: Weak Standalone Separation**:
   - Continuous facial yaw separation between upright students (mean $|\text{yaw}| = 17.46^\circ$) and turning students (mean $|\text{yaw}| = 18.73^\circ$) is only **+1.27°** (Cohen's d: **0.087**, distribution overlap: **87.64%**).
   - Head-pose yaw **CANNOT** independently detect cheating or head-turns. It is an auditable supporting cue only.
2. **Head-Pose Extreme Profile Support**:
   - `HopeNet-Yaw` operates strictly within the half-open domain $[-99.0^\circ, +99.0^\circ)$. It cannot extrapolate to extreme profiles or rear views.
   - `ResNet18-Yaw-Circular` provides full $360^\circ$ coverage, but extreme profile errors ($\ge 90^\circ$) are high (**42.17°** MAE) due to sparse training data ($N=6$).
3. **Resolution Scale Gating**:
   - Students with bounding box height $H < 120\text{ px}$ or head dimensions $< 25 \times 25\text{ px}$ lack sufficient pixels for facial orientation estimation. Gating as `HEAD_POSE_UNAVAILABLE` is mandatory.

---

## 4. Unvalidated Temporal Behaviors

1. **Phone Temporal Accuracy**:
   - **`PHONE_TEMPORAL_ACCURACY = NOT_SUPPORTED`**. No temporal video dataset with ground truth desk phone trajectories currently exists.
2. **Discussion Temporal Accuracy**:
   - **`DISCUSSION_TEMPORAL_ACCURACY = NOT_SUPPORTED`**. Macro discuss detection is integrated functionally without continuous pair-level ground truth.
3. **End-to-End Integrated Wall-Clock Runtime**:
   - **`INTEGRATED_END_TO_END_RUNTIME_NOT_YET_MEASURED = TRUE`**. Component benchmarks demonstrate feasibility, but live multi-camera RTSP pipeline latency will be measured during Stage 2.

---

## 5. Production Readiness Verdict

- **`READY_FOR_STAGE2 = YES`**
- **`PRODUCTION_READY = NO`**

The system cannot be certified for operational deployment in actual exam centers until:
1. Target-school classroom CCTV validation is conducted.
2. End-to-end RTSP pipeline latency and multi-camera streaming are benchmarked.
3. Long-duration stability tests (3+ hours continuously) are completed.
4. Human invigilator user experience (UX) and false alarm tolerance are validated.
