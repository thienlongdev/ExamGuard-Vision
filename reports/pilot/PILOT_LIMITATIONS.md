# PILOT LIMITATIONS & SCIENTIFIC INTEGRITY STATEMENT

## 1. Governance & Boundary Definitions
This document explicitly records the operational boundaries, scientific limitations, and unvalidated capabilities of the Target CCTV Pilot system as prepared.

---

## 2. Definitive Gate Verdicts

| Readiness Gate | Verdict | Authoritative Rationale |
| :--- | :--- | :--- |
| **`READY_FOR_CONTROLLED_TARGET_CCTV_PILOT`** | **YES** | All 160 baseline regression tests pass; checkpoint hashes match certified weights; camera calibration, preflight tooling, and resilience suites pass; 1080p/1440p/4K operating envelopes profiled; evidence privacy policies enforced. |
| **`DENSE_ROOM_READY`** | **NO** | Physical benchmarking demonstrates that dense classroom workloads with $\ge 20$ simultaneously tracked students cannot sustain $25 - 30\text{ FPS}$ line rate on the current full multi-cue pipeline without dropped frames or backpressure throttling. Dense room deployment requires multi-GPU scaling or detector cadence decimation. |
| **`PRODUCTION_READY`** | **NO** | Mandatory invariant. The system has not undergone real exam-day proctor user studies, multi-room domain shift evaluation across diverse school lighting conditions, formal institutional ethics approvals, or 24-hour continuous site soak testing. |

---

## 3. Explicit Model & Perception Boundaries

1. **Facial Yaw vs Bodily Posture Decoupling**:
   - `HopeNetYaw` measures horizontal yaw angle only ($[-99.0^\circ, +99.0^\circ)$).
   - Yaw **must never infer** sleep, reading, writing, or head-down behaviors. Those remain the exclusive domain of the 4-class posture model (`MobileNetV3-Small`).
2. **Ceiling High Mounting Limitations**:
   - Steep camera angles ($> 60^\circ$ pitch) compress facial landmarks to sub-pixel blobs. Facial yaw is completely unavailable in high-ceiling mounting configurations.
3. **Phone Visibility & Ambiguity**:
   - Without labeled exam hall ground-truth data, no detection accuracy percentage is claimed.
   - Phones occluded by hands, laps, or desk rims cannot be optically detected.
   - Ambiguous ownership between adjacent desks halts event emission.
4. **Anonymous Student Tracks**:
   - The system performs zero facial recognition, zero biometric identification, and zero student name binding. Tracks are ephemeral integers (`Track #1`, `Track #2`).
5. **Human Adjudication**:
   - The system records observable physical evidence; it never issues moral or disciplinary verdicts of cheating or fraud.
