# V4D Phone & Contraband Spatial Association Analysis

**Document ID**: `reports/v4d/V4D_PHONE_ASSOCIATION_ANALYSIS.md`  
**Phase**: Phase V4D — Temporal & Multi-Cue Fusion Layer  
**Date**: 2026-10-03  
**Status**: ARCHITECTURAL SPECIFICATION & SAFETY AUDIT  

---

## 1. Architectural Separation

In strict compliance with V4D core tenets:
- **Phone is NOT part of posture classification**.
- The concept `USE_PHONE` was permanently rejected from the posture taxonomy. Posture models classify body and head geometry only.
- Phone detection operates through a dedicated pipeline branch:
  $$\text{COCO Object Detector} \longrightarrow \text{Spatial Track Association} \longrightarrow \text{Temporal Persistence} \longrightarrow \text{PHONE\_ASSOCIATED Event}$$

---

## 2. Spatial Association Logic & Ambiguity Protection

Global detection of a phone in a classroom frame must **never** be naively mapped to "Student X has a phone". The association engine enforces rigorous spatial constraints:

1. **Direct Contact**: Phone bounding box overlaps student hand/lap region ($\text{IoU} \ge 0.30$ or center-point inside lower torso).
2. **Desk Proximity**: Phone bounding box located in the immediate desk bounding box linked to student track ID.
3. **Ambiguity Veto**:
   - If a detected phone overlaps multiple student bounding boxes equally, or lies equidistant between two desk neighbours:
     $$\text{Status} \longleftarrow \text{AMBIGUOUS\_ASSOCIATION}$$
   - **Crucial Rule**: `AMBIGUOUS_ASSOCIATION` vetoes event promotion. The state machine will **not** open a `PHONE_ASSOCIATED` event when ownership is geometrically uncertain.
4. **Temporal Persistence**:
   - Transient single-frame reflections or false object detections are filtered by requiring minimum candidate persistence ($D_{\text{cand}} \ge 1.0\text{ s}$).

---

## 3. Scientific Validation Limitation

> [!WARNING]
> **`PHONE_TEMPORAL_ACCURACY = NOT_SUPPORTED`**
>
> While the object association geometry and ambiguity veto have passed all functional unit tests (`tests/test_object_association.py` and `tests/test_v4d_fusion_engine.py`), no continuous temporal classroom video dataset with ground truth student-phone association trajectories exists in the repository.
> 
> Therefore:
> - Temporal accuracy, precision, and recall for phone association are **not empirically verified on live video**.
> - Live validation will be conducted during Stage 2 target-school trial deployments.
