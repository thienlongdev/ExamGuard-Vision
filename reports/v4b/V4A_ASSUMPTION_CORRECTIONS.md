# V4A Assumption & Terminology Corrections

**Document ID**: `reports/v4b/V4A_ASSUMPTION_CORRECTIONS.md`  
**Phase**: V4B — Clean Person-Crop / Head-Pose Dataset Construction  
**Author**: Data Engineering & Audit Pipeline  
**Date**: 2026-10-02  
**Status**: APPROVED & ACTIVE  

---

## 1. Executive Summary

During the V4A Strong Data Acquisition phase, several working assumptions and terminology conventions were established to guide architectural scoping and dataset search. Prior to initiating data engineering in V4B, these assumptions have been physically audited and formally revised to eliminate premature generalizations, unvalidated threshold claims, and dataset identity conflation.

---

## 2. Correction 1: Temporal Threshold Status Reclassification

### Previous V4A Convention
In V4A reports (e.g., `reports/v4a/V4_HYBRID_ARCHITECTURE_PLAN.md`, `reports/v4a/HEAD_POSE_DATA_PLAN.md`), specific duration figures were cited as examples of behavior verification rules:
- `turn_head`: duration $\ge 1.5\text{ s}$
- `head_down` / `sleep`: duration $\ge 3.0\text{ s}$
- `stand`: duration $\ge 1.0\text{ s}$

### Formal Correction
**These figures are NOT experimentally validated operational thresholds.**  
In V4B and all subsequent documentation, these values are designated as:

$$\mathbf{PROVISIONAL\_CANDIDATE\_THRESHOLD}$$

### Engineering Rationale
1. **Lack of Ground-Truth Action Durations in Single-Frame Benchmarks:** The training datasets (SCBehavior, SCB5) consist of static images. Static boxes provide zero information regarding the true temporal distribution of natural glances (saccades, momentary whiteboard checks) versus sustained lateral copying.
2. **Video Clip Constraints:** EduAction clips are short ($3\text{--}5\text{ s}$ duration), capturing isolated pre-segmented activities rather than continuous unconstrained exam sessions with natural transitions.
3. **Mandatory Future Validation:** No threshold may be labeled as "operational", "validated", or "final" until rigorous temporal evaluation on continuous CCTV video streams with human-verified start/end timestamps is performed in later phases.

---

## 3. Correction 2: AFLW vs AFLW2000 Dataset Provenance Resolution

### Previous V4A Convention
In V4A inventory logs and summaries, the head-pose dataset acquired via Google Drive archive was frequently referenced as:
> *"AFLW2000-derived head-pose corpus containing 23,080 cropped face images."*

### Physical Audit Findings
Detailed physical inspection of `datasets/raw_v4/head_pose_aflw2000/test.data.zip` (158,854,200 bytes) and its associated `.npy` metadata arrays reveals two completely distinct sub-corpora bundled together from the 3DDFA (3D Dense Face Alignment) benchmark distribution:

| Sub-Corpus Name | File Path in Zip | Physical Image Count | Ground Truth Metadata Arrays | Available Targets | Provenance / Nature |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AFLW2000-3D** | `test.data/AFLW2000-3D_crop/` | **2,000** | `AFLW2000-3D.pose.npy`<br>`AFLW2000-3D-new.pose.npy`<br>`AFLW2000-3D.pts68.npy`<br>`AFLW2000-3D_crop.roi_box.npy` | Continuous Yaw (deg), 68 3D Landmarks $(x,y,z)$, ROI BBox | First 2,000 images of AFLW fitted with 3DMM (Zhu et al., CVPR 2016). Full 3D ground truth. |
| **AFLW-GT** | `test.data/AFLW_GT_crop/` | **21,080** | `AFLW_GT_crop_yaws.npy`<br>`AFLW_GT_crop_roi_box.npy` | Continuous Yaw (deg), ROI BBox | Cropped faces from broader AFLW test/eval set. Yaw angle only; **no 3D landmarks**. |
| **Combined Total** | `test.data/` | **23,080** | — | — | Sum of 2,000 (AFLW2000-3D) + 21,080 (AFLW-GT). |

### Formal Correction
1. **Never refer to the entire 23,080-image folder as "AFLW2000".**
2. The folder identity is resolved into two distinct datasets:
   - **`AFLW2000-3D`**: 2,000 samples with 3D landmark coordinates and fitted Euler yaw.
   - **`AFLW-GT`**: 21,080 samples with yaw angle regression ground truth only.
3. Cross-dataset split safety must be enforced: because both derive from the parent AFLW collection, identical source image indices (e.g. `image00002.jpg`) must not leak across splits.

---

## 4. Correction 3: Strict Decoupling of Behavioral Posture vs Cheating Assessment

### Reaffirmed Policy
V4B produces **purely observable posture datasets**.
- Classes are descriptive physical states: `NORMAL_UPRIGHT`, `NORMAL_READ_WRITE`, `HEAD_DOWN_DEEP`, `HEAD_REST_SLEEP`, `TURN_HEAD_CLEAR`.
- No sample, file, manifest entry, or directory will use the subjective designation `CHEATING`.
- Cheating is an administrative and contextual inference performed exclusively by downstream rule and fusion engines, never an intrinsic label of a static person crop.

---

## 5. Verification Checklist

| Item | Status | Action Taken in V4B |
| :--- | :--- | :--- |
| Temporal threshold renamed | **VERIFIED** | Renamed to `PROVISIONAL_CANDIDATE_THRESHOLD` in all configs and docs. |
| AFLW2000 count corrected | **VERIFIED** | Corrected to 2,000 AFLW2000-3D and 21,080 AFLW-GT images. |
| 3D landmark availability scoped | **VERIFIED** | Restricted to the 2,000 AFLW2000-3D samples; AFLW-GT flagged as yaw-only. |
| Cheating label prohibition | **VERIFIED** | Strict adherence to physical observable posture ontology. |
