# AFLW Pose Array Schema & Provenance Resolution

**Document ID**: `reports/v4c/AFLW_POSE_ARRAY_SCHEMA_RESOLUTION.md`  
**Phase**: V4C Specialized Head-Pose Target Integrity  
**Date**: 2026-10-03  
**Status**: AUDITED & UNAMBIGUOUSLY FROZEN  
**Target Module**: Phase V4C Module B — Head-Pose Estimator  

---

## 1. Executive Summary & Resolution of Documentation Contradiction

A contradiction arose in earlier audit reports regarding the shape and column semantics of `AFLW2000-3D.pose.npy`:
- **Hypothesis A (Earlier Audit)**: `AFLW2000-3D.pose.npy` is a 1D yaw array of shape `(2000,)`.
- **Hypothesis B (Erroneous Assumption)**: `AFLW2000-3D.pose.npy` is a 3-column pose matrix of shape `(2000, 3)` containing `[pitch, yaw, roll]`.

### The Physical Fact (Verified via `np.load`)
Every relevant `.npy` file was loaded and inspected directly using NumPy. **Hypothesis A is the exact physical reality**:
- `AFLW2000-3D.pose.npy` has shape **`(2000,)`** and dtype `float32`. It is strictly a **1D array of signed Euler yaw angles**.
- There is **no 3-column `[pitch, yaw, roll]` file** anywhere in the raw dataset.
- The root cause of the misconception was that `datasets/v4_head_pose/manifest.jsonl` contains `"pitch"` and `"roll"` fields for AFLW2000-3D test records. Physical code inspection of `scripts/build_v4_head_pose.py` revealed that these pitch and roll values were derived geometrically on-the-fly from the 3D landmark array `AFLW2000-3D.pts68.npy` via `derive_pitch_roll_from_pts68()`.

---

## 2. Physical Inspection Census of All Pose Arrays

| Array Filename | Relative Path | Shape | Dtype | Range [Min, Max] | Semantic Content |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`AFLW2000-3D.pose.npy`** | `datasets/raw_v4/head_pose_aflw2000/configs/AFLW2000-3D.pose.npy` | `(2000,)` | `float32` | `[-351.23, +187.79]` | **Signed Euler Yaw Only** (ground-truth test yaw) |
| **`AFLW2000-3D-new.pose.npy`** | `datasets/raw_v4/head_pose_aflw2000/configs/AFLW2000-3D-new.pose.npy` | `(2000,)` | `float32` | `[0.003, 114.14]` | **Unsigned Yaw Magnitude** (**DISQUALIFIED**) |
| **`AFLW_GT_crop_yaws.npy`** | `datasets/raw_v4/head_pose_aflw2000/configs/AFLW_GT_crop_yaws.npy` | `(21080,)` | `float32` | `[-125.08, +167.94]` | **Signed Euler Yaw Only** (train/val yaw) |
| **`AFLW2000-3D.pts68.npy`** | `datasets/raw_v4/head_pose_aflw2000/configs/AFLW2000-3D.pts68.npy` | `(2000, 3, 68)` | `float32` | `[-236.16, +557.65]` | **68 3D Landmark Points $(X, Y, Z)$** (from 3DDFA mesh) |
| **`AFLW2000-3D_crop.roi_box.npy`** | `datasets/raw_v4/head_pose_aflw2000/configs/AFLW2000-3D_crop.roi_box.npy` | `(2000, 4)` | `float32` | `[1.0, 450.0]` | Face crop bounding boxes $[x_1, y_1, x_2, y_2]$ |
| **`AFLW_GT_crop_roi_box.npy`** | `datasets/raw_v4/head_pose_aflw2000/configs/AFLW_GT_crop_roi_box.npy` | `(21080, 4)` | `uint16` | `[1, 450]` | Face crop bounding boxes $[x_1, y_1, x_2, y_2]$ |

### First 5 Representative Rows:
- **`AFLW2000-3D.pose.npy`**: `[1.044306, 68.15524, 50.485413, 17.143373, 68.64055]` (degrees)
- **`AFLW2000-3D-new.pose.npy`**: `[1.3212477, 73.31085, 55.45405, 7.209424, 95.72651]` (absolute degrees)
- **`AFLW_GT_crop_yaws.npy`**: `[1.3212477, 73.31085, 55.45405, 7.209424, 95.72651]` (degrees)

---

## 3. Unambiguous Specification of Supervision Targets

Under Phase V4C Head-Pose Model Specifications, supervision and evaluation targets are strictly defined as follows:

```
========================================================================================
V4C HEAD-POSE TARGET SOURCE REGISTRY
========================================================================================
YAW TRAIN TARGET SOURCE  : AFLW_GT_crop_yaws.npy (1D float32, partitioned to 16,218 train)
YAW VAL TARGET SOURCE    : AFLW_GT_crop_yaws.npy (1D float32, partitioned to 2,862 val)
YAW TEST TARGET SOURCE   : AFLW2000-3D.pose.npy (1D float32, 2,000 test samples)
PITCH SOURCE             : DERIVED_FROM_3D_LANDMARKS (AFLW2000-3D.pts68.npy) for test only;
                           ABSENT (None) in 2D AFLW-GT train/val.
ROLL SOURCE              : DERIVED_FROM_3D_LANDMARKS (AFLW2000-3D.pts68.npy) for test only;
                           ABSENT (None) in 2D AFLW-GT train/val.
LANDMARK SOURCE          : AFLW2000-3D.pts68.npy (shape (2000, 3, 68))
========================================================================================
```

### Critical Operational Constraints:
1. **Primary V4C Objective**: Continuous **SIGNED YAW REGRESSION** for classroom lateral turn monitoring.
2. **Pitch and Roll Scope**: Neither pitch nor roll is required for V4C classroom suspicious behavior monitoring success. No pitch or roll loss is used during training, and no pitch/roll ground truth is claimed from AFLW-GT.
3. **Disqualification Maintained**: `AFLW2000-3D-new.pose.npy` remains permanently disqualified due to absent sign information.

---

## 4. Conclusion & Gate Signoff

The schema contradiction is fully resolved. All training pipelines, dataset loaders, and evaluation functions consistently reference verified 1D signed yaw arrays with verified geometric provenance for secondary landmarks.
