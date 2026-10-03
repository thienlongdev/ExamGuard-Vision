# AFLW / AFLW2000 Provenance & Annotation Audit

**Document ID**: `reports/v4b/AFLW_PROVENANCE_AUDIT.md`  
**Phase**: V4B — Clean Person-Crop / Head-Pose Dataset Construction  
**Author**: Data Engineering & Audit Pipeline  
**Date**: 2026-10-02  
**Status**: PHYSICALLY VERIFIED  

---

## 1. Executive Summary

A comprehensive physical provenance audit of `datasets/raw_v4/head_pose_aflw2000/` was conducted to resolve the exact origin, composition, annotation fidelity, and ground-truth targets of the acquired head-pose corpus.

### Key Finding
The local physical archive `test.data.zip` contains **23,080 images** divided strictly into two distinct datasets:
1. **`AFLW2000-3D`**: Exactly **2,000 images** accompanied by 68 3D landmark points `(2000, 3, 68)`, bounding boxes `(2000, 4)`, and continuous Euler yaw ground-truth angles `(2000,)`.
2. **`AFLW-GT`**: Exactly **21,080 images** from the standard AFLW test collection, accompanied by 21,080 bounding boxes `(21080, 4)` and continuous yaw ground-truth angles `(21080,)`. **No 3D landmarks exist for AFLW-GT.**

Therefore, referring to the entire 23,080 corpus as "AFLW2000" is factually incorrect and reflects standard bundling conventions from the 3DDFA (3D Dense Face Alignment) benchmark release.

---

## 2. Provenance Matrix

| Field | AFLW2000-3D Sub-Corpus | AFLW-GT Sub-Corpus |
| :--- | :--- | :--- |
| **Origin Repository / Source** | Cleardusk 3DDFA / 3DDFA_V2 / Hailo Model Zoo | Cleardusk 3DDFA / 3DDFA_V2 / Hailo Model Zoo |
| **Parent Academic Dataset** | AFLW (Annotated Facial Landmarks in the Wild) | AFLW (Annotated Facial Landmarks in the Wild) |
| **Primary Citation** | Zhu et al., *"Face Alignment Across Large Poses: A 3D Solution"*, CVPR 2016 | Köstinger et al., *"Annotated Facial Landmarks in the Wild"*, ICCV Workshops 2011 |
| **Physical Image Directory** | `test.data/AFLW2000-3D_crop/` inside `test.data.zip` | `test.data/AFLW_GT_crop/` inside `test.data.zip` |
| **Image Count** | **2,000** | **21,080** |
| **Image Resolution** | $450 \times 450$ px RGB Crops | $450 \times 450$ px RGB Crops |
| **Annotation File (Yaw)** | `configs/AFLW2000-3D.pose.npy` (and `AFLW2000-3D-new.pose.npy`) | `configs/AFLW_GT_crop_yaws.npy` |
| **Annotation File (BBox)** | `configs/AFLW2000-3D_crop.roi_box.npy` | `configs/AFLW_GT_crop_roi_box.npy` |
| **Annotation File (3D Points)** | `configs/AFLW2000-3D.pts68.npy` | **NONE** (Unavailable in source) |
| **3D Landmarks Available** | **YES** (68 points in 3D: $x, y, z$) | **NO** (Yaw regression only) |
| **Derived vs Physical** | 3DMM Morphable Model fitting onto AFLW faces | Cropped bounding boxes with pose regression ground truth |
| **Academic License** | Non-commercial research / academic use | Non-commercial research / academic use |

---

## 3. Physical Cryptographic Hashes

All raw files under `datasets/raw_v4/head_pose_aflw2000/` have been cryptographically hashed and verified:

| File Name | File Size (Bytes) | SHA256 Checksum |
| :--- | :--- | :--- |
| `test.data.zip` | 158,854,200 | `c33b6c472baa0aca22fc21f456bcdfb5da964459208166eb93c8fffc720ddbad` |
| `configs/AFLW2000-3D.pose.npy` | 8,128 | `901c784bc19885c9fc19971af902f1a8c5c746e9e5b959fb997895b1578d91c4` |
| `configs/AFLW2000-3D-new.pose.npy` | 8,128 | `c0aa66c3145f2a291db37a928b9fd72a7bf281244e20e114c68674b79a99c38d` |
| `configs/AFLW2000-3D.pts68.npy` | 1,632,128 | `4547d9368572548b2f9f3c54986c587508ffd12d82ed03f2a8710b1c8c065872` |
| `configs/AFLW2000-3D_crop.roi_box.npy` | 32,128 | `c82153524e8339e8e15a73d3f2ba4db69a5e664b5c2e840826b4ae8668e52bd2` |
| `configs/AFLW_GT_crop_yaws.npy` | 84,448 | `47e78540bb939235a561a767cd9677006d6d258e21ccf1b263c9260a18e04f99` |
| `configs/AFLW_GT_crop_roi_box.npy` | 168,768 | `6eca0546aac6ae481a2c7dafd4df4fc33b358861ebdaf75d6de9242d496a752e` |

---

## 4. Metadata Array Specifications

### 4.1 `AFLW2000-3D.pose.npy`
- **Shape**: `(2000,)`, `float32`
- **Semantics**: Continuous yaw angle in degrees.
- **Range**: $[-351.23^\circ, 187.79^\circ]$, Mean: $0.81^\circ$, Median: $0.78^\circ$.
- *Note on phase wrapping*: In 3DDFA benchmarks, yaw angles outside $[-90^\circ, 90^\circ]$ represent extreme profiles or wrapped Euler angles from 3DMM parameterization.

### 4.2 `AFLW2000-3D.pts68.npy`
- **Shape**: `(2000, 3, 68)`, `float32`
- **Semantics**: 68 facial landmark coordinates in 3D camera space:
  - Axis 0: Image coordinate $X \in [121.87, 345.24]$
  - Axis 1: Image coordinate $Y \in [167.16, 375.73]$
  - Axis 2: Depth coordinate $Z \in [-92.75, 85.37]$
- **Fidelity**: Provides full 3D structural ground truth. Pitch and roll can be geometrically derived from rigid 3D alignment.

### 4.3 `AFLW_GT_crop_yaws.npy`
- **Shape**: `(21080,)`, `float32`
- **Semantics**: Continuous yaw angle in degrees.
- **Range**: $[-125.08^\circ, 167.94^\circ]$, Mean: $1.00^\circ$.
- **Fidelity**: High quality yaw regression ground truth for large-scale training and evaluation.

---

## 5. Dataset Splitting & Safety Rules

1. **Shared AFLW Index Isolation**:
   Because AFLW2000-3D was created by selecting the first 2,000 images of the broader AFLW dataset, `AFLW2000-3D_crop/image00002.jpg` and `AFLW_GT_crop/image00002.jpg` originate from the same base photograph.
   - **MANDATORY LEAKAGE BARRIER**: Cross-referencing between AFLW2000-3D and AFLW-GT must guarantee that corresponding image indices are placed in the **exact same split** (or AFLW2000-3D is reserved as a pure evaluation/holdout benchmark, with training performed exclusively on non-overlapping AFLW-GT samples).
2. **Dedicated Target Paths**:
   All head-pose crops and manifests must be placed in `datasets/v4_head_pose/`, entirely separate from behavior crop classification directories.
