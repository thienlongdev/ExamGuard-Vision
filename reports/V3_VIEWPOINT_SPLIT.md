# V3 Viewpoint and Video Group Split Distribution

**Date**: 2026-10-02  
**Dataset**: `datasets/processed_v3/`  
**Manifest**: `datasets/processed_v3/manifest_v3.json`

---

## 1. Overview of Camera Viewpoints

The V3 processed dataset combines physical annotations from `scb_dataset5_full` and verified high-angle sequences from `scbehavior` (`SCB5-Turn-Bow-Head-2024-9-17`). Camera viewpoints across these datasets are classified into two primary CCTV-like configurations:

1. **Oblique Upper / Ceiling-Mounted High Angle**:
   - Installed at classroom perimeter/corners pointing downward at student rows.
   - Captures realistic occlusion, foreshortening, and multi-student interactions.
   - Present across `SCB5-Turn-Bow-Head-2024-9-17`, `SCB5-Stand-2024-9-17`, and `SCB5-Discuss-2024-9-17`.

2. **Front / Elevated Frontal Classroom View**:
   - Installed near the front board or teacher podium looking across classroom desks.
   - Captures posture details (head down, writing, reading, standing).

---

## 2. Split Distribution by Video Group Components

Group partitioning was enforced using Disjoint Set Union (DSU) across exact perceptual and cryptographic duplicate clusters, ensuring that **zero image frames or video sequences cross split boundaries**.

| Viewpoint / Domain Category | Train Groups | Val Groups | Test Groups | Total Groups | Train Images | Val Images | Test Images | Total Images |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Oblique Upper / Ceiling CCTV-like** | 82 | 14 | 14 | 110 | 3,842 | 812 | 754 | 5,408 |
| **Elevated Frontal / Perimeter View** | 69 | 12 | 11 | 92 | 2,824 | 588 | 501 | 3,913 |
| **Total** | **151** | **26** | **25** | **202** | **6,666 (71.5%)** | **1,400 (15.0%)** | **1,255 (13.5%)** | **9,321 (100.0%)** |

---

## 3. Annotation Count Distribution Across Splits

| Viewpoint Category | Train BBoxes | Val BBoxes | Test BBoxes | Total BBoxes |
| :--- | :---: | :---: | :---: | :---: |
| **Oblique Upper CCTV-like** | 24,198 | 5,592 | 5,894 | 35,684 |
| **Elevated Frontal View** | 16,208 | 3,735 | 3,572 | 23,515 |
| **Total Annotations** | **40,406** | **9,327** | **9,466** | **59,199** |

---

## 4. CCTV Holdout & Generalization Verification

- **Validation Set**: 26 entire recording groups (1,400 images, 9,327 bounding boxes), with 14 groups representing high-angle oblique views. This provides strict evaluation of behavior detection under real classroom perspective without training contamination.
- **Test Set**: 25 entire recording groups (1,255 images, 9,466 bounding boxes), with 14 groups representing high-angle oblique views.
- **Unlabeled In-the-Wild Domain Reference**: In addition to the supervised test set, 60 representative frames from the completely unlabeled `cctv_exam_monitor` dataset are evaluated qualitatively using the trained checkpoint to inspect zero-shot angle adaptation.
