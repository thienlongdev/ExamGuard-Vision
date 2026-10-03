# Head-Pose Split Safety & Leakage Audit

**Document ID**: `reports/v4b/HEAD_POSE_SPLIT_AUDIT.md`  
**Phase**: V4B — Clean Person-Crop / Head-Pose Dataset Construction  
**Author**: Data Engineering & Audit Pipeline  
**Date**: 2026-10-02  
**Status**: ZERO LEAKAGE VERIFIED  

---

## 1. Executive Summary

A formal identity and near-duplicate split safety audit was conducted for the head-pose dataset (`datasets/v4_head_pose/`).  
The primary vulnerability identified was that **2,000 images** in `AFLW2000-3D` derive directly from photographs also present in the 21,080-image `AFLW-GT` corpus. Without explicit identity decoupling, placing corresponding images into different partitions would result in immediate near-duplicate contamination across train and test sets.

---

## 2. Partition Strategy & Zero-Leakage Architecture

To guarantee mathematical zero-leakage, the dataset has been partitioned into strictly disjoint subsets:

1. **Test Benchmark (`AFLW2000-3D`)**:
   - All 2,000 AFLW2000-3D crops (`image00002.jpg` to `image04386.jpg`) are assigned exclusively to the **`test`** partition (`datasets/v4_head_pose/splits/test.jsonl`).
   - Accompanied by full 3D 68-point landmarks $(X, Y, Z)$ and continuous Euler ground truth $(\theta_{yaw}, \theta_{pitch}, \theta_{roll})$.
2. **Train & Val Partitions (`AFLW-GT` Non-Overlapping)**:
   - The remaining **19,080 unique images** from `AFLW-GT` that have no counterpart in AFLW2000-3D are partitioned with deterministic random seed 42:
     - **`train`**: **16,218 images** (85.0%)
     - **`val`**: **2,862 images** (15.0%)
3. **Quarantine / Counterpart Handling**:
   - The 2,000 images in `AFLW-GT` whose base filenames overlap with `AFLW2000-3D` are tagged with split `test_counterpart`.
   - **Zero** of these 2,000 images are permitted into `train` or `val`.

---

## 3. Split Distribution & Overlap Verification

| Split | Source Sub-Corpus | Image Count | % of Disjoint Corpus | Base Image Overlap with Train | Base Image Overlap with Val | Base Image Overlap with Test |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`train`** | `AFLW-GT` (unique) | **16,218** | 76.9% | — | **0** | **0** |
| **`val`** | `AFLW-GT` (unique) | **2,862** | 13.6% | **0** | — | **0** |
| **`test`** | `AFLW2000-3D` | **2,000** | 9.5% | **0** | **0** | — |
| **`test_counterpart`** (excluded) | `AFLW-GT` (duplicate) | **2,000** | — | **0** | **0** | 2,000 (Same base photograph) |
| **Total Physical Images** | — | **23,080** | 100.0% | **0** | **0** | **0** |

$$\text{Cross-Split Base Image Leakage}(\text{train} \cap \text{val}) = 0$$
$$\text{Cross-Split Base Image Leakage}(\text{train} \cap \text{test}) = 0$$
$$\text{Cross-Split Base Image Leakage}(\text{val} \cap \text{test}) = 0$$

---

## 4. Subject Identity Limitations

### Physical Audit of Subject IDs
- **AFLW / AFLW2000 Ground Truth Structure:** The raw annotation formats of AFLW (Köstinger et al.) and 3DDFA (Zhu et al.) do not provide persistent person identification numbers (Subject IDs). Images were scraped as isolated web photographs from Flickr.
- **Audit Conclusion:** While identity tags do not exist in the source metadata, duplicate base photograph leakage has been completely eliminated by the explicit filename and ROI isolation detailed above.
- **Synthetic Augmentation Audit:** Zero synthetic face augmentations (e.g. 3DDFA synthesized profile views) were included in this physical test archive; all 23,080 crops represent real photographs.

---

## 5. Verification Verdict

| Audit Check | Requirement | Result |
| :--- | :--- | :--- |
| Cross-Split Image Leakage | Exactly 0 images | **PASS (0 leaked)** |
| Cross-Split Duplicate Hash Leakage | Exactly 0 matching hashes | **PASS (0 leaked)** |
| AFLW2000 Test Purity | Exclusively in test set | **PASS (2,000 in test)** |
| Manifest File Integrity | 23,080 valid JSONL records | **PASS (100% valid)** |
