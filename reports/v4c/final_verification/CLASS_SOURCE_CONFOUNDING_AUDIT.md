# V4C Class-by-Source and Viewpoint Confounding Audit

**Document ID**: `reports/v4c/final_verification/CLASS_SOURCE_CONFOUNDING_AUDIT.md`  
**Phase**: V4C Specialized Posture + Head-Pose Model Verification  
**Date**: 2026-10-03  
**Status**: COMPLETE & FORENSICALLY VERIFIED  
**Source Artifact**: `datasets/v4_crop/manifest.jsonl` (20,490 Physical Records)

---

## 1. Executive Summary & Purpose

A fundamental principle of scientific machine learning evaluation is that **out-of-distribution (OOD) and cross-source generalization metrics are only valid if positive test samples physically exist in those target partitions**.

This forensic audit rigorously cross-tabulates all 20,490 normalized person crops across:
1. **Source Dataset** (`SCBehavior-HighRes` vs. `EduAction`)
2. **Camera Viewpoint** (`4K_CEILING_HIGH_ANGLE`, `QHD_FRONT_OBLIQUE`, `FRONTAL_DESK_LEVEL`)
3. **Evaluation Partition / Split** (`train`, `same_domain_val`, `high_angle_holdout`, `cross_source_holdout`, `temporal_holdout`)

---

## 2. Complete Class Distribution Matrix

The table below reports exact physical sample counts for all ontology classes across sources and splits:

| Ontology Label | Supervised in V4C | Source: SCBehavior | Source: EduAction | Total Crops | Train | Same-Domain Val | High-Angle Holdout | Cross-Source Holdout | Temporal Holdout |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`NORMAL_UPRIGHT`** | **YES** | 8,930 | 653 | **9,583** | 6,719 | 1,475 | 1,230 | 79 | 80 |
| **`NORMAL_READ_WRITE`** | **YES** | 3,188 | 637 | **3,825** | 2,925 | 564 | 176 | 83 | 77 |
| **`HEAD_REST_SLEEP`** | **YES** | **0** | 604 | **604** | 362 | 91 | **0** | 76 | 75 |
| **`TURN_HEAD_CLEAR`** | **YES** | 2,002 | **0** | **2,002** | 1,546 | 264 | 192 | **0** | **0** |
| `AMBIGUOUS_LOOKUP` | NO (Quarantine) | 1,284 | 0 | 1,284 | 1,172 | 108 | 4 | 0 | 0 |
| `DISCUSS_PAIR` | NO (Context) | 484 | 0 | 484 | 448 | 34 | 2 | 0 | 0 |
| `STAND_MACRO` | NO (Context) | 132 | 0 | 132 | 98 | 20 | 14 | 0 | 0 |
| `TALKING_CONTEXT` | NO (Context) | 0 | 651 | 651 | 390 | 105 | 0 | 78 | 78 |
| `PHONE_INTERACTION_CONTEXT` | NO (Context) | 0 | 636 | 636 | 379 | 103 | 0 | 80 | 74 |
| `DRINKING_CONTEXT` | NO (Context) | 0 | 633 | 633 | 382 | 105 | 0 | 70 | 76 |
| `COMPUTER_CONTEXT` | NO (Context) | 0 | 656 | 656 | 400 | 104 | 0 | 77 | 75 |
| **TOTAL** | — | **16,020** | **4,470** | **20,490** | **14,821** | **2,973** | **1,618** | **543** | **535** |

---

## 3. Viewpoint Cross-Tabulation

| Ontology Label | `4K_CEILING_HIGH_ANGLE` | `QHD_FRONT_OBLIQUE` | `FRONTAL_DESK_LEVEL` | Total |
| :--- | :---: | :---: | :---: | :---: |
| **`NORMAL_UPRIGHT`** | 6,348 | 2,582 | 653 | 9,583 |
| **`NORMAL_READ_WRITE`** | 860 | 2,328 | 637 | 3,825 |
| **`HEAD_REST_SLEEP`** | **0** | **0** | **604** | 604 |
| **`TURN_HEAD_CLEAR`** | 1,040 | 962 | **0** | 2,002 |
| Context & Quarantined | 224 | 1,676 | 2,576 | 4,476 |
| **TOTAL** | **8,472** | **7,548** | **4,470** | **20,490** |

---

## 4. Crucial Forensic Findings & Confounding Disclosures

### Finding 1: `HEAD_REST_SLEEP` Source and Viewpoint Monoculture
- **Source Origin**: `HEAD_REST_SLEEP` is **100% sourced from `EduAction`** (604/604 samples).
- **Viewpoint**: `HEAD_REST_SLEEP` was captured exclusively from **`FRONTAL_DESK_LEVEL`** cameras.
- **Physical Absence in High-Angle**: There are **zero** instances of `HEAD_REST_SLEEP` in `high_angle_holdout` ($N=0$).
- **Scientific Implication**:
  - `HIGH_ANGLE_HOLDOUT` **cannot and does not evaluate sleeping detection**.
  - Any evaluation claiming high-angle sleep recall is scientifically false.
  - The model's ability to detect sleeping postures from overhead ceiling cameras remains **unvalidated by empirical data** and must be tested in future on-site pilot trials.

### Finding 2: `TURN_HEAD_CLEAR` Source and Viewpoint Monoculture
- **Source Origin**: `TURN_HEAD_CLEAR` is **100% sourced from `SCBehavior-HighRes`** (2,002/2,002 samples).
- **Viewpoint**: `TURN_HEAD_CLEAR` was captured exclusively from `4K_CEILING_HIGH_ANGLE` (1,040 samples) and `QHD_FRONT_OBLIQUE` (962 samples).
- **Physical Absence in Cross-Source & Temporal**:
  - `cross_source_holdout` contains **zero** instances of `TURN_HEAD_CLEAR` ($N=0$).
  - `temporal_holdout` contains **zero** instances of `TURN_HEAD_CLEAR` ($N=0$).
- **Scientific Implication**:
  - `CROSS_SOURCE_HOLDOUT` and `TEMPORAL_HOLDOUT` **cannot and do not evaluate head-turning recall**.
  - Cross-source generalization of head-turn detection is supported only by the Head-Pose Yaw Regression module (`hopenet_yaw` / `resnet18_yaw`) evaluated on AFLW2000-3D, **not** by the 4-class posture classifier.

### Finding 3: Rigorous Handling of Physically Absent Classes
- Per Section 9 of the V4C Specification:
  - When evaluating `HIGH_ANGLE_HOLDOUT`, macro metrics are strictly computed over the 3 physically supported classes: `[NORMAL_UPRIGHT, NORMAL_READ_WRITE, TURN_HEAD_CLEAR]`.
  - When evaluating `CROSS_SOURCE_HOLDOUT` and `TEMPORAL_HOLDOUT`, macro metrics are strictly computed over the 3 physically supported classes: `[NORMAL_UPRIGHT, NORMAL_READ_WRITE, HEAD_REST_SLEEP]`.
  - `HEAD_REST_SLEEP` in High-Angle, and `TURN_HEAD_CLEAR` in Cross-Source/Temporal, are explicitly marked:
    `NOT_SUPPORTED_IN_PARTITION`
  - No synthetic zeros or fake imputation are permitted.

---

## 5. Audit Sign-Off

- **Manifest Verified**: `datasets/v4_crop/manifest.jsonl` (20,490 records intact).
- **Confounding Fully Documented**: Yes.
- **Overclaiming Prevented**: Yes. All partition metrics reflect strictly their physically supported classes.
