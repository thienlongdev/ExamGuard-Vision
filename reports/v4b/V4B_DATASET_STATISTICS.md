# V4B Complete Dataset Statistics Report

**Document ID**: `reports/v4b/V4B_DATASET_STATISTICS.md`  
**Phase**: V4B — Clean Person-Crop / Head-Pose Dataset Construction  
**Author**: Data Engineering & Audit Pipeline  
**Date**: 2026-10-03  
**Status**: COMPLETE & VERIFIED  

---

## 1. Executive Summary

This report aggregates the complete inventory and statistical telemetry across both newly constructed V4B datasets:
1. **V4 Person-Crop Dataset** (`datasets/v4_crop/`): **20,490 crops** spanning 400 SCBehavior high-res images and 350 EduAction video clips across 5 evaluation partitions.
2. **V4 Head-Pose Dataset** (`datasets/v4_head_pose/`): **23,080 crops** with continuous Euler angles and 3D landmarks, strictly partitioned into disjoint train, val, and test subsets.

---

## 2. Raw Sources vs Selected & Filtered Crops

| Source Entity | Raw Available Units | Derived / Evaluated Candidates | Selected Final Crops | Quarantined / Filtered Crops | Reduction / Filter Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SCBehavior High-Res** | 400 Images (8,083 bboxes) | 16,166 crops (tight + context) | **14,106 supervised crops** | 1,914 quarantined (ambiguous lookup, discuss, stand) | 11.8% quarantined |
| **EduAction Video Clips**| 350 MP4 clips (39,563 frames)| 39,563 raw frames | **4,470 diverse sampled crops** | 35,093 redundant / context frames | 88.7% redundancy rejected |
| **AFLW / AFLW2000-3D** | 1 Zip Archive (23,080 images)| 23,080 face crops | **23,080 continuous angle crops**| 0 discarded (full corpus preserved) | 0.0% discarded |
| **Total Pipeline** | **751 files / 3 archives** | **78,809 raw candidates** | **43,570 active manifest records**| **37,007 filtered / quarantined**| **Clean High-Fidelity Data** |

---

## 3. Person-Crop Distribution Telemetry ($N = 20,490$)

### 3.1 Per-Class Breakdown
- **`NORMAL_UPRIGHT`**: 9,583 (46.77%)
- **`NORMAL_READ_WRITE`**: 3,825 (18.67%) [Hard Negative]
- **`TURN_HEAD_CLEAR`**: 2,002 (9.77%) [Positive Lateral Glance]
- **`HEAD_REST_SLEEP`**: 604 (2.95%) [Positive Slump / Desk Rest]
- **`AMBIGUOUS_LOOKUP`**: 1,284 (6.27%) [Quarantined Rear Row]
- **`COMPUTER_CONTEXT`**: 656 (3.20%) [Quarantined Context]
- **`TALKING_CONTEXT`**: 651 (3.18%) [Quarantined Context]
- **`PHONE_INTERACTION_CONTEXT`**: 636 (3.10%) [Quarantined Context]
- **`DRINKING_CONTEXT`**: 633 (3.09%) [Quarantined Context]
- **`DISCUSS_PAIR`**: 484 (2.36%) [Quarantined Macro]
- **`STAND_MACRO`**: 132 (0.64%) [Quarantined Macro]

### 3.2 Per-Source Breakdown
- **`SCBehavior-HighRes`**: 16,020 crops (78.18%)
- **`EduAction`**: 4,470 crops (21.82%)

### 3.3 Per-Split Breakdown
- **`train`**: 14,821 crops (72.33%)
- **`same_domain_val`**: 2,973 crops (14.51%)
- **`high_angle_holdout`**: 1,618 crops (7.90%)
- **`cross_source_holdout`**: 543 crops (2.65%)
- **`temporal_holdout`**: 535 crops (2.61%)

### 3.4 Per-Viewpoint Breakdown
- **`4K_CEILING_HIGH_ANGLE`**: 8,472 crops (41.35%)
- **`QHD_FRONT_OBLIQUE`**: 7,548 crops (36.84%)
- **`FRONTAL_DESK_LEVEL`**: 4,470 crops (21.82%)

### 3.5 Per-Scale Breakdown
- **`PERSON_LARGE`** ($H \ge 300\text{ px}$): 4,066 crops (19.84%)
- **`PERSON_MEDIUM`** ($180 \le H < 300\text{ px}$): 12,128 crops (59.19%)
- **`PERSON_SMALL`** ($120 \le H < 180\text{ px}$): 3,328 crops (16.24%)
- **`PERSON_VERY_SMALL`** ($H < 120\text{ px}$): 968 crops (4.72%)

### 3.6 Per-Quality Flag Breakdown
- **`GOOD`**: 9,703 crops (47.35%)
- **`BLURRY`** ($\text{Var}(\text{Lap}) < 35$): 9,888 crops (48.26%)
- **`QUARANTINED`**: 4,476 crops (21.84%)
- **`SMALL`**: 940 crops (4.59%)
- **`EXTREME_LIGHTING`**: 53 crops (0.26%)
- **`VERY_SMALL`**: 32 crops (0.16%)
- **`UNRESOLVABLE`**: 8 crops (0.04%)

---

## 4. Head-Pose Distribution Telemetry ($N = 23,080$)

### 4.1 Split Allocations
- **`train`** (`AFLW-GT` unique): 16,218 images (70.27%)
- **`val`** (`AFLW-GT` unique): 2,862 images (12.40%)
- **`test`** (`AFLW2000-3D`): 2,000 images (8.67%)
- **`test_counterpart`** (`AFLW-GT` overlap): 2,000 images (8.67% - isolated)

### 4.2 Angle Bins
- **Frontal Yaw** ($|\theta_{yaw}| < 15^\circ$): 7,901 samples (34.23%)
- **Moderate Lateral Yaw** ($15^\circ \le |\theta_{yaw}| < 45^\circ$): 8,521 samples (36.92%)
- **Large Lateral Yaw** ($|\theta_{yaw}| \ge 45^\circ$): 6,658 samples (28.85%)
- **Downward Pitch** ($\theta_{pitch} > 15^\circ$, AFLW2000): 1,370 samples (68.50%)
- **Level Pitch** ($-10^\circ \le \theta_{pitch} \le 15^\circ$, AFLW2000): 614 samples (30.70%)
- **Upward Pitch** ($\theta_{pitch} < -10^\circ$, AFLW2000): 16 samples (0.80%)

---

## 5. License Status Distribution

| License Designation | Record Count | % of Global Inventory | Data Source Origin |
| :--- | :--- | :--- | :--- |
| **`ACADEMIC_ONLY`** | 20,490 crops | 47.03% | SCBehavior High-Res + EduAction |
| **`ACADEMIC_NON_COMMERCIAL`** | 23,080 crops | 52.97% | AFLW + AFLW2000-3D |
| **Total Audited Assets** | **43,570** | **100.00%** | All non-commercial academic research |
