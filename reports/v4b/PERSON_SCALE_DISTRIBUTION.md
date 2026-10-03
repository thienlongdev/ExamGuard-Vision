# Person Scale Distribution & Small-Person Analysis

**Document ID**: `reports/v4b/PERSON_SCALE_DISTRIBUTION.md`  
**Phase**: V4B — Clean Person-Crop / Head-Pose Dataset Construction  
**Author**: Data Engineering & Audit Pipeline  
**Date**: 2026-10-03  
**Status**: EMPIRICALLY VERIFIED  

---

## 1. Executive Summary

In Stage 1 and Stage 1.5, error audits revealed that posture and behavior detection degradation is strongly correlated with person and head pixel dimensions. Specifically, student heads smaller than approximately $15 \times 15$ px resulted in catastrophic feature collapse, high confusion between reading and head-down, and miss rates exceeding 60%.

To establish systematic benchmark baselines for future V4C models, all 20,490 crops in `datasets/v4_crop/manifest.jsonl` were stratified into four empirically grounded scale buckets derived from physical bounding box distributions:
- **`PERSON_LARGE`**: Bounding box height $H \ge 300\text{ px}$ (Head height $\ge 80\text{ px}$)
- **`PERSON_MEDIUM`**: $180\text{ px} \le H < 300\text{ px}$ (Head height $\approx 45\text{--}80\text{ px}$)
- **`PERSON_SMALL`**: $120\text{ px} \le H < 180\text{ px}$ (Head height $\approx 25\text{--}45\text{ px}$)
- **`PERSON_VERY_SMALL`**: $H < 120\text{ px}$ (Head height $< 25\text{ px}$)

---

## 2. Overall Scale Breakdown ($N = 20,490$)

| Scale Bucket | Height Range ($H$) | Crop Count | Percentage | Typical Classroom Seating Location |
| :--- | :--- | :--- | :--- | :--- |
| **`PERSON_LARGE`** | $H \ge 300\text{ px}$ | **4,066** | **19.84%** | Front rows, foreground desks, standing invigilators |
| **`PERSON_MEDIUM`** | $180 \le H < 300\text{ px}$ | **12,128** | **59.19%** | Middle rows, standard surveillance coverage, EduAction crops |
| **`PERSON_SMALL`** | $120 \le H < 180\text{ px}$ | **3,328** | **16.24%** | Rear rows, distant desks |
| **`PERSON_VERY_SMALL`**| $H < 120\text{ px}$ | **968** | **4.72%** | Extreme back corners, distal surveillance background |
| **Total** | — | **20,490** | **100.00%** | Comprehensive multi-scale coverage |

---

## 3. Scale Distribution by Class

| Ontology Label | `PERSON_LARGE` | `PERSON_MEDIUM` | `PERSON_SMALL` | `PERSON_VERY_SMALL` | Total Crops |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`NORMAL_UPRIGHT`** | 2,698 (28.2%) | 6,259 (65.3%) | 594 (6.2%) | 32 (0.3%) | **9,583** |
| **`NORMAL_READ_WRITE`** | 398 (10.4%) | 1,653 (43.2%) | 1,246 (32.6%) | 528 (13.8%) | **3,825** |
| **`TURN_HEAD_CLEAR`** | 704 (35.2%) | 808 (40.4%) | 404 (20.2%) | 86 (4.3%) | **2,002** |
| **`HEAD_REST_SLEEP`** | 0 (0.0%) | 604 (100.0%) | 0 (0.0%) | 0 (0.0%) | **604** |
| **`AMBIGUOUS_LOOKUP`** (Quarantined)| 0 (0.0%) | 0 (0.0%) | 966 (75.2%) | 318 (24.8%) | **1,284** |
| **`TALKING_CONTEXT`** (Quarantined) | 0 (0.0%) | 651 (100.0%) | 0 (0.0%) | 0 (0.0%) | **651** |
| **`PHONE_INTERACTION_CONTEXT`** | 0 (0.0%) | 636 (100.0%) | 0 (0.0%) | 0 (0.0%) | **636** |
| **`COMPUTER_CONTEXT`** | 0 (0.0%) | 656 (100.0%) | 0 (0.0%) | 0 (0.0%) | **656** |
| **`DRINKING_CONTEXT`** | 0 (0.0%) | 633 (100.0%) | 0 (0.0%) | 0 (0.0%) | **633** |
| **`DISCUSS_PAIR`** | 156 (32.2%) | 206 (42.6%) | 118 (24.4%) | 4 (0.8%) | **484** |
| **`STAND_MACRO`** | 110 (83.3%) | 22 (16.7%) | 0 (0.0%) | 0 (0.0%) | **132** |

### Strategic Insight
- Notice that **100% of `AMBIGUOUS_LOOKUP`** samples fall into `PERSON_SMALL` (966) and `PERSON_VERY_SMALL` (318). This validates the physical audit finding that small, distant students in the rear rows cannot be unambiguously labeled as alert upright attention vs reading/distraction from surveillance video, justifying their quarantine from supervised training.
- `TURN_HEAD_CLEAR` possesses strong representation across large (704), medium (808), and small (404) scales, enabling robust evaluation across distance.

---

## 4. Scale Distribution by Source Dataset

| Source Dataset | `PERSON_LARGE` | `PERSON_MEDIUM` | `PERSON_SMALL` | `PERSON_VERY_SMALL` | Total Crops |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SCBehavior-HighRes** | 4,066 (25.4%) | 7,658 (47.8%) | 3,328 (20.8%) | 968 (6.0%) | **16,020** |
| **EduAction** | 0 (0.0%) | 4,470 (100.0%) | 0 (0.0%) | 0 (0.0%) | **4,470** |

---

## 5. Scale Distribution by Camera Viewpoint

| Camera Viewpoint | `PERSON_LARGE` | `PERSON_MEDIUM` | `PERSON_SMALL` | `PERSON_VERY_SMALL` | Total Crops |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`4K_CEILING_HIGH_ANGLE`** | 2,812 (33.2%) | 5,094 (60.1%) | 524 (6.2%) | 42 (0.5%) | **8,472** |
| **`QHD_FRONT_OBLIQUE`** | 1,254 (16.6%) | 2,564 (34.0%) | 2,804 (37.1%) | 926 (12.3%) | **7,548** |
| **`FRONTAL_DESK_LEVEL`** | 0 (0.0%) | 4,470 (100.0%) | 0 (0.0%) | 0 (0.0%) | **4,470** |

- The **4K Ceiling High-Angle** camera view benefits from 4K resolution, keeping 93.3% of crops in Large and Medium categories even from an elevated angle.
- The **QHD Front Oblique** camera captures deeper depth-of-field across long classroom rows, providing 3,730 rear-row Small and Very Small student crops essential for stress-testing resolution degradation in V4C.
