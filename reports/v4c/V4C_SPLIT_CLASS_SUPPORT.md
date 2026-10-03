# V4C Split Class Support & Physical Availability Audit

**Document ID**: `reports/v4c/V4C_SPLIT_CLASS_SUPPORT.md`  
**Phase**: V4C Specialized Posture + Head-Pose Model Training & Benchmark  
**Date**: 2026-10-03  
**Status**: VERIFIED & AUDITED  

---

## 1. Primary 4-Class Posture Ontology

Per Section 1.A of the V4C Specification, the posture classifier must use exactly four supervised classes:

- `0`: **`NORMAL_UPRIGHT`**
- `1`: **`NORMAL_READ_WRITE`**
- `2`: **`HEAD_REST_SLEEP`** (Minority class, physical ground truth from EduAction)
- `3`: **`TURN_HEAD_CLEAR`** (Yaw deviation $> 35^\circ$ from SCBehavior)

> [!IMPORTANT]
> **Zero Supervised HEAD_DOWN_DEEP Samples**: The physical V4B dataset contains 0 verified supervised `HEAD_DOWN_DEEP` samples. In strict compliance with guidelines, no empty 5th class is created, and no synthetic or pseudo-labeled samples are introduced.

---

## 2. Supervised Class Support Across Evaluation Partitions

Below are the verified counts of supervised samples physically available in each split for both Representation A (Tight) and Representation B (Context):

| Split | Representation | NORMAL_UPRIGHT | NORMAL_READ_WRITE | HEAD_REST_SLEEP | TURN_HEAD_CLEAR | Total Supervised | Quarantined | Total Split Crops |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train` | **Tight (A)** | 3555 | 1649 | 362 | 773 | **6339** | 3269 | 14821 |
| `train` | **Context (B)** | 3555 | 1649 | 362 | 773 | **6339** | 3269 | 14821 |
| `same_domain_val` | **Tight (A)** | 789 | 334 | 91 | 132 | **1346** | 579 | 2973 |
| `same_domain_val` | **Context (B)** | 789 | 334 | 91 | 132 | **1346** | 579 | 2973 |
| `high_angle_holdout` | **Tight (A)** | 615 | 88 | 0 | 96 | **799** | 20 | 1618 |
| `high_angle_holdout` | **Context (B)** | 615 | 88 | 0 | 96 | **799** | 20 | 1618 |
| `cross_source_holdout` | **Tight (A)** | 79 | 83 | 76 | 0 | **238** | 305 | 543 |
| `cross_source_holdout` | **Context (B)** | 79 | 83 | 76 | 0 | **238** | 305 | 543 |
| `temporal_holdout` | **Tight (A)** | 80 | 77 | 75 | 0 | **232** | 303 | 535 |
| `temporal_holdout` | **Context (B)** | 80 | 77 | 75 | 0 | **232** | 303 | 535 |

---

## 3. Physical Absence Warnings & Metric Handling

In accordance with Section 3, 14, and 15:

1. **`high_angle_holdout`**: Originates exclusively from SCBehavior 4K ceiling cameras.
   - `HEAD_REST_SLEEP` is **PHYSICALLY ABSENT** (count = 0).
   - Supported classes: `NORMAL_UPRIGHT` (615), `NORMAL_READ_WRITE` (88), `TURN_HEAD_CLEAR` (96). Total = 799.
   - **Metric Rule**: High-angle evaluation must compute macro F1 and balanced accuracy strictly over the 3 physically present classes. Fake zero-metric penalties for absent classes are forbidden.

2. **`cross_source_holdout`**: Originates exclusively from EduAction video sequences.
   - `TURN_HEAD_CLEAR` is **PHYSICALLY ABSENT** (count = 0).
   - Supported classes: `NORMAL_UPRIGHT` (79), `NORMAL_READ_WRITE` (83), `HEAD_REST_SLEEP` (76). Total = 238.
   - **Metric Rule**: Cross-source evaluation must compute metrics strictly over the 3 physically present classes.

3. **`temporal_holdout`**: Originates exclusively from EduAction continuous clips.
   - `TURN_HEAD_CLEAR` is **PHYSICALLY ABSENT** (count = 0).
   - Supported classes: `NORMAL_UPRIGHT` (80), `NORMAL_READ_WRITE` (77), `HEAD_REST_SLEEP` (75). Total = 232.
   - **Metric Rule**: Temporal evaluation and clip aggregation are computed over the 3 physically present classes.

4. **`same_domain_val`**: Contains all 4 supervised classes.
   - `NORMAL_UPRIGHT` (789), `NORMAL_READ_WRITE` (334), `HEAD_REST_SLEEP` (91), `TURN_HEAD_CLEAR` (132). Total = 1,346.
   - Full 4-class confusion matrix, precision, recall, and balanced accuracy will be computed.

5. **Quarantined Classes Isolation**:
   - Total Quarantined: **4,476 crops** across all splits (`AMBIGUOUS_LOOKUP`, `TALKING_CONTEXT`, `PHONE_INTERACTION_CONTEXT`, `COMPUTER_CONTEXT`, `DRINKING_CONTEXT`, `DISCUSS_PAIR`, `STAND_MACRO`).
   - Verified: 100% of quarantined crops carry the `QUARANTINED` quality flag and are strictly excluded from posture model training.
