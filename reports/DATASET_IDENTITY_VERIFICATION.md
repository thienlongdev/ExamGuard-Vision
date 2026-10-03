# Dataset Identity Verification Report

**Verification Date**: 2026-10-02  
**Verification Method**: Direct physical disk inspection of files, directory trees, and local `data.yaml` files.  
**Strict Rule**: No class names or metrics are inferred from academic literature, web documentation, or previous conversational assumptions.

---

## 1. Physical Dataset Inventory Overview

| Directory Name | Physical Local Path | Detected YOLO YAML | Actual Images | Actual Labels | Source Origin | Physical Status | Identity Confidence |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| `scb_discuss` (`SCB5-Discuss-2024-9-17`) | `datasets/raw/scb_dataset3/SCB5-Discuss-2024-9-17` | `data.yaml` | 864 | 864 | HuggingFace (`wintonYF/SCB-Dataset`) | **PRESENT** | **HIGH** |
| `scb_handrise_read_write` (`SCB5-Handrise-Read-write-2024-9-17`) | `datasets/raw/scb_dataset3/SCB5-Handrise-Read-write-2024-9-17` | `data.yaml` | 6,864 | 6,864 | HuggingFace (`wintonYF/SCB-Dataset`) | **PRESENT** | **HIGH** |
| `scbehavior_bow_turn` (`SCB_BowTurnHead_20250509`) | `datasets/raw/scbehavior/SCB_BowTurnHead_20250509` | `data.yaml` | 2,410 | 2,410 | HuggingFace (`wintonYF/SCB-Dataset`) | **PRESENT** | **HIGH** |
| `cctv_exam_monitor` | `datasets/raw/cctv_exam_monitor` | None | 0 | 0 | Kaggle (`cctvdataset/cctv-exam-monitor-dataset`) | **EMPTY (Pending Download)** | N/A |
| `exam_cheating_roboflow` | `datasets/raw/exam_cheating_roboflow` | None | 0 | 0 | Roboflow Universe | **EMPTY (Pending Download)** | N/A |
| `exam_cheating_kaggle` | `datasets/raw/exam_cheating_kaggle` | None | 0 | 0 | Kaggle | **EMPTY (Pending Download)** | N/A |

---

## 2. Detailed Dataset Verification Records

### A. Dataset: `SCB5-Discuss-2024-9-17` (Sub-dataset of `scb_dataset3`)

- **Physical Path**: `C:\WorkingSpace Python\DETECTOR-YOLO\datasets\raw\scb_dataset3\SCB5-Discuss-2024-9-17`
- **Detected YAML**: `datasets/raw/scb_dataset3/SCB5-Discuss-2024-9-17/data.yaml`
- **Exact YAML Contents**:
  ```yaml
  train: /data/SCB/SCB-dataset/SCB5-Discuss-2024-9-17/images/train
  val: /data/SCB/SCB-dataset/SCB5-Discuss-2024-9-17/images/val
  nc: 1
  names: ['discuss']
  ```
- **Actual Local Class Names**:
  - `0`: `discuss`
- **Documentation / License Files**:
  - `metadata.json`: Tracked in parent directory (Academic / Non-commercial research license).
- **Directory Structure**:
  - `images/`: `train/` (691 images), `val/` (173 images)
  - `labels/`: `train/` (691 labels), `val/` (173 labels)
- **Total Images**: 864
- **Total Labels**: 864 (100% paired, 0 unpaired)
- **Total Annotations**: 5,392 (all Class 0 `discuss`)
- **Source Identity Confidence**: **HIGH**
- **Mismatch / Conflict**: None. Exactly 1 class (`discuss`).

---

### B. Dataset: `SCB5-Handrise-Read-write-2024-9-17` (Sub-dataset of `scb_dataset3`)

- **Physical Path**: `C:\WorkingSpace Python\DETECTOR-YOLO\datasets\raw\scb_dataset3\SCB5-Handrise-Read-write-2024-9-17`
- **Detected YAML**: `datasets/raw/scb_dataset3/SCB5-Handrise-Read-write-2024-9-17/data.yaml`
- **Exact YAML Contents**:
  ```yaml
  train: /data/SCB/SCB-dataset/SCB5-Handrise-Read-write-2024-9-17/images/train
  val: /data/SCB/SCB-dataset/SCB5-Handrise-Read-write-2024-9-17/images/val
  nc: 3
  names: ['hand-raising', 'read', 'write']
  ```
- **Actual Local Class Names**:
  - `0`: `hand-raising`
  - `1`: `read`
  - `2`: `write`
- **Documentation / License Files**:
  - `metadata.json`: Tracked in parent directory (Academic / Non-commercial research license).
- **Directory Structure**:
  - `images/`: `train/` (5,491 images), `val/` (1,373 images)
  - `labels/`: `train/` (5,491 labels), `val/` (1,373 labels)
- **Total Images**: 6,864
- **Total Labels**: 6,864 (100% paired, 0 unpaired)
- **Total Annotations**: 47,372
  - Class 0 (`hand-raising`): 13,453 annotations across 4,064 images
  - Class 1 (`read`): 24,078 annotations across 3,878 images
  - Class 2 (`write`): 9,841 annotations across 1,721 images
- **Source Identity Confidence**: **HIGH**
- **Mismatch / Conflict**:
  - *CRITICAL PAST CONFLICT*: Previously, `scb_dataset3` was audited using a single flattened list `['discuss', 'hand-raising', 'read', 'write']`. Because Class 0 in this subfolder is `hand-raising`, the previous flat script misattributed all 13,453 `hand-raising` annotations to `discuss`! This is now isolated and corrected.

---

### C. Dataset: `SCB_BowTurnHead_20250509` (Sub-dataset of `scbehavior`)

- **Physical Path**: `C:\WorkingSpace Python\DETECTOR-YOLO\datasets\raw\scbehavior\SCB_BowTurnHead_20250509`
- **Detected YAML**: `datasets/raw/scbehavior/SCB_BowTurnHead_20250509/data.yaml`
- **Exact YAML Contents**:
  ```yaml
  train: /root/SCB_BowTurnHead_20250509/images/train
  val: /root/SCB_BowTurnHead_20250509/images/val
  nc: 2
  names: ['BowHead', 'TurnHead']
  ```
- **Actual Local Class Names**:
  - `0`: `BowHead`
  - `1`: `TurnHead`
- **Documentation / License Files**:
  - `metadata.json`: Research Use Only.
- **Directory Structure**:
  - `images/`: `train/` (1,928 images), `val/` (482 images)
  - `labels/`: `train/` (1,928 labels), `val/` (482 labels)
- **Total Images**: 2,410
- **Total Labels**: 2,410 (100% paired, 0 unpaired)
- **Total Annotations**: 16,118
  - Class 0 (`BowHead`): 4,962 annotations across 1,001 images
  - Class 1 (`TurnHead`): 11,156 annotations across 2,098 images
- **Source Identity Confidence**: **HIGH**
- **Mismatch / Conflict**: None. Only `BowHead` and `TurnHead` exist in this archive.

---

### D. Dataset: `cctv_exam_monitor`

- **Physical Path**: `C:\WorkingSpace Python\DETECTOR-YOLO\datasets\raw\cctv_exam_monitor`
- **Detected YAML**: None
- **Total Images**: 0
- **Total Labels**: 0
- **Physical Status**: **NOT LOCALLY DOWNLOADED**
- **Expected Classes from Kaggle Registry**:
  `Correct Posture`, `LeftSideMove`, `RightSideMove`, `ForwardMove`, `BackwardMove`, `Stand`.
- **Identity Confidence**: **LOW (Pending Physical Download)**

---

### E. Datasets: `exam_cheating_roboflow` & `exam_cheating_kaggle`

- **Physical Paths**:
  - `C:\WorkingSpace Python\DETECTOR-YOLO\datasets\raw\exam_cheating_roboflow`
  - `C:\WorkingSpace Python\DETECTOR-YOLO\datasets\raw\exam_cheating_kaggle`
- **Detected YAML**: None
- **Total Images**: 0
- **Total Labels**: 0
- **Physical Status**: **NOT LOCALLY DOWNLOADED**
- **Identity Confidence**: **N/A**
