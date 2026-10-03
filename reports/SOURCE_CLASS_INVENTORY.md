# Ground-Truth Source Class Inventory (Local Data Only)

**Generated Date**: 2026-10-02  
**Inventory Scope**: Strictly compiled from verified, locally downloaded YOLO datasets and their respective physical `data.yaml` files.  
**Constraint Enforced**: Excludes all speculative, un-downloaded, or hypothetical classes.

---

## 1. Verified Local Dataset Summary

| Dataset Identifier | Physical Directory | YOLO Configuration | Image Count | Label Count | Total Annotations |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `scb_discuss` | `datasets/raw/scb_dataset3/SCB5-Discuss-2024-9-17` | `data.yaml` | 864 | 864 | 5,392 |
| `scb_handrise_read_write` | `datasets/raw/scb_dataset3/SCB5-Handrise-Read-write-2024-9-17` | `data.yaml` | 6,864 | 6,864 | 47,372 |
| `scbehavior_bow_turn` | `datasets/raw/scbehavior/SCB_BowTurnHead_20250509` | `data.yaml` | 2,410 | 2,410 | 16,118 |
| **Total Physical Assets** | — | — | **10,138** | **10,138** | **68,882** |

---

## 2. Ground-Truth Source Class Breakdown

### A. Dataset: `scb_discuss` (`SCB5-Discuss-2024-9-17`)
- **Physical YAML**: `datasets/raw/scb_dataset3/SCB5-Discuss-2024-9-17/data.yaml`
- **Class Count (`nc`)**: 1

| Class ID | Class Name | Annotations | Total Images Containing Class | Share of Dataset Annotations | Example Contact Sheet Path |
| :---: | :--- | :---: | :---: | :---: | :--- |
| `0` | **`discuss`** | 5,392 | 864 | 100.0% | [reports/class_samples_verified/scb_discuss_discuss.jpg](file:///c:/WorkingSpace%20Python/DETECTOR-YOLO/reports/class_samples_verified/scb_discuss_discuss.jpg) |

---

### B. Dataset: `scb_handrise_read_write` (`SCB5-Handrise-Read-write-2024-9-17`)
- **Physical YAML**: `datasets/raw/scb_dataset3/SCB5-Handrise-Read-write-2024-9-17/data.yaml`
- **Class Count (`nc`)**: 3

| Class ID | Class Name | Annotations | Total Images Containing Class | Share of Dataset Annotations | Example Contact Sheet Path |
| :---: | :--- | :---: | :---: | :---: | :--- |
| `0` | **`hand-raising`** | 13,453 | 4,064 | 28.4% | [reports/class_samples_verified/scb_handrise_read_write_hand_raising.jpg](file:///c:/WorkingSpace%20Python/DETECTOR-YOLO/reports/class_samples_verified/scb_handrise_read_write_hand_raising.jpg) |
| `1` | **`read`** | 24,078 | 3,878 | 50.8% | [reports/class_samples_verified/scb_handrise_read_write_read.jpg](file:///c:/WorkingSpace%20Python/DETECTOR-YOLO/reports/class_samples_verified/scb_handrise_read_write_read.jpg) |
| `2` | **`write`** | 9,841 | 1,721 | 20.8% | [reports/class_samples_verified/scb_handrise_read_write_write.jpg](file:///c:/WorkingSpace%20Python/DETECTOR-YOLO/reports/class_samples_verified/scb_handrise_read_write_write.jpg) |

---

### C. Dataset: `scbehavior_bow_turn` (`SCB_BowTurnHead_20250509`)
- **Physical YAML**: `datasets/raw/scbehavior/SCB_BowTurnHead_20250509/data.yaml`
- **Class Count (`nc`)**: 2

| Class ID | Class Name | Annotations | Total Images Containing Class | Share of Dataset Annotations | Example Contact Sheet Path |
| :---: | :--- | :---: | :---: | :---: | :--- |
| `0` | **`BowHead`** | 4,962 | 1,001 | 30.8% | [reports/class_samples_verified/scbehavior_bow_turn_BowHead.jpg](file:///c:/WorkingSpace%20Python/DETECTOR-YOLO/reports/class_samples_verified/scbehavior_bow_turn_BowHead.jpg) |
| `1` | **`TurnHead`** | 11,156 | 2,098 | 69.2% | [reports/class_samples_verified/scbehavior_bow_turn_TurnHead.jpg](file:///c:/WorkingSpace%20Python/DETECTOR-YOLO/reports/class_samples_verified/scbehavior_bow_turn_TurnHead.jpg) |

---

## 3. Overall Local Class Aggregation Across All Datasets

| Source Class Name | Datasets Containing Class | Total Annotations Across All Datasets | Total Images Containing Class |
| :--- | :--- | :---: | :---: |
| `read` | `scb_handrise_read_write` | 24,078 | 3,878 |
| `hand-raising` | `scb_handrise_read_write` | 13,453 | 4,064 |
| `TurnHead` | `scbehavior_bow_turn` | 11,156 | 2,098 |
| `write` | `scb_handrise_read_write` | 9,841 | 1,721 |
| `discuss` | `scb_discuss` | 5,392 | 864 |
| `BowHead` | `scbehavior_bow_turn` | 4,962 | 1,001 |
| **Combined** | — | **68,882** | **10,138** |

---

## 4. Un-Downloaded / Missing Candidate Datasets

The following datasets do NOT physically exist on disk and contribute **0** samples to the current inventory:
- `cctv_exam_monitor` (Kaggle: `cctvdataset/cctv-exam-monitor-dataset`) — *0 images / 0 annotations*
- `exam_cheating_roboflow` — *0 images / 0 annotations*
- `exam_cheating_kaggle` — *0 images / 0 annotations*
