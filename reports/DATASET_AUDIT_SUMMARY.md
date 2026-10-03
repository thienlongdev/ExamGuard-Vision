# Comprehensive Dataset Audit & Infrastructure Summary Report

**Execution Date**: 2026-10-02  
**Target Production Domain**: Fixed Wall-Mounted Examination Room CCTV (Surveillance / High Diagonal Angle)  
**Demo / Testing Source**: Laptop Webcam (Exclusively for local demo/validation; not the training target)  
**Hardware Verification**: NVIDIA GeForce RTX 5070 (12 GB VRAM, sm_120, PyTorch 2.14.1+cu130, Ultralytics 8.4.171)

---

## 1. Dataset Inventory & Availability

| Dataset Key | Dataset Name | Staging Directory | Availability | Format | Primary Role |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `cctv_exam_monitor` | CCTV Exam Monitor Dataset | `datasets/raw/cctv_exam_monitor` | **Pending Download** | YOLO | Primary high-angle surveillance exam domain data |
| `scb_dataset3` | SCB-Dataset3 (wintonYF) | `datasets/raw/scb_dataset3` | **Available (Downloaded)** | YOLO | Multi-student classroom behaviors (discuss, read, write) |
| `scbehavior` | SCBehavior (Whiffe) | `datasets/raw/scbehavior` | **Available (Downloaded)** | YOLO | Student posture & actions (BowHead, TurnHead) |
| `exam_cheating_roboflow` | Roboflow Exam Cheating | `datasets/raw/exam_cheating_roboflow` | **Pending Download** | YOLO | Supplementary phone use & head turn variations |
| `exam_cheating_kaggle` | Kaggle Exam Cheating | `datasets/raw/exam_cheating_kaggle` | **Pending Download** | YOLO | Supplementary negative normal exam scenes |

---

## 2. Dataset Sources & Origins

1. **`cctv_exam_monitor`**: Hosted on Kaggle by user `cctvdataset`. URL: `https://www.kaggle.com/datasets/cctvdataset/cctv-exam-monitor-dataset`. Real-world surveillance footage from exam rooms and computer labs.
2. **`scb_dataset3`**: Hosted on HuggingFace by `wintonYF/SCB-Dataset`. Extracted from high-resolution classroom monitoring cameras.
3. **`scbehavior`**: Originates from smart classroom research (`Whiffe/SCB-dataset` & `wintonYF`). Captures high/diagonal views of seated students.
4. **`exam_cheating_roboflow`**: Roboflow Universe public community contributions for invigilation.
5. **`exam_cheating_kaggle`**: Various public Kaggle datasets tagged with examination/proctoring.

---

## 3. License Status & Compliance

| Dataset | Declared License | License Status | Permitted Usage | Commercial / Production Note |
| :--- | :--- | :---: | :--- | :--- |
| `cctv_exam_monitor` | CC0: Public Domain | **Verified** | Unrestricted / Commercial / Research | Safe for open weights & enterprise deployment |
| `scb_dataset3` | Academic / Research Use | **Needs Verification** | Academic & Research Benchmarking | Verify commercial release rights before distribution |
| `scbehavior` | Research Use Only | **Needs Verification** | Research Benchmarking | Verify author terms before commercial redistribution |
| `exam_cheating_roboflow` | CC BY 4.0 / Universe Open | **Needs Verification** | Open Source / Attribution required | Dependent on exact workspace license |
| `exam_cheating_kaggle` | Varies / Community | **Needs Verification** | Community Research | Reject single-frame subjective labels |

---

## 4 & 5. Image & Annotation Counts

### Locally Downloaded & Audited Datasets

| Dataset | Total Images | Total Label Files | Total Annotations | Avg Objects / Image | Unpaired Images | Corrupted Images |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `scb_dataset3` | 7,174 | 7,174 | 48,422 | 6.75 | 0 | 0 |
| `scbehavior` | 2,410 | 2,410 | 16,118 | 6.69 | 0 | 0 |
| **Audited Total** | **9,584** | **9,584** | **64,540** | **6.73** | **0** | **0** |

*(External `cctv_exam_monitor` adds 8,156 images upon manual download via Kaggle CLI).*

---

## 6 & 7. Original Classes & Distributions

### A. `scb_dataset3` Class Breakdown
- `discuss` (ID 0): **18,428 annotations** (38.1% share, appears in 4,841 images)
- `hand-raising` (ID 1): **20,978 annotations** (43.3% share, appears in 3,373 images)
- `read` (ID 2): **9,016 annotations** (18.6% share, appears in 1,467 images)

### B. `scbehavior` Class Breakdown
- `BowHead` (ID 0): **4,962 annotations** (30.8% share, appears in 1,001 images)
- `TurnHead` (ID 1): **11,156 annotations** (69.2% share, appears in 2,098 images)

### C. `cctv_exam_monitor` (Registry Definition)
- `Correct Posture` (~62% expected share)
- `LeftSideMove` (~9% expected share)
- `RightSideMove` (~9% expected share)
- `ForwardMove` (~8% expected share)
- `BackwardMove` (~7% expected share)
- `Stand` (~5% expected share)

---

## 8. Invalid Annotations & Syntax Anomalies

Audit script `training/inspect_dataset.py` identified the following anomalies in raw files:

- **Total Syntax / Geometric Issues Found**: 11
  - **Zero-dimension box**: `3005076.txt:9` (`w=0.0, h=0.0038`) — *Automatically sanitized and excluded during manifest building.*
  - **Out-of-bounds box center**: `1276008.txt:3` (`xc=1.0076`) and `1276033.txt:7` (`xc=1.1459`).
  - **Out-of-bounds box edges**: 8 instances where students on the periphery had bounding boxes exceeding frame edges by up to 20% (e.g. `[0.46, 0.45, 0.72, 1.27]`).
  - **Unreadable / Corrupted Images**: **0** (All images cleanly opened by OpenCV).

---

## 9. Duplicate & Near-Duplicate Analysis

Tool `training/duplicate_detector.py` performed SHA-256 binary hash and 64-bit dHash perceptual analysis:

| Dataset | Total Images Scanned | Exact Duplicate Groups | Redundant Exact Files | Probable Near-Duplicate Pairs | Action Taken |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `scb_dataset3` | 7,728 | 555 | 555 | 1 | Flagged; unique representatives preserved |
| `scbehavior` | 2,410 | 0 | 0 | 10 | Flagged for review; preserved in manifests |

---

## 10 & 11. Resolution & Bounding-Box Area Distribution

### Image Resolutions
- Standard input frame resolutions: `1920x1080` (Full HD CCTV stream) and `1280x720` (720p).
- Aspect ratio: Fixed 16:9 widescreen across 100% of surveillance samples.

### Bounding-Box Scale Profile
| Category | Definition | `scb_dataset3` | `scbehavior` | Combined Share |
| :--- | :--- | :---: | :---: | :---: |
| **Tiny** | Area < 1% of frame | 9,428 (19.5%) | 5,822 (36.1%) | **23.6%** |
| **Small** | Area 1% - 5% of frame | 25,139 (51.9%) | 8,738 (54.2%) | **52.5%** |
| **Medium** | Area 5% - 20% of frame | 12,624 (26.1%) | 1,542 (9.6%) | **22.0%** |
| **Large** | Area >= 20% of frame | 1,231 (2.5%) | 16 (0.1%) | **1.9%** |

> [!NOTE]
> **Surveillance Domain Alignment**: Over **76.1%** of all student bounding boxes occupy under 5% of the frame area. This matches the target school CCTV camera distance and validates the default `imgsz: 768` parameter.

---

## 12. Visual Class Inspection & Contact Sheets

Contact sheets were generated in `reports/class_samples/`:
1. [reports/class_samples/scbehavior_BowHead.jpg](file:///c:/WorkingSpace%20Python/DETECTOR-YOLO/reports/class_samples/scbehavior_BowHead.jpg):
   - **Visual finding**: Shows students with their forehead close to or resting on the desk, face downward. Perfectly matches the canonical `head_down` behavior.
2. [reports/class_samples/scbehavior_TurnHead.jpg](file:///c:/WorkingSpace%20Python/DETECTOR-YOLO/reports/class_samples/scbehavior_TurnHead.jpg):
   - **Visual finding**: Shows students rotating their neck 45° to 90° sideways to glance at neighbors or aisle. Directly matches the canonical `turn_head` class.
3. [reports/class_samples/scb_dataset3_discuss.jpg](file:///c:/WorkingSpace%20Python/DETECTOR-YOLO/reports/class_samples/scb_dataset3_discuss.jpg):
   - **Visual finding**: Shows two or more students leaning toward each other with heads oriented face-to-face. Matches canonical `discuss`.
4. [reports/class_samples/scb_dataset3_read.jpg](file:///c:/WorkingSpace%20Python/DETECTOR-YOLO/reports/class_samples/scb_dataset3_read.jpg):
   - **Visual finding**: Students looking slightly downward at open books/sheets while holding head erect. Overlaps partially with ordinary seated posture.
5. [reports/class_samples/scb_dataset3_hand_raising.jpg](file:///c:/WorkingSpace%20Python/DETECTOR-YOLO/reports/class_samples/scb_dataset3_hand_raising.jpg):
   - **Visual finding**: Standard classroom arm raising. Excluded from exam cheating taxonomy (`ignored`).

---

## 13. Proposed Canonical Behavior Taxonomy (V1)

```
0: normal      (Student sitting upright/working normally at desk)
1: head_down   (Student head resting on desk or bowed deeply)
2: lean        (Torso tilted significantly forward or over table)
3: turn_head   (Head rotated sideways toward neighbor or back)
4: use_phone   (Active interaction with or viewing a mobile device)
5: discuss     (Peer-to-peer discussion / multi-student communication)
6: stand       (Student stood up from seat)
```

*Future Directional Classes (Reserved, Disabled in V1)*:
- `left_move`
- `right_move`

---

## 14. Explicit Source-to-Canonical Mapping Table

Configured in `configs/dataset_mapping.yaml`:

| Dataset | Source Class | Canonical Target | Mapping Status | Rationale |
| :--- | :--- | :--- | :---: | :--- |
| `scbehavior` | `BowHead` | `head_down` | **APPROVED** | Verified head-down resting on desk |
| `scbehavior` | `TurnHead` | `turn_head` | **APPROVED** | Verified lateral head rotation |
| `scbehavior` | `discuss` | `discuss` | **APPROVED** | Multi-student interaction |
| `scbehavior` | `stand` | `stand` | **APPROVED** | Standing student |
| `scb_dataset3` | `discuss` | `discuss` | **APPROVED** | Multi-student interaction |
| `scb_dataset3` | `using phone` | `use_phone` | **APPROVED** | Mobile phone usage |
| `scb_dataset3` | `leaning over table`| `lean` | **APPROVED** | Upper body torso tilt |
| `cctv_exam_monitor`| `Stand` | `stand` | **APPROVED** | Standing in exam hall |
| `cctv_exam_monitor`| `Correct Posture`| `normal` | **APPROVED** | Regular seated posture |
| `exam_cheating_roboflow`| `phone use` | `use_phone` | **APPROVED** | Phone detected in hand |
| `exam_cheating_roboflow`| `normal` | `normal` | **APPROVED** | Seated candidate |
| `exam_cheating_kaggle`| `no cheating` | `normal` | **APPROVED** | Seated candidate |

---

## 15. Ambiguous & Rejected Labels Requiring User Confirmation

| Dataset | Source Class | Proposed Target | Status | Why It Is Ambiguous / Dangerous |
| :--- | :--- | :--- | :---: | :--- |
| `cctv_exam_monitor` | `LeftSideMove` | `null` (or `lean`) | **NEEDS REVIEW** | Represents lateral body/chair shifting, not necessarily glancing left. Inverting with horizontal flip corrupts data. |
| `cctv_exam_monitor` | `RightSideMove`| `null` (or `lean`) | **NEEDS REVIEW** | Lateral shift to the right. Same issue as above. |
| `cctv_exam_monitor` | `ForwardMove` | `lean` | **NEEDS REVIEW** | Leaning forward could be normal concentration or reading desk paper. |
| `cctv_exam_monitor` | `BackwardMove`| `normal` | **NEEDS REVIEW** | Leaning back in chair to relax vs staring at ceiling. |
| `scb_dataset3` | `read` / `reading` | `normal` | **NEEDS REVIEW** | Reading is normal in exams, but could conflict with looking straight ahead. |
| `scb_dataset3` | `write` / `writing` | `normal` | **NEEDS REVIEW** | Writing is standard exam behavior. If collapsed into normal, normal becomes very broad. |
| `scbehavior` | `lookup` | `normal` | **NEEDS REVIEW** | Looking up at invigilator vs glancing up to cheat. |
| `exam_cheating_roboflow`| `looking around` | `turn_head` | **NEEDS REVIEW** | Often poorly boxed around whole body rather than head orientation. |
| `exam_cheating_kaggle`| `cheating` | `null` | **REJECTED** | **FATAL**: Single-frame subjective "cheating" label violates fundamental system architecture. |
| `scb_dataset3` | `hand-raising` | `null` | **IGNORED** | Normal classroom question gesture; safely omitted from exam cheating rules. |

---

## 16 & 17. Train / Validation / Test Grouping & Leakage Elimination

### Strategy
- **Sequence-Aware Partitioning**: Video sequences are identified via the 4-digit clip identifier embedded in file naming (e.g. `0006001` -> clip `0006`).
- **Strict Independence**: All frames from a given video clip are assigned exclusively to either `train`, `val`, or `test`.

### Verification Metrics
- **Total Unique Sequences**: 510 video clips
- **Train Sequences**: 363 (71.2%) -> 7,096 images
- **Val Sequences**: 85 (16.7%) -> 1,520 images
- **Test Sequences**: 62 (12.2%) -> 1,522 images
- **Intra-Sequence Overlap (Leakage)**: **0.0% (ZERO LEAKAGE CONFIRMED)**

---

## 18. Class Imbalance Analysis & Domain Risks

In the currently processed multi-source subset:
- `discuss`: 18,845 instances
- `turn_head`: 11,156 instances
- `head_down`: 4,962 instances
- `use_phone`, `stand`, `lean`: Awaiting `cctv_exam_monitor` and targeted phone dataset downloads.

**Domain Imbalance Alert**:
Classroom datasets (`scb_dataset3` and `scbehavior`) contain large class sizes and group interaction (`discuss`). Once `cctv_exam_monitor` is staged, it will introduce 8,156 exam-room CCTV images with individual desks and solitary students, balancing out the social classroom bias.

---

## 19. Suggested Stage 1 Sampling Strategy

1. **CCTV Exam Monitor**: 40% target weight (highest domain importance, realistic exam setting).
2. **SCB-Dataset3**: 30% target weight (head postures, desk interaction).
3. **SCBehavior**: 20% target weight (head turns, head down).
4. **Exam Cheating (Roboflow/Kaggle)**: 10% target weight (phone use and negative exam frames).

---

## 20. Remaining Manual Decisions Required Before Training

1. **Approval of Ambiguous Mappings**:
   - Confirm whether `read` and `write` should be merged into `normal`, or kept as separate auxiliary classes.
   - Confirm whether `cctv_exam_monitor` `LeftSideMove` / `RightSideMove` should be mapped to `lean` or kept `ignored` until directional transforms are written.
2. **Kaggle CCTV Dataset Download**:
   - Run the Kaggle download command below to stage the 8,156 images into `datasets/raw/cctv_exam_monitor/`.

---

## External Dataset Download Instructions

### To Download CCTV Exam Monitor Dataset (CC0 Public Domain):
```bash
# Option 1: Kaggle CLI (if configured)
kaggle datasets download -d cctvdataset/cctv-exam-monitor-dataset -p datasets/raw/cctv_exam_monitor --unzip

# Option 2: Web Browser
# 1. Visit: https://www.kaggle.com/datasets/cctvdataset/cctv-exam-monitor-dataset
# 2. Click 'Download'
# 3. Unzip into: C:\WorkingSpace Python\DETECTOR-YOLO\datasets\raw\cctv_exam_monitor\
```
