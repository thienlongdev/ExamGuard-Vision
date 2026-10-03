# Comprehensive Dataset Integrity, Verification, and Pre-Training Audit Report

**Date**: 2026-10-02  
**Target Environment**: Fixed Wall-Mounted Examination Room CCTV (Surveillance Perspective)  
**System Hardware**: NVIDIA GeForce RTX 5070 (11.94 GB Dedicated VRAM, Blackwell `sm_120`)  
**Audit Purpose**: Complete physical verification of local datasets, elimination of class mapping hallucinations, duplicate and sequence leakage eradication, batch/VRAM calibration, and mini-training sanity validation.

---

## 1. GPU Status

- **Device**: `cuda:0` — **NVIDIA GeForce RTX 5070**
- **Dedicated Video Memory**: `11.94 GB` (12,227 MiB)
- **PyTorch / CUDA Driver**: `PyTorch 2.14.1+cu130`, CUDA 13.0, cuDNN 9.x
- **Compute Capability**: `sm_120` (Blackwell generation)
- **Status**: **ACTIVE & VERIFIED** (CUDA forward, mixed-precision AMP, and backward propagation fully validated).

---

## 2. Model Checkpoint Verification

- **Target Checkpoint**: `yolo26m.pt`
- **Model Task**: `detect` (Object Detection)
- **Architecture**: `<class 'ultralytics.nn.tasks.DetectionModel'>`
- **Total Parameters**: `21,896,248` (Fused: `20,354,849`)
- **Forward Pass Benchmark**: Shape `(1, 3, 640, 640)` runs in **9.3 ms** on `cuda:0`.
- **Training API Compatibility**: Native Ultralytics `YOLO("yolo26m.pt").train()` loads and trains without fallback.
- **Fallback Candidate**: `yolo11m.pt` remains designated as a secondary fallback, but was not needed because `yolo26m.pt` is physically confirmed and fully operational.

---

## 3. Local Dataset Inventory

Physical scan of `datasets/raw/` revealed **3 distinct dataset directories**:

| Dataset Directory | Absolute Path | Format | Status |
| :--- | :--- | :---: | :---: |
| **`scb_discuss`** | `datasets/raw/scb_dataset3/SCB5-Discuss-2024-9-17` | YOLO Detection | **PHYSICALLY PRESENT** |
| **`scb_handrise_read_write`** | `datasets/raw/scb_dataset3/SCB5-Handrise-Read-write-2024-9-17` | YOLO Detection | **PHYSICALLY PRESENT** |
| **`scbehavior_bow_turn`** | `datasets/raw/scbehavior/SCB_BowTurnHead_20250509` | YOLO Detection | **PHYSICALLY PRESENT** |

---

## 4. Exact `data.yaml` Contents & Class Names

Each physically existing local dataset was inspected directly from its ground-truth YAML definition:

### A. `scb_discuss` (`SCB5-Discuss-2024-9-17/data.yaml`)
```yaml
nc: 1
names:
  - discuss
```
- Total Images: 864
- Total Labels: 864
- Total Annotations: 5,392 (100% `discuss`)

### B. `scb_handrise_read_write` (`SCB5-Handrise-Read-write-2024-9-17/data.yaml`)
```yaml
nc: 3
names:
  - hand-raising
  - read
  - write
```
- Total Images: 6,864
- Total Labels: 6,864
- Total Annotations: 47,372 (`hand-raising`: 13,453, `read`: 24,078, `write`: 9,841)

### C. `scbehavior_bow_turn` (`SCB_BowTurnHead_20250509/data.yaml`)
```yaml
nc: 2
names:
  - BowHead
  - TurnHead
```
- Total Images: 2,410
- Total Labels: 2,410
- Total Annotations: 16,118 (`BowHead`: 4,962, `TurnHead`: 11,156)

---

## 5. Previous Audit Contradictions Found

Comparison against `reports/DATASET_AUDIT_SUMMARY.md` exposed 5 critical integrity errors:

1. **Hallucinated Class Mappings**: The previous audit proposed mappings for `using phone`, `leaning over table`, `stand`, and `lookup`. Physical inspection proved **zero instances** of these classes exist in any downloaded local folder. These were paper-level concepts from the full SCB benchmark, not present in the downloaded modular releases.
2. **Harmful Sub-Dataset Flattening**: `SCB5-Discuss` and `SCB5-Handrise-Read-write` were previously flattened under the single name `scb_dataset3`. Because Class 0 in Discuss is `discuss` while Class 0 in Handrise is `hand-raising`, flattening them caused 13,453 `hand-raising` bounding boxes to be falsely labeled as `discuss`, while completely erasing the `write` class (Class 2).
3. **Class Name Casing Contradiction**: The previous report referenced `Turn Head` and `bowing head`. The actual ground-truth YAML classes are camel-case `TurnHead` and `BowHead`.
4. **Duplicate Physical Frames Across Subsets**: 2,022 images in `SCB5-Handrise-Read-write` were identical raw camera frames to images in `SCB_BowTurnHead_20250509`. Processing them as separate datasets caused identical images to receive different sequence IDs and bleed across train/val splits.
5. **Direct "Cheating" Classification**: The previous plan contemplated mapping single-frame exam cheating datasets directly to detector outputs. This violated the core system architecture (detection of observable physical behaviors only, leaving suspicious event judgment to temporal rules and human invigilators).

---

## 6. Corrected Source Class Inventory

Generated exclusively from physical annotations present on disk (68,882 total raw annotations across 10,138 raw image files):

| Dataset Key | Class ID | Source Class Name | Annotation Count | Image Count | Visual Sample Reference |
| :--- | :---: | :--- | :---: | :---: | :--- |
| `scb_discuss` | 0 | `discuss` | 5,392 | 864 | `reports/class_samples_verified/scb_discuss_discuss.jpg` |
| `scb_handrise_read_write` | 0 | `hand-raising` | 13,453 | 4,210 | `reports/class_samples_verified/scb_handrise_read_write_hand_raising.jpg` |
| `scb_handrise_read_write` | 1 | `read` | 24,078 | 5,821 | `reports/class_samples_verified/scb_handrise_read_write_read.jpg` |
| `scb_handrise_read_write` | 2 | `write` | 9,841 | 3,118 | `reports/class_samples_verified/scb_handrise_read_write_write.jpg` |
| `scbehavior_bow_turn` | 0 | `BowHead` | 4,962 | 1,842 | `reports/class_samples_verified/scbehavior_bow_turn_BowHead.jpg` |
| `scbehavior_bow_turn` | 1 | `TurnHead` | 11,156 | 2,298 | `reports/class_samples_verified/scbehavior_bow_turn_TurnHead.jpg` |

---

## 7. Verified Visual Interpretation per Class

All classes were visually audited via 25–30 sample contact sheets generated with high-contrast bounding boxes and metadata overlays:

1. **`read`**: Contact sheet confirms normal seated posture, eyes and face directed down at exam paper/desk surface. Bounding box captures torso and head. Semantically represents baseline exam posture.
2. **`write`**: Contact sheet confirms seated student holding pen/pencil writing on paper. Posture is identical to ordinary seated desk work.
3. **`BowHead`**: Contact sheet shows students bowing head deeply onto desk or resting forehead on desk. Distinct acute downward pitch compared to standard writing posture.
4. **`TurnHead`**: Contact sheet shows head yaw angles from 45° to 90° away from the student's own desk toward adjacent students, aisles, or ceiling.
5. **`discuss`**: Contact sheet shows two or more students turning upper bodies toward each other, actively engaging in multi-person interaction.
6. **`hand-raising`**: Contact sheet shows one arm extended upward in standard classroom lecture participation.

---

## 8. Final Proposed Canonical Taxonomy

The canonical taxonomy for V1 behavior detection is strictly bounded to observable physical postures:

```
0: normal      (baseline seated reading, writing, and standard desk work)
1: head_down   (deep head bow or sleeping on desk)
2: lean        (lateral or forward torso deviation from chair centerline)
3: turn_head   (lateral head yaw >45 degrees away from desk)
4: use_phone   (phone visible in hand or on desk surface)
5: discuss     (two or more students turning/interacting)
6: stand       (standing up from examination seat)
```

**Training Availability Status**:
- `0 normal`: **AVAILABLE** (from `read` and `write`)
- `1 head_down`: **AVAILABLE** (from `BowHead`)
- `2 lean`: **NO VERIFIED TRAINING SOURCE CURRENTLY AVAILABLE** (requires CCTV Exam Monitor)
- `3 turn_head`: **AVAILABLE** (from `TurnHead`)
- `4 use_phone`: **NO VERIFIED TRAINING SOURCE CURRENTLY AVAILABLE** (requires phone detection module/data)
- `5 discuss`: **AVAILABLE** (from `discuss`)
- `6 stand`: **NO VERIFIED TRAINING SOURCE CURRENTLY AVAILABLE** (requires CCTV Exam Monitor)

---

## 9. Source -> Canonical Mapping Matrix

Defined in `configs/dataset_mapping.yaml`:

| Source Dataset | Source Class | Canonical Target | Status | Rationale |
| :--- | :--- | :--- | :---: | :--- |
| `scb_discuss` | `discuss` | `discuss` (5) | **approved** | Visual contact sheet confirms 2+ students turning toward each other in conversation. |
| `scb_handrise_read_write` | `read` | `normal` (0) | **approved** | Standard seated desk reading behavior. |
| `scb_handrise_read_write` | `write` | `normal` (0) | **approved** | Standard seated desk writing behavior. |
| `scb_handrise_read_write` | `hand-raising` | `null` | **ignored** | Benign classroom participation, out of scope for suspicious exam behavior. |
| `scbehavior_bow_turn` | `BowHead` | `head_down` (1) | **approved** | Student forehead resting on or bowed deeply toward desk surface. |
| `scbehavior_bow_turn` | `TurnHead` | `turn_head` (3) | **approved** | Lateral head yaw >45 degrees away from desk. |

---

## 10. Classes Still Needing Review

These classes reside in candidate external datasets and remain flagged as `needs_review` (strictly excluded from training):

- `cctv_exam_monitor` / `LeftSideMove` (Ambiguous: lateral body displacement vs. fidgeting)
- `cctv_exam_monitor` / `RightSideMove` (Ambiguous: lateral body displacement vs. fidgeting)
- `cctv_exam_monitor` / `ForwardMove` (Must distinguish severe torso lean from ordinary forward writing lean)
- `cctv_exam_monitor` / `BackwardMove` (Reclining in chair; requires visual inspection)

---

## 11. Ignored Classes

- **`hand-raising`** (`scb_handrise_read_write`): Safely excluded from processed manifests. Images containing only `hand-raising` are retained as background negative frames without pseudo-labels.

---

## 12. Rejected Classes

- **`cheating`** (`exam_cheating_kaggle`): **STRICTLY REJECTED**. Single-frame subjective cheating annotations violate architectural doctrine. Under no circumstances will the detector emit a "cheating" class token.

---

## 13. CCTV Exam Monitor Status

- **Status**: **NOT CURRENTLY AVAILABLE LOCALLY**
- **Action Taken**: Checked `datasets/raw/cctv_exam_monitor/`. Folder is not populated.
- **Kaggle Credentials**: Kaggle CLI credentials (`~/.kaggle/kaggle.json`) are not configured in the host environment.
- **Protocol Enforced**: In accordance with instructions, no synthetic credentials were created, and no fake entries were injected into processed datasets.
- **Remediation**: Instructions provided in Section 25 for manual user download.

---

## 14. Exam Roboflow / Kaggle Status

- `datasets/raw/exam_cheating_roboflow/`: **UNAVAILABLE**
- `datasets/raw/exam_cheating_kaggle/`: **UNAVAILABLE**
- Both marked `PENDING_PHYSICAL_DOWNLOAD` in `configs/dataset_mapping.yaml`.

---

## 15. Duplicate Analysis Findings

- **Physical Identical Image Clusters**: 1,892 SHA-256 duplicate clusters exist across the raw directories.
- **Cross-Subset Duplication**: 2,022 files in `SCB5-Handrise-Read-write` share exact SHA-256 hashes with files in `SCB_BowTurnHead_20250509` (same raw video frames annotated independently for different behaviors).
- **Resolution Implemented**:
  1. All duplicate records are unified by image SHA-256 hash.
  2. Multi-task annotations are merged into a single non-contradictory frame label.
  3. Connected components grouping guarantees that any groups sharing identical frames are merged into the same split.

---

## 16. Sequence Grouping Evidence

- **Grouping Methodology**: Scene-aware grouping using 4-digit sequence prefix (`clip_{stem[:4]}` for numeric stems) and video prefix (`clip_video_{id}` for underscore stems).
- **Perceptual dHash Empirical Verification**:
  - Intra-sequence image distance: **8** (smooth continuous motion of same students).
  - Inter-sequence image distance: **25** (distinct rooms, angles, and lighting).
- **Classification**: **HEURISTIC / EMPIRICAL** (Confidence: **HIGH**). Documented in `reports/SPLIT_GROUPING_VERIFICATION.md`.

---

## 17. Train / Val / Test Partitioning

Generated in `datasets/processed/`:

| Split | Unique Frames | Sequence Clips | Normal | Head Down | Turn Head | Discuss | Total BBoxes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 5,665 | 136 | 21,836 | 2,139 | 6,052 | 3,962 | 33,989 |
| **Validation** | 1,218 | 27 | 4,714 | 142 | 775 | 446 | 6,077 |
| **Test** | 1,233 | 26 | 5,845 | 1,822 | 3,745 | 984 | 12,396 |
| **Total** | **8,116** | **189** | **32,395** | **4,103** | **10,572** | **5,392** | **52,462** |

---

## 18. Cross-Split Leakage Checks

Enforced and audited via `scripts/validate_preflight.py`:

- **Train ∩ Val Shared Frames**: **0**
- **Train ∩ Test Shared Frames**: **0**
- **Val ∩ Test Shared Frames**: **0**
- **Train ∩ Val Shared Sequence Groups**: **0**
- **Train ∩ Test Shared Sequence Groups**: **0**
- **Val ∩ Test Shared Sequence Groups**: **0**
- **Leakage Status**: **ZERO CROSS-SPLIT LEAKAGE ACHIEVED (100% ISOLATED)**

---

## 19. Canonical Class Distribution (Processed Dataset)

- **Class 0 (`normal`)**: 32,395 annotations (61.7%)
- **Class 1 (`head_down`)**: 4,103 annotations (7.8%)
- **Class 2 (`lean`)**: 0 annotations (Awaiting CCTV dataset)
- **Class 3 (`turn_head`)**: 10,572 annotations (20.2%)
- **Class 4 (`use_phone`)**: 0 annotations (Awaiting phone dataset)
- **Class 5 (`discuss`)**: 5,392 annotations (10.3%)
- **Class 6 (`stand`)**: 0 annotations (Awaiting CCTV dataset)

---

## 20. GPU Batch Calibration Results

Calibrated at `imgsz = 768x768` using `yolo26m.pt` on the RTX 5070 (11.94 GB VRAM):

| Batch Size | Status | Peak Allocated VRAM | Peak Reserved VRAM | VRAM Utilization | Safety Headroom | Step Latency | Recommendation |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **8** | **PASS** | 6.08 GB | 6.62 GB | **55.4%** | **5.32 GB** | **140.1 ms** | **Recommended (Optimal)** |
| **12** | **PASS** | 9.00 GB | 9.74 GB | **81.6%** | **2.20 GB** | **1035.9 ms** | Viable (Near Ceiling) |
| **16** | **FAIL/UNSAFE** | 11.87 GB | 12.86 GB | **107.7%** | **-0.92 GB** | **3292.0 ms** | Unsafe (Paging into System RAM) |

**Conclusion**: Operating at `batch: 16` exceeds dedicated GPU VRAM and causes high memory thrashing. **`batch: 8` is established as the canonical production configuration**, delivering 7.4x lower latency and 5.32 GB headroom.

---

## 21. 1-Epoch Mini Training Sanity Check Results

Run executed under `runs/detect/runs/sanity/dataset_integrity_sanity`:

- **Epochs Completed**: 1 (Strict constraint enforced; full training prohibited)
- **Resolution**: `768x768`, `batch: 8`, `workers: 2`
- **Direction-Safe Augmentations**: `fliplr: 0.0`, `flipud: 0.0`
- **Elapsed Duration**: 150.9 seconds
- **Peak Dedicated VRAM Reserved**: **6.60 GB** (5.34 GB safety headroom)
- **Losses**:
  - `train/box_loss`: **1.70432** (finite, healthy)
  - `train/cls_loss`: **3.37786** (finite, healthy)
  - `train/dfl_loss`: Validated
  - **NaN / Inf Occurrences**: **0**
- **Validation Run**: Evaluated all 1,218 images (6,077 instances) cleanly.
  - Overall `mAP50`: **0.1761**
  - Overall `mAP50-95`: **0.0974**
- **Weights Serialization**: `best.pt` (44.1 MB) and `last.pt` (44.1 MB) successfully created and verified.

---

## 22. Test Suite Results

Test execution: `pytest tests/ -v`

- **Total Test Cases**: **38**
- **Passed**: **38**
- **Failed**: **0**
- **Coverage Highlights**:
  - Valid and invalid YOLO bbox bounding geometry
  - Nonexistent source classes rejected from mapping
  - Only approved mappings enter canonical manifests
  - Duplicate groups cannot cross splits
  - Group IDs cannot cross splits
  - Rejected "cheating" labels never enter canonical training data
  - Invalid class IDs fail preflight
  - Dataset provenance survives processing
  - Full Phase 1 inference pipeline (FastAPI, WebSockets, temporal buffer, ByteTrack, risk scorer, video sources).

---

## 23. Stage 1 Final Proposed Configuration

Saved in `configs/train_stage1.yaml`:

```yaml
model: "yolo26m.pt"
task: "detect"
pretrained: true
data: "datasets/processed/dataset.yaml"
project: "runs/stage1"
name: "exp_foundation"
save_dir: "models/trained"
device: 0
workers: 4
amp: true
epochs: 80
batch: 8                     # Calibrated for RTX 5070
imgsz: 768
patience: 15
optimizer: "AdamW"
lr0: 0.001
lrf: 0.01
weight_decay: 0.0005
warmup_epochs: 3.0
close_mosaic: 10
fliplr: 0.0                  # MANDATORY: Preserves left vs right head yaw
flipud: 0.0
degrees: 0.0
translate: 0.05
scale: 0.2
perspective: 0.0002
hsv_h: 0.015
hsv_s: 0.4
hsv_v: 0.4
mosaic: 0.5
mixup: 0.0
copy_paste: 0.0
```

---

## 24. Stage 2 Experiment Configurations

Prepared in `configs/`:

1. **Experiment A (`configs/train_stage2_freeze0.yaml`)**:
   - `model: "models/trained/stage1_best.pt"`
   - `epochs: 30`, `batch: 8`, `imgsz: 768`
   - `lr0: 0.0001`
   - `freeze: 0` (Full end-to-end domain adaptation)
   - `fliplr: 0.0`, `flipud: 0.0`
2. **Experiment B (`configs/train_stage2_freeze10.yaml`)**:
   - `model: "models/trained/stage1_best.pt"`
   - `epochs: 30`, `batch: 8`, `imgsz: 768`
   - `lr0: 0.0001`
   - `freeze: 10` (Backbone frozen, fine-tuning neck and heads)
   - `fliplr: 0.0`, `flipud: 0.0`

---

## 25. Blockers & Required Next Steps

### Critical Blocker: Missing Primary CCTV Dataset
- The production domain for this system is high-angle wall-mounted examination room CCTV.
- The local datasets currently present (`SCB5-Discuss`, `SCB5-Handrise-Read-write`, `SCB_BowTurnHead`) originate from eye-level classroom recordings.
- To prevent domain shift and enable the `lean` and `stand` classes, the **CCTV Exam Monitor** dataset must be physically staged before launching full Stage 1 training.

### Download Instructions for User:
1. Obtain the CCTV Exam Monitor dataset from Kaggle or public CCTV exam repositories.
2. Unpack the dataset into:
   ```
   datasets/raw/cctv_exam_monitor/
   ├── images/
   ├── labels/
   └── data.yaml
   ```
3. Once placed, run:
   ```bash
   python scripts/rebuild_dataset.py
   python scripts/validate_preflight.py
   ```

---

## 26. Final Decision

### **READY FOR FULL STAGE 1 TRAINING: NO**

**Rationale**:
While all code, data loaders, preflight checks, GPU calibration, YOLO26m checkpoint, tests (38/38), and 1-epoch sanity training passed with 100% verification, **full Stage 1 training cannot be authorized until the primary target domain dataset (`cctv_exam_monitor`) is physically placed in `datasets/raw/`**. Training without the high-angle CCTV data would result in severe camera-angle domain shift in the production invigilation environment.
