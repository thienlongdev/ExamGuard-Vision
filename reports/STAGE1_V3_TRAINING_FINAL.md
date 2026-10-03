# STAGE 1 V3 FOUNDATION TRAINING FINAL REPORT

## 1. Environment
- **Python Version**: `3.13.9 (tags/v3.13.9:8183fa5, Oct 14 2025, 14:09:13) [MSC v.1944 64 bit (AMD64)]`
- **PyTorch Version**: `2.14.1+cu130`
- **CUDA Version**: `13.0`
- **Ultralytics Version**: `8.4.171`
- **GPU Hardware**: `NVIDIA GeForce RTX 5070` (11.94 GB VRAM, sm_120 Blackwell)
- **Dataset Manifest SHA-256**: `91f86447fc466a63340d9e12aaeaddb75212b17b525c5012212cced45709c2ec`
- **Git Commit**: `N/A (not a git repo)`

## 2. Exact Training Configuration
- **Model Architecture**: `yolo26m.pt` (fresh foundation initialization)
- **Input Resolution**: `768x768`
- **Batch Size**: `8` (conservative safe ceiling; ~6.53 GB peak VRAM)
- **Optimizer**: `AdamW` (`lr0: 0.001`, `lrf: 0.01`, `weight_decay: 0.0005`, `warmup_epochs: 3.0`)
- **Precision**: Mixed Precision AMP enabled
- **Schedule**: Maximum `80` epochs, early stopping `patience: 15`, `close_mosaic: 10`
- **Augmentation Policy**: `fliplr: 0.0`, `flipud: 0.0` strictly enforced; `scale: 0.2`, `translate: 0.05`, `mosaic: 0.5`, `mixup: 0.0`
- **Seed**: `42` (`deterministic: true`)

## 3. Dataset Statistics
Grounded strictly on verified physical `datasets/processed_v3/`:
- **Total Images**: 9,321 images, 59,199 bounding boxes across 202 video groups
- **Train Split**: 6,666 images (71.5%), 40,406 boxes, 151 sequence groups
- **Validation Split**: 1,400 images (15.0%), 9,327 boxes, 26 sequence groups (includes 14 held-out oblique CCTV groups)
- **Test Split**: 1,255 images (13.5%), 9,466 boxes, 25 sequence groups (includes 14 held-out oblique CCTV groups)
- **Grouping**: DSU grouped partition guaranteeing ZERO intra-sequence or duplicate frame leakage

## 4. Final Stage 1 Taxonomy
- Canonical Classes (5): `0: normal`, `1: head_down`, `2: turn_head`, `3: discuss`, `4: stand`
- Cell Phone Handling: Kept completely independent in dedicated 2-stage object association subsystem (`COCO phone + ByteTrack student id`)

## 5. Training Timeline & Duration
- **Start Time**: `2026-10-02T11:52:38.379520+00:00`
- **End Time**: `2026-10-02T13:44:50.489673+00:00`
- **Total Epochs Actually Completed**: **48**
- **Early Stopping Status**: `TRIGGERED (patience 15)`
- **Best Epoch**: **Epoch 33**
- **Peak VRAM Reserved**: `6.55 GB` (headroom: ~5.39 GB)

## 6. Final Training Losses
- **Train Box Loss**: `0.6188`
- **Train Class Loss**: `0.4288`
- **Validation Box Loss**: `0.9963`
- **Validation Class Loss**: `1.6689`

## 7. Validation Split Metrics (Evaluated at Best Epoch Checkpoint)
- **Overall Precision**: `0.5018`
- **Overall Recall**: `0.5256`
- **Overall mAP50**: `0.5033`
- **Overall mAP50-95**: `0.3624`

### Validation Per-Class Breakdown
| Class | Precision | Recall | mAP50 |
| :--- | :---: | :---: | :---: |
| **normal** | 0.6480 | 0.8059 | 0.5271 |
| **head_down** | 0.1349 | 0.0237 | 0.0411 |
| **turn_head** | 0.2080 | 0.1749 | 0.0668 |
| **discuss** | 0.6278 | 0.7333 | 0.4667 |
| **stand** | 0.8904 | 0.8900 | 0.7105 |

## 8. Locked Test Split Evaluation (Evaluated Exactly Once on best.pt)
- **Overall Precision**: `0.6158`
- **Overall Recall**: `0.5256`
- **Overall mAP50**: `0.5302`
- **Overall mAP50-95**: `0.3786`

### Test Per-Class Breakdown
| Class | Precision | Recall | mAP50 |
| :--- | :---: | :---: | :---: |
| **normal** | 0.5389 | 0.7387 | 0.4379 |
| **head_down** | 0.5345 | 0.0666 | 0.0830 |
| **turn_head** | 0.4855 | 0.1509 | 0.1612 |
| **discuss** | 0.6454 | 0.8137 | 0.4965 |
| **stand** | 0.8745 | 0.8583 | 0.7145 |

## 9. Viewpoint Domain Slicing
Comparison of model performance across held-out viewpoint partitions on the locked test set:
| Viewpoint Partition | Images | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **cctv_oblique_high_angle** | 1,215 | 0.6115 | 0.5306 | 0.5325 | 0.3801 |
| **frontal_classroom** | 40 | 0.5786 | 0.5213 | 0.5708 | 0.4015 |

## 10. Minority-Class Analysis
- **`head_down`**: Severe bottleneck class (2,521 total boxes, 1,630 train). Suffers from extremely low recall (val 0.0237, test 0.0666) driven by high false negative rates and confusion with normal writing posture.
- **`turn_head` & `discuss`**: `turn_head` is a major bottleneck (val precision 0.2080, recall 0.1749; test precision 0.4855, recall 0.1509), showing low recall and modest precision. `discuss` achieved moderate-high precision (0.6454 test) and good recall (0.8137 test).

## 11. Qualitative Zero-Shot CCTV Exam Monitor Transfer
- **Dataset Context**: Completely unlabeled exam surveillance footage. No ground truth exists; metrics represent qualitative zero-shot detections rather than accuracy.
- **Sample Tested**: 150 diverse frames evaluated qualitatively.
- **Total Raw Behavior Detections**: 384
- **Average Detection Density**: 2.6 detections/frame
- **Behavior Breakdown**:
  - `normal`: 198 detections (51.6%)
  - `head_down`: 3 detections (0.8%)
  - `turn_head`: 29 detections (7.6%)
  - `discuss`: 0 detections (0.0%)
  - `stand`: 154 detections (40.1%)

## 12. Error Patterns & Failure Modes
- Representative error galleries cataloged in `reports/stage1_v3/error_analysis/` (30 FP and 30 FN visual crops).
- Primary FP mode: Leaning forward during paper writing occasionally crossing head-down angle threshold.
- Primary FN mode: Severe occlusion in classroom back corners under steep surveillance angles.

## 13. System Integration & Regression Test Verification
- **Pipeline End-to-End Smoke Test**: `PASSED` (YOLO Detector -> ByteTrack -> Student ID -> Buffer -> Rule Engine -> FastAPI)
- **Full Pytest Regression Suite**: `PASSED (50/50 tests)`

## 14. Artifact Locations
- **Standardized Best Checkpoint**: `models/trained/stage1_best.pt`
- **Run Best Checkpoint**: `runs/stage1_v3/full_foundation/weights/best.pt`
- **Run Last Checkpoint**: `runs/stage1_v3/full_foundation/weights/last.pt`
- **Error Gallery**: `reports/stage1_v3/error_analysis/`
- **CCTV Qualitative Visualizations**: `reports/stage1_v3/cctv_qualitative/predictions/`

## 15. Next Step Recommendation & Stage 2 Readiness
### READY FOR STAGE 2: **NO**

**CRITICAL DIRECTIVE**: Stage 2 has NOT been started. Execution is paused awaiting explicit user authorization.
