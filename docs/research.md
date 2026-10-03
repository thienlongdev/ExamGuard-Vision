# ExamGuard Vision — Research, Datasets & Model Development

This document consolidates the engineering research, dataset curation, model architecture benchmarks, and training methodology established during the development of ExamGuard Vision.

---

## 1. Dataset Acquisition, Provenance & Taxonomy

### 1.1 Dataset Landscape & Primary Sources
To train robust perception models capable of operating in classroom and examination environments, diverse public and domain-specific datasets were evaluated:

| Dataset | Modality / Format | Volume | Role in System | Provenance / License |
| :--- | :--- | :--- | :--- | :--- |
| **SCB-Dataset5 (Full)** | Classroom CCTV images, bounding boxes | 24,078 images, 80k+ labels | Primary training source for macro posture and behavior | Academic research license |
| **AFLW / AFLW2000-3D** | Face crops with 3D pose annotations | 23,080 labeled face crops | Head pose yaw regression training and continuous angle benchmarking | Academic research use |
| **Roboflow Exam Candidates** | Exam room images | ~3.8k frames | Candidate screening, domain shift analysis | Public research archive |
| **Local Live Sensor Captures** | DirectShow webcam feeds | Real-time streams | Zero-PII validation, hardware calibration | Local verification |

### 1.2 Taxonomy Resolution & Conflict Quarantine
Classroom datasets frequently suffer from contradictory labels (e.g., the same person labeled simultaneously as `read`, `write`, and `normal`, or conflicting definitions of `bow_head`).

- **Unified Canonical Taxonomy**:
  1. `NORMAL_UPRIGHT`: Standard upright attentive seated posture.
  2. `NORMAL_READ_WRITE`: Downward gaze focused directly on desk exam papers.
  3. `HEAD_REST_SLEEP`: Torso/head rested fully on desk surface or arm support.
  4. `TURN_HEAD_CLEAR`: Lateral head rotation exceeding certified threshold ($> 35^\circ$).
- **Conflict Quarantine Pipeline**: A dedicated audit identified **2,939 contradictory multi-label annotations** across public subsets. These annotations were systematically quarantined into an excluded set (`annotation_conflicts_v3.json`) to prevent negative supervision and label confusion during training.

### 1.3 Head Pose Angle Conventions & Schema Harmonization
- **Angle Range**: Standardized on signed continuous degrees: $\theta_{\text{yaw}} \in [-99^\circ, +99^\circ]$ where negative values denote looking left and positive values denote looking right.
- **Euler Singularity Handling**: Raw 3DDFA optimization outputs in AFLW2000-3D contained 2 wrapping singularities (e.g., raw yaw $-351.23^\circ$ due to roll inversion). Standard periodic continuous transformation was applied:
  $$\theta' = (\theta + 180^\circ \pmod{360^\circ}) - 180^\circ$$
  restoring correct physical geometry ($+8.77^\circ$).

---

## 2. Perception Model Architectures & Benchmarks

### 2.1 Stage 1: General Object Detector (`yolo26m.pt`)
- **Role**: Full-frame object detection for student candidates (`person`) and suspicious device candidates (`cell phone`).
- **Input Resolution**: 640x640 pixels (FP32 inference).
- **Parameters**: 21,896,248 parameters (21.9M).
- **Class Dynamic Resolution**: Rather than assuming fixed class indices, target classes are dynamically resolved at runtime from `model.names` (`person` = ID 0, `cell phone` = ID 67).
- **Latency**: ~38.5 ms per frame on RTX 3050 Laptop GPU.

### 2.2 Stage 1.5: Macro Behavior Detector (`stage1_5_best.pt`)
- **Role**: Direct scene-level behavioral candidate detection across 5 macro classes (`normal`, `head_down`, `turn_head`, `discuss`, `stand`).
- **Input Resolution**: 768x768 pixels.
- **Parameters**: 21,780,598 parameters.
- **Validation**: Certified against held-out classroom splits with strict zero group and hash leakage guarantees.

### 2.3 V4C Posture Classifier (`v4_posture_best.pt`)
- **Role**: Upper-body tight person crop classification for nuanced student stance.
- **Architecture**: MobileNetV3-Small with modified linear classification head (4 classes).
- **Input Resolution**: 224x224 RGB image normalized with ImageNet statistics.
- **Parameters**: 1,521,956 parameters (1.52M).
- **Trained Checkpoint Performance**:
  - Validation Accuracy: **88.71%** (Balanced Accuracy: 87.95%)
  - Macro F1: **0.8634**
  - Per-class breakdown:
    - `NORMAL_UPRIGHT`: Precision 94.1%, Recall 89.6%, F1 0.918
    - `NORMAL_READ_WRITE`: Precision 89.0%, Recall 89.8%, F1 0.894
    - `HEAD_REST_SLEEP`: Precision 100.0%, Recall 98.9%, F1 0.995
    - `TURN_HEAD_CLEAR`: Precision 57.7%, Recall 73.5%, F1 0.647
- **Inference Speed**: ~4.1 ms per crop on GPU.

### 2.4 V4C Head Pose Yaw Estimator (`v4_headpose_yaw_best.pt`)
- **Role**: Continuous head orientation regression for lateral turn tracking.
- **Architecture**: HopeNet-Yaw utilizing a ResNet50 backbone, multi-bin classification head (66 bins, $[-99^\circ, +99^\circ]$) coupled with continuous softmax expectation regression:
  $$\hat{\theta}_{\text{yaw}} = \sum_{i=1}^{66} \text{softmax}(z)_i \cdot c_i$$
- **Parameters**: 23,643,266 parameters (23.6M).
- **Trained Checkpoint Performance**:
  - Overall Validation MAE: **5.76°** (Median Absolute Error: **4.34°**)
  - Frontal slice ($< 15^\circ$): **4.90° MAE**
  - Moderate turn slice ($30^\circ - 45^\circ$): **5.27° MAE**
  - Clear turn slice ($45^\circ - 90^\circ$): **7.04° MAE**
- **Inference Speed**: ~8.2 ms per face crop on GPU.

### 2.5 Fallback Architectures
For edge devices with severe VRAM constraints (< 2 GB), lightweight fallback checkpoints are maintained:
- `models/fallback/posture_320/best_model.pt`: MobileNetV3-Small trained on 320x320 crops.
- `models/fallback/headpose_resnet18/best_model.pt`: ResNet18 HopeNet variant (~11.2M parameters) achieving 6.82° MAE with half the computational complexity.

---

## 3. V4D Multi-Cue Temporal Fusion

### 3.1 Design Principles
Single-frame visual detections suffer from momentary occlusions, motion blur, and glance false-positives. V4D addresses this through:
1. **Sliding-Window Aggregation**: Maintains a 30-second rolling history per active student track.
2. **Multi-Cue State Integration**: Fuses detection bounding boxes, posture class probabilities, head yaw angles, and spatial phone proximity into a unified time series.
3. **Domain-Specific Veto Rules**:
   - *Read/Write Veto on Sleep*: If a student exhibits `HEAD_REST_SLEEP` posture, downward gaze angles are suppressed from triggering `SUSPICIOUS_GAZE_DOWN` events.
   - *Spatial Phone Association Gate*: Phone detections are only assigned to a student track if bounding box intersection-over-union (IoU) or proximity satisfies calibrated spatial thresholds.
4. **Observable Event State Machine**: Events progress through deterministic states (`PENDING` -> `ACTIVE` -> `COOLDOWN` -> `RESOLVED`), preventing notification spam.

### 3.2 Ablation Findings
Ablation experiments comparing raw Stage 1 detection against V4D fusion confirmed:
- **False Positive Reduction**: 74.2% drop in spurious cheating flags caused by natural exam room movement (stretching, checking watch, turning page).
- **Sustained Turn Precision**: Requiring $\ge 1.5$ seconds of continuous lateral turn ($|\text{yaw}| > 35^\circ$) eliminated 91.8% of micro-glance false triggers.
- **Evidence Traceability**: Every event links directly to the temporal window and timestamped snapshot, enabling auditable human supervisor review.
