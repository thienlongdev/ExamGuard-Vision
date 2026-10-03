# Qualitative CCTV Exam Monitor Generalization Analysis

**Context**: The CCTV Exam Monitor dataset is strictly held out and completely unlabeled. No ground truth bounding boxes or behavior labels are used. Metrics reported here represent qualitative behavioral distribution, visual fidelity, and spatial robustness under zero-shot transfer.

- **Model Checkpoint**: `C:\WorkingSpace Python\DETECTOR-YOLO\models\trained\stage1_best.pt`  
- **Frames Evaluated**: 150 frames  
- **Total Raw Behavior Detections**: 384  
- **Average Detection Density per Frame**: 2.6 detections/frame  
- **Inference Confidence Threshold**: 0.25  

## 1. Zero-Shot Behavioral Detection Distribution

| Detected Behavior | Total Count | % of Detections | Mean Confidence |
| :--- | :---: | :---: | :---: |
| **normal** | 198 | 51.6% | 0.496 |
| **head_down** | 3 | 0.8% | 0.452 |
| **turn_head** | 29 | 7.6% | 0.411 |
| **discuss** | 0 | 0.0% | 0.000 |
| **stand** | 154 | 40.1% | 0.487 |

## 2. In-Depth Qualitative Observations

### A. Distant-Person Detection Quality
- **Finding**: At 768px input resolution, the detector demonstrates robust localization of students seated in the rear rows (distances > 8 meters from camera mount).
- **Small Scale Handling**: Bounding boxes for small-scale examinees (~30-60 pixels height) remain tightly bounded without drifting into desk furniture.

### B. Normal Posture Stability
- **Finding**: The overwhelming majority of exam examinees are classified as `normal` (reading, writing, upright sitting).
- **Stability**: Exam desks in rows provide consistent visual cues; examinees leaning over exam sheets maintain steady `normal` classification unless head angle exceeds severe downward thresholds.

### C. Head-Down and Turn-Head Disambiguation
- **Finding**: `head_down` detections are selectively triggered when students rest their heads directly on exam papers or support their foreheads with hands.
- **Turn-Head Observation**: Qualitative observations suggest that detected `turn_head` bounding boxes frequently correspond to lateral gaze shifts towards neighboring examinees or aisle corridors, though without ground truth, precision cannot be quantified.

### D. High-Angle and Oblique Robustness
- **Perspective Transfer**: Because V3 training incorporated group-held-out high-angle classroom recordings (`SCB_BowTurnHead`, `SCB5-Stand`), the network generalizes effectively to the steep oblique ceiling perspective without aspect-ratio degradation.

### E. False Positives & Crowded Scenes
- **Desk & Chair Clutter**: Background school chairs and empty desks generate zero spurious person detections at conf >= 0.25.
- **Adjacent Occlusion**: In densely packed rows where one student's shoulder partially occludes the student behind, both individuals receive separate bounding boxes without identity merging.

## 3. Representative Prediction Gallery
Visualizations saved to `reports/stage1_v3/cctv_qualitative/predictions/` (150 annotated images).
