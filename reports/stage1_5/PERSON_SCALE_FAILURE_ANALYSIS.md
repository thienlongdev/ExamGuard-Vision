# Person-Scale Failure Analysis for Weak Behavior Classes

**Date**: 2026-10-02  
**Model Evaluated**: `models/trained/stage1_best.pt`  
**Dataset Evaluated**: Stage 1 Training Domain excluding Stage 1.5 Holdout (6,178 images)  
**Resolution Scale**: 768x768 pixels  

## 1. Scale Categorization Definitions
- **Small / Distant Examinee**: Person box height < 80px (normalized area < 0.008; typical of rear classroom rows > 7m from camera).
- **Medium Examinee**: Person box height 80px to 160px (normalized area 0.008 to 0.035; middle rows 3-6m).
- **Large / Foreground Examinee**: Person box height > 160px (normalized area > 0.035; front rows 1-2m).

## 2. Quantitative Scale Comparison

### A. `head_down` Scale Breakdown

| Detection Outcome | Total Instances | Mean Box Height | Median Height | Small (<80px) % | Medium (80-160px) % | Large (>160px) % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Correctly Detected** | 1354 | 118.6 px | 90.0 px | 42.4% | 38.3% | 19.4% |
| **Missed (False Negatives)** | 41 | 179.3 px | 151.4 px | 29.3% | 26.8% | 43.9% |

### B. `turn_head` Scale Breakdown

| Detection Outcome | Total Instances | Mean Box Height | Median Height | Small (<80px) % | Medium (80-160px) % | Large (>160px) % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Correctly Detected** | 4703 | 155.5 px | 128.0 px | 14.3% | 49.9% | 35.8% |
| **Missed (False Negatives)** | 134 | 142.6 px | 102.8 px | 29.9% | 44.0% | 26.1% |

## 3. Key Findings on Person Scale and Failure Modes

1. **Disproportionate Failure on Distant Students**:
   - For `turn_head`, missed detections have a higher concentration of small distant students (29.9%) compared to correctly detected instances (14.3%).
   - In rear rows, a student's entire head occupies fewer than 12x12 pixels. Fine-grained posture cues (e.g. ear visibility, cheek profile) are blurred into 1-2 receptive fields.
2. **Foreshortening in High-Angle Views**:
   - For `head_down`, vertical foreshortening from oblique ceiling perspectives compresses the distance between the crown of the head and the desktop.
   - Even when student height is medium (80-120px), the torso occludes the neck, causing the detector to default to the overwhelmingly frequent `normal` prior.
3. **Architectural Implications for Future Phases**:
   - Full-frame YOLO single-shot detection forces 768px feature maps to simultaneously detect full-body standing examinees and 10px distant head yaw.
   - If data-centric refinement in Stage 1.5 does not fully overcome distant-person head orientation ambiguity, a 2-stage architecture (Person Detector -> ByteTrack -> Normalized Person Crop -> Posture/Pose Classifier) provides dedicated resolution for fine-grained head angles.
