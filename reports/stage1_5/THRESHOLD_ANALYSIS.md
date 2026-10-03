# Stage 1.5 Operating Threshold and PR Curve Analysis

- **Model**: `best.pt`
- **Holdout Dataset**: `datasets/stage1_5_holdout/`
- **Resolution**: 768px
- **Evaluation Sweep**: Confidence thresholds from 0.05 to 0.70

## 1. Optimal F1 Operating Thresholds per Class

| Class | Optimal Conf Threshold | Max F1 Score | Precision @ Opt | Recall @ Opt | Standard @ 0.25 F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **normal** | 0.05 | 0.9109 | 0.8858 | 0.9374 | 0.9109 |
| **head_down** | 0.50 | 0.9087 | 0.9068 | 0.9106 | 0.9082 |
| **turn_head** | 0.50 | 0.8784 | 0.9031 | 0.8551 | 0.8777 |
| **discuss** | 0.60 | 0.9729 | 0.9729 | 0.9729 | 0.9697 |
| **stand** | 0.05 | 0.9704 | 0.9719 | 0.9689 | 0.9704 |

## 2. Weak-Class Confidence Sweep Breakdown

### `head_down` Precision-Recall vs Confidence
| Confidence | Precision | Recall | F1 Score |
| :---: | :---: | :---: | :---: |
| 0.05 | 0.9067 | 0.9097 | 0.9082 |
| 0.10 | 0.9067 | 0.9097 | 0.9082 |
| 0.15 | 0.9067 | 0.9097 | 0.9082 |
| 0.20 | 0.9067 | 0.9097 | 0.9082 |
| 0.25 | 0.9067 | 0.9097 | 0.9082 |
| 0.30 | 0.9067 | 0.9097 | 0.9082 |
| 0.35 | 0.9067 | 0.9097 | 0.9082 |
| 0.40 | 0.9067 | 0.9097 | 0.9082 |
| 0.50 | 0.9068 | 0.9106 | 0.9087 |
| 0.60 | 0.9286 | 0.8851 | 0.9063 |
| 0.70 | 0.9426 | 0.8383 | 0.8874 |

### `turn_head` Precision-Recall vs Confidence
| Confidence | Precision | Recall | F1 Score |
| :---: | :---: | :---: | :---: |
| 0.05 | 0.9029 | 0.8538 | 0.8777 |
| 0.10 | 0.9029 | 0.8538 | 0.8777 |
| 0.15 | 0.9029 | 0.8538 | 0.8777 |
| 0.20 | 0.9029 | 0.8538 | 0.8777 |
| 0.25 | 0.9029 | 0.8538 | 0.8777 |
| 0.30 | 0.9029 | 0.8538 | 0.8777 |
| 0.35 | 0.9029 | 0.8538 | 0.8777 |
| 0.40 | 0.9029 | 0.8538 | 0.8777 |
| 0.50 | 0.9031 | 0.8551 | 0.8784 |
| 0.60 | 0.9219 | 0.8213 | 0.8687 |
| 0.70 | 0.9421 | 0.7710 | 0.8480 |

## 3. Failure Mode Diagnosis: Separation Failure vs Threshold Failure

### Physical Diagnostic Rules
- **Threshold Failure**: If recall jumps dramatically (> 0.20 boost) when lowering confidence to 0.05-0.10 while maintaining acceptable precision, the model separated features but the default 0.25 threshold was calibrated too aggressively.
- **Model-Separation Failure**: If lowering confidence to 0.05 does NOT substantially increase recall or causes precision to collapse near zero, the detector backbone/head lacks discriminative spatial features to separate the subtle posture differences (e.g. slight head tilt vs reading) from single 2D bounding boxes.

### Diagnostic Findings
1. **`head_down`**: At conf=0.05, Recall is 0.9097 (vs 0.9097 at conf=0.25).
2. **`turn_head`**: At conf=0.05, Recall is 0.8538 (vs 0.8538 at conf=0.25).

### Recommendation for Temporal Rule Engine
- Recommended detector confidence threshold for `head_down`: **0.50**.
- Recommended detector confidence threshold for `turn_head`: **0.50**.
- Recommended detector confidence threshold for `normal`: **0.05**.
- In the multi-stage exam monitoring system, single-frame false positives at lower confidence thresholds are filtered out by the **ByteTrack temporal buffer** requiring persistent behavior across consecutive frames (e.g. 5+ frames) and cooldown hysteresis.

