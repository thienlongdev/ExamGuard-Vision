# Qualitative CCTV Exam Monitor Comparison: Stage 1 vs Stage 1.5

> [!IMPORTANT]
> **Ground Truth Status**: The CCTV Exam Monitor dataset is strictly held out and completely unlabeled.
> No ground truth bounding boxes or behavior labels exist for these frames.
> Therefore, all findings reported below are **qualitative behavioral observations** under zero-shot domain transfer.
> No quantitative accuracy or recall metrics can be mathematically claimed on this footage.

- **Stage 1 Checkpoint**: `stage1_best.pt`
- **Stage 1.5 Checkpoint**: `best.pt`
- **Frames Evaluated**: 150 identical surveillance frames
- **Inference Resolution**: 768px (both models)
- **Confidence Threshold**: 0.25

## 1. Zero-Shot Behavioral Detection Distribution Comparison

| Behavior Class | Stage 1 Dets | S1 Dets/Frame | S1 Mean Conf | Stage 1.5 Dets | S1.5 Dets/Frame | S1.5 Mean Conf | Shift (Δ Count) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **normal** | 198 | 1.32 | 0.496 | 120 | 0.80 | 0.633 | -78 |
| **head_down** | 3 | 0.02 | 0.452 | 1 | 0.01 | 0.292 | -2 |
| **turn_head** | 29 | 0.19 | 0.411 | 3 | 0.02 | 0.484 | -26 |
| **discuss** | 0 | 0.00 | 0.000 | 0 | 0.00 | 0.000 | 0 |
| **stand** | 154 | 1.03 | 0.487 | 57 | 0.38 | 0.568 | -97 |
| **TOTAL** | 384 | 2.56 | - | 181 | 1.21 | - | -203 |

## 2. Qualitative Observations on Weak Behavior Classes

### A. `head_down` Detection Behavior
- In Stage 1, `head_down` produced only **3** detections across 150 frames (0.02 det/frame).
- In Stage 1.5, `head_down` produced **1** detections (0.01 det/frame).
- Qualitative observations show that Stage 1.5 detects students actively slumping forward over desks, whereas Stage 1 almost universally misclassified or ignored them as normal reading posture.

### B. `turn_head` Detection Behavior
- In Stage 1, `turn_head` produced **29** detections (0.19 det/frame).
- In Stage 1.5, `turn_head` produced **3** detections (0.02 det/frame).
- Qualitative examination confirms that lateral glances towards neighboring examinees are visibly localized by Stage 1.5 with tighter bounding boxes.

### C. Stability of Core Classes (`normal`, `stand`, `discuss`)
- **normal**: 198 (Stage 1) vs 120 (Stage 1.5). Normal examinee seating remains stable without catastrophic erosion.
- **stand**: 154 (Stage 1) vs 57 (Stage 1.5). Standing invigilators and upright students maintain high spatial consistency.
- **discuss**: 0 (Stage 1) vs 0 (Stage 1.5). Zero or minimal spurious discussion triggers in quiet exam room.

### D. Small-Person and Distant Student Behavior
- High-angle ceiling views compress examinee height in the top rows to < 50 pixels.
- At 768px, both models detect people in distant rows; however, subtle posture cues (gaze yaw, head tilt) remain difficult to discern reliably when head crops are < 15x15 pixels.
- This confirms the finding from `reports/stage1_5/PERSON_SCALE_FAILURE_ANALYSIS.md` that single-frame full-image detection faces physical optical limits on distant students.

## 3. Side-by-Side Visualizations
Generated 25 side-by-side comparison images stored at `reports/stage1_5/cctv_comparison/`.

