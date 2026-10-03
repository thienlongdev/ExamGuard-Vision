# Stage 1 V3 Error Analysis & Failure Modes

**Sample Count**: 30 False Positives, 30 False Negatives  
**Target Focus**: `head_down`, `turn_head`, `discuss`

## 1. Summary of Dominant Error Patterns

### A. `head_down` vs `normal` Writing Confusion
- **Mechanism**: Students leaning slightly forward to write with pens exhibit head angles between 30-45 degrees, which lies on the continuous boundary between `normal` (reading/writing) and `head_down` (bowing head/sleeping).
- **Spatial Context**: Under upper oblique camera angles, foreshortening compresses the distance between forehead and desk surface.

### B. `turn_head` vs `normal` Posture Shift
- **Mechanism**: Subtle gaze shifts or partial head turns (<30 degrees) without torso reorientation.
- **Occlusion**: Distant students seated behind taller peers suffer occlusion of neck and chin.

### C. `discuss` vs `turn_head` Disambiguation
- **Mechanism**: A student turning toward a desk partner can visually resemble `turn_head` unless mouth motion or reciprocated head orientation is captured.

## 2. False Positive Catalog (Sample Extract)

| ID | Predicted Class | Ground Truth | Confidence | Type |
| :--- | :--- | :--- | :---: | :--- |
| FP-01 | `turn_head` | `background` | 0.698 | hallucination |
| FP-02 | `turn_head` | `background` | 0.582 | hallucination |
| FP-03 | `turn_head` | `background` | 0.454 | hallucination |
| FP-04 | `turn_head` | `background` | 0.535 | hallucination |
| FP-05 | `turn_head` | `normal` | 0.476 | misclassification |
| FP-06 | `discuss` | `background` | 0.392 | hallucination |
| FP-07 | `discuss` | `background` | 0.26 | hallucination |
| FP-08 | `turn_head` | `background` | 0.834 | hallucination |
| FP-09 | `turn_head` | `background` | 0.418 | hallucination |
| FP-10 | `turn_head` | `background` | 0.823 | hallucination |
| FP-11 | `turn_head` | `background` | 0.774 | hallucination |
| FP-12 | `turn_head` | `background` | 0.262 | hallucination |
| FP-13 | `turn_head` | `background` | 0.422 | hallucination |
| FP-14 | `turn_head` | `background` | 0.907 | hallucination |
| FP-15 | `turn_head` | `background` | 0.611 | hallucination |

## 3. False Negative Catalog (Sample Extract)

| ID | Missed Ground Truth Class | Type |
| :--- | :--- | :--- |
| FN-01 | `head_down` | missed_detection |
| FN-02 | `head_down` | missed_detection |
| FN-03 | `turn_head` | missed_detection |
| FN-04 | `head_down` | missed_detection |
| FN-05 | `turn_head` | missed_detection |
| FN-06 | `turn_head` | missed_detection |
| FN-07 | `turn_head` | missed_detection |
| FN-08 | `turn_head` | missed_detection |
| FN-09 | `turn_head` | missed_detection |
| FN-10 | `turn_head` | missed_detection |
| FN-11 | `turn_head` | missed_detection |
| FN-12 | `turn_head` | missed_detection |
| FN-13 | `turn_head` | missed_detection |
| FN-14 | `turn_head` | missed_detection |
| FN-15 | `turn_head` | missed_detection |
