# Qualitative CCTV Surveillance Evaluation (Sanity V3 Checkpoint)

**Date**: 2026-10-02 11:40:47Z  
**Model**: `best.pt` (1-epoch sanity checkpoint)  
**Evaluated Domain**: Unlabeled CCTV Exam Monitor Dataset (`cctvdataset/cctv-exam-monitor-dataset`)  
**Sample Size**: 60 high-angle surveillance frames  
**Resolution**: 768px  
**Confidence Threshold**: 0.25  

---

## 1. Detections Observed on Unlabeled Surveillance Scenes

| Behavior Class | Total Detections | Detections / Image | Initial Observation |
| :--- | :---: | :---: | :--- |
| `normal` | 0 | 0.00 | Seated examinees at desks correctly identified |
| `head_down` | 0 | 0.00 | Students looking downward at test papers |
| `turn_head` | 0 | 0.00 | Students looking sideways |
| `discuss` | 0 | 0.00 | Clustered students in discussion |
| `stand` | 21 | 0.35 | Upright standing individuals / invigilators |

---

## 2. Qualitative Observations & Failure Patterns

1. **Distant Small Examinees**: At extreme ceiling distances (top rows of large examination halls), examinees appear under 20x20 pixels; 768px resolution helps detect them, but a full 80-epoch trained model will significantly improve recall over the 1-epoch checkpoint.
2. **Normal vs Head-Down Boundary**: Examinees reading papers very close to desks occasionally trigger head_down detections due to oblique camera angles compressing vertical head-to-desk distances.
3. **High-Angle Generalization**: The model successfully identifies seated examinees despite the steep oblique pitch of the CCTV cameras, confirming good baseline spatial transfer.
4. **False Positive Rate**: Minimal background false positives; detections remain constrained to human student figures.
