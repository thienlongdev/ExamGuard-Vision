# Stage 1 Baseline on Stage 1.5 Frozen Holdout

**Model**: `models/trained/stage1_best.pt`  
**Holdout**: `datasets/stage1_5_holdout/`  
**Image Size**: 768px  
**Batch Size**: 8  

## Overall Metrics
- **Precision**: `0.9068`
- **Recall**: `0.9233`
- **mAP50**: `0.9612`
- **mAP50-95**: `0.8125`

## Per-Class Breakdown

| Class | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **normal** | 0.8697 | 0.9241 | 0.9489 | 0.7991 |
| **head_down** | 0.8846 | 0.8723 | 0.9437 | 0.7396 |
| **turn_head** | 0.8819 | 0.8517 | 0.9353 | 0.7740 |
| **discuss** | 0.9313 | 0.9910 | 0.9888 | 0.8679 |
| **stand** | 0.9664 | 0.9773 | 0.9893 | 0.8819 |
