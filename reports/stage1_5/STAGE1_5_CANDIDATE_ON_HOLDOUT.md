# Stage 1.5 Final Candidate Evaluation Metrics

**Checkpoint**: `best.pt` (from `runs/stage1_5/v3_5_refinement/weights/best.pt`)  
- **Resolution**: 768px  
- **Batch Size**: 4  
- **Device**: CUDA:0 (NVIDIA RTX 5070)  

---

## 1. Primary Comparison: Frozen Stage 1.5 Holdout (Unseen Holdout Groups)

- **Overall Precision**: `0.9247`
- **Overall Recall**: `0.9307`
- **Overall mAP50**: `0.9629`
- **Overall mAP50-95**: `0.8385`

| Class | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **normal** | 0.8858 | 0.9374 | 0.9472 | 0.8178 |
| **head_down** | 0.9067 | 0.9097 | 0.9443 | 0.7693 |
| **turn_head** | 0.9029 | 0.8538 | 0.9414 | 0.7969 |
| **discuss** | 0.9560 | 0.9837 | 0.9900 | 0.8958 |
| **stand** | 0.9719 | 0.9689 | 0.9917 | 0.9125 |

---

## 2. Generalization Audit: Original V3 Validation Split (Completely Unseen Groups)

- **Overall Precision**: `0.4849`
- **Overall Recall**: `0.5256`
- **Overall mAP50**: `0.4684`
- **Overall mAP50-95**: `0.3449`

| Class | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **normal** | 0.6108 | 0.7573 | 0.6369 | 0.4697 |
| **head_down** | 0.1368 | 0.0514 | 0.0162 | 0.0102 |
| **turn_head** | 0.1680 | 0.1969 | 0.0701 | 0.0423 |
| **discuss** | 0.6426 | 0.7200 | 0.7059 | 0.4771 |
| **stand** | 0.8664 | 0.9024 | 0.9127 | 0.7255 |

---

## 3. Reference Test Split (REFERENCE ONLY — NOT PRISTINE FOR STAGE 1.5)

> [!NOTE]
> Per Requirement 19, the original V3 test set is strictly a post-selection reference test,
> as its Stage 1 baseline numbers were already known and it was not used for tuning.

- **Overall Precision**: `0.6024`
- **Overall Recall**: `0.5208`
- **Overall mAP50**: `0.4995`
- **Overall mAP50-95**: `0.3730`

| Class | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **normal** | 0.5376 | 0.6926 | 0.5615 | 0.4237 |
| **head_down** | 0.4304 | 0.0721 | 0.0732 | 0.0536 |
| **turn_head** | 0.5098 | 0.1802 | 0.2173 | 0.1527 |
| **discuss** | 0.6716 | 0.7978 | 0.7572 | 0.4957 |
| **stand** | 0.8628 | 0.8615 | 0.8885 | 0.7392 |
