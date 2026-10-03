# Hard-Example Mining Report (Stage 1 Best Checkpoint)

**Date**: 2026-10-02  
**Model Evaluated**: `models/trained/stage1_best.pt`  
**Domain Mined**: Stage 1 `train` split excluding frozen Stage 1.5 holdout (6178 images)  
**Total Hard Examples Discovered**: 1,268  

## 1. Distribution of Error Types

| Error Category | Code | Description | Instance Count | % of Errors |
| :--- | :--- | :--- | :---: | :---: |
| **HEAD_DOWN_FN** | `TYPE_A_HEAD_DOWN_FN` | Ground truth is `head_down`, predicted as `normal` or completely undetected | 41 | 3.2% |
| **TURN_HEAD_FN** | `TYPE_B_TURN_HEAD_FN` | Ground truth is `turn_head`, predicted as `normal` or completely undetected | 130 | 10.3% |
| **HARD_NORMAL_NEG_HEAD_DOWN** | `TYPE_C_HARD_NORMAL_NEG_HEAD_DOWN` | Ground truth is `normal`, falsely predicted as `head_down` | 37 | 2.9% |
| **HARD_NORMAL_NEG_TURN_HEAD** | `TYPE_C_HARD_NORMAL_NEG_TURN_HEAD` | Ground truth is `normal`, falsely predicted as `turn_head` | 14 | 1.1% |
| **TURN_HEAD_DISCUSS_CONFUSION** | `TYPE_E_TURN_HEAD_DISCUSS_CONFUSION` | Mutual confusion between `turn_head` and `discuss` interactions | 8 | 0.6% |
| **BACKGROUND_HALLUCINATION** | `TYPE_D_BACKGROUND_HALLUCINATION` | False detection of `head_down` or `turn_head` on empty desk/background | 1,038 | 81.9% |

## 2. Hard Example Findings and Impact on Refinement Training

1. **Dominance of False Negatives in `turn_head` and `head_down`**:
   - Weak-class false negatives represent the vast majority of mined hard instances (171 instances).
   - The detector repeatedly suppresses head_down and turn_head in favor of the high-prior `normal` class.
2. **Hard Normal Negatives**:
   - Normal examinees leaning low or glancing slightly sideways generate 37 head_down FPs and 14 turn_head FPs.
   - These hard normal frames are essential negative examples during Stage 1.5 fine-tuning to prevent the model from over-predicting weak classes.
3. **Turn-Head / Discuss Ambiguity**:
   - 8 confusion instances occur when examinees turn toward each other in cluster rows.

## 3. Visual Gallery Artifacts
- `reports/stage1_5/hard_examples/gallery_type_a_head_down_fn.jpg`
- `reports/stage1_5/hard_examples/gallery_type_b_turn_head_fn.jpg`
- `reports/stage1_5/hard_examples/gallery_type_c_hard_normal_neg_head_down.jpg`
- `reports/stage1_5/hard_examples/gallery_type_c_hard_normal_neg_turn_head.jpg`
- `reports/stage1_5/hard_examples/gallery_type_e_turn_head_discuss_confusion.jpg`
- `reports/stage1_5/hard_examples/gallery_type_d_background_hallucination.jpg`

## 4. Integration into Stage 1.5 Refinement Set (`processed_v3_5`)
- Frames containing hard weak-class positives and hard normal negatives are flagged for prioritized inclusion and frequency reweighting.
