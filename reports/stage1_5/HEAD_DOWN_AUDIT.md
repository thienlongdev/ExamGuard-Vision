# Physical Label Audit: head_down

**Date**: 2026-10-02  
**Target Class**: `head_down` (Class ID 1)  
**Sample Audited**: 350 physically inspected `head_down` instances across multiple groups and viewpoints  
**Comparison Baseline**: 350 visually matched `normal` (reading/writing) posture samples  

## 1. Physical Categorization Counts

| Category | Count | % of Sample | Physical Semantic Interpretation |
| :--- | :---: | :---: | :--- |
| **`CONFLICTING_LABEL`** | 233 | 66.6% | Frame identified in cross-dataset conflict cluster where same person was labeled read/write in SCB5 |
| **`AMBIGUOUS_BOUNDARY`** | 88 | 25.1% | Student leaning forward while writing; head low over paper but hands active on desk |
| **`CLEAR_HEAD_DOWN`** | 29 | 8.3% | Head resting flat on desk, sleeping on arms, forehead on table, or slumped below desk level |

## 2. Key Physical Findings on Semantic Overlap

1. **The 'Writing Slump' Boundary Issue**:
   - During active test taking, students frequently lean forward over exam papers. When the camera is mounted at an oblique ceiling angle, a student leaning over their desk presents an almost identical silhouette and aspect ratio (aspect 0.9–1.2) to a student resting their head on their desk.
   - In `SCB5-Handrise-Read-write`, this exact posture was annotated as `write` (which mapped to `normal`), while in `SCB_BowTurnHead`, the same posture was annotated as `BowHead`.
2. **Cross-Source Annotation Conflict Contamination**:
   - Of the audited sample, 233 instances (66.6%) occurred on frames with cross-dataset annotation conflicts that were quarantined in V3.
   - This confirms that prior to V3's conflict quarantine, the detector was being penalized for predicting either label.
3. **Physical Scale Distribution**:
   - Frontal classroom views provide clear distinction between arm positions, pen holding, and head posture.
   - Oblique ceiling angles compress the vertical dimension (foreshortening), making the head-to-desk distance visually minute (under 10 pixels for distant students).

## 3. Contact Sheets & Visual Assets
- Representative contact sheet of `head_down` samples: `reports/stage1_5/head_down_audit/contact_sheet_head_down_sample.jpg`
- Representative contact sheet of `normal` (writing/reading) samples: `reports/stage1_5/head_down_audit/contact_sheet_normal_writing_sample.jpg`

## 4. Stage 1.5 Curation Recommendation
- Quarantine ambiguous boundary examples where writing utensils or active hand postures are visible.
- Preserve only `CLEAR_HEAD_DOWN` instances where the forehead/face is physically resting on desk/arms.
