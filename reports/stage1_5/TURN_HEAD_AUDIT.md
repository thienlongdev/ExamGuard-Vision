# Physical Label Audit: turn_head

**Date**: 2026-10-02  
**Target Class**: `turn_head` (Class ID 2)  
**Sample Audited**: 350 physically inspected `turn_head` instances across multiple groups and viewpoints  

## 1. Physical Categorization Counts

| Category | Count | % of Sample | Physical Semantic Interpretation |
| :--- | :---: | :---: | :--- |
| **`AMBIGUOUS_BOUNDARY`** | 149 | 42.6% | Subtle gaze shift (< 30°), distant small student (< 0.003 area), or rear-view ambiguity |
| **`CLEAR_TURN_HEAD`** | 111 | 31.7% | Distinct lateral head yaw (45° to 90°) looking sideways toward neighbor or aisle |
| **`CONFLICTING_LABEL`** | 90 | 25.7% | Frame identified in cross-dataset conflict cluster (e.g. discuss vs turn_head) |

## 2. Key Physical Findings on `turn_head` Bottleneck

1. **Lateral Head Orientation vs Rear-View Occlusion**:
   - In ceiling-mounted cameras, when students are viewed from behind, turning the head produces significant occlusion of facial features. The visual cue is almost entirely neck silhouette and ear visibility.
   - Subtle posture shifts (e.g. shifting body weight while keeping eyes on paper) can look like a small head turn, resulting in false positives or annotator disagreement.
2. **Small Scale & Distant Examinees**:
   - Small examinees in rear rows (area < 0.003) represent a large portion of `AMBIGUOUS_BOUNDARY` (149 instances, 42.6%).
   - At 768px input resolution, a distant head is only 8x8 to 12x12 pixels, which makes distinguishing a 30° head turn from a forward gaze extremely difficult without temporal tracking.
3. **Turn-Head vs Discuss Overlap**:
   - When two neighboring students both turn heads toward each other, one dataset labeled them `discuss` while another labeled individual students `turn_head`.

## 3. Contact Sheets & Visual Assets
- Representative contact sheet of `turn_head` samples: `reports/stage1_5/turn_head_audit/contact_sheet_turn_head_sample.jpg`

## 4. Stage 1.5 Curation Recommendation
- Prioritize `CLEAR_TURN_HEAD` with unambiguous lateral orientation.
- Down-weight distant small-box turn_head instances during single-frame training, relying on temporal tracking downstream.
