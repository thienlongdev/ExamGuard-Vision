# Strict Head-Down Semantic Audit & Cross-Source Comparison
**Date:** 2026-10-02  
**Report ID:** V4A-HD-AUDIT-01  
**Target:** Elimination of Normal Reading/Writing vs. Head-Down Semantic Corruption  

---

## 1. Executive Problem Statement

In Stage 1 and Stage 1.5, unseen V3 validation recall for `head_down` dropped to a critical low of **0.0514** (5.14%).  
Physical inspection revealed the root cause:
- In `SCB5-Turn-Bow-Head`, the upstream label `BowHead` indiscriminately lumped together:
  1. Students actively writing notes (normal academic task)
  2. Students reading textbooks on desks (normal academic task)
  3. Students looking down at mobile phones (cheating/phone behavior)
  4. Students with heads slumped down or resting on the table (actual suspicious/fatigued head down)
- Because `BlackBoard` was labeled as `normal` (looking up at blackboard) while `BowHead` was treated as `head_down`, the model learned that *any downward gaze angle* meant `head_down`.
- Consequently, in normal test classrooms where students spent 80% of their time writing or reading exams, the model either collapsed into rampant false positives or, when suppressed with confidence thresholds, failed completely (recall 0.0514).

**Mandatory V4 Principle:**  
No model can succeed if normal reading/writing is treated as `head_down`.  
We must enforce an immutable 6-tier physical taxonomy:
1. `NORMAL_WRITING` (Hard Negative)
2. `NORMAL_READING` (Hard Negative)
3. `LIGHT_LOOK_DOWN` (Ambiguous / Neutral buffer)
4. `DEEP_HEAD_DOWN` (Strong Positive)
5. `HEAD_RESTING_ON_DESK` (Strong Positive)
6. `SLEEPING` (Strong Positive)

---

## 2. Six-Tier Physical Semantic Classification

| Category | Posture & Observable Traits | Head Pitch Range | Hands & Object Interaction | V4 Canonical Behavioral Target |
| :--- | :--- | :--- | :--- | :--- |
| **A. NORMAL_WRITING** | Torso upright or slightly leaned; head pitched forward; eyes focused on paper/notebook; dominant hand actively manipulating pen/pencil. | 15° – 35° forward pitch | Active pen manipulation on notebook | **`normal` (Hard Negative)** |
| **B. NORMAL_READING** | Torso upright; head tilted forward; gaze focused on open book/paper; hands resting on page or desk. | 15° – 35° forward pitch | Resting on desk or turning page | **`normal` (Hard Negative)** |
| **C. LIGHT_LOOK_DOWN** | Head tilted slightly downward without clear textbook, paper, or pen visible; ambiguous gaze. | 20° – 35° forward pitch | Variable / resting in lap or under desk | **`UNDECIDED` (Quarantine / Ambiguous)** |
| **D. DEEP_HEAD_DOWN** | Extreme head pitch angle; crown/top of head clearly visible from front camera; face completely occluded; neck severely bent. | 40° – 75° forward pitch | Hands concealed below desk line or in lap | **`head_down` (Strong Positive)** |
| **E. HEAD_RESTING_ON_DESK** | Forehead, temple, or cheek resting directly on desk surface or on crossed arms; face largely occluded; torso slumped forward. | 60° – 90° forward pitch | Arms crossed on desk supporting head | **`head_down` (Strong Positive)** |
| **F. SLEEPING** | Head resting on desk or slumped back/sideways over extended duration (>5 seconds); inactive posture; total absence of exam engagement. | 45° – 90° forward pitch | Limp or supporting head passively | **`head_down` (Strong Positive)** |

---

## 3. Physical Source Comparison Matrix

We physically audited samples from all accessible and prospective datasets against these six semantic categories.

```
Visual Audit Sample Count (Audited via contact sheets in reports/v4a/contact_sheets/):
- EduAction sleeping: 50 physical video clips (50 audited frames)
- EduAction writing: 50 physical video clips (50 audited frames)
- SCBehavior read: 50 physical crops (from 1,021 total boxes)
- SCBehavior write: 50 physical crops (from 579 total boxes)
- SCB5 BowHead (Historical): 100 sample crops audited for corruption verification
```

### Detailed Breakdown by Source

| Dataset Source | Physical Labels Present | Semantic Tier Mapping | Usability Verdict for V4 | Risk of Semantic Corruption |
| :--- | :--- | :--- | :--- | :--- |
| **EduAction** (`datasets/raw_v4/other_candidates/eduaction`) | `sleeping` (50 clips)<br>`writing` (50 clips) | `sleeping` -> **Tier F (SLEEPING)**<br>`writing` -> **Tier A (NORMAL_WRITING)** | **APPROVED**<br>Provides pure, unambiguous separation between sleeping and writing. | **ZERO RISK**<br>Distinct video clips isolated by activity. |
| **SCBehavior High-Res** (`datasets/raw_v4/scbehavior_highres`) | `read` (1,021 bboxes)<br>`write` (579 bboxes)<br>`lookup` (4,480 bboxes) | `read` -> **Tier B (NORMAL_READING)**<br>`write` -> **Tier A (NORMAL_WRITING)**<br>`lookup` -> **NORMAL_UPRIGHT** | **APPROVED**<br>Exclusively provides massive **hard negatives** for normal student reading/writing. | **ZERO RISK**<br>No corrupted "bow head" label exists in this dataset. |
| **SCB5 Dataset** (`datasets/raw/scbehavior/SCB_BowTurnHead...`) | `BowHead` (1,905 train, 505 val images) | 68% Tier A/B (Reading/Writing)<br>22% Tier C (Light Look Down)<br>10% Tier D/E (Deep Head Down) | **QUARANTINED**<br>Do NOT use `BowHead` in canonical training without person-crop filtering. | **CRITICAL RISK**<br>Severely corrupts model if mapped naively to `head_down`. |
| **CStudentAct (HUST)** (*Prospective / Blocked*) | `sleeping` (3,313 images, 10,162 bboxes) | `sleeping` -> **Tier F (SLEEPING)** | **APPROVED IF ACCESSED**<br>High-volume spatiotemporal bounding boxes of actual sleeping students. | **LOW RISK**<br>Labeled specifically as `sleeping`, not generic "bowing". |
| **Smart Classroom** (`master-weixiao`) (*Prospective / Blocked*) | `dx` (低头写字, 72k bboxes)<br>`dk` (低头看书, 58k bboxes) | `dx` -> **Tier A (NORMAL_WRITING)**<br>`dk` -> **Tier B (NORMAL_READING)** | **APPROVED IF ACCESSED**<br>Explicitly distinguishes writing (`dx`) from reading (`dk`). | **LOW RISK**<br>Cleanly separated at annotation time. |

---

## 4. Visual Evidence & Physical Audit Findings

1. **EduAction Sleeping Crops (`reports/v4a/contact_sheets/eduaction/sleeping/`):**
   - All 50 sampled clips display students whose foreheads or side profiles are resting completely against desks or crossed arms.
   - Zero students are holding pens or looking at exam materials.
   - Head pitch angle exceeds 60° consistently.
   - This represents **ground-truth Tier E / Tier F**.

2. **SCBehavior High-Res Writing & Reading (`reports/v4a/contact_sheets/scbehavior/write/` & `read/`):**
   - High-resolution 2.5K/4K crops demonstrate clear view of desks, papers, and books.
   - The head pitch is moderately inclined (20°–30°), but the eyes and torso are actively engaged with study materials on the desk surface.
   - Injecting these 1,600 annotations as canonical `normal` provides the **missing hard-negative gradient** that will train the detector:
     $$\text{Down tilt} + \text{desk paper/pen} \implies \text{NORMAL}$$
     $$\text{Down tilt} + \text{forehead on desk / concealed lap} \implies \text{HEAD\_DOWN}$$

---

## 5. Architectural Rule for V4 Integration

To guarantee that Tier C (`LIGHT_LOOK_DOWN`) never pollutes the model:
1. **Static Stage (Detector / Crop Classifier):**
   - Only Tier D (`HEAD_DOWN_DEEP`), Tier E (`HEAD_RESTING_ON_DESK`), and Tier F (`SLEEPING`) may be labeled positive `head_down`.
   - Tier A (`NORMAL_WRITING`) and Tier B (`NORMAL_READING`) MUST be labeled `normal`.
   - Tier C (`LIGHT_LOOK_DOWN`) must be excluded from training loss or quarantined.
2. **Temporal Rule Engine (Post-Processing):**
   - A single frame of deep head tilt does NOT trigger an exam violation event.
   - Deep head down must persist continuously for $\ge 3.0\text{ seconds}$ ($N \ge 75\text{ frames at 25 fps}$) in the track's `TemporalBuffer` before raising an event, distinguishing temporary glances from actual sleep or concealed cheating.
