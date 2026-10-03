# Final Model Taxonomy Decision V3

**Date**: 2026-10-02  
**Decision Engine**: Physical Evidence Audit & Visual Verification  
**Selected Taxonomy Version**: **V1-5 (5-Class Compact Clean Taxonomy)**

---

## 1. Final Approved Taxonomy

| Canonical Class ID | Canonical Class Name | Physical Source Annotations | Approved Source Classes & Subsets | Verification Status |
| :---: | :--- | :---: | :--- | :---: |
| **0** | `normal` | 33,919+ | `read` (24,078), `write` (9,841) from `SCB5-Handrise-Read-write-2024-9-17` | **APPROVED** |
| **1** | `head_down` | 4,962 | `BowHead` (4,962) from `SCB_BowTurnHead_20250509` | **APPROVED** |
| **2** | `turn_head` | 11,156 | `TurnHead` (11,156) from `SCB_BowTurnHead_20250509` | **APPROVED** |
| **3** | `discuss` | 5,392 | `discuss` (5,392) from `SCB5-Discuss-2024-9-17` | **APPROVED** |
| **4** | `stand` | 12,969 | `stand` (12,969) from `SCB5-Stand-2024-9-17` | **APPROVED** |

---

## 2. Rationales for Approved Classes

1. **`normal` (Class 0)**:
   - *Evidence*: Seated reading (24,078 boxes) and writing (9,841 boxes) provide the exact baseline posture of an examinee taking a test at a desk.
   - *Visual Check*: Verified on `reports/scb_dataset5_full/class_samples/read.jpg` and `write.jpg`.
2. **`head_down` (Class 1)**:
   - *Evidence*: 4,962 physical boxes from `SCB_BowTurnHead_20250509`.
   - *Visual Check*: Shows students resting forehead on desk, bowed asleep, or slouching deeply, distinctly different from regular writing.
3. **`turn_head` (Class 2)**:
   - *Evidence*: 11,156 physical boxes from `SCB_BowTurnHead_20250509`.
   - *Visual Check*: Clean 45-90 degree head turn toward peers or away from desk.
4. **`discuss` (Class 3)**:
   - *Evidence*: 5,392 physical boxes from `SCB5-Discuss-2024-9-17`.
   - *Visual Check*: Groups of students turning towards each other and engaged in peer conversation.
5. **`stand` (Class 4)**:
   - *Evidence*: 12,969 physical boxes from `SCB5-Stand-2024-9-17` (newly acquired in `scb_dataset5_full`).
   - *Visual Check*: Clean upright full-body student and teacher standing postures. Crucial for detecting examinees leaving desks or standing up during an examination.

---

## 3. Omitted Classes and Detailed Reasons

| Candidate Class | Decision | Reason & Technical Justification |
| :--- | :---: | :--- |
| `use_phone` | **OMITTED** | Visual audit of phone candidate datasets demonstrates bounding boxes enclose handheld mobile phones and hands resting on tables, NOT complete student body postures. Per Section 29, Option B fallback architecture is active: person-level phone use is detected via dedicated COCO phone detector + ByteTrack spatial-temporal association + posture rules. This prevents label contamination of the behavior detector. |
| `lean` | **OMITTED** | No clean, visually verified physical annotations exist in the acquired datasets. Creating an empty or synthetic class slot is strictly prohibited per Section 15. |
| `cheating` / `abnormal` | **REJECTED** | Strictly prohibited by Section 17. Cheating is a multi-frame temporal event inferred by tracking and risk scoring, not an observable single-frame visual object class. |
| `hand-raising` | **IGNORED** | Normal classroom gesture, irrelevant for exam cheating detection. |
| `teacher` / `blackboard` / `screen` | **IGNORED** | Furniture and instructor categories, not examinee behaviors. |

---

## 4. Architectural Soundness

The selected 5-class taxonomy (`normal`, `head_down`, `turn_head`, `discuss`, `stand`) directly matches the project's Suspicious Behavior Rule engine:
- `turn_head` (extended duration) -> triggers `PEEKING_LOOKING_AROUND`
- `discuss` (multi-student interaction) -> triggers `UNAUTHORIZED_CONVERSATION`
- `stand` (leaving seat) -> triggers `ABNORMAL_ABSENCE_STANDING`
- `head_down` (head slumped) -> triggers `SLEEPING_CONCEALMENT`
- Phone presence + normal/head_down posture -> triggers `PHONE_USE` via Object Associator and Temporal Buffer.
