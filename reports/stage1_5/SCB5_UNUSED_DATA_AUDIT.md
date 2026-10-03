# Unused SCB5 Data Audit for Weak Behavior Classes

**Date**: 2026-10-02  
**Target Goal**: Search for unused groups in the full physical SCB-Dataset5 release containing verified examples of `head_down` (`BowHead`) or `turn_head` (`TurnHead`).  
**Audited Roots**:
- `datasets/raw/scb_dataset5_full/SCB-Dataset/` (9 raw subsets; 96,760 items in upstream zip)
- `datasets/raw/scbehavior/SCB_BowTurnHead_20250509/` (2,410 raw images)

---

## 1. Inventory of Subsets in `scb_dataset5_full`

Each directory in `datasets/raw/scb_dataset5_full/SCB-Dataset/` was physically inspected for annotations, YAML schemas, label syntax, and image samples:

| Subset Directory | Upstream Items | Physical Labels Found | Classes Defined | Behavior Category | Audit Verdict |
| :--- | :---: | :---: | :--- | :--- | :---: |
| `SCB5-Handrise-Read-write-2024-9-17` | 13,731 | `SCB5-Handrise-Read-write-2024-9-17.yaml` | `['hand-raising', 'read', 'write']` | Student reading/writing/hand-raising | **Fully Admitted in V3** (`read`/`write` -> `normal`) |
| `SCB5-Stand-2024-9-17` | 15,207 | `SCB5-Stand-2024-9-17.yaml` | `['stand']` | Student standing upright | **Fully Admitted in V3** (`stand` -> `stand`) |
| `SCB5-Discuss-2024-9-17` | 1,730 | `data.yaml` | `['discuss']` | Multi-student peer conversation | **Fully Admitted in V3** (`discuss` -> `discuss`) |
| `SCB5-Talk-2024-9-17` | 9,081 | `SCB5-Talk-2024-9-17.yaml` | `['talk']` (5,506 boxes) | Individual verbal speech | **QUARANTINED** (Ambiguous verbal gesture) |
| `SCB5-Talk-Teacher-Behavior-2024-9-17` | 17,843 | `SCB5-Talk-Teacher-Behavior-2024-9-17.yaml` | `['talk', 'guide', 'answer', 'On-stage interaction', 'blackboard-writing']` | Teacher lecture actions | **EXCLUDED** (Teacher/lecture pedagogy, out-of-scope) |
| `SCB5-Teacher-2024-9-17` | 14,859 | `SCB5-Teacher-2024-9-17.yaml` | `['teacher']` | Instructor entity bounding box | **EXCLUDED** (Instructor entity, non-student) |
| `SCB5-Teacher-Behavior-2024-9-17` | 11,882 | `SCB5-Teacher-Behavior-2024-9-17.yaml` | `['guide', 'answer', 'On-stage interaction', 'blackboard-writing']` | Teacher instructional behavior | **EXCLUDED** (Teacher actions, non-student) |
| `SCB5-BlackBoard-Screen` | 4,882 | `data.yaml` | `['blackBoard', 'screen']` | Classroom fixtures / objects | **EXCLUDED** (Objects/fixtures, not persons) |
| `SCB5-BlackBoard-Sreen-Teacher` | 7,545 | `data.yaml` | `['blackBoard', 'screen', 'teacher']` | Fixtures and teacher entity | **EXCLUDED** (Fixtures and instructor, not student behaviors) |

---

## 2. In-Depth Evaluation of Candidate Subset: `SCB5-Talk-2024-9-17`

`SCB5-Talk-2024-9-17` contains 4,539 images and 5,506 bounding boxes labeled `talk`:
- **Physical Semantics**: Annotations identify a seated student actively speaking or responding.
- **Physical Verification**:
  1. Hash matching against `processed_v3` reveals substantial frame sharing with `SCB5-Handrise-Read-write` and `SCB_BowTurnHead`.
  2. Bounding boxes in `SCB5-Talk` do not capture lateral head orientation (which is `TurnHead`) or sleeping/forehead-down (which is `BowHead`).
  3. Attempting to map `talk` to `turn_head` would introduce massive semantic contamination: students looking straight ahead while vocalizing would be falsely supervised as turning their head away from the exam.
  4. Attempting to map `talk` to `discuss` would conflict with the multi-student definition of `discuss` (where 2+ students interact).
- **Verdict**: **QUARANTINE**. `SCB5-Talk` must NOT be admitted into Stage 1.5 training.

---

## 3. Provenance of True Weak-Class Data (`BowHead` & `TurnHead`)

The authors of the Smart Classroom Behavior benchmark published `BowHead` and `TurnHead` in a distinct release: `SCB_BowTurnHead_20250509`:
- **Location**: `datasets/raw/scbehavior/SCB_BowTurnHead_20250509/SCB5-Turn-Bow-Head-2024-9-17/`
- **Total Images**: 2,410 images across 57 video recording groups.
- **Total Labels**: 2,410 label files (4 images had empty annotations).
- **V3 Ingestion Status**: Exactly **2,406 images** (99.8%) were already ingested into `processed_v3`.
- **Unused Groups**: Exactly **ZERO** unused groups of `BowHead` or `TurnHead` exist in the raw storage.

---

## 4. Architectural and Data-Centric Implication

Because all verified raw SCB5 data for `BowHead` and `TurnHead` is already present in the workspace, weak-class recovery cannot be achieved by simply dragging in external unverified datasets.

Instead, Stage 1.5 weak-class recovery must be achieved through **rigorous data-centric refinement**:
1. **Quarantining Conflicting Annotations**: Removing ambiguous forward slumps where `read`/`write` and `BowHead` overlap.
2. **Hard-Example Mining**: Identifying where Stage 1 best failed on the clean training domain (false negatives for head_down and turn_head, and hard normal negatives).
3. **Training-Sampler Re-weighting / Selective Focus**: Concentrating gradient updates on the clean decision boundaries of `head_down` and `turn_head`.
