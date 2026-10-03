# SCB-Dataset5 Full — Identity Verification Report

**Verification Date**: 2026-10-02 11:32:27Z  
**Verification Method**: Physical inspection of downloaded archive, subdirectories, YOLO data YAMLs, and annotation files.  
**Strict Rule**: No class names, image counts, or annotations are inferred from papers or conversational assumptions.

---

## 1. Identity Verdict

**IS THIS ACTUALLY THE FULL SCB-DATASET5?**  
**YES**

### Evidence:
1. **Full Academic Scope**: The downloaded dataset matches the official release described in arXiv:2304.02488 (*"SCB-dataset: A dataset for detecting student classroom behavior"*).
2. **Subdataset Structure**: Contains all core SCB-Dataset5 sub-releases:
   - `SCB5-Discuss-2024-9-17`
   - `SCB5-Handrise-Read-write-2024-9-17`
   - `SCB5-Stand-2024-9-17` (Physically contains student/teacher standing annotations)
   - `SCB5-Talk-2024-9-17`
   - `SCB5-Teacher-2024-9-17` / `SCB5-Teacher-Behavior-2024-9-17`
   - `SCB5-BlackBoard-Screen`
3. **Physical File Verification**:
   - Total Images: **48,369**
   - Total Labels: **48,369**
   - Total Annotations: **111,976**
   - Subdatasets: **9**
4. **License**: Academic Research / MIT (as declared on host).

---

## 2. Subdatasets in Full Package

| Subdataset | YAML File | Images | Labels | Total Boxes | Classes Defined in YAML |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `SCB5-BlackBoard-Screen` | `SCB-Dataset\SCB5-BlackBoard-Screen\data.yaml` | 2,441 | 2,441 | 5,560 | 0:blackBoard, 1:screen |
| `SCB5-BlackBoard-Sreen-Teacher` | `SCB-Dataset\SCB5-BlackBoard-Sreen-Teacher\data.yaml` | 3,771 | 3,771 | 9,572 | 0:blackBoard, 1:screen, 2:teacher |
| `SCB5-Discuss-2024-9-17` | `SCB-Dataset\SCB5-Discuss-2024-9-17\data.yaml` | 864 | 864 | 5,392 | 0:discuss |
| `SCB5-Handrise-Read-write-2024-9-17` | `SCB-Dataset\SCB5-Handrise-Read-write-2024-9-17\SCB5-Handrise-Read-write-2024-9-17.yaml` | 6,864 | 6,864 | 47,372 | 0:hand-raising, 1:read, 2:write |
| `SCB5-Stand-2024-9-17` | `SCB-Dataset\SCB5-Stand-2024-9-17\SCB5-Stand-2024-9-17.yaml` | 7,602 | 7,602 | 12,969 | 0:stand |
| `SCB5-Talk-2024-9-17` | `SCB-Dataset\SCB5-Talk-2024-9-17\SCB5-Talk-2024-9-17.yaml` | 4,539 | 4,539 | 5,506 | 0:talk |
| `SCB5-Talk-Teacher-Behavior-2024-9-17` | `SCB-Dataset\SCB5-Talk-Teacher-Behavior-2024-9-17\SCB5-Talk-Teacher-Behavior-2024-9-17.yaml` | 8,920 | 8,920 | 11,692 | 0:talk, 1:guide, 2:answer, 3:On-stage interaction, 4:blackboard-writing |
| `SCB5-Teacher-2024-9-17` | `SCB-Dataset\SCB5-Teacher-2024-9-17\SCB5-Teacher-2024-9-17.yaml` | 7,428 | 7,428 | 7,727 | 0:teacher |
| `SCB5-Teacher-Behavior-2024-9-17` | `SCB-Dataset\SCB5-Teacher-Behavior-2024-9-17\SCB5-Teacher-Behavior-2024-9-17.yaml` | 5,940 | 5,940 | 6,186 | 0:guide, 1:answer, 2:On-stage interaction, 3:blackboard-writing |
