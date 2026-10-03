# Dataset Audit Cross-Check & Discrepancy Corrections

**Audit Date**: 2026-10-02  
**Subject**: Systematic comparison between the previous report (`reports/DATASET_AUDIT_SUMMARY.md`) and the verified physical disk ground-truth (`reports/DATASET_IDENTITY_VERIFICATION.md`).

---

## 1. Summary of Identified Contradictions

| Item / Finding | Previous Audit Claim | Physical Ground-Truth Reality | Root Cause | Severity | Corrective Action |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Nonexistent Local Source Classes** | `scb_dataset3` contains `using phone`, `leaning over table`; `scbehavior` contains `stand`, `lookup`. | **None of these classes exist on disk.** Only `discuss`, `hand-raising`, `read`, `write`, `BowHead`, and `TurnHead` exist. | The previous mapping config copied academic paper taxonomies rather than inspecting downloaded files. | **CRITICAL** | Purge all nonexistent classes from active mapping rules. Mark canonical `use_phone`, `stand`, `lean` as UNAVAILABLE until physical data is staged. |
| **Subset Collision in `scb_dataset3`** | `scb_dataset3` has: 18,428 `discuss`, 20,978 `hand-raising`, 9,016 `read`, 0 `write`. | True counts: 5,392 `discuss`, 13,453 `hand-raising`, 24,078 `read`, 9,841 `write`. | `SCB5-Discuss` and `SCB5-Handrise-Read-write` were merged naively. Both used Class ID 0 for different behaviors (`discuss` vs. `hand-raising`). | **CRITICAL** | Isolate subsets into independent directories with their own `data.yaml`. Never merge folders with conflicting class ID numbering. |
| **Missing `write` Class** | Previous report listed 0 instances of `write` in `scb_dataset3`. | **9,841 instances** of `write` physically exist in `SCB5-Handrise-Read-write`. | In the flat list `[discuss, hand-raising, read, write]`, `write` was at index 3, but `data.yaml` defines `write` at index 2 (`names: [hand-raising, read, write]`). | **HIGH** | Re-audit using explicit per-folder `data.yaml` indices. |
| **`cctv_exam_monitor` Physical Presence** | Acknowledged as pending, but classes were listed in mapping table without physical data. | Directory `datasets/raw/cctv_exam_monitor` is empty (0 images). | Kaggle API credentials are not configured on the machine. | **HIGH** | Strictly mark all `cctv_exam_monitor` mappings as inactive until files are downloaded. Keep `READY FOR FULL TRAINING = NO`. |
| **Exam Cheating (Roboflow / Kaggle)** | Mapped `phone use`, `looking around`, `cheating`, `no cheating`. | Directories are completely empty (0 images). | External download required. | **HIGH** | Exclude from processed manifests. |

---

## 2. Detailed Breakdown of Contradictions

### A. Nonexistent Classes Purged from Mapping
The previous `configs/dataset_mapping.yaml` referenced the following classes that **do NOT physically exist in any local directory**:
- `using phone`
- `leaning over table`
- `stand`
- `lookup`
- `Correct Posture`
- `LeftSideMove`
- `RightSideMove`
- `ForwardMove`
- `BackwardMove`
- `cheating`
- `no cheating`

**Audit Rule Applied**: Mappings must only reference classes that physically exist in the local dataset directories. Referencing hypothetical or un-downloaded classes in active training manifests corrupts the label vocabulary.

---

### B. Class ID Collision in `scb_dataset3`
In `datasets/raw/scb_dataset3/`, two independent archives were extracted:
1. `SCB5-Discuss-2024-9-17/data.yaml`:
   ```yaml
   nc: 1
   names: ['discuss']  # Class ID 0 = discuss
   ```
2. `SCB5-Handrise-Read-write-2024-9-17/data.yaml`:
   ```yaml
   nc: 3
   names: ['hand-raising', 'read', 'write']  # Class ID 0 = hand-raising, 1 = read, 2 = write
   ```

Because both datasets assigned `0` to different behaviors, scanning both directories with a unified index caused:
- 13,453 `hand-raising` bounding boxes (ID 0) to be parsed as `discuss` (5,392 + 13,453 = 18,845 `discuss` reported).
- 24,078 `read` boxes (ID 1) to be parsed as `hand-raising`.
- 9,841 `write` boxes (ID 2) to be parsed as `read`.
- Index 3 (`write`) had 0 boxes.

**Ground-Truth Corrected Annotation Distribution**:
```
SCB5-Discuss:
  Class 0 (discuss):        5,392 annotations across 864 images

SCB5-Handrise-Read-write:
  Class 0 (hand-raising):  13,453 annotations across 4,064 images
  Class 1 (read):          24,078 annotations across 3,878 images
  Class 2 (write):          9,841 annotations across 1,721 images

SCB_BowTurnHead:
  Class 0 (BowHead):        4,962 annotations across 1,001 images
  Class 1 (TurnHead):      11,156 annotations across 2,098 images

TOTAL VERIFIED ANNOTATIONS: 68,882 across 10,138 images
```

---

## 3. Corrected Source Class Inventory

| Dataset Key | Exact Folder Name | Local YAML | Class ID | Verified Class Name | Annotation Count | Image Count | Share in Subset |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `scb_discuss` | `SCB5-Discuss-2024-9-17` | `data.yaml` | 0 | `discuss` | 5,392 | 864 | 100.0% |
| `scb_handrise_read_write` | `SCB5-Handrise-Read-write-2024-9-17` | `data.yaml` | 0 | `hand-raising` | 13,453 | 4,064 | 28.4% |
| `scb_handrise_read_write` | `SCB5-Handrise-Read-write-2024-9-17` | `data.yaml` | 1 | `read` | 24,078 | 3,878 | 50.8% |
| `scb_handrise_read_write` | `SCB5-Handrise-Read-write-2024-9-17` | `data.yaml` | 2 | `write` | 9,841 | 1,721 | 20.8% |
| `scbehavior_bow_turn` | `SCB_BowTurnHead_20250509` | `data.yaml` | 0 | `BowHead` | 4,962 | 1,001 | 30.8% |
| `scbehavior_bow_turn` | `SCB_BowTurnHead_20250509` | `data.yaml` | 1 | `TurnHead` | 11,156 | 2,098 | 69.2% |
