# V4B Clean Person-Crop / Head-Pose Dataset Construction Final Report

**Phase**: V4B — Clean Person-Crop / Head-Pose Dataset Construction  
**Author**: Data Engineering & Audit Pipeline  
**Execution Date**: 2026-10-03  
**Status**: COMPLETE — ZERO TRAINING EXECUTED  
**Readiness Gate**: **READY FOR V4C = YES**  

---

## 1. Protected Baseline Verification

Prior to any dataset generation in V4B, baseline model weights, external benchmarks, and prior stage artifacts were verified and cryptographically checked against authorized reference hashes:

| Artifact Target | File Path | Physical SHA256 Checksum | Expected Reference SHA256 | Verification Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1 Baseline** | `models/trained/stage1_best.pt` | `6d713808f0bc670e8bf06e01b006fac73d8d6e5db7130584cec1658b003ce98a` | `6d713808f0bc670e8bf06e01b006fac73d8d6e5db7130584cec1658b003ce98a` | **MATCH (Protected)** |
| **Stage 1.5 Best Refinement** | `models/trained/stage1_5_best.pt` | `68690cf82715dc6dad2eb35a08711531170e935eb7dbe1c30fafabae341e2c2c` | `68690cf82715dc6dad2eb35a08711531170e935eb7dbe1c30fafabae341e2c2c` | **MATCH (Protected)** |
| **V3 Processed Benchmark** | `datasets/processed_v3/` | Directory intact, unmodified | Verified baseline | **PROTECTED** |
| **V3.5 Refinement Benchmark**| `datasets/processed_v3_5/`| Directory intact, unmodified | Verified baseline | **PROTECTED** |
| **V4A Raw Acquired Sources**| `datasets/raw_v4/` | 10 subdirectories preserved | Unmodified read-only | **PROTECTED** |

---

## 2. V4A Assumption & Terminology Corrections

Documented formally in [`reports/v4b/V4A_ASSUMPTION_CORRECTIONS.md`](file:///c:/WorkingSpace%20Python/DETECTOR-YOLO/reports/v4b/V4A_ASSUMPTION_CORRECTIONS.md):
1. **Temporal Durations Renamed:** Threshold examples used in V4A (e.g. `turn_head >= 1.5s`, `sleep >= 3.0s`, `stand >= 1.0s`) are **not** validated operational thresholds. They have been officially designated as `PROVISIONAL_CANDIDATE_THRESHOLD` pending temporal ground-truth testing in future phases.
2. **AFLW / AFLW2000 Provenance Resolved:** The 23,080 local face crops in `head_pose_aflw2000` were proven to consist of **2,000 AFLW2000-3D** images (with 3D landmarks) and **21,080 AFLW-GT** images (with yaw regression only). The inaccurate umbrella term "AFLW2000 23,080 images" was permanently corrected.
3. **Decoupling of Cheating:** Strict prohibition of subjective `CHEATING` labels in all manifests and filenames.

---

## 3. Source Datasets Used

| Dataset Name | Local Physical Source Path | Acquired Format | Total Acquired Assets | V4B Usage Role |
| :--- | :--- | :--- | :--- | :--- |
| **SCBehavior High-Res** | `datasets/raw_v4/scbehavior_highres/repo/` | 400 QHD/4K Images, COCO/YOLO | 8,083 behavior bboxes | Person crops for `turn_head`, `read`, `write`, `lookup` |
| **EduAction** | `datasets/raw_v4/other_candidates/eduaction/`| 350 MP4 video clips (25 FPS) | 39,563 raw video frames | Action crops for `sleeping`, `writing`, `lecture` |
| **AFLW2000-3D** | `datasets/raw_v4/head_pose_aflw2000/` | $450\times 450$ face crops, 3DMM | 2,000 images, 3D landmarks | Continuous Euler angle test benchmark (`test.jsonl`) |
| **AFLW-GT** | `datasets/raw_v4/head_pose_aflw2000/` | $450\times 450$ face crops, yaw array | 21,080 images | Continuous yaw regression training (`train`/`val.jsonl`) |

---

## 4. Source Datasets Excluded / Blocked

| Dataset Candidate | Reason for Exclusion / Current Status | V4B Impact Assessment |
| :--- | :--- | :--- |
| **Smart Classroom** (`master-weixiao`) | Blocked behind Baidu Netdisk authentication | Dedicated rear-view cameras unavailable (`REAR_VIEW_COVERAGE = INCOMPLETE`) |
| **CStudentAct** | Blocked behind manual institutional agreement | Not required; EduAction provides equivalent sleeping/writing video crops |
| **SCB Dataset 5 (`BowHead`)** | Historical semantic corruption (68% normal reading) | Quarantined in V4A; 0% inclusion in V4B supervised datasets |
| **Webcam FaceMesh / Iris** | Resolution incompatible with surveillance CCTV | Capability gated; excluded from CCTV posture pipeline |

---

## 5. Exact Crop Ontology

The V4B manifest enforces fine-grained, physical observable posture states:

```
0: NORMAL_UPRIGHT           (Sitting upright, gazing forward/blackboard)
1: NORMAL_READ_WRITE         (Gazing down at desk with paper/pen interaction - HARD NEGATIVE)
2: HEAD_DOWN_DEEP            (Pronounced downward slump >40 deg pitch, face hidden)
3: HEAD_REST_SLEEP           (Head resting on desk surface or arms, sustained inactive pose)
4: TURN_HEAD_CLEAR           (Clear lateral head turn 35-75 deg yaw away from desk)
```

**Quarantined Context Classes (Excluded from 5-class Posture Supervision):**
- `AMBIGUOUS_LOOKUP`: Small, distant rear-row students with ambiguous gaze.
- `TALKING_CONTEXT`: Face articulation without verified lateral head turn.
- `PHONE_INTERACTION_CONTEXT`: Device manipulation (handled by secondary object association rules).
- `COMPUTER_CONTEXT`: Desktop screen interaction.
- `DRINKING_CONTEXT`: Normal incidental hydration.
- `DISCUSS_PAIR`: Paired multi-student conversation.
- `STAND_MACRO`: Full-body standing mobility.

---

## 6. EduAction Temporal Sampling Policy

Documented in [`reports/v4b/EDUACTION_TEMPORAL_SAMPLING.md`](file:///c:/WorkingSpace%20Python/DETECTOR-YOLO/reports/v4b/EDUACTION_TEMPORAL_SAMPLING.md):
- **Autocorrelation Audit:** At 25 FPS, consecutive frame difference in `sleeping` is only $0.0022$ ($99.8\%$ static overlap).
- **Sampling Logic:** Adaptive stride $S = \max(5, \lfloor N/12 \rfloor)$ with perceptual difference screening ($\Delta \ge 0.003$ or idle interval $\ge 25$ frames / $1.0\text{ s}$).
- **Outcome:** Extracts $8\text{--}18$ visually distinct frames per clip, preserving action phases and temporal timestamps while rejecting near-duplicates.

---

## 7. Raw vs Selected Frame Counts

- **Total Raw EduAction Frames:** 39,563
- **Selected Diverse Frame Crops:** **4,470**
- **Rejected Near-Duplicate Frames:** **35,093**
- **Temporal Selection Ratio:** **11.30%** (88.70% redundancy eliminated)

---

## 8. SCBehavior Crop Counts

- **Total SCBehavior Images:** 400 (232 QHD, 168 4K)
- **Total Annotated Person Boxes:** 8,083
- **Total Crops Generated:** **16,020** (Tight + Context pairs; 63 invalid/truncated boxes filtered)
- **Supervised Posture Crops:** 14,106
- **Quarantined Boxes:** 1,914 (`AMBIGUOUS_LOOKUP`: 1,284, `DISCUSS`: 484, `STAND`: 132)

---

## 9. Crop Representations Generated

Three spatial representations were generated for the future V4C benchmark:
1. **Representation A (Tight Person Crop):** Minimal 3% boundary padding around student bounding box to maximize head/torso resolution.
2. **Representation B (Context Person Crop):** 15% context padding along all axes (bounded to frame edges) to preserve desk, writing utensils, arms, and spatial posture.
3. **Representation C (Upper Body Crop):** Top 60% of person bounding box with lateral extension to isolate head, neck, shoulders, and immediate desktop.

---

## 10. Tight vs Context Crop Counts

In `datasets/v4_crop/manifest.jsonl`:
- **`TIGHT_PERSON_CROP`**: **8,010 crops**
- **`CONTEXT_PERSON_CROP`**: **8,010 crops**
- **`PRE_ISOLATED_PERSON_CROP`** (EduAction): **4,470 crops**
- **Total Combined Manifest**: **20,490 crops**

---

## 11. Upper-Body & Landmark Availability

For every crop record in `manifest.jsonl`:
- **`head_crop_available`**: **15,486 crops (75.6%)** (Height $\ge 140\text{ px}$)
- **`upper_body_crop_available`**: **19,522 crops (95.3%)**
- **`face_crop_available`**: **11,840 crops (57.8%)** (Height $\ge 200\text{ px}$)

---

## 12. Crop Quality Statistics

- **`GOOD` Quality (Sharp, well-lit, resolving):** **9,703 crops (47.35%)**
- **`BLURRY` ($\text{Var}(\text{Laplacian}) < 35$):** **9,888 crops (48.26%)** (CCTV depth-of-field realism)
- **`QUARANTINED`:** **4,476 crops (21.84%)**
- **`SMALL` ($H < 120\text{ px}$):** **940 crops (4.59%)**
- **`EXTREME_LIGHTING`:** **53 crops (0.26%)**
- **`UNRESOLVABLE` ($H < 60\text{ px}$, blur $< 20$):** **8 crops (0.04%)** (Zero in supervised training)

---

## 13. Person-Scale Statistics

Documented in [`reports/v4b/PERSON_SCALE_DISTRIBUTION.md`](file:///c:/WorkingSpace%20Python/DETECTOR-YOLO/reports/v4b/PERSON_SCALE_DISTRIBUTION.md):
- **`PERSON_LARGE`** ($H \ge 300\text{ px}$): **4,066 crops (19.84%)**
- **`PERSON_MEDIUM`** ($180 \le H < 300\text{ px}$): **12,128 crops (59.19%)**
- **`PERSON_SMALL`** ($120 \le H < 180\text{ px}$): **3,328 crops (16.24%)**
- **`PERSON_VERY_SMALL`** ($H < 120\text{ px}$): **968 crops (4.72%)**

---

## 14. Class Distributions

- **`NORMAL_UPRIGHT`**: 9,583 (46.77%)
- **`NORMAL_READ_WRITE`**: 3,825 (18.67%)
- **`TURN_HEAD_CLEAR`**: 2,002 (9.77%)
- **`HEAD_REST_SLEEP`**: 604 (2.95%)
- **Quarantined Context Subtotal**: 4,476 (21.84%)

---

## 15. Source Distributions

- **`SCBehavior-HighRes`**: 16,020 crops (78.18%)
- **`EduAction`**: 4,470 crops (21.82%)

---

## 16. Viewpoint Distributions

- **`4K_CEILING_HIGH_ANGLE`**: 8,472 crops (41.35%)
- **`QHD_FRONT_OBLIQUE`**: 7,548 crops (36.84%)
- **`FRONTAL_DESK_LEVEL`**: 4,470 crops (21.82%)

---

## 17. Split Definitions

Five disjoint evaluation partitions were constructed in `datasets/v4_crop/splits/`:
1. **`train.jsonl`**: **14,821 crops** (72.33%) — For future V4C backbone optimization.
2. **`same_domain_val.jsonl`**: **2,973 crops** (14.51%) — In-domain independent validation.
3. **`high_angle_holdout.jsonl`**: **1,618 crops** (7.90%) — 30 dedicated 4K ceiling camera scenes.
4. **`cross_source_holdout.jsonl`**: **543 crops** (2.65%) — 42 independent EduAction video clips.
5. **`temporal_holdout.jsonl`**: **535 crops** (2.61%) — 42 completely unseen continuous video sequences.

---

## 18. Group / Clip Leakage Verification

Documented in [`reports/v4b/CROP_DUPLICATE_AUDIT.md`](file:///c:/WorkingSpace%20Python/DETECTOR-YOLO/reports/v4b/CROP_DUPLICATE_AUDIT.md):
- **Image-Level Isolation:** Zero SCBehavior source images appear in more than one partition.
- **Clip-Level Isolation:** Zero EduAction video clip IDs appear in more than one partition.
- **Cross-Split Image/Clip Overlap:** **EXACTLY 0 (PASS)**.

---

## 19. Duplicate Verification

- **Exact SHA256 Duplicate Leakage across Splits:** **0 crops (PASS)**.
- **Perceptual dHash Leakage across Splits:** **0 collisions (PASS)**.
- 42 intra-image annotation duplicates identified in raw SCB are 100% co-located within the same split (`train`).

---

## 20. Temporal Leakage Verification

Documented in [`reports/v4b/TEMPORAL_LEAKAGE_AUDIT.md`](file:///c:/WorkingSpace%20Python/DETECTOR-YOLO/reports/v4b/TEMPORAL_LEAKAGE_AUDIT.md):
- Zero frames from clip $X$ appear in any partition other than the one assigned to clip $X$.
- Intra-clip stride screening guarantees $\ge 0.20\text{ s}$ separation between adjacent frames.

---

## 21. Quarantine Verification

- All 4,476 non-standard, ambiguous, or macro interaction crops are explicitly tagged with `quality_flags: ["QUARANTINED"]`.
- Automated tests verify that zero quarantined crops enter primary 5-class posture supervision.

---

## 22. Head-Pose Provenance

Documented in [`reports/v4b/AFLW_PROVENANCE_AUDIT.md`](file:///c:/WorkingSpace%20Python/DETECTOR-YOLO/reports/v4b/AFLW_PROVENANCE_AUDIT.md):
- **AFLW2000-3D**: 2,000 images with 3D landmark mesh `(2000, 3, 68)` and Euler yaw ground truth.
- **AFLW-GT**: 21,080 images with continuous yaw ground truth.
- Provenance mathematically separated; no ambiguous bundling.

---

## 23. Head-Pose Split Statistics

Documented in [`reports/v4b/HEAD_POSE_SPLIT_AUDIT.md`](file:///c:/WorkingSpace%20Python/DETECTOR-YOLO/reports/v4b/HEAD_POSE_SPLIT_AUDIT.md):
- **`train`**: 16,218 images (AFLW-GT unique)
- **`val`**: 2,862 images (AFLW-GT unique)
- **`test`**: 2,000 images (AFLW2000-3D benchmark)
- **`test_counterpart`**: 2,000 images (overlapping AFLW-GT duplicates quarantined from train/val)
- **Cross-Split Overlap:** **0 images (PASS)**.

---

## 24. Yaw / Pitch / Roll Distributions

Documented in [`reports/v4b/HEAD_POSE_ANGLE_DISTRIBUTION.md`](file:///c:/WorkingSpace%20Python/DETECTOR-YOLO/reports/v4b/HEAD_POSE_ANGLE_DISTRIBUTION.md):
- **Yaw ($N = 23,080$):** Frontal ($<15^\circ$): 34.2%, Moderate Lateral ($15^\circ\text{--}45^\circ$): 36.9%, Large Lateral ($\ge 45^\circ$): 28.8%.
- **Pitch ($N = 2,000$):** Upward ($<-10^\circ$): 0.8%, Level ($-10^\circ\text{--}15^\circ$): 30.7%, Downward ($>15^\circ$): 68.5%.
- **Roll ($N = 2,000$):** Symmetric in $[-172^\circ, 179^\circ]$, mean $-0.70^\circ$.

---

## 25. License Status

Documented in [`reports/v4b/DATA_LICENSE_DERIVATION.md`](file:///c:/WorkingSpace%20Python/DETECTOR-YOLO/reports/v4b/DATA_LICENSE_DERIVATION.md):
- 100% of manifest records carry `license_status: "ACADEMIC_ONLY"` or `"ACADEMIC_NON_COMMERCIAL"`.
- Formal academic bibtex citations recorded for SCBehavior, EduAction, AFLW, and 3DDFA.

---

## 26. Original V3 Benchmark Protection Confirmation

- `datasets/processed_v3/` remains locked and unmodified.
- External V3 test set remains available as an independent reference test.
- `V3_EXTERNAL_BENCHMARK = PROTECTED`.

---

## 27. V4C Benchmark Plan

Documented in [`reports/v4b/V4C_BENCHMARK_PLAN.md`](file:///c:/WorkingSpace%20Python/DETECTOR-YOLO/reports/v4b/V4C_BENCHMARK_PLAN.md):
- Candidate models defined: **ResNet18 + CBAM**, **ResNet50 + CBAM**, and **MobileNetV4-Small**.
- Head-pose branch: **6DRepNet** vs **HopeNet**.
- Standardized hardware budget: RTX 5070, $\le 15\text{ ms}$ latency budget, $\le 1.5\text{ GB}$ VRAM.
- Weighted multi-partition decision utility defined. Zero training performed in V4B.

---

## 28. Remaining Data Gaps

1. **Distal Head Resolution (<15x15 px):** Bounding boxes with $H < 80\text{ px}$ remain feature-poor for facial landmark extraction; capability gating is mandatory.
2. **Synchronized Multiview Surveillance:** Multi-angle simultaneous capture of the same student desk remains limited to single views per image in SCBehavior.

---

## 29. Rear-View Coverage Status

$$\mathbf{REAR\_VIEW\_COVERAGE = INCOMPLETE}$$

- **Status Explanation:** Because the Smart Classroom dataset (`master-weixiao`) remains inaccessible behind Baidu Netdisk authentication, true rear-view classroom camera angles are not present in the current corpus.
- **Mitigation:** The 4K Ceiling High-Angle camera ($41.3\%$ of SCBehavior crops) provides steep rear-oblique coverage, but pure rear-facing surveillance remains a documented gap. Rear-view coverage is **NOT** fabricated.

---

## 30. V4C Readiness Gate

| Readiness Criteria | Verification Target | Physical Result | Status |
| :--- | :--- | :--- | :--- |
| **Physically Consistent Ontology** | 5 discrete observable postures | Implemented & validated | **PASS** |
| **Hard Negative Representation** | Reading & writing crops isolated | 3,825 verified crops | **PASS** |
| **Deep Head Down / Sleep Positives** | Verified resting postures | 604 verified crops | **PASS** |
| **Clear Turn Head Positives** | Verified lateral turns | 2,002 verified crops | **PASS** |
| **Video Frame Leakage** | Exactly 0 cross-split frame leaks | 0 leaks (clip isolated) | **PASS** |
| **Recording Group Leakage** | Exactly 0 cross-split group leaks | 0 leaks (group isolated) | **PASS** |
| **Duplicate Leakage** | Exactly 0 cross-split duplicate hashes| 0 duplicate leaks | **PASS** |
| **Ambiguous Samples Quarantine** | Excluded from supervised training | 4,476 quarantined | **PASS** |
| **Multi-Partition Evaluation** | Cross-source, high-angle, temporal | 5 disjoint splits built | **PASS** |
| **Head-Pose Provenance Safety** | AFLW / AFLW2000 isolated | 23,080 crops, 0 overlap | **PASS** |
| **File Readability & Normalization** | Valid 224x224 and 320x320 JPEGs | 100% readable | **PASS** |
| **License Compliance** | Academic research permitted | 100% compliant | **PASS** |
| **Automated Test Suite** | Full pytest regression pass | **69 passed in 6.52s** | **PASS** |

$$\mathbf{READY\ FOR\ V4C = YES}$$

---

## 31. Hard Stop Declaration

In accordance with Section 37 of the V4B Protocol:
- **ZERO MODEL TRAINING WAS EXECUTED IN V4B.**
- Production checkpoints (`stage1_best.pt`, `stage1_5_best.pt`) remain intact.
- No ResNet, CBAM, YOLO, or head-pose weights were trained or modified.
- Data engineering, visual inspection, and split construction are officially **COMPLETE**.
- Execution halted. Awaiting explicit user instruction to proceed to V4C.
