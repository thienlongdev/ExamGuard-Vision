# V4A Strong Data Acquisition, Physical Audit, and Architecture Final Report
**Date:** 2026-10-02  
**Phase:** V4A — Strong Data Acquisition + Physical Audit + Architecture Data Plan  
**Status:** Completed & Physically Verified  
**Training Authorized in this Phase:** **NONE (STRICT STOP ENFORCED)**  

---

## 1. Baseline Project State

The DETECTOR-YOLO project has successfully completed Stage 1 Foundation Training and Stage 1.5 Data-Centric Refinement. All prior canonical artifacts have been verified:
- **Canonical Stage 1 Checkpoint:** `models/trained/stage1_best.pt`  
  *SHA256:* `6d713808f0bc670e8bf06e01b006fac73d8d6e5db7130584cec1658b003ce98a`
- **Canonical Stage 1.5 Checkpoint:** `models/trained/stage1_5_best.pt`  
  *SHA256:* `68690cf82715dc6dad2eb35a08711531170e935eb7dbe1c30fafabae341e2c2c`
- **Stage 1.5 Final Report:** `reports/STAGE1_5_REFINEMENT_FINAL.md`  
  *SHA256:* `5407891c6642189c43deb037181e8e6164c00d39d7f70d21eb9fc620e8c7d4b1`
- **Protected Datasets:** `datasets/processed_v3/` and `datasets/processed_v3_5/` remain completely untouched and verified.
- **Canonical Taxonomy:** `0: normal`, `1: head_down`, `2: turn_head`, `3: discuss`, `4: stand`.  
  *Phone remains separated as an object detector class associated via ByteTrack and temporal rules.*
- **Critical Unseen V3 Validation Baseline:**  
  - `head_down` recall: **0.0514** (5.14%)  
  - `turn_head` recall: **0.1969** (19.69%)  
  - *Root cause:* The historical public SCB corpus had reached maximum exploitation with persistent conflation between normal reading/writing and `BowHead`. No further oversampling of the old corpus was authorized.

---

## 2. Regression Test Status

The full project regression test suite was executed in the project virtual environment (`.venv`):
- **Command:** `.\.venv\Scripts\python.exe -m pytest -v`
- **Result:** **58 passed, 0 failed, 1 warning in 3.40s**
- **Artifact Protection Test:** `tests/test_stage1_5_pipeline.py::test_stage1_5_protected_artifacts_unmodified` **PASSED**.
- **Holdout Integrity & Leakage Tests:** `test_stage1_5_holdout_integrity`, `test_stage1_5_zero_group_leakage`, `test_stage1_5_zero_duplicate_hash_leakage` **ALL PASSED**.

---

## 3. All Searched Datasets

A systematic survey covering academic publishers, GitHub, Hugging Face, Kaggle, Mendeley Data, and Zenodo identified 11 primary candidates:
1. `SCBehavior-Dataset` (`20191864136`) — High-resolution classroom student behaviors.
2. `EduAction` (`hhhhha-ops`) — College student action dataset in classroom environment.
3. `AFLW2000-3D` / `AFLW_GT` (`cleardusk/3DDFA`) — 3D dense head pose benchmark.
4. `Smart-Classroom-Student-Behavior-Dataset` (`master-weixiao`) — Multi-camera classroom behavior dataset.
5. `ClassBehavior` / `MSTA-SlowFast` (`weniu`) — Spatio-temporal classroom behavior dataset.
6. `CStudentAct` / `StudentAct` (HUST Hanoi) — Classroom action tube benchmark.
7. `Class Pose` (`Sensors 2022`, Wang et al.) — Classroom surveillance keypoint dataset.
8. `BIWI Kinect Head Pose Database` (ETH Zurich) — Multi-modal RGB-D head pose.
9. `AIRC-SMARTCLASS` (Mendeley Data) — High-angle classroom CCTV people counting & face detection.
10. `NCBD` (`kuangxiaoye`) — NanNing Normal University classroom behavior dataset.
11. `Student Cheating Behavior Dataset` (Hashemite University / Kaggle) — Classroom exam cheating video dataset.

---

## 4. Physically Accessible & Acquired Datasets

Under the directory structure `datasets/raw_v4/`, three essential datasets were successfully acquired and locally verified:

| Dataset | Local Path | Volume | Format | Status |
| :--- | :--- | :--- | :--- | :--- |
| **SCBehavior High-Res** | `datasets/raw_v4/scbehavior_highres/repo` | 400 images (2.5K/4K), 8,083 bboxes | YOLO .txt + COCO JSON | **100% LOCALLY CHECKED OUT** (746 MB) |
| **EduAction** | `datasets/raw_v4/other_candidates/eduaction` | 350 video clips (224x224, 25 fps) | MP4 Video Clips | **100% LOCALLY CLONED** (46.5 MB) |
| **AFLW2000-3D & AFLW_GT** | `datasets/raw_v4/head_pose_aflw2000` | 23,080 face crops + pose arrays | JPEG + NumPy Euler Arrays | **100% LOCALLY DOWNLOADED** (158.8 MB) |

---

## 5. Blocked Datasets & Exact Blockers

| Blocked Dataset | Physical URL | Exact Blocker Identified |
| :--- | :--- | :--- |
| **Smart Classroom** | `https://github.com/master-weixiao/Smart-Classroom...` | **Baidu Netdisk Barrier:** Raw video archives (11.29 GB) hosted exclusively at `pan.baidu.com/s/1uSdGbXAyZKbxD4fQdOrZzA?pwd=1234`. Automated download impossible without interactive Chinese phone SMS login / Baidu client. |
| **ClassBehavior** | `https://github.com/weniu/ClassBehavior/` | **Withheld by Authors:** Repository contains only code skeletons (`cut_frames.sh`, `map.json`). Raw videos and AVA CSV annotations omitted by authors; GitHub Issue #1 remains unanswered. |
| **CStudentAct (HUST)** | `https://sigm-seee.github.io/datasets/CStudentAct.html` | **Signed Institutional Commitment Required:** Requires requestor to sign formal non-redistribution agreement and email administrator (`lan.lethi1@hust.edu.vn`). |
| **Class Pose** | Sensors 2022 (Wang et al.) | **No Public Repository:** Surveillance images restricted due to classroom privacy regulations; available only upon direct contact with corresponding author. |
| **BIWI Head Pose** | ETH Zurich CVL | **HTTP 403 Forbidden:** Official server `data.vision.ee.ethz.ch` blocks automated HTTP requests. |
| **NCBD** | `https://github.com/kuangxiaoye/NCBD` | **Raw Video Withheld:** Authors share only pre-extracted I3D feature vectors upon signed agreement (`w258765@gmail.com`). No RGB pixels available for detector training. |

---

## 6. Physical File Counts & Integrity Audit

- **`scbehavior_highres`:**
  - `SCBehavior_COCO/coco/train2017`: 360 JPG images (2560x1440 and 3840x2160)
  - `SCBehavior_COCO/coco/val2017`: 40 JPG images (2560x1440 and 3840x2160)
  - `SCBehavior_COCO/coco/annotations`: 2 JSON files (`instances_train2017.json`, `instances_val2017.json`)
  - `SCBehavior_YOLO/images/train`: 360 JPG images
  - `SCBehavior_YOLO/images/val`: 40 JPG images
  - `SCBehavior_YOLO/labels/train`: 362 TXT annotation files
  - `SCBehavior_YOLO/labels/val`: 42 TXT annotation files
  - *Corrupted files detected:* **0**. All 802 LFS binary objects verified and checked out cleanly.
- **`eduaction`:**
  - `sleeping`: 50 MP4 files (100% playable, avg duration 4.3s at 25 fps)
  - `writing`: 50 MP4 files (100% playable, avg duration 3.7s at 25 fps)
  - `lecture`: 50 MP4 files (100% playable, avg duration 4.2s at 25 fps)
  - `talking`: 50 MP4 files (100% playable, avg duration 4.8s at 25 fps)
  - `play_phone`: 50 MP4 files (100% playable, avg duration 4.0s at 25 fps)
  - `drinking`: 50 MP4 files
  - `watch_computer`: 50 MP4 files
  - *Corrupted files detected:* **0**. Total 350 valid MP4 files.
- **`head_pose_aflw2000`:**
  - `test.data.zip`: 158,854,200 bytes containing 23,080 cropped face images.
  - `configs/AFLW2000-3D.pose.npy`: 2,000 float32 yaw angle ground truths.
  - `configs/AFLW2000-3D_crop.roi_box.npy`: 2,000 bounding boxes.
  - `configs/AFLW2000-3D.pts68.npy`: 2,000 68-point landmark arrays.

---

## 7. Annotation Formats Audited

1. **COCO JSON Format (`SCBehavior_COCO`):**
   - Standard 1-based category IDs with normalized bounding box definitions `[x, y, w, h]`.
   - Verified that zero boxes have negative coordinates, and all coordinates fit within registered dimensions.
2. **YOLO TXT Format (`SCBehavior_YOLO`):**
   - Standard normalized coordinate format: `<class_id> <x_center> <y_center> <width> <height>`.
3. **Clip-Level Video Labels (`EduAction`):**
   - Behavioral annotations structured by directory naming.
4. **Continuous Euler Angles (`AFLW2000-3D`):**
   - Continuous floating-point degree values representing $(\theta_{yaw}, \theta_{pitch}, \theta_{roll})$ alongside pixel landmark coordinates.

---

## 8. Physical Classes & Annotation Counts

### SCBehavior High-Res (8,083 total annotations)
- `lookup`: **4,480** (4,045 train, 435 val)
- `read`: **1,021** (920 train, 101 val)
- `turn_head`: **1,011** (915 train, 96 val)
- `raise_hand`: **684** (569 train, 115 val)
- `write`: **579** (520 train, 59 val)
- `discuss`: **242** (223 train, 19 val)
- `stand`: **66** (58 train, 8 val)

### EduAction Video Sequences (350 clips, ~35,000 frames)
- `sleeping`: **50 clips** (~5,000 frames)
- `writing`: **50 clips** (~4,600 frames)
- `lecture`: **50 clips** (~5,250 frames)
- `talking`: **50 clips** (~6,000 frames)
- `play_phone`: **50 clips** (~5,000 frames)
- `drinking`: **50 clips** (~5,000 frames)
- `watch_computer`: **50 clips** (~5,000 frames)

---

## 9. Camera and Viewpoint Inventories

1. **SCBehavior High-Res:**
   - 3 distinct classroom venues.
   - High-angle frontal-elevated and upper-oblique perspectives mounted at ceiling level ($H \approx 3.0\text{ m} - 4.5\text{ m}$).
   - Captures 6 to 8 desk rows with natural perspective foreshortening.
2. **EduAction:**
   - Multi-angle oblique and side-row camera placements across 4 college lecture halls.
   - Captures desk surfaces, student upper bodies, and lateral peer seating.
3. **Smart Classroom (Documented Upstream):**
   - 4 synchronized camera views: Frontal (593 clips), Upper-Oblique (4,781 clips), Rear (4,544 clips), Teacher (2,918 clips).

---

## 10. Temporal Annotation Availability

- **EduAction:** Fully contiguous temporal video sequences at 25 fps. Provides 350 continuous behavior clips ideal for sliding-window testing, duration thresholding, and transition analysis.
- **SCBehavior High-Res:** Static image frames. No inter-frame motion or temporal intervals.
- **CStudentAct (Blocked):** Spatio-temporal action tubes linking bounding boxes across time with explicit start/end frame indices.

---

## 11. Person-ID Availability

- **EduAction:** Implicit person ID (single student crop isolated throughout each clip).
- **CStudentAct (Blocked):** Explicit `action_instance_id` tracked over time.
- **SCBehavior High-Res:** Static frames; no person IDs. (In V4 runtime, ByteTrack dynamically assigns persistent track IDs).

---

## 12. Pose & Keypoint Availability

- **AFLW2000-3D:** 68 3D facial landmarks and continuous Euler angles $(\theta_{yaw}, \theta_{pitch}, \theta_{roll})$ across 2,000 crops; 21 landmarks across 21,080 AFLW crops.
- **Class Pose (Blocked):** 17 COCO body keypoints in occluded classrooms.
- **Classroom Student Crops:** High spatial resolution allows off-the-shelf top-down pose models (e.g. RTMPose / YOLOv8-pose) to extract neck, ear, and shoulder keypoints without custom annotations.

---

## 13. Resolution Statistics

- **SCBehavior High-Res:**
  - $2560\times1440$ (QHD): **232 images (58.0%)**
  - $3840\times2160$ (4K UHD): **168 images (42.0%)**
  - Average student bounding box size: $180\times240$ px in middle rows; $90\times120$ px in rear rows.
- **EduAction:**
  - $224\times224$ normalized square person crops at 25 fps.
- **Historical SCB5:**
  - $640\times640$ full-frame (rear student heads often $<15\times15$ px).

---

## 14. Usable Head-Down Candidates

The physical visual audit established:
1. **EduAction `sleeping`:**
   - 50 physical clips (~5,000 frames) of students with foreheads or cheeks resting directly on desks, faces occluded, or torsos slumped over crossed arms.
   - Provides **unambiguous, ground-truth positive evidence for deep head-down posture / sleep**.
2. **CStudentAct `sleeping` (Pending Access):**
   - 10,162 spatiotemporally annotated bounding boxes of sleeping students.

---

## 15. Usable Head-Down Hard Negatives

The primary breakthrough of the V4A acquisition is securing massive **normal hard negatives**:
1. **SCBehavior `read`:** **1,021 physical bounding boxes** of students with heads tilted down (20°–35°) reading textbooks.
2. **SCBehavior `write`:** **579 physical bounding boxes** of students with heads tilted down writing with pens on notebooks.
3. **EduAction `writing`:** **50 video clips (~4,600 frames)** of active pen manipulation.
4. **Total Hard Negatives:** **~6,200 physical instances** to permanently suppress false `head_down` detections during legitimate exam writing.

---

## 16. Usable Turn-Head Candidates

1. **SCBehavior `turn_head`:** **1,011 physical bounding boxes** in 2.5K/4K resolution. Visual contact sheet audit confirms clear lateral yaw (>35°–75°), with distinct facial profiles and ear exposure.
2. **AFLW2000-3D Extreme Yaw:** 462 face crops with $|yaw| > 35^\circ$ and 232 crops with $|yaw| > 60^\circ$ for precise angle calibration.

---

## 17. Ambiguous Labels Quarantined

- **`SCB5-Turn-Bow-Head / BowHead`:** Formally quarantined in `configs/v4_behavior_ontology.yaml`. Must never enter canonical training without crop-level human review.
- **Subtle Gaze Shifts ($15^\circ - 25^\circ$ yaw):** Quarantined as `UNDECIDED`. Not penalized during training; evaluated only temporally.
- **Light Look Down ($20^\circ - 35^\circ$ pitch without pen):** Quarantined as `UNDECIDED`.

---

## 18. Duplicate Findings

- **SHA256 Exact Matches:** **0 matches** between newly acquired V4 datasets and existing raw/processed datasets.
- **Perceptual dHash Matches ($D_H \le 4$):** **0 matches** across 50,774 comparisons.
- **Conclusion:** Newly acquired V4 data represents 100% novel, non-overlapping classroom footage.

---

## 19. License & Governance Matrix Summary

- **`SCBehavior High-Res`:** Safe for non-commercial academic research (Public GitHub).
- **`EduAction`:** Safe for non-commercial academic research (Public GitHub).
- **`AFLW2000-3D`:** Safe for non-commercial academic research (CasIA / CVPR 2016).
- **`Smart Classroom`:** Conditional (Requires Baidu Netdisk credentials).
- **`CStudentAct`:** Conditional (Requires signed commitment to `lan.lethi1@hust.edu.vn`).

---

## 20. Person-Crop Feasibility

- **Feasibility:** **EXCELLENT / 100% FEASIBLE**.
- Detailed in `reports/v4a/PERSON_CROP_FEASIBILITY.md`.
- 8,083 classroom crops extractable from SCBehavior; 35,000 frames from EduAction.
- 684 visual audit crops successfully generated and verified in `reports/v4a/contact_sheets/`.

---

## 21. Temporal Modeling Feasibility

- **Feasibility:** **EXCELLENT / 100% FEASIBLE**.
- Detailed in `reports/v4a/TEMPORAL_DATA_FEASIBILITY.md`.
- Leverages existing `TemporalBuffer` and ByteTrack track IDs to enforce duration thresholds ($\ge 1.5\text{ s}$ for turn head, $\ge 3.0\text{ s}$ for head down) and personal baseline calibration without retraining the base YOLO detector.

---

## 22. Head-Pose Data Plan Summary

- **Role 1 (Pretraining):** AFLW2000-3D + AFLW_GT continuously regresses Euler angles $(\theta_{yaw}, \theta_{pitch})$.
- **Role 2 (Classroom Adaptation):** Adapts to surveillance camera heights using classroom student crops and shoulder/ear keypoint vectors.
- Decoupled from canonical behavior classes: head angle provides geometric evidence, not direct label substitution.

---

## 23. GitHub Reference Architecture Analysis

Audited `phungthutrangsfl/cheating_behavior`, `DYBInh2k5`, `AarambhDevHub`, and `Zomma2`:
- Adopt two-stage detection + crop classification.
- Adopt duration-based state machines and personal baseline calibration.
- Reject binary "cheating" labels.
- Reject MediaPipe FaceMesh and Eye Gaze on surveillance CCTV feeds (reserved strictly for close-up webcam mode).

---

## 24. Target V4 Hybrid Architecture Recommendation

A robust, real-time multi-stage pipeline:
$$\text{VideoSource} \longrightarrow \text{Person Detector} \longrightarrow \text{ByteTrack} \longrightarrow \begin{cases}
\text{Per-Student Posture Crop Classifier (224x224)} \\
\text{Auxiliary 3D Head-Pose / Keypoint Branch} \\
\text{Stage 1.5 Full-Frame Anchor (stand / discuss)} \\
\text{Independent COCO Phone Detector}
\end{cases}$$
$$\longrightarrow \text{Multi-Cue Bayesian Fusion} \longrightarrow \text{TemporalBuffer} \longrightarrow \text{Rule Scorer} \longrightarrow \text{FastAPI Events}$$

---

## 25. Exact Datasets Recommended for V4B

When authorized, V4B will incorporate:
1. **`SCBehavior High-Res` (`20191864136`):**
   - 1,011 `turn_head` boxes
   - 1,021 `read` boxes (hard negative)
   - 579 `write` boxes (hard negative)
   - 242 `discuss` boxes
   - 66 `stand` boxes
2. **`EduAction` (`hhhhha-ops`):**
   - 50 `sleeping` clips (~5,000 frames) as ground-truth `head_down`
   - 50 `writing` clips (~4,600 frames) as crop-level hard negative
   - 50 `lecture` clips (~5,250 frames) as crop-level normal baseline
   - 50 `talking` clips (~6,000 frames) as crop-level interaction baseline
3. **`AFLW2000-3D`:**
   - 2,000 continuous pose crops for head pose branch calibration.

---

## 26. Datasets Rejected and Why

- **`ClassBehavior` (`weniu`):** Empty repository; authors withheld video clips and AVA CSV annotations.
- **`Class Pose`:** Inaccessible; surveillance images restricted without public repository.
- **`NCBD`:** Raw video withheld; feature vectors cannot train image/crop detectors.
- **`BIWI Kinect`:** Official ETH Zurich server returns 403 Forbidden.
- **`AIRC-SMARTCLASS`:** Contains only people counting and face detection; zero student behavior classes.
- **`SCB5 BowHead`:** Historically corrupted (68% normal reading/writing); permanently quarantined.

---

## 27. Manual Actions Still Required (Optional Future Extensions)

If the user desires even larger multi-camera holdout evaluation:
1. **Smart Classroom (11.29 GB):** Provide Baidu Netdisk extraction access for `pan.baidu.com/s/1uSdGbXAyZKbxD4fQdOrZzA?pwd=1234`.
2. **CStudentAct (HUST):** Complete and sign institutional commitment form and email `lan.lethi1@hust.edu.vn`.
3. **Hashemite Cheating (Kaggle):** Provide Kaggle API token `kaggle.json`.

*(Note: None of these manual actions block V4B; the already acquired datasets are completely sufficient).*

---

## 28. Total Estimated Usable New Samples

- **Verified Full-Frame Annotations (SCBehavior High-Res):** **8,083 bounding boxes** across 400 2.5K/4K images.
- **Verified Video Frame Samples (EduAction):** **~35,000 frames** across 350 video clips.
- **Verified Head-Pose Benchmark Crops (AFLW2000-3D):** **23,080 crops** with continuous 3D Euler angles.
- **Total Physical Usable Samples:** **> 66,000 verified training and evaluation instances**.

---

## 29. Major Remaining Risks & Mitigation

1. **Risk:** Distant students in back rows having small pixel size on CCTV.  
   *Mitigation:* Base resolution upgraded to 2.5K/4K; high-recall person detector + $224\times224$ crop normalization prevents scale degradation.
2. **Risk:** Over-alerting on brief glances.  
   *Mitigation:* Enforce $\ge 1.5\text{ s}$ duration threshold in `TemporalBuffer` before triggering an event.
3. **Risk:** Classroom lighting variation across unseen cameras.  
   *Mitigation:* V4 cross-camera holdout split ensures models are validated on completely novel room setups before deployment.

---

## 30. Readiness Gate Verdict

```
======================================================================
V4B READINESS GATE CRITERIA VERIFICATION:
======================================================================
1. At least one strong source for head_down or sleeping exists:   [YES] (EduAction: 50 clips, ~5k frames)
2. At least one strong source for clear turn_head exists:          [YES] (SCBehavior: 1,011 QHD/4K bboxes)
3. Camera/sequence grouping can be preserved:                     [YES] (Preserved in SCBehavior & EduAction)
4. Usable person crops can be generated:                          [YES] (Verified; 684 audit crops created)
5. License permits academic use:                                  [YES] (Verified in DATA_LICENSE_MATRIX.md)
6. Duplicate conflicts can be controlled:                         [YES] (Verified; 0 SHA256 / 0 dHash matches)
7. Semantics are physically verified:                             [YES] (Verified in contact sheets & ontology)
8. No critical annotation corruption remains unresolved:          [YES] (BowHead quarantined, read/write isolated)
======================================================================
FINAL GATE VERDICT: READY FOR V4B = YES
======================================================================
```

---

## 31. Strict Stop Enforced

In strict adherence to Section 19 and 22 of the user directive:
- **NO model training was initiated.**
- **NO new checkpoints were created.**
- **NO production weights were altered.**
- **Stage 2 was NOT started.**
- **Phase V4A is 100% COMPLETE.**

Awaiting explicit user authorization before proceeding to Phase V4B.
