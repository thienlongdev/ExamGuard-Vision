# V4A Additional Dataset Search & Evaluation Report
**Date:** 2026-10-02  
**Phase:** V4A — Strong Data Acquisition + Physical Audit + Architecture Data Plan  
**Status:** Completed Search & Physical Verification  

---

## 1. Search Methodology & Scope

In accordance with Section 3 of the V4A authorization directive, comprehensive systematic searches were executed across GitHub, Hugging Face, Kaggle, Mendeley Data, Zenodo, and academic publishers (MDPI, IEEE, Springer, ACM, arXiv) targeting student behavior, cheating posture, exam surveillance, head pose estimation, and classroom activity recognition.

Search queries utilized:
- `"student behavior dataset classroom"`
- `"exam cheating behavior dataset"`
- `"classroom surveillance behavior"`
- `"turn head classroom dataset"`
- `"head down student dataset"`
- `"student posture dataset"`
- `"classroom pose dataset"`
- `"exam monitoring dataset"`
- `"multi camera student behavior"`
- `"sleeping classroom dataset"`
- `"student action recognition dataset"`

---

## 2. Comprehensive Candidate Evaluation Records

### Candidate 1: SCBehavior-Dataset (High-Resolution Classroom Behaviors)
* **Dataset Name:** SCBehavior-Dataset
* **URL:** `https://github.com/20191864136/SCBehavior-Dataset`
* **Paper / Reference:** Research on Student Behavior Detection in Smart Classroom (2024)
* **License:** Public Open Academic Access (No commercial claim, source repository unencumbered)
* **Accessibility:** **ACCESSIBLE** (Full Git LFS checkout completed locally into `datasets/raw_v4/scbehavior_highres/repo`)
* **Volume:** 400 high-resolution physical images (360 train, 40 val) across 8,083 physical annotations
* **Resolution:** 2560x1440 (58.0%) and 3840x2160 (4K UHD, 42.0%)
* **Camera Viewpoints:** Elevated classroom perspective, multi-row desk coverage, frontal and upper-oblique
* **Annotation Type:** Object bounding boxes in both YOLO format (.txt) and COCO JSON format (`instances_train2017.json`, `instances_val2017.json`)
* **Physical Classes:**
  1. `write` (579 annotations)
  2. `read` (1,021 annotations)
  3. `lookup` (4,480 annotations)
  4. `turn_head` (1,011 annotations)
  5. `raise_hand` (684 annotations)
  6. `stand` (66 annotations)
  7. `discuss` (242 annotations)
* **Person IDs:** Not tracked across video (static frame annotations)
* **Potential Use:**
  - `turn_head` (1,011 boxes): Direct supervision for lateral head turn
  - `read` (1,021 boxes) & `write` (579 boxes): **Essential NORMAL hard negatives** to prevent head_down false positives
  - `discuss` (242 boxes) & `stand` (66 boxes): full-frame classroom context
* **Known Issues:** The upstream `.gitattributes` initially pointed to an LFS pointer hash, requiring attribute fixing to trigger git-lfs pull. 0 duplicate images detected against previous SCB5 datasets.
* **Decision:** **ACCEPT** (Highest-value static classroom dataset acquired).

---

### Candidate 2: EduAction (College Student Action Dataset)
* **Dataset Name:** EduAction
* **URL:** `https://github.com/hhhhha-ops/EduAction-A-college-student-action-dataset-for-classroom-attention-estimation`
* **Paper / Reference:** *EduAction: A college student action dataset in the classroom for attention estimation*
* **License:** Public Academic Research (GitHub open repository)
* **Accessibility:** **ACCESSIBLE** (Cloned cleanly into `datasets/raw_v4/other_candidates/eduaction`, 46.5 MB)
* **Volume:** 350 physical video clips (MP4), 50 clips per class
* **Resolution:** 224x224 (pre-cropped person-centric video sequences)
* **Camera Viewpoints:** Classroom seating rows, multi-angle oblique and frontal crops
* **Annotation Type:** Clip-level behavior video sequences (duration 3-5 seconds, ~100 frames per clip at 25 fps)
* **Physical Classes:**
  1. `sleeping` (50 clips, ~5,000 frames)
  2. `writing` (50 clips, ~4,600 frames)
  3. `lecture` (50 clips, ~5,250 frames)
  4. `talking` (50 clips, ~6,000 frames)
  5. `play_phone` (50 clips, ~5,000 frames)
  6. `drinking` (50 clips)
  7. `watch_computer` (50 clips)
* **Person IDs:** Implicit (each clip isolates a single student crop over time)
* **Potential Use:**
  - `sleeping`: Ground-truth positive for `HEAD_DOWN_DEEP` / `SLEEPING`
  - `writing`: Clean crop-level normal hard negative
  - `lecture`: Crop-level normal baseline
  - `talking`: Temporal interaction / pairwise feature
* **Known Issues:** Fixed 224x224 crop format; cannot train full-frame bounding box detectors directly, but perfect for person-crop classifier and temporal posture classification.
* **Decision:** **ACCEPT** (Resolves the critical missing ground-truth `sleeping` / deep head-down posture).

---

### Candidate 3: AFLW2000-3D & AFLW_GT Head Pose Benchmark
* **Dataset Name:** AFLW2000-3D / AFLW Benchmark
* **URL:** `http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm` / `https://github.com/cleardusk/3DDFA`
* **Paper / Reference:** *Face Alignment Across Large Poses: A 3D Solution* (CVPR 2016 / TPAMI 2017)
* **License:** Non-commercial academic research use only (Flickr original images subject to standard research fair use)
* **Accessibility:** **ACCESSIBLE** (Test archive `test.data.zip` [158.8 MB] and ground truth numpy configs downloaded into `datasets/raw_v4/head_pose_aflw2000/`)
* **Volume:** 23,080 cropped face images (2,000 in AFLW2000-3D with 68 3D landmarks + yaw/pitch/roll pose, 21,080 in AFLW GT)
* **Resolution:** Face crops (variable resolution ~200x200 to 450x450, standardized 120x120/224x224)
* **Camera Viewpoints:** Full 3D pose distribution (yaw angles spanning -90° to +90°, pitch -60° to +60°)
* **Annotation Type:** Continuous 3D Euler angles (yaw, pitch, roll in degrees), 68 3D facial landmarks, ROI bounding boxes
* **Physical Classes:** Continuous head pose angles + landmark coordinates
* **Person IDs:** Static in-the-wild crops
* **Potential Use:**
  - Pretraining/benchmarking head pose estimation branch (yaw/pitch regression or binned angle classification).
  - Physical verification of lateral turn threshold (>35° yaw) and downward pitch (>30° pitch).
* **Known Issues:** Synthetic large-pose fitting on 2D landmarks; face crops only, not full classroom scenes.
* **Decision:** **ACCEPT** (Pretraining & calibration data for Head-Pose branch).

---

### Candidate 4: Smart Classroom Student Behavior Dataset (master-weixiao)
* **Dataset Name:** Smart-Classroom-Student-Behavior-Dataset
* **URL:** `https://github.com/master-weixiao/Smart-Classroom-Student-Behavior-Dataset`
* **Paper / Reference:** National Public Resource Service Platform (Minister-level Exemplary Classes 2019)
* **License:** Academic research only. Strictly prohibited for commercial use.
* **Accessibility:** **BLOCKED / PARTIALLY ACCESSIBLE**
  - GitHub repo contains only README, metrics curves, and training plots.
  - Raw multi-camera video archives (11.29 GB across 12,836 clips) are hosted exclusively behind Baidu Netdisk (`pan.baidu.com/s/1uSdGbXAyZKbxD4fQdOrZzA`, code `1234`), requiring Chinese SMS/Baidu client authentication.
* **Volume (Documented):** 12,836 clips across 4 viewpoints (Frontal: 593, Upper-Oblique: 4,781, Rear: 4,544, Teacher: 2,918)
* **Resolution:** 800x450 at 25 fps
* **Camera Viewpoints:** 4 synchronized/labeled perspectives: 正面 (front), 斜上方 (upper-oblique), 后方 (rear), 教师 (teacher)
* **Annotation Type:** YOLOv5 frame bounding boxes
* **Classes (Documented):** dx (head down writing: 72,462), dk (head down reading: 58,932), tt (head up listening: 117,528), zt (turn head: 5,339), js (raise hand: 4,183), zl (stand: 4,101), xt (group discussion: 4,663), jz (teacher guidance: 680)
* **Potential Use:** Exceptional multi-camera diversity and clear separation of writing/reading from listening/turning.
* **Known Issues:** Unreachable without manual interactive Baidu Netdisk extraction.
* **Decision:** **CONDITIONAL** (Top priority for manual acquisition if user has Baidu Netdisk credentials).

---

### Candidate 5: ClassBehavior / MSTA-SlowFast (Hunan Normal University)
* **Dataset Name:** ClassBehavior (MSTA-SlowFast)
* **URL:** `https://github.com/weniu/ClassBehavior/`
* **Paper / Reference:** *MSTA-SlowFast: A Student Behavior Detector for Classroom Environments* (Sensors 2023, 23(11), 5205; DOI: 10.3390/s23115205)
* **License:** Stated open in paper, but code repo has standard Apache/SlowFast headers
* **Accessibility:** **BLOCKED / WITHHELD BY AUTHORS**
  - Upstream GitHub repository contains only code skeletons (`cut_frames.sh`, `map.json`, `myava.txt`), but zero video clips and zero AVA CSV annotation files.
  - GitHub Issue #1 ("Can you supply your dataset and core?") remains open with 0 replies from authors.
* **Volume (Paper):** 7 classroom behaviors annotated in AVA spatio-temporal format
* **Classes:** `lookup`, `bow`, `turn`, `talk`, `stand`, `handsup`, `lie`
* **Decision:** **REJECT / BLOCKED** (Cannot be acquired; data withheld by authors despite paper availability claim).

---

### Candidate 6: CStudentAct & StudentAct (HUST Hanoi University of Science and Technology)
* **Dataset Name:** CStudentAct / StudentAct
* **URL:** `https://sigm-seee.github.io/datasets/CStudentAct.html`
* **Paper / Reference:** Hanoi University of Science and Technology, Signal & Information Mining (SIGM) Lab
* **License:** Strictly Non-Commercial Academic Research. Redistribution explicitly forbidden.
* **Accessibility:** **BLOCKED / MANUAL AGREEMENT REQUIRED**
  - Requires requestor to sign formal commitment agreement and email dataset administrator (`lan.lethi1@hust.edu.vn`).
* **Volume:**
  - Raising_hand: 296 images, 39 action instances, 3,991 bboxes
  - Using_phone: 3,125 images, 26 action instances, 11,998 bboxes
  - Sleeping: 3,313 images, 31 action instances, 10,162 bboxes
  - Standing: 4,623 images, 46 action instances, 6,626 bboxes
* **Resolution:** Full HD (1920x1080)
* **Camera Viewpoints:** Multi-camera classroom setup (ceiling-mounted oblique and wall angles)
* **Annotation Type:** Action tube spatio-temporal bounding boxes linked frame-by-frame (`frame_id`, `action_instance_id`, `x`, `y`, `w`, `h`)
* **Potential Use:** 10,162 verified `sleeping` bounding boxes across linked action instances.
* **Decision:** **CONDITIONAL** (Requires manual email request with signed institutional commitment form).

---

### Candidate 7: Class Pose (Pose Mask Classroom Dataset)
* **Dataset Name:** Class Pose
* **Paper / Reference:** *Pose Mask: A Model-Based Augmentation Method for 2D Pose Estimation in Classroom Scenes Using Surveillance Images* (Sensors 2022, 22, 8331; Wang et al.)
* **License:** Research use only; privacy restricted
* **Accessibility:** **BLOCKED / NO PUBLIC REPO**
  - No public repository or download URL exists. Restricted due to classroom privacy regulations. Available only upon direct request to corresponding author.
* **Volume:** 1,000 surveillance images with dense keypoint annotations under severe occlusion
* **Resolution:** Surveillance resolution (varying CCTV)
* **Annotation Type:** 2D Human Pose Keypoints (COCO keypoint format)
* **Decision:** **REJECT / BLOCKED** (Inaccessible without institutional author contact).

---

### Candidate 8: BIWI Kinect Head Pose Database
* **Dataset Name:** Biwi Kinect Head Pose Database (ETH Zurich)
* **URL:** `https://data.vision.ee.ethz.ch/cvl/gfanelli/kinect_head_pose_db.tgz`
* **Paper / Reference:** *Random Forests for Real Time 3D Face Analysis* (IJCV 2013)
* **License:** Non-commercial academic research and education use
* **Accessibility:** **BLOCKED / 403 FORBIDDEN**
  - Official ETH Zurich server returns HTTP 403 Forbidden to automated download requests. Hugging Face repository (`ETHZurich/biwi_kinect_head_pose`) hosts loader scripts pointing to the same forbidden URL.
* **Volume:** 15,000 frames across 20 subjects (RGB + Depth + 3D pose angles)
* **Resolution:** 640x480
* **Decision:** **BLOCKED** (Server forbidden; AFLW2000-3D serves equivalent role without access barriers).

---

### Candidate 9: AIRC-SMARTCLASS (Mendeley Data)
* **Dataset Name:** AIRC-SMARTCLASS
* **URL:** `https://data.mendeley.com/datasets/2bpt8w5w6c/1`
* **Paper / Reference:** University of Transport Technology, Hanoi (2024)
* **License:** CC BY 4.0
* **Accessibility:** Accessible via Mendeley Data direct download
* **Volume:** 12,429 Full-HD frames (1920x1080) from 45 video clips
* **Annotation Type:** YOLO bounding boxes for: `person` (230,409 boxes), `face` (62,001 boxes), and air-conditioner displays
* **Classes:** Person counting and face detection only (NO fine-grained behavior classes)
* **Decision:** **REJECT for Behavior / CONDITIONAL for CCTV Person Detection Benchmark** (No posture or cheating behavior labels).

---

### Candidate 10: NCBD (NanNing Normal University Classroom Behavior Dataset)
* **Dataset Name:** NCBD
* **URL:** `https://github.com/kuangxiaoye/NCBD`
* **Paper / Reference:** *Research and application of student classroom behavior recognition in an open environment* (Thesis 2024, Wang Ye)
* **License:** MIT (code), Research agreement (data)
* **Accessibility:** **BLOCKED**
  - Raw video files withheld due to privacy. Only extracted I3D feature vectors shared upon signed agreement emailed to `w258765@gmail.com`.
* **Volume:** 1,024 video clips (2K PTZ camera)
* **Classes:** Listen, Read, Turn Head, Write, Yawn, Eat, Drink, Use Phone, Use Computer
* **Decision:** **REJECT** (Raw pixels/frames not released; extracted feature vectors cannot train image/crop detectors).

---

### Candidate 11: Hashemite University Student Cheating Behavior Dataset (Kaggle)
* **Dataset Name:** Student Cheating Behavior Dataset
* **URL:** `https://www.kaggle.com/datasets/student-cheating-behavior-dataset/student-cheating-behavior-dataset`
* **Paper / Reference:** *Advances in Contextual Action Recognition: Automatic Cheating Detection Using Machine Learning Techniques*
* **License:** Unknown / Academic
* **Accessibility:** **CONDITIONAL** (Requires Kaggle API key `~/.kaggle/kaggle.json` or manual browser download)
* **Volume:** 37 video sequences (1920x1080 Full HD, 24 fps, Canon EOS 70D, 8 subjects)
* **Classes:** Exchanging exam papers, looking at another student's exam paper, using cheat sheet, using mobile phone, not cheating
* **Decision:** **CONDITIONAL** (Requires Kaggle credentials; high potential for temporal exam cheating evaluation).

---

## 3. Search & Acquisition Summary Table

| Source Name | Primary URL | License | Physical Status | Volume | Resolution | Behavioral Value | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SCBehavior-Dataset** | `20191864136/SCBehavior-Dataset` | Open Academic | **ACCESSIBLE (Locally Verified)** | 400 images / 8,083 bboxes | 2560x1440 & 4K | turn_head, read, write, discuss, stand | **ACCEPT** |
| **EduAction** | `hhhhha-ops/EduAction...` | Open Academic | **ACCESSIBLE (Locally Verified)** | 350 MP4 clips / ~35k frames | 224x224 crops | sleeping, writing, talking, phone | **ACCEPT** |
| **AFLW2000-3D** | `cleardusk/3DDFA` | Research Only | **ACCESSIBLE (Locally Verified)** | 23,080 face crops + pose | 120x120 to 450x450 | Continuous yaw/pitch/roll | **ACCEPT** |
| **Smart Classroom** | `master-weixiao/Smart-Classroom...` | Academic Only | **BLOCKED (Baidu Netdisk)** | 12,836 clips (11.3 GB) | 800x450 | Multi-camera classroom | **CONDITIONAL** |
| **CStudentAct** | `sigm-seee.github.io/CStudentAct` | Institutional Research | **BLOCKED (Signed Agreement)** | 11k bboxes / action tubes | 1920x1080 | sleeping, standing, phone | **CONDITIONAL** |
| **ClassBehavior** | `weniu/ClassBehavior` | Withheld | **BLOCKED (Empty Repo)** | 0 clips released | N/A | Withheld by authors | **REJECT** |
| **Class Pose** | Sensors 2022 (Wang et al.) | Restricted | **BLOCKED (No Public Repo)** | 1,000 images | CCTV | Pose keypoints only | **REJECT** |
| **BIWI Head Pose** | ETH Zurich / HF | Non-Commercial | **BLOCKED (403 Forbidden)** | 15,000 frames | 640x480 | Head pose | **BLOCKED** |
| **AIRC-SMARTCLASS** | Mendeley Data | CC BY 4.0 | Accessible | 12,429 frames | 1920x1080 | People/Face only (No behavior) | **REJECT (Behavior)** |
| **NCBD** | `kuangxiaoye/NCBD` | MIT / Restricted | **BLOCKED (Features only)** | 1,024 clips | 2K | Extracted features only | **REJECT** |
| **Hashemite Cheating** | Kaggle | Academic | **CONDITIONAL (Kaggle Auth)** | 37 videos | 1920x1080 | Cheating actions | **CONDITIONAL** |
