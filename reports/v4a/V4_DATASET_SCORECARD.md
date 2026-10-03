# V4 Dataset Evaluation Scorecard
**Date:** 2026-10-02  
**Report ID:** V4A-SCORECARD-01  
**Scoring Paradigm:** Descriptive & Evidence-Based Criteria (No Arbitrary Numerical Ranks)  

---

## 1. Evaluation Methodology

In accordance with Section 15 of the V4A directive, every candidate source is evaluated across 12 descriptive axes grounded in physically verified local data or documented upstream telemetry.

Columns evaluated:
1. **Physical Access:** Status of acquiring binary media and annotations without fabricated credentials.
2. **Annotation Reliability:** Physical precision, bounding box validity, and freedom from label conflation.
3. **Head-Down Value:** Presence of deep head-down posture, sleeping, or forehead-on-desk instances.
4. **Turn-Head Value:** Presence of clear lateral yaw rotations (>35°) distinguishable from subtle glances.
5. **Cross-Camera Value:** Camera diversity, perspectives (high-angle, rear, upper-oblique).
6. **Temporal Value:** Contiguous video sequences, action intervals, and duration support.
7. **Pose Value:** Availability of human keypoints or continuous Euler angles.
8. **Person-ID Value:** Presence of consistent tracking identifiers across time.
9. **Resolution:** Physical pixel dimensions and spatial fidelity.
10. **License Clarity:** Openness and precision of academic research rights.
11. **Duplicate Risk:** Probability of identity or frame overlap with existing project corpora.
12. **Recommended Role:** Strategic placement in future V4 architecture.

---

## 2. Comprehensive Descriptive Scorecard

| Dataset | Physical Access | Annotation Reliability | Head-Down Value | Turn-Head Value | Cross-Camera Value | Temporal Value | Pose Value | Person-ID Value | Resolution | License Clarity | Duplicate Risk | Recommended Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SCBehavior High-Res** (`20191864136`) | **ACCESSIBLE**<br>(802 files checked out via Git-LFS) | **HIGH**<br>(8,083 valid COCO/YOLO bboxes; clean class definitions) | **HARD NEGATIVE**<br>(Supplies 1,021 read and 579 write hard negatives) | **VERY HIGH**<br>(1,011 validated clear turn_head annotations) | **MODERATE**<br>(3 elevated classroom setups) | **NONE**<br>(Static frames only) | **NONE**<br>(Bounding boxes only) | **NONE**<br>(No tracking IDs) | **VERY HIGH**<br>(2560x1440 58%, 3840x2160 42%) | **HIGH**<br>(Open GitHub repo for academic use) | **ZERO**<br>(0 SHA256 matches, 0 perceptual matches) | **CANONICAL FULL-FRAME & CROP TRAINING** (Primary static behavior anchor) |
| **EduAction** (`hhhhha-ops`) | **ACCESSIBLE**<br>(350 MP4 video clips cloned locally) | **HIGH**<br>(Clear clip-level behavior isolation; verified visually) | **VERY HIGH**<br>(50 clips of actual sleeping on desk; ~5,000 frames) | **SUPPORTING**<br>(50 talking clips with dynamic peer orientation) | **MODERATE**<br>(4 college classroom angles) | **VERY HIGH**<br>(Contiguous 3–5s clips at 25 fps; ~100 frames/clip) | **MODERATE**<br>(Suitable for keypoint extraction) | **HIGH**<br>(Single student isolated per clip) | **MODERATE**<br>(224x224 person crops) | **HIGH**<br>(Open GitHub research repository) | **ZERO**<br>(Completely independent college recording) | **PER-STUDENT CROP CLASSIFIER & TEMPORAL EVALUATION** (Primary dynamic sleep/head-down anchor) |
| **AFLW2000-3D** (`cleardusk/3DDFA`) | **ACCESSIBLE**<br>(Downloaded 158.8 MB test archive & configs) | **VERY HIGH**<br>(3DMM fitted Euler angles & 68 landmarks) | **MODERATE**<br>(Covers downward pitch angles up to 60°) | **VERY HIGH**<br>(Ground-truth yaw from -90° to +90°) | **SYNTHETIC / DIVERSE**<br>(Full 3D angle sphere) | **NONE**<br>(Static crops) | **MAXIMUM**<br>(Continuous Euler angles + 68 landmarks) | **NONE**<br>(Isolated face crops) | **MODERATE**<br>(120x120 to 450x450 crops) | **HIGH**<br>(Standard academic research benchmark) | **ZERO**<br>(Standard CVPR benchmark) | **HEAD-POSE BRANCH PRETRAINING & YAW CALIBRATION** |
| **Smart Classroom** (`master-weixiao`) | **BLOCKED**<br>(Behind Baidu Netdisk authentication) | **HIGH (DOCUMENTED)**<br>(YOLOv5 labels; separate reading, writing, listening, turning) | **HARD NEGATIVE**<br>(72k writing + 58k reading hard negatives) | **HIGH**<br>(5,339 turn_head annotations) | **MAXIMUM**<br>(4 synchronized cameras: front, rear, upper-oblique, teacher) | **VERY HIGH**<br>(12,836 clips of 1–174s duration) | **MODERATE**<br>(Full body surveillance) | **IMPLICIT**<br>(Requires multi-camera ByteTrack) | **MODERATE**<br>(800x450 at 25 fps) | **RESTRICTED**<br>(Non-commercial academic use only) | **ZERO**<br>(National Public Platform source) | **RESERVED FOR V4B CROSS-CAMERA HOLDOUT** (If user provides Baidu Netdisk access) |
| **CStudentAct** (HUST) | **BLOCKED**<br>(Requires signed institutional commitment) | **VERY HIGH (DOCUMENTED)**<br>(Spatio-temporal action tubes linked frame-by-frame) | **VERY HIGH**<br>(10,162 bboxes of verified sleeping) | **LOW**<br>(Focuses on phone, hand, sleep, stand) | **HIGH**<br>(Multiple classroom camera viewpoints) | **MAXIMUM**<br>(Action instances linked over time) | **MODERATE**<br>(Full-HD video) | **MAXIMUM**<br>(Explicit action instance IDs) | **HIGH**<br>(Full-HD 1920x1080) | **RESTRICTED**<br>(Strictly non-commercial; no redistribution) | **ZERO**<br>(HUST original recording) | **RESERVED FOR SPATIO-TEMPORAL BENCHMARKING** (If user signs commitment) |
| **Hashemite Cheating** (Kaggle) | **CONDITIONAL**<br>(Requires Kaggle API authentication) | **HIGH (DOCUMENTED)**<br>(5 specific exam cheating actions) | **MODERATE**<br>(Cheating actions; looking at neighbor) | **VERY HIGH**<br>(Peeking at another student's exam paper) | **MODERATE**<br>(Fixed camera at 3m distance) | **VERY HIGH**<br>(37 continuous video sequences) | **MODERATE**<br>(Canon 70D high quality) | **HIGH**<br>(Trackable students at fixed desks) | **HIGH**<br>(Full-HD 1920x1080 at 24 fps) | **MODERATE**<br>(Academic research) | **ZERO**<br>(Hashemite Univ recording) | **EXAM CHEATING TEMPORAL EVALUATION** |
| **ClassBehavior** (`weniu`) | **BLOCKED**<br>(Empty repository; data withheld) | **UNKNOWN**<br>(Zero annotations published) | **UNKNOWN** | **UNKNOWN** | **UNKNOWN** | **UNKNOWN** | **UNKNOWN** | **UNKNOWN** | **UNKNOWN** | **UNCLEAR** | **N/A** | **REJECTED** (Withheld by authors) |
| **Class Pose** (Sensors 2022) | **BLOCKED**<br>(No public repo) | **HIGH**<br>(Manual keypoints in dense classrooms) | **LOW** | **LOW** | **HIGH**<br>(CCTV surveillance) | **NONE**<br>(Static frames) | **MAXIMUM**<br>(COCO 17 keypoints under occlusion) | **NONE** | **VARIABLE** | **RESTRICTED** | **N/A** | **REJECTED** (Inaccessible without author outreach) |
