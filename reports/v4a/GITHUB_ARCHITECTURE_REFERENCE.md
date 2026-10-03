# GitHub Open-Source Architecture Reference Analysis
**Date:** 2026-10-02  
**Report ID:** V4A-GH-ARCH-01  
**Objective:** Critical Architectural Audit of Open-Source Proctoring and Cheating Detection Repositories  

---

## 1. Architectural Survey Overview

To ground the proposed V4 hybrid architecture in proven engineering patterns while avoiding known pitfalls, four prominent open-source repositories were audited in depth alongside supplementary proctoring projects.

Repositories audited:
1. **`phungthutrangsfl/cheating_behavior`**: Two-stage detection + classification pipeline with visual validation.
2. **`DYBInh2k5/He_Thong_Giam_Sat_Thi_Online_Voi_AI`**: Complete AI online proctoring system with MediaPipe head pose & thresholding.
3. **`AarambhDevHub/exam-cheating-detection`**: Multi-modal proctoring system (gaze, mouth, multi-face, audio with Whisper).
4. **`Zomma2/Cheating-Detection`**: Cloud-distributed exam proctoring architecture (AWS Lambda, SageMaker, EFS, VAD).

---

## 2. Deep-Dive Repository Audits

### 2.1 Repository 1: `phungthutrangsfl/cheating_behavior`
* **Source:** `https://github.com/phungthutrangsfl/cheating_behavior`
* **Core Pipeline:**
  $$\text{Full Frame} \xrightarrow{\text{YOLOv11 Detection (Person)}} \text{Person BBox Crop (224x224)} \xrightarrow{\text{YOLOv11l-cls / ResNet}} \text{Binary/Posture Label}$$
* **What it Does Well:**
  - **Clean Decoupling:** Decouples the localization problem from the fine-grained classification problem. The full-frame detector is tasked solely with finding people (a task at which COCO-pretrained models excel), while a dedicated $224\times224$ classifier specializes in subtle posture features.
  - **Dynamic Crop Classification:** Uses a modern lightweight classifier (`yolo11l-cls.pt` or ResNet) with PyTorch AMP and cosine learning rate scheduling.
* **What is CCTV-Compatible:**
  - Highly robust to varying CCTV distances. As long as person bounding boxes exceed $\sim 80\times80$ px, resizing to $224\times224$ provides standardized input to the posture network.
* **What is Webcam-Only:**
  - The dataset linked in `Link dataset.txt` comes from Roboflow mock classrooms with single desks close to the camera.
* **What Lacks Validation:**
  - Zero temporal smoothing: performs frame-by-frame inference without tracking IDs or rolling buffers, causing label flicker on borderline postures.
  - Binary classification: collapses cheating into a single subjective class, lacking fine-grained semantic explanation (e.g. cannot tell whether a student is turning head or reaching for phone).
* **What Should NOT be Copied:**
  - Do NOT adopt binary "cheating vs. normal" classification. Observable physical behaviors must remain decoupled from subjective policy judgements.

---

### 2.2 Repository 2: `DYBInh2k5/He_Thong_Giam_Sat_Thi_Online_Voi_AI`
* **Source:** `https://github.com/DYBInh2k5/He_Thong_Giam_Sat_Thi_Online_Voi_AI`
* **Core Pipeline:**
  $$\text{Webcam Frame} \xrightarrow{\text{MediaPipe FaceMesh (468 pts)}} \text{6 Canonical 3D Points} \xrightarrow{\text{cv2.solvePnP}} \text{Euler Angles } (\theta_y, \theta_p) \xrightarrow{\text{Debounced FSM}} \text{Alert}$$
* **What it Does Well:**
  - **Rigorous 3D Pose Geometry:** Solves perspective-n-point using 6 canonical facial landmarks (nose tip, chin, eye corners, mouth corners) against standard 3D anthropometric face model:
    - Yaw threshold: $|\theta_y| > 25^\circ$
    - Pitch threshold: $|\theta_p| > 20^\circ$
  - **Stateful Violation Detector:** Uses sliding duration timers (`look_away_duration = 2.0s`, `face_missing_duration = 3.0s`, `cooldown = 5.0s`), effectively filtering out natural blinks and momentary shifts.
* **What is CCTV-Compatible:**
  - The temporal duration logic (`ViolationDetector` with timestamp-based state accumulation) is directly applicable to our ByteTrack tracks.
* **What is Webcam-Only:**
  - **MediaPipe FaceMesh is strictly webcam-only.** On surveillance cameras where faces are $<40\times40$ pixels or viewed from steep downward angles, FaceMesh landmarks fail completely.
* **What Lacks Validation:**
  - Fixed angular thresholds (25° yaw) without personal baseline calibration. A student seated at an angle naturally triggers continuous false alarms.
* **What Should NOT be Copied:**
  - Do NOT apply MediaPipe FaceMesh to multi-person CCTV frames.

---

### 2.3 Repository 3: `AarambhDevHub/exam-cheating-detection`
* **Source:** `https://github.com/AarambhDevHub/exam-cheating-detection`
* **Core Pipeline:**
  $$\text{Webcam Stream} \longrightarrow \begin{cases}
  \text{EyeTracker (EAR \& Horizontal Gaze Ratio)} \\
  \text{MouthMonitor (Lip Distance \& Openness)} \\
  \text{MultiFaceDetector (Count > 1)} \\
  \text{AudioMonitor (PyAudio + OpenAI Whisper for speech)}
  \end{cases} \longrightarrow \text{Event Dashboard}$$
* **What it Does Well:**
  - **Multi-Modal Sensing:** Combines visual gaze, mouth articulation, and acoustic speech recognition.
  - **Eye Aspect Ratio (EAR) & Gaze Tracking:** Computes pupil displacement relative to eye corners to detect subtle sideways peeking without head rotation.
  - **Audio VAD + Whisper:** Records ambient speech and transcribes spoken words during exam sessions.
* **What is CCTV-Compatible:**
  - The architectural concept of an optional acoustic channel (room microphone logging speech energy or whisper transcriptions during silent exams) and reporting dashboard.
* **What is Webcam-Only:**
  - Eye gaze tracking (EAR) and mouth openness monitoring require high pixel density on the eye region ($>50\times30$ px per eye). On CCTV, eyes are unresolvable dots.
* **What Lacks Validation:**
  - High CPU overhead: Running MediaPipe FaceMesh + Whisper continuously introduces latency without hardware acceleration.
* **What Should NOT be Copied:**
  - Do NOT attempt eye-gaze tracking on surveillance camera footage.

---

### 2.4 Repository 4: `Zomma2/Cheating-Detection`
* **Source:** `https://github.com/Zomma2/Cheating-Detection`
* **Core Pipeline:**
  $$\text{Video Ingestion} \xrightarrow{\text{AWS API Gateway}} \text{Lambda Dispatch} \longrightarrow \begin{cases}
  \text{Pose Estimation on SageMaker} \\
  \text{YOLO Object Detection (Cell Phone)} \\
  \text{Voice Activity Detection (VAD)}
  \end{cases} \xrightarrow{\text{Sync EFS/S3}} \text{SES Incident Report}$$
* **What it Does Well:**
  - **Distributed Cloud Microservices:** Separates intensive GPU tasks (Pose Estimation on SageMaker) from lightweight orchestration (Lambdas).
  - **Dedicated Phone/Object Branch:** Uses an independent YOLO detector for cell phone discovery, cleanly isolating phone detection from human pose.
  - **Automated Evidence Bundling:** Generates signed PDF incident reports with timestamped frame crops and confidence metrics sent via email.
* **What is CCTV-Compatible:**
  - Decoupling object detection (phone) from behavior modeling.
  - Automated PDF report generation with forensic evidence snapshots.
* **What Lacks Validation:**
  - Network latency: Per-frame cloud roundtrips through API Gateway make real-time 30 fps proctoring cost-prohibitive. Edge inference is superior for CCTV.
* **What Should NOT be Copied:**
  - Do NOT offload individual frame inference to serverless cloud functions. Edge processing via TensorRT/ONNX on local GPU is essential for low-latency multi-stream CCTV.

---

## 3. Synthesis: Key Architectural Principles Extracted for V4

| Extracted Design Pattern | Source Repository | V4 Architectural Role | CCTV Applicability |
| :--- | :--- | :--- | :--- |
| **Two-Stage Pipeline** (Person Detect $\rightarrow$ Crop $\rightarrow$ Posture Classifier) | `phungthutrangsfl` | Core posture classification for `normal`, `head_down`, `turn_head` | **HIGH** (Eliminates full-frame scale degradation) |
| **Duration Timers & Cooldown State Machine** | `DYBInh2k5` | Core temporal engine (`TemporalBuffer` + debouncing) | **HIGH** (Prevents false alarms from fleeting glances) |
| **Head-Pose Branch (Euler Yaw/Pitch)** | `DYBInh2k5` & `cleardusk` | Secondary verification for lateral turns ($|\theta_y| > 35^\circ$) | **HIGH** (via Crop Keypoints / 3D Head Pose) |
| **Dedicated Phone Object Detector** | `Zomma2` | Independent COCO phone detector + ByteTrack association | **HIGH** (Preserves clean separation between objects & behavior) |
| **Personal Baseline Calibration** | Derived from `DYBInh2k5` limitation | Dynamic normalization of seating angle during initial 60s | **HIGH** (Accommodates side-row students) |
| **Eye Gaze / EAR Tracking** | `AarambhDevHub` | Webcam Close-Up Mode only | **WEBCAM ONLY** (Disabled in CCTV mode) |
| **Forensic Evidence Logging & PDF Reporting** | `Zomma2` / `AarambhDevHub` | Post-session audit trail with timestamped image crops | **HIGH** (Essential for human exam review) |
