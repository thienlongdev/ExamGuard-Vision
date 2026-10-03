# Sensor Capability Matrix by Camera Operational Mode
**Date:** 2026-10-02  
**Report ID:** V4A-CAM-MATRIX-01  
**Target:** Explicit Resolution & Pixel Density Gating Across Operational Camera Regimes  

---

## 1. Operational Camera Regimes Defined

Classroom proctoring systems operate across three vastly different optical regimes:

1. **`WEBCAM_CLOSE_RANGE`**:
   - Distance: 0.4 m – 1.0 m from student face.
   - Typical Hardware: Laptop internal webcam or external USB camera.
   - Head Bounding Box Size: $250\times250$ to $600\times600$ pixels.
   - Eye Region Size: $60\times30$ to $120\times60$ pixels per eye.
   - Context: Remote online exams (1 student per camera).

2. **`CCTV_MEDIUM_RANGE`**:
   - Distance: 2.0 m – 5.0 m from students.
   - Typical Hardware: Front-of-room ceiling camera, side wall-mounted camera (front 3–4 desk rows).
   - Head Bounding Box Size: $60\times60$ to $150\times150$ pixels.
   - Eye Region Size: $8\times4$ to $18\times8$ pixels per eye.
   - Context: Classroom surveillance covering 8–20 students.

3. **`CCTV_LONG_RANGE`**:
   - Distance: 5.0 m – 15.0 m from students.
   - Typical Hardware: Rear high-ceiling dome camera, broad lecture hall wide-angle camera (rear desk rows).
   - Head Bounding Box Size: $20\times20$ to $50\times50$ pixels.
   - Eye Region Size: $\le 4\times2$ pixels (unresolvable sub-pixel features).
   - Context: Full auditorium / examination hall covering 40–100 students.

---

## 2. Module Capability & Feasibility Matrix

| Module / Sensor Channel | `WEBCAM_CLOSE_RANGE` (0.4m–1.0m) | `CCTV_MEDIUM_RANGE` (2.0m–5.0m) | `CCTV_LONG_RANGE` (5.0m–15.0m) | Critical Failure Mode if Misapplied |
| :--- | :--- | :--- | :--- | :--- |
| **Person Detection (Full Body / Upper Torso)** | **USABLE** (Single student upper torso) | **OPTIMAL** (Standard multi-person bounding boxes) | **OPTIMAL** (Requires multi-scale / high-res $1280\times1280$) | Missing small students in rear rows if inference resolution is too low ($640\times640$). |
| **Macro Behavior Detection (`stand`, `discuss`)** | **CONDITIONAL** (Student standing may exit webcam FOV) | **OPTIMAL** (Clear view of student rising from chair & desk pairs) | **OPTIMAL** (Full room overview captures room movement) | Occlusion from adjacent rows in flat angles. |
| **3D Head Pose (Yaw / Pitch)** | **OPTIMAL** (Dense facial landmarks or PnP geometry) | **USABLE** (Silhouette & ear/nose keypoints or CNN crop) | **CONDITIONAL** (Coarse angle bins only: forward vs side vs down) | Fitting 468-point 3D FaceMesh on distant CCTV fails completely. |
| **Dense Facial Landmarks (468 pts / 68 pts)** | **OPTIMAL** (High precision MediaPipe FaceMesh) | **CONDITIONAL** (Works only on front 2 rows under bright lighting) | **DISABLE** (Pixel density $<30\text{ px}$ prevents landmark convergence) | Severe landmark jitter and false face detections on clothes/backpacks. |
| **Eye Gaze / Iris Tracking (EAR / Gaze Vector)** | **OPTIMAL** (Accurate pupil displacement tracking) | **DISABLE / UNUSABLE** (Eyes lack sufficient iris-to-sclera pixels) | **DISABLE / UNUSABLE** (Sub-pixel eyes; total failure) | Generating phantom gaze shifts from image noise/compression artifacts. |
| **Mouth Movement / Lip Articulation** | **OPTIMAL** (Lip distance & aspect ratio for talking) | **CONDITIONAL** (Resolvable only under high-res zoom) | **DISABLE** (Mouth is a 2-pixel line) | False talking alerts from chewing or breathing. |
| **Contraband / Phone Detection** | **OPTIMAL** (Phones held near webcam screen/lap) | **OPTIMAL** (Phones on desk or manipulated in hands) | **CONDITIONAL** (Small black rectangular phones blend into desks) | High false positive rate on calculators, pencil cases, or black notebooks. |
| **Body Pose Keypoints (COCO 17 pts: shoulders, elbows, wrists)** | **CONDITIONAL** (Lower body occluded by laptop) | **OPTIMAL** (Torso, shoulders, arms, hands on desk clearly visible) | **USABLE** (Top-down pose estimators detect shoulder line & neck angle) | Wrist occlusion behind front-seat backrests. |
| **Per-Student Crop Classifier (224x224)** | **OPTIMAL** (Pristine crop clarity) | **OPTIMAL** (Rescales $100\times120$ to $224\times224$ cleanly) | **USABLE** (Rescaling $35\times50$ preserves macro posture: slouch vs upright) | Excessive blur if camera sensor lacks optical sharpness. |
| **Audio Channel (Room Mic / USB Mic)** | **OPTIMAL** (Captures single student speaking / whispering) | **USABLE** (Captures overall classroom noise & whispered collaboration) | **USABLE** (Room-wide acoustic energy thresholding) | Ambient air conditioning or projector fan noise triggering false sound alerts. |
| **Temporal Buffers & Rule Engine** | **OPTIMAL** (Per-student sliding window state machine) | **OPTIMAL** (Tracks 15–30 students simultaneously with ByteTrack) | **OPTIMAL** (Tracks 50+ students with temporal voting) | ID-switches in crowded aisles corrupting temporal duration counters. |

---

## 3. Dynamic Configuration Policy Engine

To prevent catastrophic system failure, the V4 pipeline includes an automatic **Camera Mode Configurator** (`configs/camera.yaml`):

```yaml
# Dynamic Runtime Feature Gating by Camera Mode
camera_modes:
  webcam_online_exam:
    person_detector: true
    face_mesh: true
    eye_gaze_tracker: true
    mouth_movement: true
    crop_classifier: true
    head_pose_3d: true
    body_keypoints: false
    macro_stand_discuss: false
    phone_detector: true
    audio_monitor: true

  cctv_classroom_surveillance:
    person_detector: true
    face_mesh: false        # HARD DISABLED: Prevents landmark collapse
    eye_gaze_tracker: false # HARD DISABLED: Sub-pixel eye resolution
    mouth_movement: false   # HARD DISABLED: Unresolvable on CCTV
    crop_classifier: true   # ENABLED: 224x224 student posture classifier
    head_pose_3d: true      # ENABLED: CNN-based crop yaw/pitch estimator
    body_keypoints: true    # ENABLED: Shoulder/neck orientation
    macro_stand_discuss: true # ENABLED: Full-frame stand/discuss model
    phone_detector: true    # ENABLED: Object detector + ByteTrack
    audio_monitor: true     # OPTIONAL: Room acoustic energy threshold
```

**Golden Rule of V4:**  
Never execute webcam-specific facial micro-feature modules (Eye Gaze, Iris Tracking, 468 FaceMesh) on classroom CCTV surveillance feeds. High-level body posture, head crop classification, neck-to-shoulder keypoints, and temporal duration are the only reliable CCTV signals.
