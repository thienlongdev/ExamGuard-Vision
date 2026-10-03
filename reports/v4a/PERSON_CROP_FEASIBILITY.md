# Person-Crop Dataset Feasibility Analysis
**Date:** 2026-10-02  
**Report ID:** V4A-CROP-FEAS-01  
**Target:** Assessing Feasibility of Secondary Per-Student Crop Classification (Normal, Head-Down, Turn-Head)  

---

## 1. Motivation: Why Per-Student Crops?

In single-stage full-frame object detection (e.g. standard YOLO on full $1920\times1080$ or $2560\times1440$ classroom views), the model must simultaneously perform:
1. Student localization across crowded rows (30–60 students per frame).
2. Complex fine-grained posture classification (`normal` vs `head_down` vs `turn_head`).

When students sit in distant back rows, their entire body spans only $40\times80$ pixels, and their head spans only $20\times20$ pixels. In full-frame training, these features are heavily downsampled through strided convolutions, losing subtle head pitch and yaw gradients.

A **two-stage architecture**—where a robust Person Detector/Tracker (Stage A) extracts stabilized $224\times224$ crops of each student, followed by a dedicated Crop Classifier (Stage B, e.g. YOLO-cls, MobileNetV4, or ResNet-CBAM)—is proven in open-source benchmarks (e.g. `phungthutrangsfl/cheating_behavior`) to substantially outperform single-stage detectors on subtle postures.

---

## 2. Dataset Feasibility Categorization

Each acquired and prospective dataset is classified according to its box generation capability:
- `DIRECT_BOXES`: Pre-existing student bounding boxes with behavior labels.
- `POSE_PERSON_BOX`: Person boxes derived from keypoint bounding envelopes.
- `ACTION_TUBE_BOX`: Spatio-temporal tube boxes linked frame-by-frame.
- `TRACK_ID_BOX`: Bounding boxes with consistent person/track IDs across time.
- `NO_BOX`: Unbounded full-frame video or classification-only without coordinates.
- `MANUAL_REQUIRED`: Requires manual annotation or re-labeling.

| Dataset Source | Feasibility Category | Native Crop Dimensions | Usability for Student Crops | Estimated Usable Crops |
| :--- | :--- | :--- | :--- | :--- |
| **SCBehavior High-Res** (`datasets/raw_v4/scbehavior_highres`) | **`DIRECT_BOXES`** | Varied (from $96\times96$ to $400\times380$ px; native 2.5K/4K) | **IMMEDIATE**<br>Pre-annotated COCO/YOLO boxes directly delineate student upper body and desk interaction. | **~8,083 total crops**<br>• `normal`: ~6,000 (read/write/lookup)<br>• `turn_head`: ~1,011<br>• `discuss`: ~242<br>• `stand`: ~66 |
| **EduAction** (`datasets/raw_v4/other_candidates/eduaction`) | **`PRE_CROPPED`** | Fixed $224\times224$ px | **IMMEDIATE**<br>Already centered on student crops across video frames. | **~35,000 video frames** across 350 clips<br>• `sleeping` (head_down): ~5,000 frames<br>• `writing` (normal hard neg): ~4,600 frames<br>• `lecture` (normal): ~5,250 frames<br>• `talking` (discuss): ~6,000 frames |
| **CStudentAct (HUST)** (*Prospective / Blocked*) | **`ACTION_TUBE_BOX` + `TRACK_ID_BOX`** | Full HD (1920x1080) source bounding boxes | **EXCELLENT (IF ACCESSED)**<br>Explicit `(frame_id, action_instance_id, x, y, w, h)` provides temporal student crops. | **~32,000 linked boxes**<br>• `sleeping`: 10,162 bboxes<br>• `standing`: 6,626 bboxes<br>• `using_phone`: 11,998 bboxes |
| **Smart Classroom** (`master-weixiao`) (*Prospective / Blocked*) | **`DIRECT_BOXES`** | 800x450 frames, student boxes ~60x80 to 120x150 px | **HIGH (IF ACCESSED)**<br>Over 250,000 pre-labeled student boxes across 4 camera angles. | **>200,000 crops** across writing, reading, listening, turning. |
| **AFLW2000-3D / AFLW_GT** (`datasets/raw_v4/head_pose_aflw2000`) | **`FACE_CROP`** | 120x120 to 450x450 px | **AUXILIARY ONLY**<br>Face/head crops only (no desks, no exam bodies). Cannot train full student body classifier. | **23,080 face crops** for head pose pretraining. |
| **Existing Clean SCB5** (`datasets/processed_v3_5`) | **`DIRECT_BOXES`** | 640x640 normalized boxes | **HIGH (FILTERED)**<br>Cleaned BlackBoard, Stand, Talk, and filtered TurnHead boxes. | **~18,000 student crops** (excluding quarantined BowHead). |

---

## 3. Physical Head & Crop Pixel Size Distribution

From the physical audit of `datasets/raw_v4/scbehavior_highres` (2560x1440 and 3840x2160 images):

| Camera Distance Zone | Bounding Box Width ($w$) | Bounding Box Height ($h$) | Head Region Sub-Crop | Crop Feasibility at $224\times224$ Target |
| :--- | :--- | :--- | :--- | :--- |
| **Front Row (Close)** | 250 – 420 px | 300 – 480 px | ~150 – 200 px | **OPTIMAL** (Downsampled slightly to 224; pristine facial and hand details preserved) |
| **Middle Rows (Medium)** | 140 – 250 px | 180 – 300 px | ~80 – 140 px | **EXCELLENT** (Direct match to $224\times224$ with minimal scaling artifacts) |
| **Back Rows (Distant)** | 90 – 140 px | 90 – 160 px | ~45 – 75 px | **USABLE** (Upsampled to 224; head orientation easily resolved due to 2.5K base resolution) |

Contrast this with 640x640 full-frame images from previous datasets:
- In 640x640 images, back row students were $20\times30$ px with heads of $10\times10$ px, where bicubic upsampling produces only an unrecognizable blur.
- High-resolution 2.5K/4K acquisition ensures that **even the smallest student crop retains sufficient pixel density** for ResNet / MobileNet feature extraction.

---

## 4. Potential Crop Taxonomy for Future V4B

When authorized in V4B, the secondary crop dataset can be compiled into a standard image classification directory structure:

```
datasets/processed_v4_crops/
    train/
        normal/       (reading, writing, upright listening)
        head_down/    (deep head down, forehead on desk, sleeping)
        turn_head/    (clear lateral yaw > 35°)
    val_same_domain/
    val_cross_camera/
```

**CRITICAL SAFEGUARD:**  
In accordance with Section 10 and 19 of the directive:  
**DO NOT build or export this training crop dataset during Phase V4A.**  
All code and tools are audited for feasibility; final extraction awaits explicit V4B authorization.
