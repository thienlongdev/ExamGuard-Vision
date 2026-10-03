# Target V4 Hybrid Architecture Specification
**Date:** 2026-10-02  
**Report ID:** V4A-HYBRID-ARCH-01  
**Architecture Paradigm:** Multi-Stage Spatial-Temporal Hybrid with Decoupled Behavioral, Geometric, and Object Cues  

---

## 1. System Architecture Diagram

```
                               ┌──────────────────────┐
                               │     VideoSource      │
                               │  (RTSP / Video / CAM)│
                               └──────────┬───────────┘
                                          │
                                          ▼
                               ┌──────────────────────┐
                               │   Person Detector    │
                               │  (YOLOv8x / COCO)    │
                               └──────────┬───────────┘
                                          │
                                          ▼
                               ┌──────────────────────┐
                               │      ByteTrack       │
                               │ (Persistent TrackIDs)│
                               └──────────┬───────────┘
                                          │
             ┌────────────────────────────┼───────────────────────────┐
             │                            │                           │
             ▼                            ▼                           ▼
 ┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
 │  Per-Student Crop    │    │ Full-Frame Behavior  │    │ Phone/Object Detector│
 │  (224x224 Upper Body)│    │ (Stage 1.5 Anchor)   │    │ (COCO Cellphone Det) │
 └──────────┬───────────┘    └──────────┬───────────┘    └──────────┬───────────┘
            │                           │                           │
     ┌──────┴──────┐                    │ (Full Scene BBoxes)       │ (Phone BBoxes)
     │             │                    │                           │
     ▼             ▼                    ▼                           ▼
┌──────────┐ ┌──────────┐          ┌─────────┐                 ┌─────────┐
│ Posture  │ │Head-Pose │          │ Stand & │                 │ Spatial │
│Classifier│ │  Branch  │          │ Discuss │                 │ Student │
│(3-Class) │ │(Yaw/Pitch│          │ Evidence│                 │ Assoc.  │
└────┬─────┘ └────┬─────┘          └────┬────┘                 └────┬────┘
     │            │                     │                           │
     └──────┬─────┴─────────────────────┴───────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Multi-Cue Bayesian Fusion                    │
│      Fuses Posture + Head Angle + Full-Frame Context + Phone     │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                   TemporalBuffer (Sliding Window)                │
│    Maintains 150-frame history per track ID with personal median  │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Deterministic Rule Scorer                      │
│      Duration checks (>=1.5s), debounce, and risk level bounds   │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FastAPI / Evidence Vault                     │
│      Emits REST events, stores JPEG crops, updates dashboard     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Specifications

### Module 1: Video Ingestion (`VideoSource`)
* **Role:** Unified frame capture abstraction across RTSP CCTV, recorded MP4 exam files, and USB webcams.
* **Input:** Raw stream or file.
* **Output:** Timestamped BGR frame `(H, W, 3)` + Frame ID + Camera ID.
* **Inference Cost:** ~0 ms (I/O bound).
* **Fallback Behavior:** Reconnection loop with exponential backoff on RTSP stream drop.

---

### Module 2: Primary Person Detector (`PersonDetector`)
* **Role:** High-recall student localization in crowded classroom scenes.
* **Model:** YOLOv8x / YOLO11x pretrained on COCO class 0 (`person`) or dedicated classroom crowd model.
* **Input:** Full-frame RGB image (downscaled to $1280\times1280$ for high-resolution CCTV).
* **Output:** Person bounding boxes $[x_1, y_1, x_2, y_2]$ with localization confidence $s_{det} > 0.40$.
* **Training Data:** COCO + CrowdHuman + AIRC-SMARTCLASS person annotations.
* **Inference Cost:** ~18–25 ms (TensorRT FP16 on RTX 3080/4090).
* **Fallback Behavior:** If small students in rear rows missed, multi-scale sliding window or SAHI (Slicing Aided Hyper Inference) can be activated for 4K feeds.

---

### Module 3: Persistent Multi-Object Tracker (`ByteTrack`)
* **Role:** Maintains persistent student IDs across frames through desk occlusions, head movement, and momentary drops.
* **Input:** Detected person bounding boxes across consecutive frames.
* **Output:** Stable `track_id` per student, velocity vector, and bounding box history.
* **Inference Cost:** < 2 ms (CPU).
* **Fallback Behavior:** Re-ID association buffer retains track states for up to 30 frames of occlusion before assigning a new ID.

---

### Module 4: Per-Student Crop Normalization (`StudentCrop`)
* **Role:** Extracts normalized upper-body student crops centered on head, shoulders, and desk workspace.
* **Input:** Full frame + person bounding box $[x_1, y_1, x_2, y_2]$.
* **Output:** Standardized $224\times224\times3$ image tensor.
* **Expansion Logic:** Bounding box expanded by 10% vertically and horizontally to prevent clipping of tilted heads or hands resting on desk.

---

### Module 5: Dedicated Posture Classifier (`PostureClassifier`)
* **Role:** Fine-grained posture classification on normalized student crops.
* **Taxonomy:**
  - `0: normal` (upright listening, normal reading, normal writing)
  - `1: head_down` (forehead on desk, resting on crossed arms, deep downward slouch $>45^\circ$)
  - `2: turn_head` (lateral head rotation $>35^\circ$ gazing away from own desk)
* **Model Architecture:** MobileNetV4-Small, ConvNeXt-Femto, or YOLOv11-cls with CBAM attention module.
* **Input:** $224\times224\times3$ student crop tensor (batched across all active classroom tracks, e.g. batch size 32).
* **Output:** Softmax probabilities $[p_{normal}, p_{head\_down}, p_{turn\_head}]$.
* **Training Data:** SCBehavior High-Res crops (read/write hard negatives + turn_head positives) + EduAction crops (sleeping positives + writing negatives).
* **Inference Cost:** ~4–6 ms for batch of 32 crops on GPU.
* **Fallback Behavior:** If crop confidence is below 0.50, posture reverts to `normal` (fail-safe against false accusations).

---

### Module 6: Auxiliary Head-Pose Branch (`HeadPoseBranch`)
* **Role:** Continuous geometric confirmation of head orientation $(\theta_{yaw}, \theta_{pitch})$.
* **Model:** Lightweight 3D Head Pose regressor (e.g. HopeNet-lite or 6DRepNet).
* **Input:** Upper third of student crop (head region $128\times128$).
* **Output:** Continuous Euler angles in degrees $(\theta_{yaw}, \theta_{pitch}, \theta_{roll})$.
* **Training Data:** AFLW2000-3D, AFLW_GT, 300W-LP pretraining.
* **Inference Cost:** ~3 ms for batch of 32 on GPU.
* **Fallback Behavior:** If face landmarks or head features unresolvable due to extreme distance, module outputs `None`, and fusion relies solely on `PostureClassifier`.

---

### Module 7: Full-Frame Behavior Anchor (`FullFrameDetector`)
* **Role:** Preserves established macro-behavior detection for `stand` (rising from desk, walking) and `discuss` (pairwise student clusters).
* **Model:** Canonical Stage 1.5 model (`stage1_5_best.pt`), operated at high confidence ($\tau > 0.55$).
* **Input:** Full frame $640\times640$ or $1280\times1280$.
* **Output:** Full-frame bounding boxes for `stand` and `discuss`.
* **Inference Cost:** ~12 ms.
* **Fallback Behavior:** If full-frame model yields no detections, default seated `normal` assumed.

---

### Module 8: Dedicated Phone/Object Detector (`PhoneDetector`)
* **Role:** Discovers physical smartphone devices and unauthorized electronic gadgets.
* **Model:** YOLOv8m COCO class 67 (`cell phone`) + custom exam contraband weights.
* **Input:** Full frame or student lap crops.
* **Output:** Phone bounding boxes $[x_{p1}, y_{p1}, x_{p2}, y_{p2}]$.
* **Inference Cost:** ~10 ms (or run every 3rd frame to conserve GPU).
* **Association:** Checked via IoU and distance against student bounding boxes (`ObjectAssociator`).

---

### Module 9: Multi-Cue Bayesian Fusion Engine (`CueFusion`)
* **Role:** Fuses posture probability, head yaw angle, full-frame macro classes, and phone association into a unified per-student frame state.
* **Logic:**
  $$\text{Score}_{turn\_head} = w_1 \cdot p_{cls}(turn\_head) + w_2 \cdot \max\left(0, \frac{|\theta_{yaw}| - 25°}{20°}\right)$$
  $$\text{Score}_{head\_down} = w_1 \cdot p_{cls}(head\_down) + w_2 \cdot \max\left(0, \frac{\theta_{pitch} - 35°}{20°}\right)$$
  $$\text{Score}_{discuss} = \max\left(s_{yolo}(discuss), \text{PairwiseConvergence}(track_i, track_j)\right)$$
  $$\text{Score}_{stand} = s_{yolo}(stand)$$
  $$\text{Score}_{phone} = s_{obj}(phone) \cdot \mathbb{I}(\text{AssociatedWith}(track_i))$$

---

### Module 10: Temporal Buffer & State Machine (`TemporalBuffer`)
* **Role:** Rolling window history (150 frames, 5.0 seconds at 30 fps) per student track ID.
* **Calculates:**
  - Behavior percentage over rolling window ($P_c = \frac{N_c}{W}$).
  - Continuous behavior duration $D_c(t)$.
  - Personal baseline posture adaptation (running median subtraction).

---

### Module 11: Deterministic Rule Scorer (`RuleScorer`)
* **Role:** Maps temporal metrics to discrete risk levels (`LOW`, `MEDIUM`, `HIGH`) and triggers human review events.
* **Rules:**
  - `PHONE_DETECTED`: Instantaneous alert (Risk: **HIGH**).
  - `DEEP_HEAD_DOWN_PERSISTENT`: Duration $\ge 3.0\text{ s}$ (Risk: **MEDIUM**).
  - `TURN_HEAD_PERSISTENT`: Duration $\ge 1.5\text{ s}$ (Risk: **MEDIUM** / **HIGH** if repeated $>3$ times).
  - `ACTIVE_DISCUSSION`: Pairwise convergence for $\ge 2.0\text{ s}$ (Risk: **MEDIUM**).
  - `STANDING_UP`: Duration $\ge 1.0\text{ s}$ (Risk: **LOW** / **MEDIUM** depending on exam phase).
* **Debounce & Cooldown:** Enforces 5.0s cooldown per track ID to prevent alert spam.

---

### Module 12: API & Evidence Persistence (`FastAPI / Storage`)
* **Role:** Broadcasts real-time events over WebSockets/REST, caches JPEG forensic crops with bounding box overlays, and enables proctor human review.
* **Verified Endpoints:** `/api/v1/health`, `/api/v1/dashboard/summary`, `/api/v1/events`, `/api/v1/events/{id}/review`.

---

## 3. Computational Budget & Throughput Target

On a single standard workstation GPU (NVIDIA RTX 3080 / RTX 4070 / T4 Cloud):

| Pipeline Stage | Model / Component | Execution Frequency | Latency (ms) |
| :--- | :--- | :--- | :--- |
| **Ingestion** | OpenCV VideoSource | Every frame (30 fps) | 1.0 ms |
| **Full Detection** | YOLOv8 Person + Macro Behavior | Every frame | 14.0 ms |
| **Tracking** | ByteTrack | Every frame | 1.5 ms |
| **Crop Classifier** | MobileNetV4-cls (Batch 32) | Every 2nd frame | 5.0 ms |
| **Head-Pose Branch** | Lite Pose (Batch 32) | Every 2nd frame | 3.5 ms |
| **Phone Detector** | COCO Phone | Every 3rd frame | 8.0 ms (avg 2.7 ms/frame) |
| **Temporal & Rules** | CPU Buffer & Scorer | Every frame | 0.8 ms |
| **Total per Frame** | **Full Hybrid Pipeline** | **Real-Time** | **~26.5 ms (~37.7 FPS)** |

**Target Verified:** Real-time throughput $\ge 30\text{ FPS}$ on Full HD video streams.
