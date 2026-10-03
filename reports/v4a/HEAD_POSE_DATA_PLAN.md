# Head-Pose Data Plan & Architecture Pipeline
**Date:** 2026-10-02  
**Report ID:** V4A-POSE-PLAN-01  
**Target:** Explicit Decoupling of Generic 3D Head Pose from Classroom Behavior Supervision  

---

## 1. Core Principle & Decoupling Strategy

A common failure mode in academic cheating detection projects is naively assigning "cheating" labels directly to face datasets (e.g. labeling an image of a person looking right as "cheating").  
In reality:
- Head pose is a **geometric property** $(\text{yaw } \theta_y, \text{pitch } \theta_p, \text{roll } \theta_r)$.
- Cheating is a **contextual and temporal behavioral assessment** conditioned on classroom environment, desk location, exam materials, and duration.

Therefore, V4 strictly separates head pose data into two distinct operational roles:

```
┌────────────────────────────────────────────────────────┐
│ ROLE 1: Generic Head-Pose Pretraining                  │
│ • Datasets: AFLW2000-3D, AFLW_GT, (300W-LP, BIWI)      │
│ • Ground Truth: Continuous Euler angles (yaw, pitch)   │
│ • Output: 3D orientation vector / angle predictions    │
└───────────────────────────┬────────────────────────────┘
                            │ (Frozen / Fine-Tuned Backbone)
                            ▼
┌────────────────────────────────────────────────────────┐
│ ROLE 2: Classroom Student Crop Adaptation              │
│ • Datasets: SCBehavior High-Res, EduAction crops       │
│ • Ground Truth: Observed classroom context             │
│ • Output: Calibrated student head/torso state          │
└───────────────────────────┬────────────────────────────┘
                            │ (Angle Evidence + Context)
                            ▼
┌────────────────────────────────────────────────────────┐
│ Multi-Cue Temporal Fusion (Rule Engine / Scorer)       │
│ • Evaluates: (Yaw > 35° & Duration > 1.5s) → turn_head │
│ • Evaluates: (Pitch > 40° & No pen) → head_down        │
└────────────────────────────────────────────────────────┘
```

---

## 2. Role 1: Generic Head-Pose Pretraining

### 2.1 Dataset Inventory & Status

| Dataset | Volume | Resolution | Annotation Format | Access Status in V4A |
| :--- | :--- | :--- | :--- | :--- |
| **AFLW2000-3D** | 2,000 face crops | ~200x200 to 450x450 | 68 3D landmarks + fitted yaw, pitch, roll angles (deg) | **ACQUIRED LOCALLY** (`datasets/raw_v4/head_pose_aflw2000/`) |
| **AFLW_GT** | 21,080 face crops | 120x120 crops | 21 landmarks + yaw ground truth | **ACQUIRED LOCALLY** (`datasets/raw_v4/head_pose_aflw2000/`) |
| **300W-LP** | 61,225 synthetic crops | 120x120 crops | 3DMM fitted coefficients (-90° to +90° full range) | Pretraining backbone weights accessible |
| **BIWI Kinect** | 15,000 frames (20 subjects) | 640x480 RGB+D | Depth sensor 3D head center & Euler angles | Server 403; optional secondary benchmark |

### 2.2 Mathematical Representation
The generic head-pose estimator predicts Euler angles:
$$\mathbf{y} = \begin{bmatrix} \theta_{yaw} \\ \theta_{pitch} \\ \theta_{roll} \end{bmatrix} \in [-90^\circ, +90^\circ]^3$$

Loss formulation (e.g. HopeNet / 6DRepNet / MobileNet-V3):
$$\mathcal{L}_{pose} = \alpha \mathcal{L}_{MSE}(\mathbf{\hat{y}}, \mathbf{y}) + \beta \mathcal{L}_{bce\_binned}(\mathbf{\hat{p}}_{bin}, \mathbf{p}_{bin})$$
where yaw and pitch are partitioned into continuous angles plus discrete bins (e.g. 66 bins of 3 degrees each).

---

## 3. Role 2: Classroom Student Crop Fine-Tuning & Adaptation

Generic face models trained on close-up selfie datasets (AFLW, 300W) suffer domain collapse when exposed to:
1. Surveillance cameras positioned 3 to 7 meters away.
2. Steep downward CCTV pitch angles (30°–60° camera inclination).
3. Back-of-head views where facial landmarks are completely occluded.

### 3.1 Classroom Adaptation Sources

1. **SCBehavior High-Resolution (`datasets/raw_v4/scbehavior_highres`):**
   - 400 images containing 8,083 localized student bounding boxes in 2.5K/4K resolution.
   - Extracts 1,011 `turn_head` student crops and 1,600 `read`/`write` student crops.
   - Provides realistic CCTV illumination, motion blur, and desk occlusions.
2. **EduAction (`datasets/raw_v4/other_candidates/eduaction`):**
   - 350 video clips at 224x224 person-crop resolution.
   - Provides dynamic temporal transitions from upright listening (`lecture`) to resting on desk (`sleeping`) and peer conversation (`talking`).

### 3.2 Dual-Cue Integration: Head vs. Torso Orientation

In real exam surveillance:
- If a student looks at a neighbor, **both head yaw and eye gaze** deviate.
- If the camera only captures the back of the student's head, facial landmarks are unavailable.
- Therefore, the classroom adaptation branch incorporates **Keypoint / Body Pose** (ear, shoulder, and nose vectors):
  $$\Delta \theta = \arctan2(y_{ear} - y_{nose}, x_{ear} - x_{nose})$$
  $$\theta_{torso} = \arctan2(y_{shoulder\_R} - y_{shoulder\_L}, x_{shoulder\_R} - x_{shoulder\_L})$$
  $$\text{Relative Head Turn} = |\theta_{head} - \theta_{torso}|$$

If $|\theta_{head} - \theta_{torso}| > 35^\circ$, the student has turned their neck relative to their desk alignment, providing an invariant signal regardless of camera mounting angle!

---

## 4. Operational Invariant: No Label Merging

Under no circumstances will AFLW2000 or generic head-pose data be mixed into the canonical YOLO training set `data.yaml`.
- The canonical YOLO behavior model trains ONLY on verified classroom surveillance scenes.
- The head-pose branch exists as an auxiliary feature extractor (either as a lightweight secondary ONNX model or multi-task head).
- Evidence from both streams is fused in the rule engine via probabilistic Bayesian combination:
  $$P(\text{TurnHead} \mid \text{BBox}, \text{Pose}, t) = \sigma\left( w_1 \cdot s_{yolo} + w_2 \cdot \frac{|\theta_{yaw}| - 25°}{20°} + w_3 \cdot \text{Duration}(t) \right)$$
