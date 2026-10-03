# V4D Viewpoint Capability & Scale Gating Matrix

**Document ID**: `reports/v4d/V4D_CAPABILITY_MATRIX.md`  
**Phase**: Phase V4D — Temporal & Multi-Cue Fusion Layer  
**Date**: 2026-10-03  
**Status**: AUTHORITATIVE CAPABILITY SPECIFICATION  

---

## 1. Camera Viewpoint Capability Profiles

Surveillance camera mount angles dramatically alter cue observability. V4D enforces profile-based capability gating to prevent firing unsupported modules on unresolvable perspectives:

| Camera Viewpoint Profile | Posture Supported | Head-Pose Supported | Phone Association Supported | Face Landmarks Supported | Minimum BBox Height ($H$) | Minimum Head Crop Dimension |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HIGH_ANGLE_CCTV** | **YES** | **NO** (Steep crown angle masks face) | **YES** | **NO** | $\ge 120\text{ px}$ | N/A |
| **FRONT_OBLIQUE_CCTV** | **YES** | **YES** (Profile / frontal visible) | **YES** | **NO** | $\ge 120\text{ px}$ | $\ge 25 \times 25\text{ px}$ |
| **FRONTAL_DESK_LEVEL** | **YES** | **YES** (Full frontal geometry) | **YES** | **YES** | $\ge 100\text{ px}$ | $\ge 20 \times 20\text{ px}$ |
| **LOW_RESOLUTION** | **YES** | **NO** (Insufficient facial resolution) | **NO** | **NO** | $\ge 80\text{ px}$ | N/A |
| **REAR_OBLIQUE** | **YES** (Body turn only) | **NO** (Occipital / rear head only) | **YES** | **NO** | $\ge 120\text{ px}$ | N/A |

---

## 2. Scale & Resolution Gating Policy

### 2.1. Posture Classifier Eligibility
- **Minimum Person Crop Height**: $H \ge 120\text{ px}$
- **Minimum Person Crop Width**: $W \ge 40\text{ px}$
- Crops below these boundaries are marked `POSTURE_CLASSIFIER_ELIGIBLE = FALSE` and yield `ObservationStatus.UNAVAILABLE`.
- **Resolution Dispatch**:
  - Primary Model: **224x224** (`models/trained/v4_posture_best.pt`)
  - High-Resolution Candidate: **320x320** (`runs/v4c/C1_mobilenet_v3_small_tight_person_crop_320/best_model.pt`)
  - Governance Rule: `use_320_fallback` defaults to `False`. 320 is exposed via capability interface (`HIGH_RES_FALLBACK_AVAILABLE = True`) but requires evidence-backed camera configuration to activate.

### 2.2. Head-Pose Eligibility & Dispatch
- **Prerequisite**: BBox height $H \ge 120\text{ px}$ AND Head crop $\ge 25 \times 25\text{ px}$ AND frontal/profile face detected with confidence $\ge 0.25$.
- **Primary Domain**: $[-99.0^\circ, +99.0^\circ)$ dispatched to **`HopeNet-Yaw`** (`models/trained/v4_headpose_yaw_best.pt`).
- **Circular Fallback Domain**: Outside $[-99.0^\circ, +99.0^\circ)$ dispatched to **`ResNet18-Yaw-Circular`** (`runs/v4c/headpose_resnet18_yaw/best_model.pt`) with discounted reliability ($0.50$).
- **Strict Missing Cue Rule**: If face is not physically resolvable (rear view, severe occlusion, tiny face), the system emits **`HEAD_POSE_UNAVAILABLE`** (`yaw_deg = None`). It **NEVER** fabricates a default angle such as $\text{yaw} = 0^\circ$.

---

## 3. Comprehensive Event Capability Matrix

| Event / Behavioral Cue | Implemented in V4D | Physically Validated on Video Data | Functionally Tested Only | Scientific Status & Limitation |
| :--- | :---: | :---: | :---: | :--- |
| **`SUSTAINED_HEAD_REST`** | **YES** | **YES** (20 physical clips, 242 frames) | - | High physical accuracy (85%-100% clip recall, 0 false alarms on writing). 100% EduAction source. |
| **`SUSTAINED_LATERAL_HEAD_ORIENTATION`** | **YES** | **NO** | **YES** | Zero positive turn-head temporal holdout clips. Classroom yaw has weak standalone separation (+1.27°, 87.6% overlap). |
| **`PHONE_ASSOCIATED`** | **YES** | **NO** | **YES** | Spatial association and ambiguity veto validated functionally; no temporal video GT for desk phone persistence. |
| **`DISCUSSION_CANDIDATE`** | **YES** | **NO** | **YES** | Macro discuss detector integrated functionally; no temporal pair-level conversation GT. |
| **`STANDING`** | **YES** | **NO** | **YES** | Macro stand detector integrated functionally; context-dependent risk. |
| **Missing Cue Robustness** | **YES** | **YES** (Synthetic + Holdout) | - | Proven graceful degradation across intermittent dropouts and absent sensors. |
| **FPS Invariance** | **YES** | **YES** (15, 25, 30 FPS) | - | Timestamp-first sliding window guarantees identical physical activation delays. |
