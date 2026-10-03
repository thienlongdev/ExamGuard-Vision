# CCTV CAMERA CAPABILITY MAP & RESOLUTION GATING SPECIFICATION

## 1. Overview & Scientific Boundary
This document defines the physical resolution boundaries under which specialized neural perception models (general detection, posture classification, facial yaw estimation, and phone association) operate reliably.

> [!IMPORTANT]
> **No Fake Accuracy**: In accordance with system governance invariants, these capability classifications reflect **empirical visibility coverage and physical scale gating**, NOT synthetic behavioral accuracy claims. No universal accuracy is claimed on unseen CCTV domains without ground-truth annotations.

---

## 2. Person Scale & Posture Capability Map

The posture model (`MobileNetV3-Small`, input resolution $224 \times 224$, 4-class ontology: `NORMAL_UPRIGHT`, `NORMAL_READ_WRITE`, `HEAD_REST_SLEEP`, `TURN_HEAD_CLEAR`) operates on tight person bounding box crops.

### Person Height Empirical Scale Buckets
| Person Bounding Box Height | Posture Capability | Reliability Category | Tracking Stability | Fusion Status | Operational Guidance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$< 60\text{ px}$** | UNAVAILABLE | UNRELIABLE | DEGRADED | EXCLUDED (Weight 0.0) | Severe downsampling loss; posture classification disabled to prevent spurious inference. |
| **$60 - 89\text{ px}$** | LIMITED | LOW_FIDELITY | ACCEPTABLE | DISCOUNTED (Weight 0.4) | Severe blur upon upscale to 224px. Useful only as secondary confirming cue. |
| **$90 - 119\text{ px}$** | LIMITED | MODERATE | STABLE | STANDARD (Weight 0.7) | Coarse posture discernible (torso angle, head on desk). Deep head-down cues may be muffled. |
| **$120 - 179\text{ px}$** | FULL | HIGH | ROBUST | OPTIMAL (Weight 1.0) | Primary posture operational zone. Upright vs head-rest vs turning clearly separated. |
| **$180 - 299\text{ px}$** | FULL | HIGH | EXCELLENT | OPTIMAL (Weight 1.0) | High fidelity. Posture features fully resolved; minimal interpolation artifacts. |
| **$\ge 300\text{ px}$** | FULL | EXCELLENT | EXCELLENT | OPTIMAL (Weight 1.0) | Close-range or high-resolution zoom. Posture classification at peak fidelity. |

### Minimum Usable Person Pixel Height
- **Operational Floor**: **$60\text{ px}$** (below which posture inference is bypassed).
- **Full Fidelity Threshold**: **$120\text{ px}$** (qualifies camera for `POSTURE_CAPABILITY: FULL`).

---

## 3. Head Scale & Facial Yaw Capability Map

The head-pose model (`HopeNetYaw`, ResNet-50 backbone, 66-bin continuous expectation, valid range $[-99.0^\circ, +99.0^\circ)$) estimates horizontal yaw angle from head crops.

### Head Crop Empirical Size Buckets
| Head Crop Bounding Box Size | Head-Pose Capability | Physical Suitability | Yaw Estimation State | Operational Guidance |
| :--- | :--- | :--- | :--- | :--- |
| **$< 15 \times 15\text{ px}$** | UNAVAILABLE | UNRESOLVED | DISABLED | Facial features completely sub-pixel; head-pose inference skipped entirely. |
| **$15 \times 15 - 24 \times 24\text{ px}$** | LIMITED | POOR | SUPPRESSED | Coarse blob only; facial symmetry undetectable. Yaw variance high; gated out from event opening. |
| **$25 \times 25 - 39 \times 39\text{ px}$** | LIMITED | MARGINAL | BOUNDED | Coarse yaw usable with conservative angular threshold ($\ge 35^\circ$ lateral offset required). |
| **$40 \times 40 - 59 \times 59\text{ px}$** | FULL | GOOD | ACTIVE | Clear profile and facial landmarks. Primary lateral head turning operational zone. |
| **$\ge 60 \times 60\text{ px}$** | FULL | OPTIMAL | ROBUST | High-resolution facial pose. Stable continuous yaw expectation across full $\pm 90^\circ$ span. |

### Minimum Usable Head Crop Size
- **Operational Floor**: **$25 \times 25\text{ px}$** (head-pose enabled with conservative confidence gating).
- **Full Fidelity Threshold**: **$40 \times 40\text{ px}$** (qualifies camera for `HEADPOSE_CAPABILITY: FULL`).

---

## 4. Camera Ingestion Capability Classification Rules

For any camera profile or calibration window, the runtime capability summary is computed strictly from physical visibility distributions:

```
IF (fraction of visible tracks with height >= 120 px) >= 0.70:
    POSTURE_CAPABILITY = FULL
ELSE IF (fraction of visible tracks with height >= 60 px) >= 0.50:
    POSTURE_CAPABILITY = LIMITED
ELSE:
    POSTURE_CAPABILITY = UNAVAILABLE

IF (fraction of visible tracks with head crop >= 40x40 px) >= 0.60:
    HEADPOSE_CAPABILITY = FULL
ELSE IF (fraction of visible tracks with head crop >= 25x25 px) >= 0.40:
    HEADPOSE_CAPABILITY = LIMITED
ELSE:
    HEADPOSE_CAPABILITY = UNAVAILABLE

IF (fraction of tracks with track lifespan >= 3.0 sec) >= 0.80:
    TRACKING_CAPABILITY = SUPPORTED
ELSE:
    TRACKING_CAPABILITY = DEGRADED
```

---

## 5. Phone Visibility Boundary
- Primary detector: `yolo26m.pt` (COCO class 67: cell phone, runtime resolution 640px).
- Practical physical visibility threshold: **$20 \times 20\text{ px}$** bounding box in source frame.
- Phones resting flat on exam desks are typically $25 \times 45\text{ px}$ at 1080p, $35 \times 60\text{ px}$ at 1440p, and $55 \times 90\text{ px}$ at 4K.
- Lap-level phones, hands beneath desks, and rear-angled camera views obscure phones below detection limits.
- Ambiguous ownership (`PHONE_ASSOCIATION_AMBIGUOUS`) suppresses event opening to prevent false associations.
