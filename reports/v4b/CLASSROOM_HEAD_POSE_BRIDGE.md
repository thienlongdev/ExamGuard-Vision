# Classroom-Specific Head-Pose Bridge Feasibility

**Document ID**: `reports/v4b/CLASSROOM_HEAD_POSE_BRIDGE.md`  
**Phase**: V4B — Clean Person-Crop / Head-Pose Dataset Construction  
**Author**: Data Engineering & Architecture  
**Date**: 2026-10-02  
**Status**: FEASIBILITY APPROVED  

---

## 1. Executive Summary

A critical architectural question investigated in V4B is whether generic head-pose models (trained on cropped facial datasets like AFLW / AFLW2000-3D) can bridge effectively into classroom surveillance CCTV environments (such as SCBehavior High-Res and EduAction).

### Core Policy Rule
> [!CAUTION]
> **NO PSEUDO-ANGLE FABRICATION:**  
> SCBehavior classroom person crops **do NOT have ground-truth continuous Euler angles**.  
> We will **NOT** invent, pseudo-label, or synthesize numerical yaw/pitch degrees for classroom images.  
> Classroom annotations remain strictly discrete classification targets:
> - `TURN_HEAD_CLEAR`
> - `NORMAL_READ_WRITE`
> - `NORMAL_UPRIGHT`
> - `HEAD_DOWN_DEEP` / `HEAD_REST_SLEEP`

---

## 2. The Bridge Concept for Future V4C

Rather than attempting to train a single monolithic network to simultaneously predict bounding boxes, continuous head angles, and macro-behaviors, the recommended V4C architecture decouples the geometric feature representation from classroom decision logic:

```
[ Classroom CCTV Person Crop ]
              │
              ├───► [ Posture Classifier (ResNet+CBAM) ] ───► P(Normal), P(TurnHead), P(HeadDown)
              │
              └───► [ Generic Head-Pose Branch (AFLW-pretreated) ]
                                    │
                                    ├───► Predicted Continuous Yaw θ_yaw
                                    └───► Predicted Continuous Pitch θ_pitch
                                                    │
                                                    ▼
                                    [ Calibration & Multi-Cue Fusion ]
                                                    │
                                                    ▼
                                    Calibrated Classroom Event Verification
```

---

## 3. Feasibility Analysis

### 3.1 Resolution & Keypoint Feasibility
- **SCBehavior High-Res:** High-resolution QHD ($2560 \times 1440$) and 4K ($3840 \times 2160$) capture yields student face crops between $40 \times 40$ px and $120 \times 120$ px for students in the front/middle rows. At this scale, coarse facial geometry (eye line, nose profile, ear visibility) is clearly resolved.
- **EduAction Clips:** Crops are pre-isolated at $224 \times 224$ px, where student heads occupy approximately $60 \times 60$ to $90 \times 90$ px. This is well within the effective range of lightweight head-pose estimators (e.g. 6DRepNet / FSA-Net backbones).
- **Distal Students (<15x15 px head scale):** As documented in Stage 1.5, generic face-landmark and fine head-pose models collapse when heads are smaller than $15 \times 15$ px. The architecture must incorporate **capability gating** (`HEAD_POSE_ELIGIBLE` flag), falling back strictly to upper-body posture classification for distant seats.

### 3.2 Viewpoint Offset Calibration
Surveillance cameras view classrooms from elevated oblique ceiling mounts (typically $20^\circ \text{--} 35^\circ$ downward tilt).
- Consequently, an upright student gazing at the front blackboard presents an apparent downward pitch to the ceiling camera.
- **Bridge Solution:** The generic head-pose model outputs raw camera-relative angles $(\theta_{yaw}^{cam}, \theta_{pitch}^{cam})$. The classroom bridge calibrates these against the classroom desk plane:

$$\theta_{pitch}^{desk} = \theta_{pitch}^{cam} - \alpha_{mount}$$

Where $\alpha_{mount}$ is the camera depression angle estimated during camera calibration or learned as a per-camera bias.

### 3.3 Semantic Verification against `TURN_HEAD_CLEAR`
The discrete annotations from SCBehavior (`turn_head`) provide an empirical benchmark to test calibration:
- When a crop is labeled `TURN_HEAD_CLEAR` by human annotators, the continuous yaw $|\theta_{yaw}|$ predicted by the head-pose branch should exhibit a statistically significant shift compared to `NORMAL_READ_WRITE` and `NORMAL_UPRIGHT`.
- In V4C, this relationship will be evaluated via ROC/PR curves comparing predicted yaw magnitude against `TURN_HEAD_CLEAR` binary targets.

---

## 4. Bridge Implementation Roadmap (V4C Scope)

| Phase | Component | Input | Output / Role |
| :--- | :--- | :--- | :--- |
| **V4B (Current)** | Head-Pose Manifest | AFLW / AFLW2000-3D | Clean, uncorrupted continuous angle training data |
| **V4B (Current)** | Crop Manifest | SCBehavior / EduAction | Fine-grained observable posture classes with quality flags |
| **V4C (Future)** | Head-Pose Backbones | Face/Head Crops | Continuous $(\theta_{yaw}, \theta_{pitch})$ estimation |
| **V4C (Future)** | Cross-Modal Calibration | CCTV Crops + Head-Pose | Map angle outputs to behavior likelihoods without pseudo-labels |
| **V4D (Future)** | Multi-Cue Bayesian Fusion | Posture + Pose + Temporal | Robust violation scoring with capability gating |
