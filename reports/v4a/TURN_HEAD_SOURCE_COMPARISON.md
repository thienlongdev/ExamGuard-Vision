# Strict Turn-Head Semantic Audit & Cross-Source Comparison
**Date:** 2026-10-02  
**Report ID:** V4A-TH-AUDIT-01  
**Target:** Distinguishing Actionable Lateral Head Yaw from Subtle Gaze Shifts, Discussion, and Distance Artifacts  

---

## 1. Executive Problem Statement

In Stage 1 and Stage 1.5, unseen V3 validation recall for `turn_head` was **0.1969** (19.69%).  
Physical inspection revealed key failure modes:
1. **Distance degradation:** In rear and high-angle CCTV feeds, distant students occupy only $20 \times 20$ to $40 \times 40$ pixels. Subtle head orientation cannot be resolved by standard convolutional feature pyramids.
2. **Semantic conflation:** In previous datasets, minor natural gaze shifts (10°–20° off-center) were labeled inconsistently—sometimes as `normal`, sometimes as `TurnHead`.
3. **Mutual Discussion vs. Turning Head:** When two students face each other during group work, some frames labeled them as `TurnHead`, while others labeled them as `Talk`/`Discuss`, creating contradictory loss gradients.

**Mandatory V4 Principle:**  
Supervision for `turn_head` must be restricted to clear, observable physical yaw rotation, explicitly separated from subtle glances, torso reorientation, and multi-student discussion.

---

## 2. Six-Tier Semantic Classification for Lateral Orientation

| Category | Posture & Observable Traits | Head Yaw Range | Torso Yaw Range | V4 Canonical Behavioral Target |
| :--- | :--- | :--- | :--- | :--- |
| **A. SUBTLE_GAZE_SHIFT** | Student's eyes or head angle shifts slightly off-center (e.g. checking clock, looking at side of desk, glancing at adjacent row). Torso fully forward. | 15° – 25° yaw | < 15° | **`UNDECIDED` / `normal`**<br>(Do NOT supervise as turn_head; high risk of false alarms) |
| **B. CLEAR_HEAD_YAW** | Unambiguous lateral rotation of head. Profile or three-quarter view of face is clearly visible. Nose vector points sharply left or right (>35°). Torso remains largely oriented toward desk. | 35° – 75° yaw | < 25° | **`turn_head` (PRIMARY SUPERVISION)**<br>Ideal target for exam surveillance. |
| **C. HEAD_AND_TORSO_TURN** | Head rotates laterally accompanied by active shoulder/torso twist towards aisle or neighbor's workspace. | 45° – 90° yaw | 30° – 60° | **`turn_head` (SECONDARY SUPERVISION)**<br>Strong physical indicator of looking at neighbor's paper. |
| **D. TURNING_AROUND** | Extreme body rotation (>90°) where student twists back to face the desk behind them. Back of head or rear ear visible. | 90° – 160° yaw | > 60° | **`turn_head` (HIGH SEVERITY)**<br>Clear misconduct indicator. |
| **E. DISCUSSION_ORIENTATION** | Two or more students turn towards each other simultaneously, exhibiting reciprocal posture, mouth movement, and mutual attention. | 30° – 75° (mutual) | Convergent | **`discuss` (NOT turn_head)**<br>Routed to interaction module. |
| **F. DISTANT_UNRESOLVABLE** | Student is located in back 3 rows of classroom. Head bounding box is $<30\times30$ pixels. Head angle is blurred or pixelated. | Indeterminate | Indeterminate | **`EXCLUDE / QUARANTINE`**<br>(Never use for training; causes gradient noise) |

---

## 3. Physical Source Comparison Matrix

```
Visual Audit Sample Count (Audited via contact sheets in reports/v4a/contact_sheets/):
- SCBehavior High-Res turn_head: 50 physical crops (from 1,011 total bboxes)
- SCBehavior High-Res discuss: 50 physical crops (from 242 total bboxes)
- AFLW2000-3D extreme yaw: 50 physical face crops (|yaw| > 45°)
- AFLW2000-3D frontal: 50 physical face crops (|yaw| < 15°)
- EduAction talking: 50 physical video clips (50 audited frames)
```

| Dataset Source | Physical Labels Present | Semantic Tier Breakdown | Usability for `turn_head` Training | Verification Findings |
| :--- | :--- | :--- | :--- | :--- |
| **SCBehavior High-Res** (`datasets/raw_v4/scbehavior_highres`) | `turn_head` (1,011 bboxes)<br>`discuss` (242 bboxes) | 78% Tier B (Clear Yaw)<br>16% Tier C (Head+Torso)<br>6% Tier A (Subtle)<br>0% Tier F (High-res 2.5K/4K avoids pixel blur) | **HIGH VALUE (APPROVED)**<br>Primary source for full-frame classroom turn_head detector. | Due to 2560x1440 and 3840x2160 resolution, even rear students have head crops $>50\times50$ px, resolving facial profiles crisply. Clean separation from `discuss`. |
| **AFLW2000-3D** (`datasets/raw_v4/head_pose_aflw2000`) | Continuous yaw angles (-90° to +90°)<br>2,000 AFLW2000 + 21k AFLW crops | Ground-truth 3D angles:<br>462 crops with $|yaw| > 35°$<br>232 crops with $|yaw| > 60°$ | **HIGH VALUE (HEAD POSE BRANCH)**<br>Continuous regression / classification pretraining. | Direct numerical ground truth for yaw angle. Allows calibrating the exact $\ge 35°$ threshold for `turn_head`. |
| **EduAction** (`datasets/raw_v4/other_candidates/eduaction`) | `talking` (50 clips)<br>`lecture` (50 clips) | `talking` -> Tier E (Discussion / Peer orient)<br>`lecture` -> Frontal baseline | **SUPPORTING VALUE (INTERACTION)**<br>Supplies dynamic mouth & head turning in pairs. | Clips show students orienting towards neighbors during conversation, helping separate solo cheating turn from interactive discussion. |
| **SCB5 Dataset** (`datasets/raw/scbehavior/SCB_BowTurnHead...`) | `TurnHead` (1,905 train images) | 65% Tier B<br>20% Tier A<br>15% Tier F (Low-res 640x640 crops) | **CONDITIONAL / SECONDARY**<br>Contains noisy subtle glances and pixelated distant heads. | Must filter out boxes with area $< 900\text{ px}^2$ ($<30\times30$). |
| **Hashemite Cheating** (*Prospective / Kaggle*) | "Looking at another student's exam paper" (37 videos) | 90% Tier B & Tier C | **HIGH POTENTIAL**<br>Real exam paper cheating enactment. | Directly captures the intent of peeking at an adjacent exam paper. |

---

## 4. Visual Evidence & Physical Audit Findings

1. **SCBehavior High-Res Turn Head (`reports/v4a/contact_sheets/scbehavior/turn_head/`):**
   - High-resolution contact sheets confirm that `turn_head` boxes capture students with unmistakable sideways head orientation (>40° yaw).
   - In 42 out of 50 sampled crops, both ears are not visible; only the side profile, ear, and cheek are prominent.
   - Distinct from `discuss`: In `discuss`, the bounding boxes encompass pairs of interacting students or single students actively gesturing towards a partner.

2. **AFLW2000-3D Extreme Yaw (`reports/v4a/contact_sheets/aflw2000/turn_head_extreme/`):**
   - Head crops audited with $|yaw| > 45°$ exhibit the exact facial keypoint geometry corresponding to exam cheating head turns:
     - Nose tip displaced past the lateral eye boundary.
     - Eye aspect ratio compressed on the far side due to foreshortening.
     - Ear landmark fully exposed on the near side.

---

## 5. Architectural Decision for V4 Turn-Head Pipeline

Because full-frame detectors struggle with small head pixel sizes, the V4 architecture must employ a **dual-stage confirmation**:
1. **Primary Stage (Full-Frame YOLO):**
   - Proposes candidate `turn_head` bounding box on upper body.
2. **Secondary Stage (Crop Head-Pose / Keypoint Estimator):**
   - Extracts student head crop (resized to $128\times128$ or $224\times224$).
   - Computes continuous Euler yaw angle $\theta_{yaw}$.
   - Evaluates:
     $$\text{Violation Probability} = \begin{cases} 
     0.0 & \text{if } |\theta_{yaw}| < 25° \quad (\text{Tier A}) \\
     \frac{|\theta_{yaw}| - 25°}{20°} & \text{if } 25° \le |\theta_{yaw}| \le 45° \\
     1.0 & \text{if } |\theta_{yaw}| > 45° \quad (\text{Tier B/C})
     \end{cases}$$
3. **Temporal Debounce:**
   - Event is triggered only if $|\theta_{yaw}| > 35°$ for $\ge 1.5\text{ seconds}$ ($N \ge 38\text{ consecutive frames at 25 fps}$).
   - Fleeting glances ($< 1.0\text{ s}$) are logged as benign telemetry, preventing teacher alert fatigue.
