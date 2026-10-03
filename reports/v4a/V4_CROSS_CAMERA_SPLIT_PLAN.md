# V4 Cross-Camera Generalization & Split Strategy Plan
**Date:** 2026-10-02  
**Report ID:** V4A-SPLIT-PLAN-01  
**Target:** Elimination of Sequence Leakage and Verification of Unseen-Camera Generalization  

---

## 1. Executive Context: The Root of Evaluation Failures

In naive dataset splitting (e.g. random 80/20 train/val splits), adjacent frames from the same continuous video clip end up in both training and validation sets.  
Because consecutive frames at 25 fps share identical background walls, desks, lighting, student clothing, and hairstyles:
- The model memorizes specific students and room geometries rather than learning generalizable posture representations.
- Validation mAP appears artificially high (e.g. 0.85–0.95).
- However, when deployed on a novel classroom camera or perspective, performance collapses catastrophically (as seen in our Stage 1.5 audit: unseen V3 validation recall was 0.0514 for `head_down` and 0.1969 for `turn_head`).

**Mandatory V4 Principle:**  
No frame, student identity, or recording session may cross between splits.  
Evaluation must test four distinct generalization regimes:
1. `SAME_DOMAIN_HOLDOUT`: Unseen students in known room setup.
2. `CROSS_CAMERA_HOLDOUT`: Unseen camera angle of known room.
3. `HIGH_ANGLE_HOLDOUT`: Steep downward ceiling CCTV (30°–60° inclination).
4. `REAR_VIEW_HOLDOUT`: Camera positioned behind students looking forward.

---

## 2. Dataset Viewpoint & Session Inventory

| Dataset Source | Viewpoint Types | Camera Count | Session / Group Identifiers | Generalization Role |
| :--- | :--- | :--- | :--- | :--- |
| **SCBehavior High-Res** (`datasets/raw_v4/scbehavior_highres`) | Upper-oblique, high-angle classroom, multi-row elevated | 3 distinct classroom setups | Distinct image prefixes (e.g. `01.jpg` – `0100.jpg`, `0200.jpg`, `0300.jpg`, `0400.jpg`) | Split by classroom session ID into Train and Cross-Room Holdout |
| **EduAction** (`datasets/raw_v4/other_candidates/eduaction`) | Oblique desk angle, side-row perspective, student-centric | 4 distinct college lecture rooms | Clip numeric sequences (`sleep (1)` through `sleep (50)`) | Split by subject/session groups (clips 1–35 Train, 36–50 Unseen Holdout) |
| **AFLW2000-3D** (`datasets/raw_v4/head_pose_aflw2000`) | Multi-angle synthetic & in-the-wild face poses | Continuous 3D spherical angles | 2,000 distinct benchmark crops | Fixed external benchmark for pose accuracy |
| **Smart Classroom** (*Prospective / Blocked*) | 4 explicit labeled angles:<br>• 正面 (Frontal)<br>• 斜上方 (Upper-Oblique)<br>• 后方 (Rear)<br>• 教师 (Teacher) | 4 synchronized camera feeds | Explicit camera metadata (`front`, `diagonal_upper`, `rear`, `teacher`) | Gold standard for complete Camera Holdout (`rear` and `diagonal_upper` held out) |
| **CStudentAct** (*Prospective / Blocked*) | Ceiling-mounted oblique surveillance | Multi-camera Full-HD | Linked action instances with person IDs | Gold standard for Person-ID disjoint holdouts |

---

## 3. Four-Tier V4 Evaluation Architecture

When V4B dataset compilation is authorized, the validation split will NOT be a single monolithic directory. It will be partitioned into four distinct evaluation suites:

```
datasets/processed_v4/
    ├── train/                        # Aggregated diverse training corpus
    ├── val_same_domain/              # Unseen clips/frames from same room setups
    ├── val_cross_camera/             # Completely held-out camera perspectives
    ├── val_high_angle/               # Exclusively steep downward CCTV feeds (>45°)
    └── val_rear_view/                # Exclusively rear perspectives (backs of heads)
```

### Detailed Split Definition

```
1. Suite A: Same-Domain Holdout (Sanity Check)
   - Composition: 15% of clips from each training classroom/session.
   - Purpose: Verifies in-distribution learning capacity and convergence.
   - Target Metric: mAP50 > 0.85, Recall > 0.80 across all classes.

2. Suite B: Cross-Camera Holdout (Generalization Gate)
   - Composition: Entire camera sessions never seen during training.
   - Purpose: Tests resilience to unfamiliar lighting, wall color, and desk layout.
   - Target Metric: mAP50 > 0.65, Recall > 0.50 (especially turn_head and head_down).

3. Suite C: High-Angle CCTV Holdout (Steep Overhead Gate)
   - Composition: High-angle cameras where foreshortening distorts head-to-shoulder aspect ratios.
   - Target Metric: Head-down recall > 0.60.

4. Suite D: Rear-View Holdout (Back-of-Head Gate)
   - Composition: Footage where students face away from camera; faces largely invisible.
   - Target Metric: Tests whether keypoints and ear/shoulder vectors successfully detect turn_head without facial landmarks.
```

---

## 4. Group Leakage Prevention Invariants

To guarantee absolute scientific integrity:
1. **Hash-Based Disjoint Verification:**
   - Pre-commit check verifying that SHA256 hashes of all training images have zero intersection with any validation suite ($H_{train} \cap H_{val} = \emptyset$).
2. **Perceptual Hash Distance Threshold:**
   - Perceptual difference $D_{dhash}(I_{train}, I_{val}) > 6$ bits, ensuring near-duplicate frames from the same 2-second interval do not leak across splits.
3. **Session ID Clustering:**
   - Grouping regex (`group_id = file.split('_')[0]`) enforced by unit tests (`test_split_dataset_grouping_no_leakage.py`).

**V4A Status:**  
Split design finalized. No physical split created yet in Phase V4A in accordance with the Phase mandate.
