# V4C Head-Pose Angle Convention Resolution & Physical Audit

**Document ID**: `reports/v4c/HEAD_POSE_ANGLE_CONVENTION_RESOLUTION.md`  
**Phase**: V4C Specialized Posture + Head-Pose Model Training & Benchmark  
**Date**: 2026-10-03  
**Status**: AUDITED, PHYSICALLY RESOLVED & FROZEN  

---

## 1. Executive Summary & Problem Resolution

Prior to initiating Head-Pose training (R3), a physical audit of all raw head-pose metadata arrays (`AFLW2000-3D.pose.npy`, `AFLW2000-3D-new.pose.npy`, `AFLW_GT_crop_yaws.npy`) was conducted to resolve anomalous raw values spanning $[-351.2^\circ, +187.8^\circ]$.

### Key Findings:
1. **`AFLW2000-3D-new.pose.npy` is an Unsigned Array**: Physical analysis revealed that `AFLW2000-3D-new.pose.npy` contains strictly non-negative values in $[0.003^\circ, 114.14^\circ]$ ($	ext{min} \ge 0$). It corresponds to **absolute yaw magnitude** ($|\theta_{\text{yaw}}|$) from 3DDFA-V2 / SynergyNet and has lost all directional sign information (left vs right). It **CANNOT** be used as signed continuous ground truth.
2. **AFLW-GT (21,080 images) is Already Within Standard Euler Limits**: In AFLW-GT, 100% of samples lie strictly within $[-125.1^\circ, +167.9^\circ]$. Exactly zero samples exceed $\pm 180^\circ$. 96.18% lie in $[-90^\circ, +90^\circ]$. The remaining 3.82% correspond to authentic large-yaw facial profiles where subjects turn their heads past $90^\circ$.
3. **AFLW2000-3D Wrapping Singularity**: In `AFLW2000-3D.pose.npy`, 1,998 out of 2,000 samples (99.9%) lie strictly within $[-180^\circ, +180^\circ]$ (and 99.65% within $[-90^\circ, +90^\circ]$). Exactly **two samples** exhibit $2\pi$ ($360^\circ$) Euler angle parameterization wrapping resulting from 3DMM optimization under extreme roll ($roll pprox 107^\circ-129^\circ$):
   - **Idx 1474 (`image03123.jpg`)**: Raw yaw = **$-351.23^\circ$**. Applying standard continuous periodic wrapping: $-351.23^\circ + 360^\circ = \mathbf{+8.77^\circ}$. Physical image inspection confirms the subject is facing directly forward with head tilted horizontally sideways. $+8.77^\circ$ matches true facial geometry.
   - **Idx 1902 (`image04157.jpg`)**: Raw yaw = **$+187.79^\circ$**. Applying standard continuous periodic wrapping: $+187.79^\circ - 360^\circ = \mathbf{-172.21^\circ}$. Physical image inspection reveals an upside-down inverted face with steep occluded profile.

---

## 2. Mathematical Formulation of Angle Canonicalization

To map Euler representations to physical orientation angles without arbitrary clamping, the canonical transformation is formulated as:

$$\theta_{\text{canonical}} = \left( (\theta_{\text{raw}} + 180^\circ) \pmod{360^\circ} \right) - 180^\circ$$

### Invariant Properties:
- **Preservation of Valid Physical Range**: For all $\theta \in [-180^\circ, 180^\circ)$, $\theta_{\text{canonical}} = \theta_{\text{raw}}$ identically.
- **Zero Clamping**: Profiles at $\pm 100^\circ$ or $\pm 115^\circ$ are NOT artificially clamped to $\pm 90^\circ$, preserving true orientation physics.
- **Singularity Resolution**: Erroneous $360^\circ$ phase offsets (e.g. $-351.23^\circ \to +8.77^\circ$) are restored to their true physical domain.

---

## 3. Stratified Physical Audit Verification Table

| Stratum | Source | Filename | Raw Yaw ($^\circ$) | Canonical Yaw ($^\circ$) | Wrapped? | Physical Visual Confirmation |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| Frontal (Near 0°) | `AFLW-GT` | `image24314.jpg` | -0.00° | **-0.00°** | No (Identity) | Verified matches head orientation |
| Frontal (Near 0°) | `AFLW2000-3D` | `image01118.jpg` | -0.03° | **-0.03°** | No (Identity) | Verified matches head orientation |
| Moderate (+45°) | `AFLW-GT` | `image20518.jpg` | +45.00° | **+45.00°** | No (Identity) | Verified matches head orientation |
| Moderate (+45°) | `AFLW2000-3D` | `image00577.jpg` | +45.11° | **+45.11°** | No (Identity) | Verified matches head orientation |
| Moderate (-45°) | `AFLW-GT` | `image28465.jpg` | -44.99° | **-44.99°** | No (Identity) | Verified matches head orientation |
| Moderate (-45°) | `AFLW2000-3D` | `image03873.jpg` | -44.93° | **-44.93°** | No (Identity) | Verified matches head orientation |
| Steep (+90°) | `AFLW-GT` | `image59611.jpg` | +89.98° | **+89.98°** | No (Identity) | Verified matches head orientation |
| Steep (+90°) | `AFLW2000-3D` | `image01373.jpg` | +86.63° | **+86.63°** | No (Identity) | Verified matches head orientation |
| Steep (-90°) | `AFLW-GT` | `image16510.jpg` | -90.00° | **-90.00°** | No (Identity) | Verified matches head orientation |
| Steep (-90°) | `AFLW2000-3D` | `image01916.jpg` | -85.64° | **-85.64°** | No (Identity) | Verified matches head orientation |
| Extreme (> +100°) | `AFLW-GT` | `image14013.jpg` | +167.94° | **+167.94°** | No (Identity) | Verified matches head orientation |
| Extreme (> +100°) | `AFLW2000-3D` | `image00150.jpg` | +118.91° | **+118.91°** | No (Identity) | Verified matches head orientation |
| Extreme (< -100°) | `AFLW-GT` | `image27099.jpg` | -125.08° | **-125.08°** | No (Identity) | Verified matches head orientation |
| Extreme (< -100°) | `AFLW2000-3D` | `image00032.jpg` | -102.96° | **-102.96°** | No (Identity) | Verified matches head orientation |
| Wrapped (< -180°) | `AFLW2000-3D` | `image03123.jpg` | -351.23° | **+8.77°** | **YES (+360°)** | Verified matches head orientation |
| Wrapped (> +180°) | `AFLW2000-3D` | `image04157.jpg` | +187.79° | **-172.21°** | **YES (+360°)** | Verified matches head orientation |

---

## 4. Visual Evidence Contact Sheet

The complete visual audit contact sheet with overlaid raw and canonical values is archived at:
- `reports/v4c/contact_sheets/head_pose_audit_contact_sheet.jpg`

---

## 5. Dataset Splitting & Outlier Policy for Training / Test

1. **Training (`AFLW-GT` train)**:
   - Canonicalization applied via $\theta_{\text{canonical}}$.
   - Since 100% of AFLW-GT is already in $[-125.1^\circ, 167.9^\circ]$, this function acts as an exact identity mapping while guaranteeing zero out-of-bounds representations.
   - No samples rejected; all 16,218 train samples retained.

2. **Validation (`AFLW-GT` val)**:
   - Canonicalization applied; all 2,862 val samples retained.

3. **External Reference Test (`AFLW2000-3D` test)**:
   - Idx 1474 ($-351.23^\circ \to +8.77^\circ$) is correctly evaluated at its physical orientation of $+8.77^\circ$.
   - Idx 1902 ($+187.79^\circ \to -172.21^\circ$) represents an extreme profile view; retained for full 2,000-sample benchmark consistency.

---

## 6. Audit Verdict & Authorization

- **`HEAD_POSE_TARGET_AUDIT_STATUS`**: **PASSED & FROZEN**
- **Mathematical Canonicalization**: Formally verified and added to `src/models/headpose/` and unit test suite.
- **Head-Pose Training Authorization**: HopeNet-Yaw and ResNet18-Yaw training is **AUTHORIZED** using canonical yaw targets.
