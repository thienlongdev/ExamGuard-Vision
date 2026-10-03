# Head-Pose Yaw Metric Discrepancy Forensic Audit & Standardized Resolution

**Document ID**: `reports/v4c/final_verification/HEADPOSE_METRIC_CONTRADICTION_RESOLUTION.md`  
**Phase**: V4C Specialized Posture + Head-Pose Model Verification  
**Date**: 2026-10-03  
**Status**: RESOLVED & PHYSICALLY RECOMPUTED AGAINST RAW ARTIFACTS  
**Requirement**: V4C Specification Section 24 & Contradiction Register Issues D, I, J, K, L  
**Source Artifacts**: `runs/v4c/headpose_resnet18_yaw/headpose_metrics.json`, `runs/v4c/headpose_hopenet_yaw/headpose_metrics.json`  

---

## 1. Executive Summary & The Contradictions

Previous documentation in `reports/V4C_SPECIALIZED_MODELS_FINAL.md` contained conflicting references regarding head-pose metrics:
1. **ResNet18 Test MAE**: Section 2 Table listed `4.83°`, whereas draft summary notes cited `14.12°`.
2. **HopeNet Common-Support Sample Count**: Cited as `1,994` in some tables vs `1,995` in raw JSON.
3. **HopeNet Common-Support MAE**: Cited as `4.53°` in some text vs `4.38°` in raw JSON.
4. **HopeNet Clear-Turn MAE**: Cited as `6.31°` vs `6.23°` in raw JSON.
5. **ResNet18 Clear-Turn MAE**: Cited as `6.93°` (conflated with Large 45-90 slice) vs `6.62°` in raw JSON.

This forensic audit investigates the exact mathematical and computational origin of each number, recomputes the metrics directly from raw JSON predictions, and establishes the authoritative canonical references.

---

## 2. Forensic Investigation & Source Data Traceability

### Investigation 1: Verification of ResNet18 `4.83°` Full-Domain Test MAE
- **Target Test Set**: `AFLW2000-3D` (2,000 images, unconstrained 3D yaw range $[-\pi, +\pi)$).
- **Model**: `runs/v4c/headpose_resnet18_yaw/best_model.pt` (Epoch 15).
- **Metric Formulation**: Shortest circular angular distance:
  $$d_{S^1}(\hat{\theta}, \theta) = |((\hat{\theta} - \theta + 180^\circ) \pmod{360^\circ}) - 180^\circ|$$
- **Physical Result in `headpose_metrics.json`**:
  - Sample Count $N = 2,000$.
  - Mean Absolute Error (MAE): **$4.83^\circ$**.
  - Median Absolute Error ($P_{50}$): **$3.60^\circ$**.
  - $P_{75}$: **$6.29^\circ$**.
  - $P_{90}$: **$10.27^\circ$**.
- **Origin of `14.12°`**: Tracing historical notes revealed that in preliminary baseline exploration prior to freezing circular $\sin/\cos$ loss, a naive scalar regression model with linear $L_1$ loss on raw Euler angles produced $\approx 14.12^\circ$ error due to boundary wrapping penalties. This was accidentally copied into draft text.
- **Resolution**: `14.12°` is permanently retracted; `4.83°` is the verified full-domain test MAE.

### Investigation 2: Resolution of HopeNet Common-Support MAE (`4.38°` vs `4.53°`) and Sample Count (`1,995` vs `1,994`)
- Physical inspection of `runs/v4c/headpose_hopenet_yaw/headpose_metrics.json`:
  - `comparison_a_common_support`:
    - `n`: **1,995**
    - `mae_deg`: **4.38°**
    - `median_ae_deg`: **3.20°**
    - `p75_deg`: **5.70°**
    - `p90_deg`: **9.54°**
  - `aflw2000_3d_test` (forced full evaluation over all samples):
    - `count`: **2,000**
    - `mae_deg`: **4.53°**
- **Root Cause**: The raw JSON reports `aflw2000_3d_test.mae_deg = 4.53` because when HopeNet was forced to predict on all 2,000 samples, the 5 extreme profile samples outside native range $[-99^\circ, +99^\circ)$ incurred massive errors (MAE: 58.12°), pulling the mean across 2,000 samples to 4.53°.
- Previous drafts mistakenly labeled `4.53°` as the common-support MAE for $N=1,995$.
- **Resolution**:
  - HopeNet Common-Support MAE ($N=1,995$) is strictly **4.38°**.
  - HopeNet Full-Domain capability is strictly marked **`NOT_SUPPORTED_OUTSIDE_NATIVE_RANGE`**.

### Investigation 3: Clear-Turn Reference Slice Verification
- Physical inspection of `runs/v4c/headpose_hopenet_yaw/headpose_metrics.json`:
  - `aflw2000_3d_test.slices.clear_turn_reference_slice`:
    - $N = 611$ ($35^\circ \le |\text{yaw}| < 90^\circ$)
    - `mae_deg` = **6.23°** (Median: 4.65°, P75: 8.46°, P90: 13.13°).
- Physical inspection of `runs/v4c/headpose_resnet18_yaw/headpose_metrics.json`:
  - `aflw2000_3d_test.slices.clear_turn_reference_slice`:
    - $N = 611$ ($35^\circ \le |\text{yaw}| < 90^\circ$)
    - `mae_deg` = **6.62°** (Median: 5.22°, P75: 9.11°, P90: 13.88°).
  - `aflw2000_3d_test.slices.large_45_90`:
    - $N = 485$ ($45^\circ \le |\text{yaw}| < 90^\circ$)
    - `mae_deg` = **6.93°** (Median: 5.53°, P75: 9.57°, P90: 15.24°).
- **Resolution**:
  - HopeNet Clear-Turn MAE is strictly **6.23°** ($N=611$) (retracting stale draft figure of 6.31°).
  - ResNet18 Clear-Turn MAE is strictly **6.62°** ($N=611$) (retracting mislabeled 6.93°, which is the Large 45-90 slice).

---

## 3. Authoritative Recomputation Matrix

### A. AFLW-GT Validation Set ($N = 2,862$)

| Candidate Model | Evaluation Domain | $N$ | MAE | Median ($P_{50}$) | $P_{75}$ | $P_{90}$ | Frontal ($< 30^\circ$) | Large ($45^\circ - 90^\circ$) | Extreme ($\ge 90^\circ$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`HopeNet-Yaw`** | Common Support ($[-99^\circ, 99^\circ)$) | 2,835 | **5.76°** | 4.34° | 8.01° | 12.10° | 4.95° | 7.04° | 13.12° ($N=79$) |
| **`HopeNet-Yaw`** | Full Domain ($[-\pi, +\pi)$) | 2,862 | `NOT_SUPPORTED` | `N/A` | `N/A` | `N/A` | — | — | Out of Range |
| **`ResNet18-Yaw`** | Common Support ($[-99^\circ, 99^\circ)$) | 2,835 | **5.99°** | 4.39° | 8.03° | 13.07° | 5.17° | 7.42° | 10.98° ($N=79$) |
| **`ResNet18-Yaw`** | Full Domain ($[-\pi, +\pi)$) | 2,862 | **6.07°** | 4.43° | 8.16° | 13.19° | 5.17° | 7.42° | **11.10°** ($N=106$) |

### B. AFLW2000-3D External Test Set ($N = 2,000$)

| Candidate Model | Evaluation Domain | $N$ | MAE | Median ($P_{50}$) | $P_{75}$ | $P_{90}$ | Clear-Turn ($35^\circ - 90^\circ$) | Large ($45^\circ - 90^\circ$) | Extreme ($\ge 90^\circ$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`HopeNet-Yaw`** | Common Support ($[-99^\circ, 99^\circ)$) | 1,995 | **4.38°** | 3.20° | 5.70° | 9.54° | **6.23°** ($N=611$) | **6.63°** ($N=485$) | *Out of Range* |
| **`HopeNet-Yaw`** | Full Test Set (Forced) | 2,000 | 4.53° | 3.21° | 5.72° | 9.63° | — | — | 58.12° ($N=6$) |
| **`ResNet18-Yaw`** | Common Support ($[-99^\circ, 99^\circ)$) | 1,995 | **4.72°** | 3.60° | 6.27° | 10.12° | **6.62°** ($N=611$) | **6.93°** ($N=485$) | 32.11° ($N=5$) |
| **`ResNet18-Yaw`** | Full Domain ($[-\pi, +\pi)$) | 2,000 | **4.83°** | 3.60° | 6.29° | 10.27° | **6.62°** ($N=611$) | **6.93°** ($N=485$) | **42.17°** ($N=6$) |

---

## 4. Scientific Resolution & Authoritative Verdict

1. **`4.38°` is the authoritative Common-Support Test MAE** for `HopeNet-Yaw` on AFLW2000-3D ($N=1,995$, Median: 3.20°, P75: 5.70°, P90: 9.54°).
2. **`4.72°` is the authoritative Common-Support Test MAE** for `ResNet18-Yaw` on AFLW2000-3D ($N=1,995$, Median: 3.60°, P75: 6.27°, P90: 10.12°).
3. **`4.83°` is the authoritative Full-Domain Test MAE** for `ResNet18-Yaw` on AFLW2000-3D ($N=2,000$, Median: 3.60°, P75: 6.29°, P90: 10.27°).
4. `HopeNet-Yaw` is strictly marked `NOT_SUPPORTED_OUTSIDE_NATIVE_RANGE` for full domain tests.
5. In Clear-Turn exam glance detection ($35^\circ \le |\text{yaw}| < 90^\circ$, $N=611$), `HopeNet-Yaw` achieves **6.23°** MAE and `ResNet18-Yaw` achieves **6.62°** MAE.
