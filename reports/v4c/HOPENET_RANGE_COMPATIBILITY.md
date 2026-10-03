# V4C HopeNet Target-Range Compatibility Audit & Resolution

**Status**: FROZEN & AUDITED (HALF-OPEN BOUNDARY UPDATED)  
**Author**: Antigravity Machine Learning Safety & Mathematics Team  
**Date**: 2026-10-03  
**Target Module**: Phase V4C Module B — Head-Pose Estimator Architectures  
**Reference Document**: `configs/v4_fusion_contract.yaml`

---

## 1. Executive Summary

This audit investigates the compatibility between the local HopeNet architecture (`src/models/headpose/hopenet.py`) and the physical dataset distribution of AFLW-GT and AFLW2000-3D. 

We identified two critical boundary and topology issues in legacy HopeNet:
1. **Silent Clamping**: Angles outside the configured range were clamped via `torch.clamp(bins, 0, 65)`, corrupting extreme profile labels.
2. **Discrete Upper Boundary Overflow**: With `num_bins = 66`, `min_angle = -99.0`, and `bin_width = 3.0`, evaluating an exact angle of $+99.0^\circ$ produces `floor((99.0 - (-99.0)) / 3.0) = 66`, which overflows the valid bin index range `0..65`. Thus, HopeNet's mathematically valid native support is strictly the **half-open interval `[-99.0, +99.0)`**, NOT closed `[-99.0, +99.0]`.

Furthermore, because HopeNet uses a linear expectation over discrete bin centers, expanding bins to $[-180^\circ, +180^\circ)$ suffers from a circular mean collapse at the $\pm 180^\circ$ boundary.

**Resolution Adopted (Strategy B)**:
1. HopeNet-Yaw is strictly confined to its native half-open support range **`[-99.0^\circ, +99.0^\circ)`**.
2. Boundary behavior is strictly enforced with zero silent clipping:
   - `-99.0^\circ \to \text{valid (bin 0)}`
   - `-98.999^\circ \to \text{valid (bin 0)}`
   - `0.0^\circ \to \text{valid (bin 33)}`
   - `+98.999^\circ \to \text{valid (bin 65)}`
   - `+99.0^\circ \to \text{OUT OF RANGE (raises ValueError)}`
   - `> +99.0^\circ \to \text{OUT OF RANGE (raises ValueError)}`
   - `< -99.0^\circ \to \text{OUT OF RANGE (raises ValueError)}`
3. Samples outside `[-99.0^\circ, +99.0^\circ)` are explicitly excluded from HopeNet training and logged transparently.
4. Candidate B (`ResNet18-Yaw-Circular`) with continuous $(\sin\theta, \cos\theta)$ regression provides universal, non-discontinuous coverage over the entire $[-180^\circ, +180^\circ)$ domain.

---

## 2. Physical Architecture Audit of Local HopeNet

Physical inspection of `src/models/headpose/hopenet.py`:
- **Backbone**: ResNet-50 (ImageNet pretrained, conv1 through avgpool).
- **Classification Head**: `nn.Linear(2048, 66)`.
- **Number of Yaw Bins**: $66$.
- **Configured Native Range**: `[-99.0^\circ, +99.0^\circ)` (half-open).
- **Bin Width**: $w = \frac{99.0 - (-99.0)}{66} = \frac{198.0}{66} = 3.0^\circ$.
- **Bin Centers**: $[-97.5^\circ, -94.5^\circ, \dots, +94.5^\circ, +97.5^\circ]$.
- **Inference Formulation**: Softmax expected value:
  $$\hat{\theta} = \sum_{i=0}^{65} P_i \cdot c_i, \quad P_i = \text{Softmax}(\text{logits})_i$$
- **Corrected Boundary Validation**:
  ```python
  def angle_to_bin(self, angles: torch.Tensor, strict: bool = True) -> torch.Tensor:
      if strict:
          out_of_range_mask = (angles < self.min_angle) | (angles >= self.max_angle)
          if out_of_range_mask.any():
              raise ValueError("HopeNet range violation: angle outside [-99.0, +99.0)")
      bins = torch.floor((angles - self.min_angle) / self.bin_width).long()
      return bins
  ```

---

## 3. Physical Dataset Distribution Audit (Half-Open Support)

We performed an exact distribution census across train, val, and test splits after canonicalization to $[-180^\circ, +180^\circ)$ evaluated against the half-open interval `[-99.0^\circ, +99.0^\circ)`:

| Angular Interval | AFLW-GT Train ($N=16,218$) | AFLW-GT Val ($N=2,862$) | AFLW2000-3D Test ($N=2,000$) |
| :--- | :--- | :--- | :--- |
| **$\|yaw\| < 15^\circ$** (Frontal Center) | 5,187 (31.98%) | 946 (33.05%) | 893 (44.65%) |
| **$15^\circ \le \|yaw\| < 45^\circ$** (Moderate / Classroom) | 6,121 (37.74%) | 1,081 (37.77%) | 616 (30.80%) |
| **$45^\circ \le \|yaw\| < 90^\circ$** (Large / Side Profile) | 4,232 (26.09%) | 729 (25.47%) | 485 (24.25%) |
| **$90^\circ \le \|yaw\| < 120^\circ$** (Steep Profile) | 676 (4.17%) | 106 (3.70%) | 5 (0.25%) |
| **$120^\circ \le \|yaw\| < 150^\circ$** (Back-Quarter Profile) | 1 (0.01%) | 0 (0.00%) | 0 (0.00%) |
| **$\|yaw\| \ge 150^\circ$** (Turning Away) | 1 (0.01%) | 0 (0.00%) | 1 (0.05%) |
| **Supported by HopeNet `[-99^\circ, +99^\circ)`** | **16,016 (98.75%)** | **2,835 (99.06%)** | **1,995 (99.75%)** |
| **Excluded (Outside `[-99^\circ, +99^\circ)`)** | **202 (1.25%)** | **27 (0.94%)** | **5 (0.25%)** |
| **Samples with exact $yaw == +99.000^\circ$** | **0 (0.00%)** | **0 (0.00%)** | **0 (0.00%)** |
| **Domain Minimum** | $-125.08^\circ$ | $-110.65^\circ$ | $-172.21^\circ$ |
| **Domain Maximum** | $+167.94^\circ$ | $+116.12^\circ$ | $+118.91^\circ$ |

---

## 4. Evaluation of Potential Strategies

### Strategy A: Extend Yaw Bins to Full Range $[-180^\circ, +180^\circ)$
- **Formulation**: 120 bins of $3^\circ$ covering $[-180^\circ, +180^\circ)$.
- **Fatal Mathematical Defect**: HopeNet's continuous expectation is a *linear* expected value:
  $$\mathbb{E}[\theta] = \sum_{i=0}^{119} P_i \cdot c_i$$
  On a circle $S^1$, a linear expectation is undefined and produces catastrophic artifacts. For example, if a student looks away at $\theta = 180^\circ$, a bimodal prediction with $P(178.5^\circ) = 0.5$ and $P(-178.5^\circ) = 0.5$ yields:
  $$\mathbb{E}[\theta] = 0.5 \times (+178.5^\circ) + 0.5 \times (-178.5^\circ) = 0.0^\circ$$
  The model would predict that a student looking completely backwards is looking directly forward into the camera!
- **Verdict**: **REJECTED**.

### Strategy B: Native Half-Open Support `[-99^\circ, +99^\circ)` with Explicit Audit (SELECTED)
- **Formulation**: Train HopeNet strictly on the $98.75\%$ of AFLW-GT training samples within `[-99.0^\circ, +99.0^\circ)`.
- **Zero Silent Clipping**: `HopeNetYaw.angle_to_bin` enforces `strict=True`, raising `ValueError` on any angle $< -99^\circ$ or $\ge +99^\circ$.
- **Fair Two-Tier Benchmark**:
  - **Comparison A (Common Support)**: Both models evaluated on the shared $1,995 / 2,000$ test samples ($99.75\%$) in `[-99^\circ, +99^\circ)`.
  - **Comparison B (Full Domain)**: ResNet18-Yaw-Circular evaluated on all 2,000 samples. HopeNet marked `NOT_SUPPORTED_OUTSIDE_NATIVE_RANGE`.
- **Verdict**: **ACCEPTED & IMPLEMENTED**.

---

## 5. Classroom-Relevant Reference Slices & Extreme Profile Policy

Slices are **evaluation metrics only**, NOT operational behavioral thresholds.

| Reference Slice | Angular Interval Definition | Test Sample Count ($N$) | Physical Classroom Semantics |
| :--- | :--- | :--- | :--- |
| **`FRONTAL_MODERATE_REFERENCE_SLICE`** | $\|yaw\| \le 45.0^\circ$ | 1,509 (75.45%) | Student looking forward or downward at exam paper |
| **`CLEAR_TURN_REFERENCE_SLICE`** | $35.0^\circ \le \|yaw\| < 90.0^\circ$ | 611 (30.55%) | Clear lateral head turn towards neighbor or aisle |
| **`ONTOLOGY_TURN_REFERENCE_SLICE`** | $35.0^\circ \le \|yaw\| \le 75.0^\circ$ | 513 (25.65%) | Core human head turn range without full body twist |
| **`EXTREME_PROFILE_SLICE`** | $\|yaw\| \ge 90.0^\circ$ | 6 (0.30%) | Severe profile / turning away backwards |

Every slice reports: $N$, MAE, Median Absolute Error, P75, and P90 using shortest angular distance.

---

## 6. Pitch and Roll Provenance Audit

- **AFLW-GT**: Contains **Signed Yaw Only** (`AFLW_GT_crop_yaws.npy`). Pitch and roll are **physically absent**.
- **AFLW2000-3D**: `AFLW2000-3D.pose.npy` is a 1D float32 array of **Signed Yaw Only**. Secondary pitch and roll in `manifest.jsonl` were derived geometrically from 3D facial landmarks (`AFLW2000-3D.pts68.npy`).
- Full provenance details are recorded in [`AFLW_POSE_ARRAY_SCHEMA_RESOLUTION.md`](file:///c:/WorkingSpace%20Python/DETECTOR-YOLO/reports/v4c/AFLW_POSE_ARRAY_SCHEMA_RESOLUTION.md).

---

## 7. Mathematical Pre-Train Gate Status

```
[PASS] Canonicalization audit accepted and frozen
[PASS] Shortest angular distance metric implemented and verified
[PASS] HopeNet half-open interval [-99.0, +99.0) enforced
[PASS] Boundary +99.0 strictly rejected with ValueError (zero silent clipping)
[PASS] ResNet18-Yaw circular sin/cos regression protected with epsilon
[PASS] AFLW pose array schema contradiction fully resolved (1D yaw arrays confirmed)
[PASS] Structured multi-slice metrics (N, MAE, Med, P75, P90) implemented
[PASS] Comparison A (Common Support) and Comparison B (Full Domain) separated
[PASS] Full unit test suite passes (89 tests passing, 0 failing)
```

**HEAD_POSE_PRETRAIN_FINAL_GATE = PASS**
