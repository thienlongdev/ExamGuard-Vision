# V4C Head-Pose Circular Regression Design Report

**Status**: FROZEN & VALIDATED  
**Author**: Antigravity Machine Learning Safety & Mathematics Team  
**Date**: 2026-10-03  
**Target Module**: Phase V4C Module B — Head-Pose Yaw Regression  
**Reference Document**: `configs/v4_fusion_contract.yaml`

---

## 1. Executive Summary & The Blocker

In conventional scalar regression, Euler yaw angle $\theta \in [-180^\circ, +180^\circ)$ is predicted directly as an unbounded or linearly clamped real number $\hat{\theta} \in \mathbb{R}$. While canonicalization solves the input Euler phase ambiguity ($360^\circ k$ equivalent angles), **canonicalization alone does not make scalar regression loss functions mathematically safe**.

### The $\pm 180^\circ$ Discontinuity Blocker
Consider a head rotated almost completely around to the boundary:
- **Ground Truth Target**: $\theta = +179.0^\circ$
- **Network Prediction**: $\hat{\theta} = -179.0^\circ$
- **Naive Scalar Error**: $|\hat{\theta} - \theta| = |-179.0 - 179.0| = 358.0^\circ$
- **True Shortest Angular Distance**: $d_{S^1}(\hat{\theta}, \theta) = 2.0^\circ$

Under ordinary regression losses (MSE, L1, SmoothL1), the network receives a gradient of magnitude corresponding to $358^\circ$, pushing the parameters in the *opposite* direction across the entire front of the face ($358^\circ$ arc) rather than taking the true $2^\circ$ shortest geodesic path across the back boundary.

**Mathematical Rule**: Naive scalar SmoothL1 or MSE across $[-180^\circ, +180^\circ)$ is **strictly prohibited**. No yaw regression training may commence without a periodicity-safe formulation and shortest angular error metric.

---

## 2. Shortest Angular Distance Metric

All evaluation metrics (MAE, Median Absolute Error, P50, P75, P90, and distribution slices) are defined using the canonical shortest angular difference on the circle $S^1$:

$$\Delta_{\text{deg}} = ((\hat{\theta} - \theta + 180^\circ) \pmod{360^\circ}) - 180^\circ$$

$$\text{Error}_{\text{deg}} = |\Delta_{\text{deg}}|$$

### Mathematical Properties:
1. **Bounded Range**: $\text{Error}_{\text{deg}} \in [0^\circ, 180^\circ]$ for all $\hat{\theta}, \theta \in \mathbb{R}$.
2. **Symmetry**: $|\Delta(\hat{\theta}, \theta)| = |\Delta(\theta, \hat{\theta})|$.
3. **Geodesic Invariance**: Resolves any arbitrary multi-turn Euler phase wrapping:
   - Target $+179^\circ$, Pred $-179^\circ \implies \text{Error} = 2.0^\circ$
   - Target $-179^\circ$, Pred $+179^\circ \implies \text{Error} = 2.0^\circ$
   - Target $+45^\circ$, Pred $+50^\circ \implies \text{Error} = 5.0^\circ$
   - Target $-90^\circ$, Pred $+90^\circ \implies \text{Error} = 180.0^\circ$
   - Target $0^\circ$, Pred $360^\circ \implies \text{Error} = 0.0^\circ$

---

## 3. Comparison of Training Target Formulations

We evaluated two mathematically sound approaches for yaw training:

| Dimension | Option A: Circular Sin/Cos Regression | Option B: Periodic Loss on Scalar Angle |
| :--- | :--- | :--- |
| **Output Representation** | 2D vector $\hat{\mathbf{v}} = (\hat{s}, \hat{c}) \in \mathbb{R}^2$ | 1D scalar $\hat{\theta} \in \mathbb{R}$ |
| **Topology** | Embeds $S^1 \hookrightarrow \mathbb{R}^2$ smoothly | Maps simply-connected feature space to non-simply-connected circle $S^1$ |
| **Boundary Discontinuity** | **Zero discontinuity** anywhere on $S^1$ | Modulo / atan2 discontinuity in loss backward graph |
| **Gradient Smoothness** | Infinitely differentiable ($C^\infty$) vector field | Derivative singularity at $\Delta\theta = \pm 180^\circ$ |
| **Numerical Stability** | Completely stable; bounded targets $\in [-1, 1]$ | Susceptible to gradient explosion near wrapping boundary |
| **Decoding** | $\hat{\theta} = \text{atan2}(\hat{s}, \hat{c}) \cdot \frac{180}{\pi}$ | Identity $\hat{\theta}$ |

### Why Option A was Selected:
1. **Topological Purity**: By Whitney's Embedding Theorem, the circle $S^1$ cannot be continuously embedded into $\mathbb{R}^1$ without introducing a cut (boundary). Any continuous mapping $f: \mathcal{X} \to \mathbb{R}$ attempting to represent angle values will inevitably suffer from a cliff where the output wraps from $+180^\circ$ to $-180^\circ$. By outputting $(\hat{s}, \hat{c}) \in \mathbb{R}^2$, the neural network output space is topologically homeomorphic to the ambient space containing $S^1$.
2. **Smooth Loss Landscape**: The loss operates directly in unit-circle coordinate space:
   $$\mathbf{v}_{\text{target}} = (\sin(\theta_{\text{rad}}), \cos(\theta_{\text{rad}}))$$
   $$\hat{\mathbf{u}} = \frac{\hat{\mathbf{v}}}{\|\hat{\mathbf{v}}\|_2 + \epsilon}$$
   $$\mathcal{L} = \|\hat{\mathbf{v}} - \mathbf{v}_{\text{target}}\|^2_2 + (1.0 - \hat{\mathbf{u}} \cdot \mathbf{v}_{\text{target}})$$
   The second term $(1 - \cos(\hat{\theta} - \theta))$ behaves as $\frac{\Delta\theta^2}{2}$ for small errors (Smooth L2) and plateaus smoothly at $2.0$ for complete inversion ($\Delta\theta = 180^\circ$), ensuring well-behaved gradients for all sample orientations.

---

## 4. Implementation Details in V4C Codebase

### Encoding & Decoding (`src/models/headpose/headpose_estimator.py`)
- `encode_yaw_sin_cos(yaw_deg)`:
  $$\text{rad} = \text{deg} \cdot \frac{\pi}{180^\circ}, \quad \mathbf{v} = (\sin(\text{rad}), \cos(\text{rad}))$$
- `decode_yaw_sin_cos(sin_val, cos_val)`:
  $$\text{rad} = \text{atan2}(\sin, \cos), \quad \text{deg} = \text{rad} \cdot \frac{180^\circ}{\pi}$$
  Followed by `canonicalize_yaw(deg)` to guarantee canonical output in $[-180^\circ, +180^\circ)$.
- **Strict Validation**: All helpers explicitly reject `NaN`, `+Inf`, and `-Inf` with `ValueError`.

### Baseline Model Architecture (`ResNet18Yaw`)
- Backbone: Standard ResNet18 feature extractor (512-dim avgpool).
- Head: `nn.Linear(512, 2)` producing unconstrained vector $\hat{\mathbf{v}} = (\hat{s}, \hat{c})$.
- Forward pass normalizes $\hat{\mathbf{v}}$ to unit sphere and decodes to continuous canonical yaw in degrees via native `torch.atan2`.
- Output: `(raw_vec, canonical_yaw)`.

---

## 5. Verification & Test Evidence

All mathematical properties have been verified with automated unit tests in `tests/test_headpose_angle_canonicalization.py`:
- `test_shortest_angular_difference_exact_cases`: Passed (all 5 exact prompt cases verified).
- `test_shortest_angular_difference_signed_and_arrays`: Passed (NumPy and Torch vectorization verified).
- `test_shortest_angular_difference_nan_inf_rejection`: Passed (strict exception on invalid floats).
- `test_circular_mae`: Passed across $\pm 180^\circ$ boundary.
- `test_encode_decode_yaw_sin_cos`: Passed round-trip tests with precision $\le 10^{-4}$ degrees.
- `test_resnet18_yaw_circular_architecture`: Passed shape and range invariants.

**Conclusion**: The circular sin/cos regression design is mathematically complete, verified by test suite, and authorized for head-pose model training.
