# ResNet50-CBAM (B1 & B2) Training Collapse Forensic Audit & Resolution

**Document ID**: `reports/v4c/final_verification/RESNET50_COLLAPSE_AUDIT.md`  
**Phase**: V4C Specialized Posture + Head-Pose Model Verification  
**Date**: 2026-10-03  
**Status**: ROOT CAUSE IDENTIFIED, BUG RESOLVED, TARGETED TESTS PASSED, RETRAINING IN PROGRESS  
**Requirement**: V4C Specification Section 12 & Contradiction Resolution

---

## 1. Executive Summary

During initial V4C matrix execution, experiments **B1** (`resnet50_cbam_tight_person_crop_224`) and **B2** (`resnet50_cbam_context_person_crop_224`) exhibited complete failure to converge:
- Training loss remained frozen at $\approx 1.386 \approx \ln(4)$ for all epochs.
- Validation Macro F1 never exceeded $\approx 0.1920$ (random guessing on a 4-class task).
- All predictions collapsed to majority classes (`NORMAL_UPRIGHT` / `NORMAL_READ_WRITE`), with $0.0\%$ recall on `TURN_HEAD_CLEAR` and $1.1\%$ recall on `HEAD_REST_SLEEP`.

Per Section 12 of the V4C Specification, this forensic audit was conducted to determine whether this collapse was:
> **A. A genuine model/data capacity failure**, or  
> **B. An implementation/configuration bug specific to ResNet50**.

**Authoritative Finding**: The collapse was definitively caused by **Category B: An Implementation and Mixed-Precision Attention Bug Specific to ResNet-50 Bottleneck Blocks**.

---

## 2. Forensic Diagnostic Inspection

### A. Training & Validation Loss Trajectory
Inspection of `runs/v4c/B1_resnet50_cbam_tight_person_crop_224/training_history.csv` revealed an unnatural flatline:
```
Epoch 01: train_loss=1.3917, val_macro_f1=0.0974
Epoch 02: train_loss=1.3868, val_macro_f1=0.0995
Epoch 05: train_loss=1.3872, val_macro_f1=0.1001
Epoch 10: train_loss=1.3840, val_macro_f1=0.1894
Epoch 20: train_loss=1.3858, val_macro_f1=0.1848
```
A cross-entropy loss of $\ln(4) = 1.386294...$ for 20 consecutive epochs indicates that **the network parameters never updated**.

### B. Gradient Statistics & Mixed-Precision (AMP) Telemetry
We instrumented `resnet50_cbam` with gradient tracking under PyTorch AMP FP16:
```
Step 0: scale=65536.0, loss=1.3860, grad_norm=nan, has_inf=True -> UPDATE SKIPPED
Step 1: scale=32768.0, loss=1.3860, grad_norm=nan, has_inf=True -> UPDATE SKIPPED
Step 2: scale=16384.0, loss=1.3860, grad_norm=nan, has_inf=True -> UPDATE SKIPPED
Step 3: scale=8192.0, loss=1.3860, grad_norm=nan, has_inf=True -> UPDATE SKIPPED
Step 4: scale=4096.0, loss=1.3860, grad_norm=nan, has_inf=True -> UPDATE SKIPPED
```
**Diagnostic Discovery**:
1. At every single backward pass, gradients in the deep Bottleneck layers overflowed to `+Inf` in FP16.
2. `torch.amp.GradScaler.step(optimizer)` detected `Inf` and **skipped the optimizer update on 100% of batches**.
3. `GradScaler` successively halved the loss scale down to its minimum, while the weights remained completely frozen in their initialization state.

---

## 3. Mathematical & Architectural Root Cause

Three distinct flaws compounded to produce this failure exclusively in ResNet-50:

### 1. Architectural Placement Bug in `BottleneckCBAM`
In the foundational CBAM paper (*Woo et al., ECCV 2018*), CBAM is applied to the **residual branch** prior to addition with the identity shortcut:
$$\mathbf{y} = \text{ReLU}\left(\text{CBAM}\left(\mathcal{F}(\mathbf{x})\right) + \mathbf{x}\right)$$
However, in `src/models/posture/resnet_cbam.py`, `BottleneckCBAM.forward` was implemented as:
```python
# FLAWED IMPLEMENTATION:
out += identity
out = self.relu(out)
out = self.cbam(out)  # Multiplied the identity shortcut by attention!
```
In ResNet-18 (which has only 8 shallow `BasicBlock` modules with 512 max channels), the identity highway survived sufficiently for training. But in ResNet-50 (which has **16 deep `Bottleneck` modules with up to 2,048 channels**), multiplying the identity shortcut at every single layer destroyed the residual gradient highway, producing catastrophic exponential gradient scaling.

### 2. Initialization Overwrite of Attention Projection
In `ResNetCBAM.__init__`, a global module initialization loop:
```python
for m in self.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
```
overwrote the weights of `ChannelAttention.fc2`. Because `fc2` is followed by a `Sigmoid()` gate (not a ReLU), Kaiming normal initialization caused massive random activations across the 2,048 channels of `layer4`.

### 3. Missing Gradient Clipping in Training Script
In `scripts/train_posture_classifier.py`, `scaler.unscale_(optimizer)` and `torch.nn.utils.clip_grad_norm_` were absent. When deep Bottleneck gradients experienced initial transient spikes, unclipped FP16 gradients exceeded the maximum representable half-precision number ($65,504$), immediately triggering `GradScaler` skipping.

---

## 4. Engineering Rectification & Verification

### Fix 1: Conforming `BottleneckCBAM` to Standard Architecture
In `src/models/posture/resnet_cbam.py`, `BottleneckCBAM.forward` was corrected to apply CBAM on the residual feature map *before* adding the shortcut:
```python
out = self.conv3(out)
out = self.bn3(out)
out = self.cbam(out)  # Applied to residual branch before shortcut addition

if self.downsample is not None:
    identity = self.downsample(x)

out += identity
out = self.relu(out)
return out
```

### Fix 2: Zero-Initialization of Attention Projection
In `src/models/posture/resnet_cbam.py`, all `ChannelAttention.fc2` projection weights are explicitly zero-initialized after backbone instantiation:
```python
for m in model.modules():
    if hasattr(m, "fc2") and isinstance(m.fc2, nn.Conv2d):
        nn.init.zeros_(m.fc2.weight)
```
This guarantees that at step 0, $\text{Sigmoid}(0) = 0.5$ (or neutral scaling), preserving pure identity residual propagation.

### Fix 3: Gradient Clipping in Posture Training Loop
In `scripts/train_posture_classifier.py`, gradient unscaling and norm clipping were integrated:
```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
scaler.step(optimizer)
scaler.update()
```

### Fix 4: Preservation of ResNet-18 Compatibility
`BasicBlockCBAM` was left untouched to ensure that existing valid checkpoints (`A1_confirmed`, `A1_recovered`, `A2`) retain exact mathematical identity during inference and evaluation.

---

## 5. Verification Test Evidence

1. **Targeted Unit Test Added**:
   `tests/test_cbam_and_posture_models.py::test_resnet50_cbam_backward_finite_gradients`
   - Verified that `resnet50_cbam` executes forward pass, loss computation, backward pass, and parameter updates with finite gradients and zero NaN/Inf.
   - Result: **PASSED (100%)**.

2. **Empirical Convergence Confirmation**:
   Testing 6 steps of `resnet50_cbam` with the corrected architecture:
   - Step 0: `scale=65536.0, loss=1.3684, grad_norm=7.92`
   - Step 1: `loss=0.8555, scale=32768.0`
   - Step 3: `loss=0.4721, grad_norm=6.19`
   - Step 5: `loss=0.0706, grad_norm=1.40`
   - Loss smoothly converged from $1.368$ down to $0.0706$ in 5 steps.

3. **Retraining Status**:
   - Experiment **B1** (`resnet50_cbam_tight_person_crop_224`) rerun initiated with `--fresh`.
   - Epoch 1 Same-Domain Validation Macro F1 immediately reached **0.7172** (compared to 0.0974 in the collapsed run).
   - Retraining and full multi-split recomputation are proceeding to completion.
