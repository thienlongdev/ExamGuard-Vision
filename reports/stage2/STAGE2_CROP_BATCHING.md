# Stage 2 Crop Scheduling & Vectorized Batching
**Phase**: Stage 2 Full End-to-End Orchestration  
**Status**: Authoritative Scheduling Specification  
**Date**: 2026-10-03  

---

## 1. Cadence-Driven Scheduling

Executing deep neural networks for every person on every video frame creates severe compute bottlenecks. Stage 2 employs deterministic, cadence-driven per-track scheduling configured in `configs/stage2_pipeline.yaml`:

| Modality | Model | Configured Cadence | Interval | Rationale |
| :--- | :--- | :---: | :---: | :--- |
| **Object Detection** | Stage 1 YOLO | 30.0 Hz | 33.3 ms | Tracks motion, bounding boxes, and student entry/exit |
| **Posture Classification** | MobileNetV3-Small | 10.0 Hz | 100.0 ms | Sufficient to capture posture transitions while saving 66% compute |
| **Head-Pose Estimation** | HopeNet-Yaw | 6.0 Hz | 166.7 ms | Balances yaw orientation tracking against heavy ResNet50 backbone |
| **Phone Spatial Association**| Geometric Associator | 6.0 Hz | 166.7 ms | Matches phone detection cadence |

Between scheduled inferences, the sliding-window buffer bridges active probabilities to avoid treating skipped frames as negative evidence.

---

## 2. Dynamic GPU Batching & Vectorization

Instead of executing one CUDA call per student, `CropScheduler` collates all eligible student crops within the current scheduling cycle into dynamic tensors:
- **Posture Batch**: `(B_p, 3, 224, 224)` up to `max_posture_batch_size: 32`
- **Head-Pose Batch**: `(B_h, 3, 224, 224)` up to `max_headpose_batch_size: 32`

### Exact Track-to-Batch Index Mapping
```python
posture_scheduled_tracks = [(track_id, tensor), ...]
t_ids = [item[0] for item in batch_slice]
tensors = torch.stack([item[1] for item in batch_slice], dim=0).to(device)

with torch.inference_mode():
    outputs = posture_predictor.predict_tensor(tensors, student_ids=t_ids)

for i, out in enumerate(outputs):
    track_id = t_ids[i]
    posture_results[track_id] = cue
```
This guarantees that prediction outputs can never be misattributed to the wrong track ID, verified by `test_batch_track_mapping_preservation`.

---

## 3. Capability Gating & Sub-Resolution Policy

1. **Posture Crop Gating**: Person crops with height $< 60\text{ px}$ are marked `SUB_RESOLUTION` and assigned `status = UNAVAILABLE`.
2. **Head Crop Gating**: If head region dimensions are $< 20\text{ px}$, head-pose estimation is gated with `HeadPoseSupportStatus.FACE_UNRESOLVABLE` and assigned `status = UNAVAILABLE`. Yaw is strictly set to `None`—**never** defaulted to `yaw = 0.0`.
3. **Resolution Policy**: Primary posture inference runs at **224x224**. The 320x320 alternative remains gated **OFF** by default.
