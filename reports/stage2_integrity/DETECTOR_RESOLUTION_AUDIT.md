# Detector Resolution & Capability Scale Audit

## 1. Resolution Decoupling & Provenance Reconciliation

Historically, Stage 2 configurations assigned a uniform 640 px resolution across all detector components. This was scientifically incorrect because:
1. `models/trained/stage1_5_best.pt` (custom 5-class student behavior model) was physically trained, validated, and certified at **768 px**.
2. Reducing `stage1_5_best.pt` to 640 px without an empirical labeled revalidation study creates an unverified runtime variant (`RUNTIME_RESOLUTION_VARIANT_UNVALIDATED_FOR_ACCURACY`).
3. General object detection via COCO-pretrained `yolo26m.pt` is standardized and certified at **640 px**.

### Corrective Architecture
The pipeline configuration (`configs/stage2_pipeline.yaml`) now explicitly decouples perception branches:
- `general_object_detector.image_size: 640`
- `macro_behavior_detector.image_size: 768`

Neither branch forces its resolution onto the other.

---

## 2. Scale Gating & Capability Boundary Reconciliation

V4C empirical evaluations established that very small person crops suffer severe feature degradation and unreliable posture/headpose predictions. The previously permissive thresholds (e.g., person height >= 60 px treated as full confidence) have been reconciled with V4C findings:

| Category | Bounding Box Condition | Status Output | Reliability Weight | Action |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Posture Support** | Person height $\ge 120$ px | `POSTURE_PRIMARY_224_ELIGIBLE` | 1.00 | Full-confidence primary 224 inference |
| **Degraded Posture Support** | $60 \text{ px} \le \text{height} < 120 \text{ px}$ | `POSTURE_LOW_RESOLUTION` | 0.60 | Reliability discounted by 40% |
| **Sub-Resolution Posture** | Person height $< 60$ px | `NOT_ELIGIBLE` / `UNAVAILABLE` | 0.00 | Bypassed; no inference evaluated |
| **Primary Head-Pose Support** | Head width $\ge 25$ px and height $\ge 25$ px | `WITHIN_PRIMARY_SUPPORT` | 1.00 | Standard HopeNet-Yaw estimation |
| **Unresolvable Head-Pose** | Head dimension $< 25$ px | `FACE_UNRESOLVABLE` | 0.00 | Bypassed; yaw remains `None`, never zero |

### Gating Policies Enforced
- **Missing Head-Pose is UNAVAILABLE**: When a face is unresolvable due to distance, extreme occlusion, or sub-resolution, `yaw_deg` is explicitly set to `None` with `support_status = FACE_UNRESOLVABLE`. It is **strictly prohibited** to impute `yaw = 0.0°`.
- **Adaptive 320 Fallback**: Kept `OFF` by default in production configurations until specifically certified for the deployment camera viewpoint.
- **Config-Driven**: All thresholds are read from `configs/stage2_pipeline.yaml` (`min_person_height_px: 120`, `low_res_person_height_px: 60`, `min_head_dimension_px: 25`).

---

## 3. Verification Verdicts
- `DETECTOR_RESOLUTION_PROVENANCE_VERIFIED = YES`
- `SCALE_GATING_CORRECTNESS_VERIFIED = YES`
- `IMPUTATION_DEFECTS_RESOLVED = YES`
