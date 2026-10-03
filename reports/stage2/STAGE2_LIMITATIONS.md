# Stage 2 Scientific & Engineering Limitations Report

## 1. Ethical & Legal Boundaries
1. **Observable Physical Evidence Only**: The software is an automated observation tool designed to assist human invigilators. It does not possess cognitive or legal capability to assess intent.
2. **Prohibited Terminology**: Under strict product rules, the system NEVER introduces or infers:
   - `CHEATING`, `CHEATER`, `GUILTY`, or `FRAUD`
   - These terms are permanently forbidden from model classes, risk classes, database columns, API payloads, and UI displays.
3. **Sole Human Authority**: All disciplinary assessments and decisions remain the exclusive responsibility of human invigilators.

---

## 2. Model & Algorithmic Scientific Limitations

### 2.1 Posture Perception Ceiling
- The primary posture classifier (`MobileNetV3-Small 224`) achieves 95.83% test accuracy on static cropped images.
- However, under severe occlusions (e.g., student seated behind tall desktop monitors or obscured by adjacent students), posture misclassifications occur.
- **Sleep Recall Ceiling**: In the V4D isolated evaluation, temporal clip recall for `HEAD_REST_SLEEP` reached **85.0% (17/20)**. The 3 missed clips failed because the static perception model never produced consecutive `HEAD_REST_SLEEP` predictions above the confidence threshold, demonstrating that temporal fusion is strictly bounded by perception accuracy.

### 2.2 Yaw Angle Physical Interpretation Limits
- Head-pose yaw represents **horizontal angular orientation only** relative to the camera vector.
- **Strict Invariance Rule**: Yaw angle **MUST NOT** be used to infer:
  - Head-down posture
  - Sleeping or resting
  - Reading examination papers
  - Writing answers
- Head-pose networks degrade significantly on rear-view angles ($|\text{yaw}| > 90^\circ$) or when the face occupies fewer than $20\times 20$ pixels. In these regimes, yaw is declared `UNAVAILABLE` rather than estimated as zero.

### 2.3 Phone Spatial Association Caveats
- Mobile phones are detected via the full-frame object detector and associated with student bounding boxes via 2D bounding box geometry and center distance.
- In dense seating arrangements where students sit shoulder-to-shoulder, a phone placed on a shared desk may be equidistant between two candidates, triggering `AMBIGUOUS_ASSOCIATION`.
- The system correctly suppresses false alarms by rejecting ambiguous associations, but this means subtle phone usage concealed under a desk or between students may remain unassociated.

---

## 3. Evaluation & Validation Limitations

### 3.1 Unlabeled Test Video Protocols
- Physical classroom videos used during Stage 2 benchmarking (`writing (1).mp4`, `lecture (10).mp4`) are **unlabeled video streams**.
- In strict adherence to scientific integrity (Item 50):
  > Do NOT calculate precision/recall from unlabeled integration video.
  > These videos validate runtime throughput, memory bounds, and end-to-end integration stability ONLY.

### 3.2 Synthetic Load Scaling (Mode C)
- Mode C evaluates 20 simultaneous student tracks by tiling real student crops into a 4x5 spatial grid.
- This provides an accurate stress-test of GPU batching, memory footprint, and queue backpressure under high track counts.
- **Limitation**: Mode C is a **synthetic workload benchmark** and must NEVER be cited as physical classroom detection accuracy.

### 3.3 Scope of Calibrated Temporal Events
- Only `SUSTAINED_HEAD_REST` has undergone exhaustive raw-metric temporal calibration across 60 ground-truth video clips (20 positive, 40 negative).
- `SUSTAINED_LATERAL_HEAD_ORIENTATION` and `PHONE_ASSOCIATED` temporal events operate on sound engineering heuristics, but have not yet undergone large-scale multi-annotator temporal ROC calibration.

---

## 4. Deployment Readiness Verdicts

| Readiness Metric | Verdict | Justification |
| :--- | :--- | :--- |
| **READY_FOR_TARGET_CCTV_PILOT** | **YES** | The end-to-end pipeline operates reliably at line rate, maintains bounded memory, handles stream drops cleanly, exposes validated APIs/WebSockets, and passes all 146 unit/integration tests with bit-exact model checkpoints. |
| **PRODUCTION_READY** | **NO** | Production deployment requires formal on-site CCTV camera calibration, multi-room pilot trials, operator usability feedback, long-duration (24-hour) soak tests, and institutional privacy compliance review. |
