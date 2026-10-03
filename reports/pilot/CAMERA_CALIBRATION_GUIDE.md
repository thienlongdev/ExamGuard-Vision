# CCTV CAMERA CALIBRATION & SITE ONBOARDING GUIDE

## 1. Overview
Before any examination hall CCTV camera may be ingested by the live monitoring pipeline, it must undergo offline physical calibration. Calibration measures real-world video stream characteristics, student visibility, resolution scale gating, and geometric viability without modifying neural network weights or inventing unmeasured physical distances.

---

## 2. Calibration Command Execution
Run the standalone calibration CLI:
```powershell
.\.venv\Scripts\python.exe scripts/calibrate_pilot_camera.py `
    --source "samples/sample_exam.mp4" `
    --camera-id "cam_hall_01_front" `
    --viewpoint "FRONT_OBLIQUE" `
    --frames 120 `
    --duration 60.0
```

### Supported Parameters
| Flag | Description | Default | Valid Range / Options |
| :--- | :--- | :--- | :--- |
| `--source` | Path to recorded video file, webcam index, or local simulated stream | `samples/sample_exam.mp4` | Local path or stream URI |
| `--camera-id` | Unique alphanumeric camera identifier | `cam_hall_01` | String `[a-zA-Z0-9_-]+` |
| `--source-type` | Ingestion transport protocol | `video_file` | `video_file`, `webcam`, `rtsp` |
| `--viewpoint` | Mounting perspective category | `FRONT_OBLIQUE` | `CEILING_HIGH`, `FRONT_OBLIQUE`, `SIDE_OBLIQUE`, `REAR_OBLIQUE`, `DESK_LEVEL`, `UNKNOWN` |
| `--frames` | Frame sample budget for statistical aggregation | `90` | 30 to 300 frames |
| `--duration` | Time window cutoff (seconds) | `60.0` | 10.0 to 300.0 seconds |
| `--output-dir` | Output artifacts destination | `runs/pilot/calibration/<camera_id>/` | Writable directory path |

---

## 3. Measured Physical Telemetry
During the calibration window, the calibrator samples:
1. **Stream & Network Delivery**:
   - Source resolution (width, height)
   - Actual received FPS vs nominal stream FPS
   - Inter-frame arrival interval distribution (jitter indicator)
   - Video decode latency distribution
2. **Perception & Tracking Visibility**:
   - Mean person detections per frame
   - Active student track count distribution
   - Mean track lifespan (continuity verification)
   - Person bounding box height and width distributions
   - Head region crop dimensions (proxy for facial yaw resolution)
   - Phone candidate bounding box dimensions
3. **Execution & Hardware Pressure**:
   - Ingestion queue depth
   - Dropped frames count under backpressure
   - Whole-loop end-to-end processing latency (Mean, P50, P90, P95, Max)
   - GPU VRAM allocated and reserved (MB)
   - Host process RSS memory (MB)

---

## 4. Empirical Capability Classification
The calibration engine maps observed physical dimensions directly into operational capability levels without fabricating accuracy metrics:

### Posture Capability
- **FULL**: $\ge 70\%$ of visible student tracks have bounding box height $\ge 120\text{ px}$.
- **LIMITED**: $20\% \le (\text{tracks with height } \ge 60\text{ px}) < 70\%$. Low-resolution classification activated with reduced fusion reliability weight.
- **UNAVAILABLE**: $< 20\%$ of student tracks have height $\ge 60\text{ px}$ or posture explicitly disabled in profile.

### Head-Pose Capability
- **FULL**: Front or oblique viewpoint, face resolvable, and $\ge 75\%$ of head crops $\ge 25\times 25\text{ px}$.
- **LIMITED**: Front or oblique viewpoint, with $30\% \le (\text{head crops } \ge 25\times 25\text{ px}) < 75\%$.
- **UNAVAILABLE**: `CEILING_HIGH` or `REAR_OBLIQUE` viewpoints (where facial yaw is physically obscured), or $< 30\%$ of head crops $\ge 25\times 25\text{ px}$.

### Phone Capability
- **FULL**: Phone bounding boxes detected with median width $\ge 25\text{ px}$ and height $\ge 25\text{ px}$.
- **LIMITED**: Median phone width $\ge 15\text{ px}$.
- **UNAVAILABLE**: Median phone width $< 15\text{ px}$ or phone branch disabled.

---

## 5. Camera Geometry Diagnostics & Operator Warnings
When the calibration toolkit detects geometric or visibility constraints, it flags them explicitly in the calibration report:
- `HIGH_ANGLE_WARNING`: Camera pitch exceeds $45^\circ$ downward or viewpoint is `CEILING_HIGH`. Head-pose yaw estimation is automatically gated off.
- `LOW_PERSON_PIXEL_COVERAGE`: $> 40\%$ of student bounding boxes are below $60\text{ px}$ in height. Recommends optical zoom or physical relocation.
- `LOW_HEADPOSE_COVERAGE`: Head regions are below the $25\text{ px}$ threshold required by the HopeNet continuous expectation model.
- `HIGH_OCCLUSION_RISK`: Excessive overlap detected between adjacent student bounding boxes.
- `PHONE_VISIBILITY_LIMITED`: Phone objects are below practical detection resolution ($< 20\text{ px}$).

---

## 6. Examination ROI & Seat Zone Setup
To ignore non-exam areas (such as entrance doors, hallways, or teacher podiums), configure `roi_polygon` in the camera profile:
```yaml
roi_polygon:
  - [120, 150]
  - [1800, 150]
  - [1800, 1020]
  - [120, 1020]
```
For individual desk localization and spatial evidence grouping, define optional seat zones:
```yaml
seat_zones:
  - zone_id: "ROW1_SEAT1"
    desk_polygon: [[150, 300], [350, 300], [350, 480], [150, 480]]
    seat_polygon: [[180, 460], [320, 460], [320, 600], [180, 600]]
```
*Note: Seat zones group spatial evidence and assist invigilator orientation; they NEVER bind biometric identities or student names.*
