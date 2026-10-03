# Exam Suspicious Behavior Detection System

An enterprise-oriented, multi-stage computer vision and temporal behavior analysis platform designed for examination room monitoring.

> **CRITICAL ARCHITECTURAL PRINCIPLE**:  
> This system **NEVER** classifies a student as "cheating" directly from a single frame or detection.  
> The AI detects **observable behaviors and physical objects** (e.g., cell phone, head turn, leaving seat).  
> Multi-object tracking (ByteTrack) + temporal sliding-window analysis + configurable rule engines produce **"suspicious events"** with calibrated risk levels (`LOW`, `MEDIUM`, `HIGH`).  
> A **human invigilator** makes the final judgment (Confirm / Dismiss).

---

## 1. System Architecture

```
Video Source (Webcam / RTSP CCTV / Video File)
    |
    v
VideoSource Abstraction (Timestamp & Frame Metadata)
    |
    v
Frame Sampling & Inference Rate Limiter
    |
    +--------------------------------+
    |                                |
    v                                v
Object Detector             Behavior Detector
(COCO: person, phone)       (Custom YOLO / Taxonomy)
    |                                |
    +---------------+----------------+
                    |
                    v
          ByteTrack Multi-Object Tracker
          (Persistent Student Track IDs)
                    |
                    v
        Secondary Object Association
        (Phone-to-Student Spatial Binding)
                    |
                    +------------------------+
                    |                        |
                    |                 Optional Head Pose
                    |            (Disabled by default for CCTV)
                    |                        |
                    +-----------+------------+
                                |
                                v
                    Signal Fusion Engine
                 (StudentObservation per ID)
                                |
                                v
                    Sliding Temporal Buffer
             (Duration, Frequency, State Transitions)
                                |
                                v
                      Behavior Rule Engine
                 (Configurable YAML Thresholds)
                                |
                                v
                        Risk / Suspicion Scorer
                     (LOW / MEDIUM / HIGH Levels)
                                |
                                v
                      Suspicious Event Manager
                    (Debounce & Cooldown Engine)
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
    Evidence Capture                       FastAPI Backend
(Async Snapshot + Rolling Clip)       (REST Endpoints & WebSocket)
                                                   |
                                                   v
                                          Invigilator Dashboard
                                            (Human Review UI)
```

---

## 2. Camera-Source Agnostic Design

> **Production vs. Development Note**:  
> "The demo uses a laptop webcam as a substitute video source.  
> The target deployment uses wall-mounted CCTV/IP cameras via RTSP.  
> The downstream AI pipeline is unchanged."

All video feeds flow through a unified `VideoSource` abstraction (`src/video/base.py`). Downstream processing (tracking, temporal rules, scoring, evidence capture) is completely decoupled from the camera hardware.

---

## 3. Directory Layout

```
DETECTOR-YOLO/
├── configs/
│   ├── camera.yaml            # Video source settings (webcam / RTSP CCTV / file)
│   ├── classes.yaml           # Object & behavior taxonomies + dataset mappings
│   ├── inference.yaml         # Detection confidence, target inference FPS, device
│   ├── tracking.yaml          # ByteTrack thresholds, buffer, and association params
│   ├── risk_rules.yaml        # Temporal behavior rules, thresholds, severity, debounce
│   └── evidence.yaml          # Snapshot and rolling clip storage parameters
├── src/
│   ├── video/                 # Camera-source agnostic video ingestion
│   │   ├── base.py            # VideoFrame & VideoSource ABC
│   │   ├── webcam.py          # WebcamSource (cv2.VideoCapture with direct metadata)
│   │   ├── video_file.py      # VideoFileSource (repeatable offline file testing)
│   │   ├── rtsp.py            # RTSPSource (RTSP IP camera with backoff reconnect)
│   │   └── factory.py         # Video source factory
│   ├── detection/             # Framework-independent detection layer
│   │   ├── types.py           # BBox, Detection, BehaviorDetection dataclasses
│   │   ├── object_detector.py # ObjectDetector ABC + YOLOObjectDetector (person, phone)
│   │   └── behavior_detector.py # BehaviorDetector ABC + YOLOBehaviorDetector adapter
│   ├── tracking/              # Persistent student tracking
│   │   ├── tracker.py         # Track dataclass & BaseTracker ABC
│   │   └── bytetrack.py       # ByteTrackTracker adapter with persistent student IDs
│   ├── analysis/              # Spatial association, temporal buffer, fusion
│   │   ├── object_association.py # Associates detected phones to student tracks
│   │   ├── head_pose.py       # HeadPoseEstimator ABC + OptionalHeadPose (graceful fallback)
│   │   ├── fusion.py          # FusionEngine -> StudentObservation per student
│   │   └── temporal_buffer.py # Timestamp-based sliding window history per track ID
│   ├── behavior/              # Rule engine & risk scoring
│   │   ├── rules.py           # BehaviorRuleEngine (YAML-driven duration/frequency rules)
│   │   ├── scorer.py          # RiskScorer (LOW, MEDIUM, HIGH scoring)
│   │   └── event_manager.py   # EventManager (deduplication, debounce, lifecycle, callbacks)
│   ├── evidence/              # Evidence capture
│   │   ├── snapshot.py        # Asynchronous JPEG snapshot saving
│   │   └── clip_recorder.py   # Rolling circular buffer for pre/post-event MP4 clips
│   ├── api/                   # FastAPI backend & WebSocket
│   │   ├── schemas.py         # Pydantic models for API responses & events
│   │   ├── websocket.py       # WebSocket ConnectionManager for real-time live event broadcast
│   │   ├── static_ui.py       # Embedded HTML/JS dashboard for human review & live status
│   │   └── main.py            # FastAPI app: /health, /api/events, /ws/events, human review
│   └── pipeline.py            # ExamMonitoringPipeline orchestrator
├── scripts/
│   └── run_demo.py            # Unified CLI entrypoint (--source, --serve, --show)
├── samples/
│   ├── generate_sample_video.py # Synthetic exam video generator for offline testing
│   └── sample_exam.mp4        # Sample test clip
├── tests/
│   ├── test_temporal_buffer.py    # Unit tests for duration, frequency, cleanup
│   ├── test_rules_scorer.py       # Unit tests for rule triggers, scoring, cooldowns
│   ├── test_object_association.py # Unit tests for phone-to-person geometric binding
│   ├── test_video_source.py       # Unit tests for video source abstraction
│   ├── test_api.py                # Unit tests for FastAPI REST endpoints
│   └── test_pipeline_smoke.py     # End-to-end integration smoke test with synthetic frames
├── requirements.txt
└── README.md
```

---

## 4. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.13 AMD64 on Windows)
- Git

### 1. Initialize Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. GPU / CUDA Acceleration (Optional)
If an NVIDIA GPU with CUDA is available:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
Then configure `device: "cuda"` or `device: "0"` in `configs/inference.yaml` or pass `--device cuda` on the CLI.

---

## 5. Running the System

### A. Laptop Webcam Demo (Development)
Runs the live webcam feed with real-time OpenCV annotations and the web dashboard:
```powershell
python scripts/run_demo.py --source webcam --input 0 --show --serve --port 8050
```
Open your browser at:
`http://localhost:8050` to access the real-time Invigilator Review Dashboard.

### B. Offline Video File Testing
Runs repeatable testing on an exam recording:
```powershell
python scripts/run_demo.py --source video --input samples/sample_exam.mp4 --show --serve --port 8050
```

### C. Wall-Mounted School CCTV / RTSP Stream (Production)
Connects directly to an IP camera in an examination room:
```powershell
python scripts/run_demo.py --source rtsp --input "rtsp://admin:password@192.168.1.100:554/stream1" --serve --port 8050
```

---

## 6. Running Tests

Run the complete automated test suite (all unit tests, spatial association, temporal buffer, rule engine, and integration smoke tests):
```powershell
pytest tests/ -v
```

---

## 7. Model Conventions & Future Training Roadmap

- **Object Detector**: Default `yolo11n.pt` / `yolov8n.pt` downloaded on first run and cached locally.
- **Behavior Detector**: Adapter configured in `src/detection/behavior_detector.py`. Looks for `models/trained/behavior_best.pt`. If missing, degrades gracefully into a heuristic adapter with an explicit non-faking warning.
- **Stage 1 Training (Upcoming Phase)**:
  - Dataset: CCTV Exam Monitor + SCB-Dataset3 / SCBehavior.
  - Direction-safe augmentations (`fliplr: 0.0`, `flipud: 0.0`).
- **Stage 2 Fine-Tuning (Upcoming Phase)**:
  - Target school CCTV camera angle domain adaptation with lower learning rate.

---

## 8. Limitations & Scope

1. **Human Confirmation Requirement**: All automated detections are strictly alerts for human review. Status defaults to `new`; only human invigilators can transition an alert to `confirmed` or `dismissed`.
2. **Head Pose Constraint**: MediaPipe / FaceMesh is disabled by default (`head_pose.enabled: false`) because faces on wall-mounted CCTV cameras are typically <50 pixels, where gaze/eye-tracking is physically unreliable.
3. **Hardware Fallback**: When no CUDA GPU is detected, the pipeline automatically runs in CPU mode with configurable frame sampling (`target_inference_fps: 12.0`) to maintain real-time responsiveness.

---

## 9. Dataset Pipeline & Model Training Infrastructure

```
Raw Datasets (datasets/raw/)
    ↓
Audit & Validation (training/inspect_dataset.py, training/validate_dataset.py)
    ↓
Visual Class Inspection (training/visual_sample_inspector.py)
    ↓
Explicit Class Mapping (configs/dataset_mapping.yaml, training/class_mapping.py)
    ↓
Normalization & Grouped Split (training/split_dataset.py, training/build_manifest.py)
    ↓
Stage 1 Foundation Training (training/train_stage1.py, configs/train_stage1.yaml)
    ↓
Stage 2 Domain Adaptation (training/train_stage2.py, configs/train_stage2.yaml)
    ↓
Evaluation & Domain Breakdown (training/evaluate.py)
```

### Target Domain vs. Demo Source
- **Training Target**: Fixed wall-mounted examination room CCTV cameras (high diagonal angle, multiple students, occlusion, small person bounding boxes <5% of frame).
- **Demo / Testing Source**: Laptop webcam (strictly for local interactive testing and invigilator dashboard visualization). Models must never be trained or optimized exclusively on close-up webcam footage.

### Direction-Sensitive Augmentation Constraint
All Stage 1 and Stage 2 training configurations strictly enforce:
```yaml
fliplr: 0.0
flipud: 0.0
```
Horizontal flipping transforms lateral movement semantics (e.g., leaning/glancing left vs. right) and corrupts directional behavior learning unless a custom label-aware permutation pipeline is implemented.

