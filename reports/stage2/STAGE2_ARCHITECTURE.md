# Stage 2 Architecture & Integrated Pipeline Design
**Phase**: Stage 2 Full End-to-End Orchestration  
**Status**: Authoritative Architecture Specification  
**Date**: 2026-10-03  

---

## 1. Architectural Overview

The Stage 2 Orchestration integrates video ingestion, deep neural object detection, multi-object tracking, cadence-driven GPU crop scheduling, spatial object association, temporal multi-cue fusion, event lifecycle management, risk aggregation, evidence recording, and REST/WebSocket API presentation into a unified real-time pipeline.

```
+-----------------------------------------------------------------------------------+
|                                   VideoSource                                     |
|                     (WebcamSource / VideoFileSource / RTSPSource)                 |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                             Frame Decode & Timestamp                              |
|          (Monotonic timestamp_sec, variable FPS invariance, frame dropping)       |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                               Full-Frame Detector                                 |
|          (Stage 1 YOLO: Person & Phone Detections @ 640x640, GPU CUDA)            |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                                ByteTrack Tracker                                  |
|         (Persistent Student Track IDs, Kalman Filtering, Lost Track Handling)     |
+-----------------------------------------+-----------------------------------------+
                                          |
                     +--------------------+--------------------+
                     |                    |                    |
                     v                    v                    v
+--------------------------+ +-------------------------+ +--------------------------+
|  Posture Crop Extraction | |   Head Crop Extraction  | | Phone Spatial Associator |
| & Batched GPU Inference  | | & Batched Head-Pose GPU | |  (Spatial Containment,   |
|  (MobileNetV3 224 @ 10Hz)| |   (HopeNet-Yaw @ 6Hz)   | |  IoU, Ambiguity Check)   |
+--------------------------+ +-------------------------+ +--------------------------+
                     |                    |                    |
                     +--------------------+--------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                               UnifiedTrackUpdate                                  |
|        (Canonical Data Contract per Track per Frame, Strictly Formatted)          |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                            V4D MultiCueFusionEngine                               |
|        (Sliding Window Buffer, Reliability Model, Competing Negative Evidence)    |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                           Event Lifecycle State Machine                           |
|       (Hysteresis Enter/Exit, Episode Deduplication, Emits OPEN/UPDATE/CLOSE)     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                                  Risk Aggregator                                  |
|         (Configured Risk Score 0-100, Provisional Levels: LOW, MEDIUM, HIGH)      |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                             Integrated Evidence Manager                           |
|       (Bounded Snapshots on OPEN/Escalation, Rolling MP4 Clips, Metadata JSON)    |
+-----------------------------------------+-----------------------------------------+
                                          |
                     +--------------------+--------------------+
                     |                                         |
                     v                                         v
+-----------------------------------------+ +---------------------------------------+
|             FastAPI Backend             | |          WebSocket Broadcaster        |
|    (/health, /api/cameras, /api/events, | |         (/ws/events: Real-Time        |
|     /api/system/status, /api/models)    | |            Lifecycle Stream)          |
+-----------------------------------------+ +---------------------------------------+
```

---

## 2. Immutable Scientific & Product Governance

1. **Observable Facts Only**: The system detects observable evidence. It **NEVER** declares `CHEATING`, `CHEATER`, `GUILTY`, or `FRAUD`. Disciplinary judgment rests exclusively with human invigilators.
2. **Frozen 4-Class Posture Ontology**:
   - `0: NORMAL_UPRIGHT`
   - `1: NORMAL_READ_WRITE`
   - `2: HEAD_REST_SLEEP`
   - `3: TURN_HEAD_CLEAR`
3. **Decoupled Phone Branch**: Phone detection remains an independent spatial association branch; never merged into posture taxonomy.
4. **Yaw Meaning**: Yaw represents continuous horizontal orientation only. It is **NEVER** used to infer vertical head-down, reading, writing, or sleeping.
5. **Capability Gating**: Sub-resolution crops or unresolvable faces are marked `UNAVAILABLE`, never fabricated as `yaw = 0.0`.

---

## 3. Subsystem Interoperability

- **Orchestration Layer** (`src/orchestration/stage2_pipeline.py`): Coordinates the frame loop, thread boundaries, and timing instrumentation.
- **Model Registry** (`src/orchestration/model_registry.py`): Singleton cache ensuring neural network checkpoints are loaded exactly once and warmed up on CUDA.
- **Crop Scheduler** (`src/orchestration/crop_scheduler.py`): Vectorizes inference by collating eligible student crops into GPU batches while strictly preserving track-to-batch index mapping.
- **Phone Associator** (`src/orchestration/phone_associator.py`): Enforces ambiguity gating; borderline proximity between multiple students prevents false positive alert attribution.
- **Fusion Engine** (`src/fusion/`): The sole authority on temporal state, buffering, and event transitions.
