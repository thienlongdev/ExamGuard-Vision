# V4D Multi-Cue Temporal Fusion Architecture Specification

**Document ID**: `reports/v4d/V4D_FUSION_ARCHITECTURE.md`  
**Phase**: Phase V4D — Temporal & Multi-Cue Fusion Layer  
**Date**: 2026-10-03  
**Status**: AUTHORITATIVE ARCHITECTURAL SPECIFICATION  

---

## 1. System Context & Architectural Role

The V4D Temporal & Multi-Cue Fusion Layer sits between upstream perception/tracking modules and downstream event/evidence storage. It transforms noisy, instantaneous single-frame predictions into temporally-stable, observable factual event streams.

```
+-------------------------------------------------------------------------+
|                       Surveillance Video Feed                          |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                  Upstream Perception & Tracking Layer                   |
|  - Full-Frame YOLO Detector (Students, Objects, Contraband)             |
|  - ByteTrack Associator (Persistent Track IDs)                          |
|  - MobileNetV3-Small Posture Crop Classifier (224x224 / 320x320)        |
|  - Head-Pose Estimator (HopeNet-Yaw [-99,+99) / ResNet18-Circular)      |
|  - Secondary Desk Contraband / Phone Spatial Associator                |
|  - Full-Frame Macro Behavior Detector (Stage 1.5 Discuss/Stand)         |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                    Capability & Scale Gating Engine                     |
|  - Camera Viewpoint Profile Enforcement (HIGH_ANGLE, FRONT_OBLIQUE)     |
|  - Resolution Eligibility (H >= 120px for Posture, 25x25px Head Crop)   |
|  - Head-Pose Domain Dispatch (HopeNet primary vs ResNet18 fallback)     |
|  - Missing Cue Safety: Explicit AVAILABLE / UNAVAILABLE / NOT_EVALUATED|
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                   Timestamp-First TemporalBuffer                        |
|  - Bounded Sliding Window per Track ID (max_samples=300, horizon=30s)  |
|  - Timestamp-based duration calculation (variable FPS, dropped frames)  |
|  - Out-of-order sorting & duplicate timestamp deduplication            |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                      Multi-Cue Fusion Engine                            |
|  - Correlated Cue Protection (posture turn + continuous yaw discount)  |
|  - Competing Evidence Veto (NORMAL_READ_WRITE suppresses SLEEP)         |
|  - Auditable Reliability Weighting Model (crop scale, blur, ambiguity)  |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                Temporal Event State Machine Engine                      |
|  - Lifecycle: INACTIVE -> CANDIDATE -> ACTIVE -> COOLDOWN -> INACTIVE   |
|  - Asymmetric Hysteresis (enter threshold != exit threshold)            |
|  - Event Deduplication (1 sustained episode = 1 event, not 300 rows)    |
|  - Track Identity Isolation (closure on tracking discontinuity)        |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                      Risk Aggregation Engine                            |
|  - Multi-cue concurrence escalation                                     |
|  - Single-frame noise protection (cap < 25.0, cannot reach MEDIUM/HIGH) |
|  - Temporal persistence factor & recurrence escalation                  |
|  - Risk Levels: LOW, MEDIUM, HIGH (Degree of evidence, NOT guilt)       |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                  Downstream Evidence & API Dispatch                     |
|  - Observable Fact Event Stream (/api/events, /ws/events)              |
|  - Evidence Capture Triggers (Snapshots, Video Clips)                   |
|  - Human Invigilator Review Interface (AI never declares guilt)         |
+-------------------------------------------------------------------------+
```

---

## 2. Core Package Organization (`src/fusion/`)

The fusion layer is organized as a decoupled, modular Python package:

| Module File | Purpose & Responsibilities |
| :--- | :--- |
| `src/fusion/types.py` | Strict dataclasses, Enums, frozen 4-class ontology, event families, risk levels, and contract schemas. |
| `src/fusion/capability.py` | Viewpoint capability profiles, resolution scale gating, and head-pose model dispatch. |
| `src/fusion/reliability.py` | Auditable evidence weighting, blur/scale penalties, and correlated cue discounting. |
| `src/fusion/temporal_buffer.py` | Timestamp-indexed sliding window buffer per track with out-of-order, duplicate, and gap resilience. |
| `src/fusion/cue_state.py` | Instantaneous and smoothed per-track multi-cue representation. |
| `src/fusion/fusion_engine.py` | Multi-cue fusion coordinator, competing evidence resolution, and bridge for intermittent sampling. |
| `src/fusion/event_engine.py` | Per-track event state machines, hysteresis, deduplication, and track expiration handling. |
| `src/fusion/risk_aggregator.py` | Maps duration, cue concurrence, reliability, and recurrence into calibrated risk scores and levels. |
| `src/fusion/replay.py` | 100% deterministic offline replay harness executing over recorded prediction traces. |
| `src/fusion/__init__.py` | Clean symbol exposure for integration. |

---

## 3. Strict Ontological Governance

In accordance with strict ethical and legal compliance:
1. **Zero Cheating Semantics**:
   - The words `CHEATING`, `CHEATER`, `FRAUD`, and `GUILTY` do not exist anywhere in the code, database, API schemas, or event outputs.
   - All events describe observable physical behaviors: `SUSTAINED_HEAD_REST`, `SUSTAINED_LATERAL_HEAD_ORIENTATION`, `PHONE_ASSOCIATED`, `DISCUSSION_CANDIDATE`, `STANDING`.
2. **Risk Is Not Guilt**:
   - Risk levels `LOW`, `MEDIUM`, `HIGH` represent the persistence and concurrence of observable suspicious indicators.
   - Disciplinary determination is strictly reserved for human exam invigilators.
