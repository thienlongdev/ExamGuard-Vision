# Stage 2 Tracking Integration & Identity Continuity
**Phase**: Stage 2 Full End-to-End Orchestration  
**Status**: Authoritative Tracking Specification  
**Date**: 2026-10-03  

---

## 1. Tracking Responsibilities & Ownership Key

ByteTrack (`src.tracking.bytetrack.ByteTrackTracker`) provides persistent multi-object tracking across video frames. The assigned integer `track_id` serves as the universal ownership key across the entire pipeline:
- Posture inference history
- Continuous head-pose trajectory
- Secondary phone association
- Temporal sliding window buffer
- Event lifecycle state machines
- Snapshot and video evidence records

```
             Track ID (Universal Key)
                        |
      +-----------------+-----------------+
      |                 |                 |
      v                 v                 v
Posture State    Head-Pose State    Phone State
      |                 |                 |
      +-----------------+-----------------+
                        |
                        v
              UnifiedTrackUpdate
                        |
                        v
            TrackEventStateMachine
                        |
                        v
              Evidence & Alerts
```

---

## 2. Track State Transition Policies

| State Transition | Detection Condition | Pipeline Action | Continuity Guarantee |
| :--- | :--- | :--- | :--- |
| **New Track** | Unmatched high-confidence detection ($conf \ge 0.50$) | Allocate new `TrackCadenceState`, initialize temporal buffer | Clean start, no inherited history |
| **Temporary Occlusion** | Unmatched track for $< 2.0\text{ s}$ | Maintain Kalman filter position prediction, bridge posture via sliding window | Preserves episode continuity through momentary dips |
| **Continuity Break** | Frame gap $> 2.0\text{ s}$ ($>\text{tolerance}$) | Force close active events (`TRACK_CONTINUITY_BREAK`), mark continuity invalid | Prevents event state jumping to a different student |
| **Track Expiration** | Unseen for $> 15.0\text{ s}$ ($>\text{eviction}$) | Evict track buffer, force close active events, delete scheduler state | Eliminates dead-track memory accumulation |

---

## 3. ID Switch Protection

To prevent alert pollution caused by identity swaps:
1. `MultiCueFusionEngine` checks `check_identity_continuity(track_id, timestamp)`. If a large physical or temporal gap occurs, `continuity_valid = False` immediately closes active events.
2. Expired tracks are purged atomically across `TemporalBuffer`, `EventEngine`, and `CropScheduler`.
