# Stage 2 Event & Evidence Lifecycle
**Phase**: Stage 2 Full End-to-End Orchestration  
**Status**: Authoritative Event & Evidence Specification  
**Date**: 2026-10-03  

---

## 1. Event State Machine & Deduplication

Every observable behavior family (`SUSTAINED_HEAD_REST`, `SUSTAINED_LATERAL_HEAD_ORIENTATION`, `PHONE_ASSOCIATED`, `DISCUSSION_CANDIDATE`, `STANDING`) is managed by an independent `TrackEventStateMachine`:

```
           +----------------+
           |    INACTIVE    |
           +-------+--------+
                   | score >= enter_thresh & not vetoed
                   v
           +----------------+
           |   CANDIDATE    | <---+ (score drops or vetoed: return to INACTIVE)
           +-------+--------+
                   | duration >= min_candidate_duration
                   v
           +----------------+
           |     ACTIVE     | ---> Generates OPEN action
           |   (Episode)    | ---> Generates UPDATE action (duration advances)
           +-------+--------+
                   | score < exit_thresh or vetoed
                   v
           +----------------+
           |    COOLDOWN    | ---> Generates CLOSE action
           +-------+--------+
                   | elapsed >= cooldown_seconds
                   v
           +----------------+
           |    INACTIVE    |
           +----------------+
```

### Deduplication Guarantee
A continuous physical episode (e.g. resting head on desk for 15 seconds) generates exactly **1 OPEN**, zero to several periodic **UPDATE** milestones, and **1 CLOSE** transition. It **never** generates hundreds of duplicate event rows.

---

## 2. Evidence Manager Actions

`IntegratedEvidenceManager` connects directly to event state transitions:

1. **OPEN Transition**:
   - Triggers asynchronous JPEG capture of the opening frame (`{event_id}_open.jpg`).
   - Starts rolling clip extraction from the circular pre-event buffer (`pre_event_seconds: 3.0s`).
   - Attaches `open_snapshot_path` and `clip_path` to event metadata.
2. **UPDATE Transition**:
   - If the event risk escalates to `HIGH` risk level, captures an escalation snapshot (`{event_id}_escalation.jpg`).
3. **CLOSE Transition**:
   - Finalizes rolling MP4 video clip with post-event padding (`post_event_seconds: 3.0s`).
   - Saves audit metadata JSON to `storage/evidence/metadata/{event_id}_metadata.json`.
   - Cleans up internal tracking dictionaries to prevent memory leaks.

---

## 3. Privacy & Storage Bounding

- **No Identity Recognition**: All evidence is keyed strictly by anonymous `track_id`, `event_id`, and `camera_id`. Facial recognition and student identification are prohibited.
- **Snapshot Cap**: Enforces `max_snapshots_per_event: 3` to prevent disk saturation.
- **Asynchronous Disk I/O**: Snapshots and MP4 encoding execute in background thread pools, isolating file I/O latency from inference loops.
