# Track State Metadata & Continuity Audit

## 1. Defect Description
In previous Stage 2 orchestration code, track state passed to the fusion layer was hardcoded on every single frame:
```python
track_age_frames = 1
time_since_seen_sec = 0.0
```
This hardcoding broke temporal continuity tracking. Tracks that existed for hundreds of frames appeared as brand new tracks of age 1, preventing the system from distinguishing transient detection blips from persistent student presence.

---

## 2. Corrective Implementation
The pipeline orchestrator (`Stage2Pipeline`) now maintains a persistent per-track metadata store (`self._track_metadata: Dict[int, Dict[str, Any]]`).

### Lifecycle Management
1. **New Track Registration**:
   - `first_seen_timestamp = ts`
   - `last_seen_timestamp = ts`
   - `track_age_frames = 1`
   - `time_since_seen_sec = 0.0`
   - `continuity_status = "CONTINUOUS"`

2. **Active Track Update (per frame)**:
   - When detected in current frame:
     - `track_age_frames += 1`
     - `time_since_seen_sec = 0.0`
     - `last_seen_timestamp = ts`

3. **Occluded / Missing Track Update**:
   - When a track is lost or missing:
     - `time_since_seen_sec = ts - last_seen_timestamp`
     - If `time_since_seen_sec > continuity_tolerance_sec` (2.0s):
       `continuity_status = "DISCONTINUOUS"`

4. **Expired Track Cleanup**:
   - When tracks are confirmed lost and removed from ByteTrack:
     - All associated active events are gracefully closed.
     - Associated metadata is pruned from `self._track_metadata`.
     - Rolling crop scheduler state is cleared via `cleanup_expired_tracks()`.

---

## 3. Regression Testing Verification
Verified in `tests/test_stage2_integrity.py`:
- `test_real_track_age_and_time_since_seen`: Confirms that a track present over consecutive frames increments `track_age_frames` monotonically (e.g. Frame 1 age = 1, Frame 2 age = 2, Frame 3 age = 3), and when temporarily missing, `time_since_seen_sec` accurately reflects elapsed time.
- `test_memory_state_cleanup`: Confirms that when tracks expire, metadata dictionaries are completely evicted without memory leakage.

---

## 4. Status Flag
- `TRACK_STATE_INTEGRATION_FIXED = YES`
