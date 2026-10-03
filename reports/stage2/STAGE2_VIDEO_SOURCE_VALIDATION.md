# Stage 2 Video Source Validation & Ingestion Semantics
**Phase**: Stage 2 Full End-to-End Orchestration  
**Status**: Authoritative Ingestion Specification  
**Date**: 2026-10-03  

---

## 1. VideoSource Abstraction & Agnostic Interface

Stage 2 decouples all downstream perception and temporal processing from physical camera hardware via `src.video.base.VideoSource`. Every frame yields a canonical `VideoFrame`:

```python
@dataclass
class VideoFrame:
    frame: np.ndarray        # (H, W, 3) BGR image array
    timestamp: float        # Monotonic or source timestamp in seconds
    frame_idx: int          # Monotonically increasing counter
    fps: float              # Source nominal FPS
    width: int              # Width in pixels
    height: int             # Height in pixels
    source_id: str          # Unique camera identifier
```

---

## 2. Ingestion Source Implementations

| Source Type | Implementation Class | Timestamping Mechanism | Target Use Case |
| :--- | :--- | :--- | :--- |
| **Local Video File** | `VideoFileSource` | Precise source PTS / elapsed playback time | Offline benchmark, replay audit, regression |
| **USB / Laptop Webcam** | `WebcamSource` | Monotonic clock arrival (`time.monotonic()`) | Local live pilot testing |
| **RTSP Stream** | `RTSPSource` | Monotonic clock arrival with reconnect loop | Production exam room CCTV integration |

---

## 3. Real Timestamp Policy & Temporal Robustness

1. **Duration Semantics**: Downstream temporal buffers rely strictly on `timestamp_sec`. Durations are **never** calculated as `frame_count / nominal_FPS`.
2. **Variable FPS Invariance**: The pipeline is verified invariant to fluctuating FPS (15 Hz, 25 Hz, 30 Hz) in `test_fps_invariance_trigger_time`. Event activation triggers at exact physical time boundaries regardless of frame pacing.
3. **Stream Interruption & Reconnection**: RTSP reconnections re-establish the stream without resetting active track histories unless the gap exceeds `track_continuity_tolerance_sec` (2.0s). Gaps exceeding 2.0s cleanly force-close open events to prevent state jumping across different persons.
4. **Duplicate & Stale Timestamps**: `TrackObservationBuffer` detects near-duplicate timestamps ($\le 1.0\text{ ms}$) and updates observations in place rather than appending redundant records.
