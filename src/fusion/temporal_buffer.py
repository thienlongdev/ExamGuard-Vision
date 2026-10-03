"""
V4D Timestamp-First Sliding Window Temporal Buffer
=================================================
Maintains bounded observation history per student track ID.
Strictly relies on timestamp_sec for all durations and windowing.
Robust against variable FPS, dropped frames, out-of-order timestamps,
duplicate timestamps, gaps, NaN/Inf inputs, and track expiration.
"""

from collections import deque
import bisect
import math
import logging
from typing import Dict, List, Optional, Tuple, Any

from src.fusion.types import (
    UnifiedTrackUpdate,
    ObservationStatus,
    PostureCue,
    HeadPoseCue,
    PhoneCue,
    MacroBehaviorCue,
)

logger = logging.getLogger(__name__)


class TrackObservationBuffer:
    """Sliding window history for a single persistent track ID."""

    def __init__(
        self,
        track_id: int,
        time_horizon_seconds: float = 30.0,
        max_samples: int = 300,
        duplicate_tolerance_sec: float = 0.001,
    ):
        self.track_id = track_id
        self.time_horizon_seconds = time_horizon_seconds
        self.max_samples = max_samples
        self.duplicate_tolerance_sec = duplicate_tolerance_sec

        # Ordered list of UnifiedTrackUpdate
        self._history: List[UnifiedTrackUpdate] = []
        self._timestamps: List[float] = []

    def push(self, update: UnifiedTrackUpdate) -> bool:
        """Insert a validated update, maintaining chronological order."""
        t = update.timestamp_sec
        if math.isnan(t) or math.isinf(t):
            logger.warning(f"Track {self.track_id}: Rejected non-finite timestamp {t}")
            return False

        if not self._timestamps:
            self._history.append(update)
            self._timestamps.append(t)
            return True

        # Check for duplicates or near-duplicates
        idx = bisect.bisect_left(self._timestamps, t)
        if idx > 0 and abs(self._timestamps[idx - 1] - t) <= self.duplicate_tolerance_sec:
            self._history[idx - 1] = update
            return True
        if idx < len(self._timestamps) and abs(self._timestamps[idx] - t) <= self.duplicate_tolerance_sec:
            self._history[idx] = update
            return True

        # Check for out-of-order insertion
        if idx == len(self._timestamps):
            self._history.append(update)
            self._timestamps.append(t)
        else:
            self._history.insert(idx, update)
            self._timestamps.insert(idx, t)

        # Enforce time horizon and max sample limits
        latest_t = self._timestamps[-1]
        cutoff_t = latest_t - self.time_horizon_seconds

        # Prune older than time horizon
        while self._timestamps and self._timestamps[0] < cutoff_t:
            self._timestamps.pop(0)
            self._history.pop(0)

        # Prune exceeding max samples
        while len(self._history) > self.max_samples:
            self._timestamps.pop(0)
            self._history.pop(0)

        return True

    @property
    def latest_timestamp(self) -> Optional[float]:
        return self._timestamps[-1] if self._timestamps else None

    @property
    def sample_count(self) -> int:
        return len(self._history)

    def get_window(self, window_seconds: float) -> List[UnifiedTrackUpdate]:
        """Return updates within [latest_timestamp - window_seconds, latest_timestamp]."""
        if not self._history:
            return []
        latest_t = self._timestamps[-1]
        cutoff = latest_t - window_seconds
        start_idx = bisect.bisect_left(self._timestamps, cutoff)
        return self._history[start_idx:]

    def duration_in_state(
        self,
        predicate_fn,
        window_seconds: float = 10.0,
        continuous_only: bool = False,
    ) -> float:
        """Calculate elapsed duration (seconds) where predicate_fn(update) is True.
        
        Uses true timestamp differences, NOT frame counting.
        """
        window = self.get_window(window_seconds)
        if not window:
            return 0.0

        if len(window) == 1:
            return 0.1 if predicate_fn(window[0]) else 0.0

        if continuous_only:
            # Measure unbroken streak up to latest update
            if not predicate_fn(window[-1]):
                return 0.0
            streak_start = window[-1].timestamp_sec
            for i in range(len(window) - 1, -1, -1):
                if predicate_fn(window[i]):
                    streak_start = window[i].timestamp_sec
                else:
                    break
            return window[-1].timestamp_sec - streak_start

        # Cumulative duration within window
        duration = 0.0
        for i in range(len(window) - 1):
            curr_u = window[i]
            next_u = window[i + 1]
            if predicate_fn(curr_u):
                dt = next_u.timestamp_sec - curr_u.timestamp_sec
                if 0.0 < dt < 5.0:  # Ignore massive gaps
                    duration += dt
        return duration

    def smoothed_posture_probabilities(self, window_seconds: float = 0.5) -> Dict[str, float]:
        """Compute time-weighted mean probabilities for posture over recent window.
        
        If window has no posture observations (e.g. intermittent cadence), expands
        window up to 2.0s to avoid treating missing samples as negative evidence.
        """
        window = self.get_window(window_seconds)
        probs = {
            "NORMAL_UPRIGHT": 0.0,
            "NORMAL_READ_WRITE": 0.0,
            "HEAD_REST_SLEEP": 0.0,
            "TURN_HEAD_CLEAR": 0.0,
        }
        valid_updates = [u for u in window if u.posture.status == ObservationStatus.AVAILABLE and u.posture.probabilities]
        if not valid_updates and window_seconds < 2.0:
            # Expand window to bridge intermittent sampling (e.g. every 3 frames)
            window = self.get_window(2.0)
            valid_updates = [u for u in window if u.posture.status == ObservationStatus.AVAILABLE and u.posture.probabilities]

        if valid_updates:
            for u in valid_updates:
                for k in probs:
                    probs[k] += u.posture.probabilities.get(k, 0.0)
            for k in probs:
                probs[k] /= len(valid_updates)
        return probs


class ScopedTrackDict(dict):
    """Dictionary supporting both (camera_id, track_id) composite keys and integer track_id lookups."""

    def __getitem__(self, key: Any) -> TrackObservationBuffer:
        if super().__contains__(key):
            return super().__getitem__(key)
        if isinstance(key, int):
            for (cam, tid), val in self.items():
                if tid == key:
                    return val
        raise KeyError(key)

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: Any) -> bool:
        if super().__contains__(key):
            return True
        if isinstance(key, int):
            for k in self.keys():
                tid = k[1] if isinstance(k, tuple) else k
                if tid == key:
                    return True
        return False


class TemporalBuffer:
    """Multi-track sliding-window temporal observation coordinator with camera-scoping."""

    def __init__(
        self,
        time_horizon_seconds: float = 30.0,
        max_samples_per_track: int = 300,
        eviction_inactive_seconds: float = 15.0,
        track_continuity_tolerance_sec: float = 2.0,
        duplicate_tolerance_sec: float = 0.001,
    ):
        self.time_horizon_seconds = time_horizon_seconds
        self.max_samples_per_track = max_samples_per_track
        self.eviction_inactive_seconds = eviction_inactive_seconds
        self.track_continuity_tolerance_sec = track_continuity_tolerance_sec
        self.duplicate_tolerance_sec = duplicate_tolerance_sec

        self._tracks: ScopedTrackDict = ScopedTrackDict()
        self._last_seen: Dict[Tuple[str, int], float] = {}

    def _resolve_key(self, track_id: int, camera_id: str = "cam_0") -> Tuple[str, int]:
        return (camera_id, track_id)

    def push(self, update: UnifiedTrackUpdate) -> bool:
        """Push an update for a student track."""
        if not update.is_valid():
            return False

        t_id = update.track_id
        cam_id = getattr(update, "camera_id", "cam_0") or "cam_0"
        key = (cam_id, t_id)

        if key not in self._tracks:
            self._tracks[key] = TrackObservationBuffer(
                track_id=t_id,
                time_horizon_seconds=self.time_horizon_seconds,
                max_samples=self.max_samples_per_track,
                duplicate_tolerance_sec=self.duplicate_tolerance_sec,
            )

        success = self._tracks[key].push(update)
        if success:
            self._last_seen[key] = update.timestamp_sec
        return success

    def get_track_history(self, track_id: int, window_seconds: float, camera_id: str = "cam_0") -> List[UnifiedTrackUpdate]:
        """Retrieve recent updates for a track."""
        key = (camera_id, track_id)
        if key in self._tracks:
            return self._tracks[key].get_window(window_seconds)
        if track_id in self._tracks:
            return self._tracks[track_id].get_window(window_seconds)
        return []

    def check_identity_continuity(self, track_id: int, current_timestamp: float, camera_id: str = "cam_0") -> Tuple[bool, float]:
        """Check if track was lost or has a gap exceeding continuity tolerance."""
        key = (camera_id, track_id)
        last_t = self._last_seen.get(key)
        if last_t is None:
            # Fallback search
            for (c, tid), lt in self._last_seen.items():
                if tid == track_id:
                    last_t = lt
                    break
        if last_t is None:
            return True, 0.0
        gap = current_timestamp - last_t
        continuous = (gap <= self.track_continuity_tolerance_sec)
        return continuous, gap

    def evict_inactive_tracks(self, current_timestamp: float, camera_id: Optional[str] = None) -> List[int]:
        """Evict tracks not seen for eviction_inactive_seconds."""
        evicted = []
        for key, last_t in list(self._last_seen.items()):
            cam, t_id = key if isinstance(key, tuple) else ("cam_0", key)
            if camera_id is not None and cam != camera_id:
                continue
            if (current_timestamp - last_t > self.eviction_inactive_seconds) or (last_t - current_timestamp > 30.0):
                evicted.append(t_id)
                self._tracks.pop(key, None)
                self._last_seen.pop(key, None)
        return sorted(list(set(evicted)))

    def evict_expired(self, current_timestamp: float, camera_id: Optional[str] = None) -> List[int]:
        """Alias for evict_inactive_tracks."""
        return self.evict_inactive_tracks(current_timestamp, camera_id=camera_id)

    def active_track_ids(self, camera_id: Optional[str] = None) -> List[int]:
        ids = []
        for k in self._tracks.keys():
            cam, tid = k if isinstance(k, tuple) else ("cam_0", k)
            if camera_id is None or cam == camera_id:
                ids.append(tid)
        return sorted(list(set(ids)))

    def clear(self, camera_id: Optional[str] = None) -> None:
        """Clear all or camera-specific tracking observation state."""
        if camera_id is None:
            self._tracks.clear()
            self._last_seen.clear()
        else:
            to_remove = [k for k in list(self._tracks.keys()) if (isinstance(k, tuple) and k[0] == camera_id)]
            for k in to_remove:
                self._tracks.pop(k, None)
                self._last_seen.pop(k, None)

