"""Temporal buffer maintaining observation history per student track ID.

Uses timestamp-based sliding windows (NOT frame counts) to compute behavior
durations, transition frequencies, and patterns.
"""

from collections import deque
from dataclasses import dataclass
import logging
from typing import Dict, List, Optional
import numpy as np

from src.analysis.fusion import StudentObservation

logger = logging.getLogger(__name__)


class TemporalBuffer:
    """Sliding-window temporal observation history indexed by track ID."""

    def __init__(
        self,
        max_history_seconds: float = 300.0,
        eviction_inactive_seconds: float = 60.0,
    ):
        self.max_history_seconds = max_history_seconds
        self.eviction_inactive_seconds = eviction_inactive_seconds
        # Mapping: track_id -> deque of StudentObservation
        self._buffers: Dict[int, deque[StudentObservation]] = {}
        # Mapping: track_id -> last_seen_timestamp
        self._last_seen: Dict[int, float] = {}

    def push(self, observation: StudentObservation) -> None:
        """Add a new observation for a student track."""
        t_id = observation.track_id
        if t_id not in self._buffers:
            self._buffers[t_id] = deque()

        buf = self._buffers[t_id]
        buf.append(observation)
        self._last_seen[t_id] = observation.timestamp

        # Prune records older than max_history_seconds
        cutoff = observation.timestamp - self.max_history_seconds
        while buf and buf[0].timestamp < cutoff:
            buf.popleft()

    def get_history(self, track_id: int, window_seconds: float) -> List[StudentObservation]:
        """Get all observations for track_id within [now - window_seconds, now]."""
        if track_id not in self._buffers or not self._buffers[track_id]:
            return []

        buf = self._buffers[track_id]
        latest_time = buf[-1].timestamp
        cutoff = latest_time - window_seconds
        return [obs for obs in buf if obs.timestamp >= cutoff]

    def duration_behavior(
        self,
        track_id: int,
        behavior: str,
        window_seconds: float = 30.0,
        continuous_only: bool = False,
    ) -> float:
        """Calculate the duration (in seconds) the student exhibited the specified behavior.

        Args:
            track_id: Student track ID.
            behavior: Target behavior name (e.g. 'use_phone', 'turn_head', 'discuss').
            window_seconds: Evaluation time window looking back from the latest frame.
            continuous_only: If True, only measures the current unbroken continuous streak.

        Returns:
            Duration in seconds.
        """
        history = self.get_history(track_id, window_seconds)
        if not history:
            return 0.0

        if len(history) == 1:
            return 0.1 if history[0].behavior == behavior else 0.0

        if continuous_only:
            if history[-1].behavior != behavior:
                return 0.0
            streak_start = history[-1].timestamp
            for i in range(len(history) - 1, -1, -1):
                if history[i].behavior == behavior:
                    streak_start = history[i].timestamp
                else:
                    break
            return history[-1].timestamp - streak_start

        # Cumulative duration within window
        total_duration = 0.0
        for i in range(len(history) - 1):
            obs = history[i]
            next_obs = history[i + 1]
            if obs.behavior == behavior:
                dt = next_obs.timestamp - obs.timestamp
                # Cap dt to avoid huge gaps inflating duration if student disappeared briefly
                total_duration += min(dt, 2.0)

        return total_duration

    def count_behavior(
        self,
        track_id: int,
        behavior: str,
        window_seconds: float = 60.0,
    ) -> int:
        """Count the number of distinct occurrences / transitions into the behavior within window.

        Args:
            track_id: Student track ID.
            behavior: Target behavior name (e.g. 'turn_head').
            window_seconds: Time window in seconds.

        Returns:
            Number of distinct occurrences.
        """
        history = self.get_history(track_id, window_seconds)
        if not history:
            return 0

        occurrences = 0
        in_behavior = False

        for obs in history:
            if obs.behavior == behavior:
                if not in_behavior:
                    occurrences += 1
                    in_behavior = True
            else:
                in_behavior = False

        return occurrences

    def percentage_behavior(
        self,
        track_id: int,
        behavior: str,
        window_seconds: float = 60.0,
    ) -> float:
        """Calculate percentage of observed time in the window spent exhibiting the behavior."""
        history = self.get_history(track_id, window_seconds)
        if not history:
            return 0.0

        matches = sum(1 for obs in history if obs.behavior == behavior)
        return (matches / len(history)) * 100.0

    def recent_phone_presence(
        self,
        track_id: int,
        window_seconds: float = 10.0,
    ) -> bool:
        """Check if a cell phone was detected near this student within recent window."""
        history = self.get_history(track_id, window_seconds)
        return any(obs.phone_present for obs in history)

    def evict_expired(self, current_time: float) -> int:
        """Remove tracks that have not been observed for > eviction_inactive_seconds.

        Prevents memory leaks in continuous production CCTV operation.

        Returns:
            Count of evicted track IDs.
        """
        cutoff = current_time - self.eviction_inactive_seconds
        expired_ids = [
            t_id for t_id, last_t in self._last_seen.items() if last_t < cutoff
        ]

        for t_id in expired_ids:
            self._buffers.pop(t_id, None)
            self._last_seen.pop(t_id, None)

        if expired_ids:
            logger.debug(f"Evicted {len(expired_ids)} expired student tracks from temporal buffer.")

        return len(expired_ids)

    def active_track_ids(self) -> List[int]:
        """Return list of currently active student track IDs."""
        return list(self._buffers.keys())
