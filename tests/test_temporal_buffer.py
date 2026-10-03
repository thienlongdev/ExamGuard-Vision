"""Tests for TemporalBuffer sliding-window calculations and track lifecycle."""

import time
import pytest
from src.analysis.fusion import StudentObservation
from src.analysis.temporal_buffer import TemporalBuffer
from src.detection.types import BBox


def make_obs(track_id: int, timestamp: float, behavior: str, phone_present: bool = False):
    return StudentObservation(
        track_id=track_id,
        timestamp=timestamp,
        bbox=BBox(10, 10, 100, 100),
        behavior=behavior,
        behavior_confidence=0.9,
        phone_present=phone_present,
    )


def test_temporal_buffer_push_and_history():
    buf = TemporalBuffer(max_history_seconds=30.0)
    t0 = 1000.0

    buf.push(make_obs(1, t0, "normal"))
    buf.push(make_obs(1, t0 + 1.0, "normal"))
    buf.push(make_obs(1, t0 + 2.0, "turn_head"))

    hist = buf.get_history(1, window_seconds=5.0)
    assert len(hist) == 3
    assert hist[-1].behavior == "turn_head"


def test_duration_behavior():
    buf = TemporalBuffer(max_history_seconds=60.0)
    t0 = 1000.0

    # 4 seconds of phone usage sampled every 1 second
    buf.push(make_obs(1, t0, "use_phone"))
    buf.push(make_obs(1, t0 + 1.0, "use_phone"))
    buf.push(make_obs(1, t0 + 2.0, "use_phone"))
    buf.push(make_obs(1, t0 + 3.0, "use_phone"))
    buf.push(make_obs(1, t0 + 4.0, "use_phone"))

    duration = buf.duration_behavior(1, "use_phone", window_seconds=10.0)
    assert duration >= 3.9  # Approximately 4.0 seconds


def test_count_behavior_occurrences():
    buf = TemporalBuffer(max_history_seconds=60.0)
    t0 = 1000.0

    # Student turns head 3 separate times
    # Sequence: turn_head -> normal -> turn_head -> normal -> turn_head
    buf.push(make_obs(1, t0 + 0.0, "turn_head"))
    buf.push(make_obs(1, t0 + 1.0, "normal"))
    buf.push(make_obs(1, t0 + 2.0, "turn_head"))
    buf.push(make_obs(1, t0 + 3.0, "normal"))
    buf.push(make_obs(1, t0 + 4.0, "turn_head"))

    count = buf.count_behavior(1, "turn_head", window_seconds=10.0)
    assert count == 3


def test_percentage_behavior():
    buf = TemporalBuffer(max_history_seconds=60.0)
    t0 = 1000.0

    buf.push(make_obs(1, t0 + 0.0, "normal"))
    buf.push(make_obs(1, t0 + 1.0, "turn_head"))
    buf.push(make_obs(1, t0 + 2.0, "normal"))
    buf.push(make_obs(1, t0 + 3.0, "turn_head"))

    pct = buf.percentage_behavior(1, "turn_head", window_seconds=10.0)
    assert pct == 50.0


def test_recent_phone_presence():
    buf = TemporalBuffer(max_history_seconds=60.0)
    t0 = 1000.0

    buf.push(make_obs(1, t0 + 0.0, "normal", phone_present=False))
    buf.push(make_obs(1, t0 + 1.0, "normal", phone_present=True))

    assert buf.recent_phone_presence(1, window_seconds=5.0) is True
    assert buf.recent_phone_presence(2, window_seconds=5.0) is False


def test_eviction_expired_tracks():
    buf = TemporalBuffer(max_history_seconds=60.0, eviction_inactive_seconds=10.0)
    t0 = 1000.0

    buf.push(make_obs(1, t0, "normal"))
    buf.push(make_obs(2, t0 + 15.0, "normal"))

    # Current time is t0 + 15.0. Track 1 last seen at t0 (15s ago > 10s cutoff)
    evicted = buf.evict_expired(t0 + 15.0)
    assert evicted == 1
    assert 1 not in buf.active_track_ids()
    assert 2 in buf.active_track_ids()
