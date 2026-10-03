"""
Tests for V4D Timestamp-First Sliding Window Temporal Buffer
===========================================================
Covers:
- Timestamp ordering & insertion
- Duplicate timestamp handling
- NaN & Inf rejection
- Out-of-order timestamp insertion
- Track expiry & bounded memory
- Duration calculations
"""

import math
import pytest
from src.fusion.types import (
    UnifiedTrackUpdate,
    ObservationStatus,
    PostureCue,
    TrackingState,
)
from src.fusion.temporal_buffer import TemporalBuffer, TrackObservationBuffer


def make_update(track_id: int, timestamp: float, sleep_prob: float = 0.0, upright_prob: float = 1.0):
    probs = {
        "NORMAL_UPRIGHT": upright_prob,
        "NORMAL_READ_WRITE": 0.0,
        "HEAD_REST_SLEEP": sleep_prob,
        "TURN_HEAD_CLEAR": 0.0,
    }
    return UnifiedTrackUpdate(
        track_id=track_id,
        timestamp_sec=timestamp,
        posture=PostureCue(
            status=ObservationStatus.AVAILABLE,
            probabilities=probs,
            confidence=max(sleep_prob, upright_prob),
        ),
        tracking=TrackingState(
            track_id=track_id,
            timestamp_sec=timestamp,
            bbox=(10, 10, 100, 200),
        ),
    )


def test_temporal_buffer_timestamp_ordering():
    buf = TrackObservationBuffer(track_id=1, time_horizon_seconds=10.0)
    # Insert out-of-order: 2.0, 1.0, 3.0
    u2 = make_update(1, 2.0)
    u1 = make_update(1, 1.0)
    u3 = make_update(1, 3.0)

    buf.push(u2)
    buf.push(u1)
    buf.push(u3)

    assert buf._timestamps == [1.0, 2.0, 3.0]
    assert buf.sample_count == 3
    assert buf.latest_timestamp == 3.0


def test_temporal_buffer_duplicate_timestamp():
    buf = TrackObservationBuffer(track_id=1, time_horizon_seconds=10.0)
    u1 = make_update(1, 1.0, sleep_prob=0.1)
    u1_dup = make_update(1, 1.0, sleep_prob=0.9)

    buf.push(u1)
    buf.push(u1_dup)

    assert buf.sample_count == 1
    assert buf._timestamps == [1.0]
    assert buf._history[0].posture.probabilities["HEAD_REST_SLEEP"] == 0.9


def test_temporal_buffer_nan_and_inf_rejection():
    buf = TemporalBuffer()
    u_nan = make_update(1, float("nan"))
    u_inf = make_update(1, float("inf"))
    u_valid = make_update(1, 5.0)

    assert buf.push(u_nan) is False
    assert buf.push(u_inf) is False
    assert buf.push(u_valid) is True
    assert 1 in buf.active_track_ids()


def test_temporal_buffer_track_expiry():
    buf = TemporalBuffer(eviction_inactive_seconds=5.0)
    buf.push(make_update(1, 10.0))
    buf.push(make_update(2, 14.0))

    # At t=16.0: track 1 has been inactive for 6.0s (> 5.0s cutoff), track 2 inactive for 2.0s
    evicted = buf.evict_inactive_tracks(current_timestamp=16.0)
    assert evicted == [1]
    assert buf.active_track_ids() == [2]


def test_temporal_buffer_duration_in_state():
    buf = TrackObservationBuffer(track_id=1, time_horizon_seconds=10.0)
    # 3 seconds of sleep from t=1.0 to t=4.0 sampled every 1.0s
    for t in [1.0, 2.0, 3.0, 4.0]:
        buf.push(make_update(1, t, sleep_prob=0.8))

    dur = buf.duration_in_state(
        predicate_fn=lambda u: u.posture.probabilities.get("HEAD_REST_SLEEP", 0) > 0.5,
        window_seconds=5.0,
    )
    assert abs(dur - 3.0) < 0.01
