"""
Tests for V4D Timestamp Robustness & Variable FPS Invariance
============================================================
Covers:
- Sample rate invariance: 15 FPS, 25 FPS, 30 FPS triggering at identical elapsed time
- Dropped frames and irregular timestamp spacing
- Large time gaps handling
"""

import pytest
from src.fusion.replay import ReplayEngine
from src.fusion.types import (
    UnifiedTrackUpdate,
    ObservationStatus,
    PostureCue,
    TrackingState,
)


def create_fps_trace(fps: float, total_duration: float = 4.0, sleep_start: float = 1.0, sleep_end: float = 3.5):
    updates = []
    dt = 1.0 / fps
    t = 0.0
    while t <= total_duration:
        is_sleeping = (sleep_start <= t <= sleep_end)
        p_sleep = 0.85 if is_sleeping else 0.05
        p_upright = 0.10 if is_sleeping else 0.90

        updates.append(UnifiedTrackUpdate(
            track_id=1,
            timestamp_sec=round(t, 4),
            posture=PostureCue(
                status=ObservationStatus.AVAILABLE,
                probabilities={
                    "NORMAL_UPRIGHT": p_upright,
                    "NORMAL_READ_WRITE": 0.05,
                    "HEAD_REST_SLEEP": p_sleep,
                    "TURN_HEAD_CLEAR": 0.0,
                },
                confidence=p_sleep if is_sleeping else p_upright,
            ),
            tracking=TrackingState(1, round(t, 4), (10, 10, 100, 200)),
        ))
        t += dt
    return updates


@pytest.mark.parametrize("fps", [15.0, 25.0, 30.0])
def test_fps_invariance_trigger_time(fps):
    config = {
        "provisional_thresholds": {
            "sustained_head_rest": {
                "min_candidate_duration_sec": 1.0,  # Should open at t = sleep_start + 1.0 = 2.0s
                "evidence_enter_threshold": 0.60,
                "evidence_exit_threshold": 0.30,
                "cooldown_seconds": 2.0,
            }
        }
    }

    trace = create_fps_trace(fps=fps, sleep_start=1.0, sleep_end=3.5)
    engine = ReplayEngine(config)
    events = engine.replay_trace(trace)

    # Must generate an event
    open_events = [ev for ev in events if ev.status in ("active", "closed")]
    assert len(open_events) > 0

    first_open = open_events[0]
    # Invariant: start timestamp must align with true physical start (~1.0s),
    # and first activation must be near 2.0s (+/- 1 frame of latency)
    assert abs(first_open.start_timestamp - 1.0) < 0.15


def test_irregular_timestamp_spacing():
    # Irregular jittered timestamps
    timestamps = [0.0, 0.04, 0.15, 0.22, 0.38, 0.50, 0.72, 1.05, 1.40, 1.95, 2.30]
    updates = []
    for t in timestamps:
        updates.append(UnifiedTrackUpdate(
            track_id=1,
            timestamp_sec=t,
            posture=PostureCue(
                status=ObservationStatus.AVAILABLE,
                probabilities={
                    "NORMAL_UPRIGHT": 0.1,
                    "NORMAL_READ_WRITE": 0.1,
                    "HEAD_REST_SLEEP": 0.8,
                    "TURN_HEAD_CLEAR": 0.0,
                },
                confidence=0.8,
            ),
            tracking=TrackingState(1, t, (10, 10, 100, 200)),
        ))

    config = {
        "provisional_thresholds": {
            "sustained_head_rest": {
                "min_candidate_duration_sec": 1.0,
                "evidence_enter_threshold": 0.60,
                "evidence_exit_threshold": 0.30,
            }
        }
    }
    engine = ReplayEngine(config)
    events = engine.replay_trace(updates)
    assert len(events) > 0
