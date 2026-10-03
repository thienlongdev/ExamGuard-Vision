"""
Tests for V4D Missing-Cue Robustness & Graceful Degradation
==========================================================
Covers:
- System stability when individual cues (head-pose, posture, phone, macro) are unavailable
- Intermittent cue dropouts (cadence subsampling)
- Ensuring missing cues never fabricate false negative or default zero evidence
"""

import pytest
from src.fusion.replay import ReplayEngine
from src.fusion.types import (
    UnifiedTrackUpdate,
    ObservationStatus,
    PostureCue,
    HeadPoseCue,
    PhoneCue,
    TrackingState,
)


def test_missing_headpose_degradation():
    # Only posture is available; head-pose is completely UNAVAILABLE
    updates = []
    for i in range(25):
        t = i * 0.1
        updates.append(UnifiedTrackUpdate(
            track_id=1,
            timestamp_sec=t,
            tracking=TrackingState(1, t, (10, 10, 100, 200)),
            posture=PostureCue(
                status=ObservationStatus.AVAILABLE,
                probabilities={
                    "NORMAL_UPRIGHT": 0.1,
                    "NORMAL_READ_WRITE": 0.0,
                    "HEAD_REST_SLEEP": 0.0,
                    "TURN_HEAD_CLEAR": 0.85,
                },
                confidence=0.85,
            ),
            headpose=HeadPoseCue(status=ObservationStatus.UNAVAILABLE),
        ))

    engine = ReplayEngine()
    events = engine.replay_trace(updates)
    # Posture alone can still trigger lateral head orientation event
    assert any(ev.event_type == "SUSTAINED_LATERAL_HEAD_ORIENTATION" for ev in events)
    ev = [e for e in events if e.event_type == "SUSTAINED_LATERAL_HEAD_ORIENTATION"][0]
    assert ev.cue_availability["headpose"] == "UNAVAILABLE"


def test_intermittent_cadence_subsampling():
    # Posture sampled every 3 frames (10 Hz), Tracking every frame (30 Hz)
    updates = []
    dt = 1.0 / 30.0
    for i in range(60):
        t = i * dt
        has_posture = (i % 3 == 0)
        p_cue = PostureCue(
            status=ObservationStatus.AVAILABLE if has_posture else ObservationStatus.UNAVAILABLE,
            probabilities={"NORMAL_UPRIGHT": 0.1, "NORMAL_READ_WRITE": 0.1, "HEAD_REST_SLEEP": 0.8, "TURN_HEAD_CLEAR": 0.0} if has_posture else {},
            confidence=0.8 if has_posture else 0.0,
        )
        updates.append(UnifiedTrackUpdate(
            track_id=1,
            timestamp_sec=round(t, 4),
            tracking=TrackingState(1, round(t, 4), (10, 10, 100, 200)),
            posture=p_cue,
        ))

    engine = ReplayEngine({
        "provisional_thresholds": {
            "sustained_head_rest": {
                "min_candidate_duration_sec": 1.0,
                "evidence_enter_threshold": 0.50,
                "evidence_exit_threshold": 0.30,
            }
        }
    })
    events = engine.replay_trace(updates)
    # The temporal buffer should bridge the 3-frame gaps cleanly
    assert len(events) > 0
    assert any(ev.event_type == "SUSTAINED_HEAD_REST" for ev in events)
