"""
Tests for V4D Temporal Event State Machine & Deduplication Engine
=================================================================
Covers:
- Event lifecycle transitions (INACTIVE -> CANDIDATE -> ACTIVE -> COOLDOWN -> INACTIVE)
- Hysteresis behavior (different enter and exit thresholds)
- Cooldown debounce (prevents re-opening during cooldown)
- Event deduplication (sustained behavior updates single event instead of spamming duplicates)
- Track continuity protection (force closure on track loss)
"""

import pytest
from src.fusion.event_engine import EventEngine, TrackEventStateMachine
from src.fusion.cue_state import PerTrackCueState
from src.fusion.types import EventFamily, EventLifecycleState, ObservationStatus


@pytest.fixture
def state_machine():
    return TrackEventStateMachine(
        track_id=1,
        event_family=EventFamily.SUSTAINED_HEAD_REST,
        min_candidate_duration=1.5,
        enter_threshold=0.60,
        exit_threshold=0.30,
        cooldown_seconds=3.0,
    )


def test_event_lifecycle_and_deduplication(state_machine):
    cue_state = PerTrackCueState(
        track_id=1,
        last_update_timestamp=10.0,
        posture_status=ObservationStatus.AVAILABLE,
    )

    # 1. Below enter threshold: remains INACTIVE
    ev, act = state_machine.process_frame(10.0, evidence_score=0.40, is_vetoed=False, cue_state=cue_state)
    assert act == "NONE"
    assert state_machine.state == EventLifecycleState.INACTIVE

    # 2. Exceeds enter threshold at t=11.0: transitions to CANDIDATE
    ev, act = state_machine.process_frame(11.0, evidence_score=0.75, is_vetoed=False, cue_state=cue_state)
    assert act == "NONE"
    assert state_machine.state == EventLifecycleState.CANDIDATE

    # 3. Before min_candidate_duration (1.5s): still CANDIDATE at t=12.0 (elapsed 1.0s)
    ev, act = state_machine.process_frame(12.0, evidence_score=0.80, is_vetoed=False, cue_state=cue_state)
    assert act == "NONE"
    assert state_machine.state == EventLifecycleState.CANDIDATE

    # 4. Meets min_candidate_duration at t=12.5 (elapsed 1.5s): Promotes to ACTIVE, emits OPEN
    ev, act = state_machine.process_frame(12.5, evidence_score=0.85, is_vetoed=False, cue_state=cue_state)
    assert act == "OPEN"
    assert ev is not None
    assert state_machine.state == EventLifecycleState.ACTIVE
    event_id = ev.event_id

    # 5. Sustained behavior across t=13.0, 14.0, 15.0: DEDUPLICATION (emits UPDATE, same event_id!)
    for t in [13.0, 14.0, 15.0]:
        ev, act = state_machine.process_frame(t, evidence_score=0.70, is_vetoed=False, cue_state=cue_state)
        assert act == "UPDATE"
        assert ev.event_id == event_id
        assert ev.last_update_timestamp == t

    # 6. Hysteresis: score drops to 0.40 (below enter 0.60, but above exit 0.30) -> remains ACTIVE!
    ev, act = state_machine.process_frame(16.0, evidence_score=0.40, is_vetoed=False, cue_state=cue_state)
    assert act == "UPDATE"
    assert state_machine.state == EventLifecycleState.ACTIVE

    # 7. Score drops to 0.20 (below exit threshold 0.30): Closes event, transitions to COOLDOWN
    ev, act = state_machine.process_frame(17.0, evidence_score=0.20, is_vetoed=False, cue_state=cue_state)
    assert act == "CLOSE"
    assert ev.status == "closed"
    assert ev.end_timestamp == 17.0
    assert state_machine.state == EventLifecycleState.COOLDOWN

    # 8. During cooldown (t=18.0, elapsed 1.0s < 3.0s cooldown): cannot open even with high score
    ev, act = state_machine.process_frame(18.0, evidence_score=0.90, is_vetoed=False, cue_state=cue_state)
    assert act == "NONE"
    assert state_machine.state == EventLifecycleState.COOLDOWN

    # 9. After cooldown (t=20.5, elapsed 3.5s > 3.0s): returns to INACTIVE
    ev, act = state_machine.process_frame(20.5, evidence_score=0.10, is_vetoed=False, cue_state=cue_state)
    assert state_machine.state == EventLifecycleState.INACTIVE


def test_track_continuity_force_close():
    engine = EventEngine({
        "provisional_thresholds": {
            "sustained_head_rest": {
                "min_candidate_duration_sec": 1.0,
                "evidence_enter_threshold": 0.50,
                "evidence_exit_threshold": 0.30,
            }
        }
    })

    # Start event
    cue_active = PerTrackCueState(
        track_id=1,
        last_update_timestamp=1.0,
        posture_status=ObservationStatus.AVAILABLE,
        posture_probs={"HEAD_REST_SLEEP": 0.8},
        continuity_valid=True,
    )
    engine.process_cue_state(cue_active)

    cue_active.last_update_timestamp = 2.5
    res = engine.process_cue_state(cue_active)
    assert any(act == "OPEN" for ev, act in res)

    # Next update has continuity_valid = False (track gap)
    cue_broken = PerTrackCueState(
        track_id=1,
        last_update_timestamp=5.0,
        continuity_valid=False,
    )
    res_close = engine.process_cue_state(cue_broken)
    assert any(act == "CLOSE" for ev, act in res_close)
    closed_ev = [ev for ev, act in res_close if act == "CLOSE"][0]
    assert closed_ev.evidence_summary["closure_reason"] == "TRACK_CONTINUITY_BREAK"
