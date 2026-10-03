"""
Tests for V4D Multi-Cue Temporal Fusion Engine
=============================================
Covers:
- Missing cue safety (missing cues do not fabricate negative evidence)
- Competing negative evidence (NORMAL_READ_WRITE suppresses HEAD_REST_SLEEP)
- Correlated cue handling (turn posture + yaw)
- Phone ambiguous association safety
"""

import pytest
from src.fusion.fusion_engine import MultiCueFusionEngine
from src.fusion.types import (
    UnifiedTrackUpdate,
    ObservationStatus,
    PostureCue,
    HeadPoseCue,
    PhoneCue,
    PhoneAssociationStatus,
    TrackingState,
)


@pytest.fixture
def engine():
    config = {
        "temporal_buffer": {
            "time_horizon_seconds": 10.0,
            "max_samples_per_track": 100,
        },
        "provisional_thresholds": {
            "sustained_head_rest": {
                "read_write_veto_threshold": 0.50,
            },
            "sustained_lateral_head_orientation": {
                "supporting_yaw_deg_threshold": 25.0,
            }
        },
        "reliability_weights": {
            "posture": {"base_weight": 1.0},
            "head_pose": {"base_weight": 0.60, "correlated_with_turn_penalty": 0.30},
            "phone": {"base_weight": 1.0, "ambiguous_association_penalty": 0.50},
        }
    }
    return MultiCueFusionEngine(config)


def test_missing_cue_safety(engine):
    # Update with ONLY tracking; posture, headpose, phone are UNAVAILABLE
    u = UnifiedTrackUpdate(
        track_id=1,
        timestamp_sec=1.0,
        tracking=TrackingState(1, 1.0, (10, 10, 100, 200)),
        posture=PostureCue(status=ObservationStatus.UNAVAILABLE),
        headpose=HeadPoseCue(status=ObservationStatus.UNAVAILABLE),
        phone=PhoneCue(status=ObservationStatus.UNAVAILABLE),
    )
    state = engine.update_track(u)
    assert state is not None
    assert state.posture_status == ObservationStatus.UNAVAILABLE
    assert state.headpose_status == ObservationStatus.UNAVAILABLE
    assert state.phone_status == ObservationStatus.UNAVAILABLE
    # Missing headpose does NOT fabricate yaw = 0
    assert state.smoothed_yaw_deg is None


def test_read_write_veto_on_sleep(engine):
    # Strong NORMAL_READ_WRITE alongside some sleep probability
    probs = {
        "NORMAL_UPRIGHT": 0.1,
        "NORMAL_READ_WRITE": 0.70,   # Strong reading/writing
        "HEAD_REST_SLEEP": 0.20,
        "TURN_HEAD_CLEAR": 0.0,
    }
    u = UnifiedTrackUpdate(
        track_id=1,
        timestamp_sec=1.0,
        tracking=TrackingState(1, 1.0, (10, 10, 100, 200)),
        posture=PostureCue(
            status=ObservationStatus.AVAILABLE,
            probabilities=probs,
            confidence=0.70,
        ),
    )
    state = engine.update_track(u)
    assert state is not None
    # Suppression must be ACTIVE
    assert state.read_write_suppression_active is True
    assert state.read_write_score >= 0.50


def test_phone_ambiguous_association_penalty(engine):
    u = UnifiedTrackUpdate(
        track_id=1,
        timestamp_sec=1.0,
        tracking=TrackingState(1, 1.0, (10, 10, 100, 200)),
        phone=PhoneCue(
            status=ObservationStatus.AVAILABLE,
            detected=True,
            association_confidence=0.60,
            association_status=PhoneAssociationStatus.AMBIGUOUS_ASSOCIATION,
        ),
    )
    state = engine.update_track(u)
    assert state is not None
    assert state.phone_association_status == "AMBIGUOUS_ASSOCIATION"
    # Reliability penalized due to ambiguity
    assert state.phone_reliability < 0.60
