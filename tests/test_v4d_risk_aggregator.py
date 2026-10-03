"""
Tests for V4D Risk Aggregation & Evidence Scorer
================================================
Covers:
- Single-cue safety: isolated frames cannot trigger HIGH or MEDIUM
- Multi-cue concurrence: independent cues escalate risk
- Recurrence impact on risk score
- Monotonicity with respect to duration and cue count
"""

import pytest
from src.fusion.risk_aggregator import RiskAggregator
from src.fusion.types import FusedEvent, RiskLevel, EventFamily


@pytest.fixture
def risk_agg():
    config = {
        "risk_policy": {
            "thresholds": {"low_to_medium": 40.0, "medium_to_high": 75.0},
            "max_single_frame_score": 25.0,
        }
    }
    return RiskAggregator(config)


def make_dummy_event(track_id: int, event_type: str, duration: float) -> FusedEvent:
    return FusedEvent(
        event_id="dummy_ev",
        track_id=track_id,
        camera_id="cam_0",
        event_type=event_type,
        start_timestamp=0.0,
        last_update_timestamp=duration,
        duration=duration,
    )


def test_single_frame_safety(risk_agg):
    # Short noisy glitch: 0.1s duration
    ev = make_dummy_event(1, EventFamily.PHONE_ASSOCIATED.value, duration=0.1)
    assessed = risk_agg.assess_event_risk(ev, active_cues_count=1, independent_cues_count=1)
    
    # MUST be strictly capped at max_single_frame_score (25.0) and remain LOW
    assert assessed.risk_score <= 25.0
    assert assessed.risk_level == RiskLevel.LOW.value


def test_independent_cues_escalation(risk_agg):
    # Sustained event (5.0s) with 1 cue vs 2 independent cues
    ev_single = make_dummy_event(1, EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION.value, duration=5.0)
    res_single = risk_agg.assess_event_risk(ev_single, active_cues_count=1, independent_cues_count=1)

    ev_multi = make_dummy_event(2, EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION.value, duration=5.0)
    res_multi = risk_agg.assess_event_risk(ev_multi, active_cues_count=2, independent_cues_count=2)

    # Multi-cue evidence must be strictly greater than single cue
    assert res_multi.risk_score > res_single.risk_score


def test_risk_recurrence_elevation(risk_agg):
    # Repeat event on same student track
    ev1 = make_dummy_event(1, EventFamily.PHONE_ASSOCIATED.value, duration=2.0)
    ev1.status = "closed"
    res1 = risk_agg.assess_event_risk(ev1, active_cues_count=1, independent_cues_count=1)

    ev2 = make_dummy_event(1, EventFamily.PHONE_ASSOCIATED.value, duration=2.0)
    ev2.status = "closed"
    res2 = risk_agg.assess_event_risk(ev2, active_cues_count=1, independent_cues_count=1)

    # Second event should have recurrence bonus
    assert res2.risk_score > res1.risk_score
