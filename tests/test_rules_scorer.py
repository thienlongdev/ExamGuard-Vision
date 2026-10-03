"""Tests for Rule Engine, Risk Scorer, and Event Manager with debounce."""

import os
import pytest
from src.analysis.fusion import StudentObservation
from src.analysis.temporal_buffer import TemporalBuffer
from src.behavior.rules import BehaviorRuleEngine, RuleDefinition
from src.behavior.scorer import RiskScorer
from src.behavior.event_manager import EventManager
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


def test_rule_matching_phone_use():
    engine = BehaviorRuleEngine()
    engine.add_rule(
        RuleDefinition(
            rule_id="prolonged_phone_use",
            name="Prolonged Phone",
            behavior="use_phone",
            min_duration_seconds=3.0,
            require_phone=True,
            severity="high",
            score=85.0,
        )
    )

    buf = TemporalBuffer()
    t0 = 100.0

    # Under threshold: 2 seconds
    buf.push(make_obs(1, t0, "use_phone", phone_present=True))
    buf.push(make_obs(1, t0 + 1.0, "use_phone", phone_present=True))
    buf.push(make_obs(1, t0 + 2.0, "use_phone", phone_present=True))

    matches = engine.evaluate_track(1, buf, t0 + 2.0)
    assert len(matches) == 0

    # Over threshold: 4 seconds
    buf.push(make_obs(1, t0 + 3.0, "use_phone", phone_present=True))
    buf.push(make_obs(1, t0 + 4.0, "use_phone", phone_present=True))

    matches = engine.evaluate_track(1, buf, t0 + 4.0)
    assert len(matches) == 1
    assert matches[0].rule_id == "prolonged_phone_use"
    assert matches[0].severity == "high"


def test_risk_scoring_levels():
    scorer = RiskScorer(low_threshold=25.0, medium_threshold=50.0, high_threshold=75.0)

    # Empty matches -> NORMAL
    ass_none = scorer.score_matches(1, [])
    assert ass_none.risk_level == "NORMAL"
    assert ass_none.score == 0.0

    # Low score match -> LOW
    engine = BehaviorRuleEngine()
    engine.add_rule(RuleDefinition(rule_id="r_low", name="R Low", score=30.0, severity="low"))
    matches_low = [RuleDefinition(rule_id="r_low", name="R Low", score=30.0, severity="low")]
    # Create rule match
    from src.behavior.rules import RuleMatch
    rm_low = RuleMatch("r_low", "R Low", 1, "low", 30.0, {}, 100.0)
    ass_low = scorer.score_matches(1, [rm_low])
    assert ass_low.risk_level == "LOW"
    assert ass_low.score == 30.0

    # Medium score match -> MEDIUM
    rm_med = RuleMatch("r_med", "R Med", 1, "medium", 60.0, {}, 100.0)
    ass_med = scorer.score_matches(1, [rm_med])
    assert ass_med.risk_level == "MEDIUM"

    # High score match -> HIGH
    rm_high = RuleMatch("r_high", "R High", 1, "high", 85.0, {}, 100.0)
    ass_high = scorer.score_matches(1, [rm_high])
    assert ass_high.risk_level == "HIGH"


def test_event_manager_debounce_cooldown():
    em = EventManager(camera_id="cam-1", cooldown_seconds=10.0)
    scorer = RiskScorer(high_threshold=75.0)
    from src.behavior.rules import RuleMatch

    rm = RuleMatch("prolonged_phone_use", "Phone Use", 7, "high", 85.0, {}, 100.0)
    ass = scorer.score_matches(7, [rm])

    # 1. First alert at t=100s -> creates new SuspiciousEvent
    ev1 = em.process_assessment(ass, timestamp=100.0)
    assert ev1 is not None
    assert ev1.track_id == 7
    assert ev1.status == "new"

    # 2. Second alert at t=104s (within 10s cooldown) -> suppressed
    ev2 = em.process_assessment(ass, timestamp=104.0)
    assert ev2 is None

    # 3. Third alert at t=112s (>10s after t=100s) -> allowed
    ev3 = em.process_assessment(ass, timestamp=112.0)
    assert ev3 is not None
    assert ev3.event_id != ev1.event_id


def test_event_manager_human_review_update():
    em = EventManager(camera_id="cam-1")
    scorer = RiskScorer()
    from src.behavior.rules import RuleMatch

    rm = RuleMatch("phone_use", "Phone", 5, "high", 85.0, {}, 100.0)
    ass = scorer.score_matches(5, [rm])
    ev = em.process_assessment(ass, 100.0)

    assert ev.status == "new"

    # Human confirms
    success = em.update_event_status(ev.event_id, "confirmed", reviewer_notes="Student was checking phone")
    assert success is True
    assert ev.status == "confirmed"
    assert ev.reviewer_notes == "Student was checking phone"
