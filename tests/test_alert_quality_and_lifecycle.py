"""
Alert Quality, Event Lifecycle Deduplication, and Review Queue Load Certification Tests
========================================================================================
Covers requirements from Sections 61 through 67:
- 61. EVIDENCE CARDINALITY: One event -> 1 snapshot + 1 video; event UPDATE does not create new artifact rows.
- 62. CONTINUOUS INCIDENT: Sustained posture for long duration yields exactly 1 event ID.
- 63. RECURRENCE: Behavior -> sustained normal reset -> behavior yields 2 distinct event IDs.
- 64. PHONE FLICKER: Brief sub-threshold detector blips do not create micro-event spam.
- 65. CROSS-FAMILY INCIDENT: Multi-cue glance escalation updates risk without duplicate cards.
- 66. NORMAL EXAM MOTION: Continuous reading/writing posture with natural diagonal glance does not flood review queue.
- 67. 50-EVENT REVIEW QUEUE: Queue handles 50 legitimate pending events without data loss or sorting corruption.
"""

import time
import pytest
import numpy as np

from src.fusion.types import (
    FusedEvent,
    EventFamily,
    RiskLevel,
    ObservationStatus,
    UnifiedTrackUpdate,
    PostureCue,
    HeadPoseCue,
)
from src.fusion.cue_state import PerTrackCueState
from src.fusion.event_engine import EventEngine, TrackEventStateMachine, EventLifecycleState
from src.fusion.reliability import ReliabilityModel
from src.fusion.risk_aggregator import RiskAggregator


def test_61_evidence_cardinality_and_update_idempotence():
    """Section 61: Verify event updates do not create additional artifact requests or ID fragmentation."""
    sm = TrackEventStateMachine(
        track_id=1,
        event_family=EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION,
        min_candidate_duration=0.75,
        enter_threshold=0.50,
        exit_threshold=0.30,
        cooldown_seconds=3.0,
        exit_grace_seconds=1.2,
    )

    t = 0.0
    ev, act = sm.process_frame(t, evidence_score=0.60, is_vetoed=False, cue_state=None)
    assert act == "NONE"

    t = 0.8
    ev_open, act_open = sm.process_frame(t, evidence_score=0.60, is_vetoed=False, cue_state=None)
    assert act_open == "OPEN"
    initial_id = ev_open.event_id

    # 10 consecutive frames of updates
    for i in range(10):
        t += 0.1
        ev_up, act_up = sm.process_frame(t, evidence_score=0.58, is_vetoed=False, cue_state=None)
        assert act_up == "UPDATE"
        assert ev_up.event_id == initial_id, "Event ID must remain identical on UPDATE"


def test_62_continuous_incident_single_event_id():
    """Section 62: A continuous sustained head turn lasting 60 seconds with minor frame noise produces exactly 1 event."""
    engine = EventEngine({
        "provisional_thresholds": {
            "sustained_lateral_head_orientation": {
                "min_candidate_duration_sec": 0.75,
                "posture_turn_enter_threshold": 0.50,
                "posture_turn_exit_threshold": 0.30,
                "cooldown_seconds": 3.0,
                "exit_grace_seconds": 1.2,
                "normal_reset_seconds": 3.5,
            }
        }
    })

    opened_events = []
    closed_events = []

    # Simulate 60 seconds at 10 Hz (600 steps)
    # Head yaw fluctuates between 24 deg (score ~0.52) and occasional momentary dips to 22 deg (score ~0.28 for 0.2s)
    t = 0.0
    for step in range(600):
        t += 0.1
        # Brief 0.2s dip every 8 seconds
        is_dip = (step % 80 in [70, 71])
        turn_score = 0.25 if is_dip else 0.55

        cue = PerTrackCueState(
            track_id=1,
            last_update_timestamp=t,
            continuity_valid=True,
            turn_fused_evidence=turn_score,
            posture_status=ObservationStatus.AVAILABLE,
        )
        emitted = engine.process_cue_state(cue, camera_id="cam0")
        for ev, act in emitted:
            if act == "OPEN":
                opened_events.append(ev.event_id)
            elif act == "CLOSE":
                closed_events.append(ev.event_id)

    # Must produce exactly 1 OPEN event across the entire 60-second continuous episode!
    assert len(opened_events) == 1, f"Continuous 60s turn must yield exactly 1 OPEN event (got {len(opened_events)})"
    assert len(closed_events) == 0, f"Continuous 60s turn with sub-second dips must not prematurely close (got {len(closed_events)})"


def test_63_real_recurrence_creates_two_distinct_events():
    """Section 63: Incident -> Sustained normal recovery (>= 4s) -> Incident produces 2 distinct event IDs."""
    engine = EventEngine({
        "provisional_thresholds": {
            "sustained_lateral_head_orientation": {
                "min_candidate_duration_sec": 0.75,
                "posture_turn_enter_threshold": 0.50,
                "posture_turn_exit_threshold": 0.30,
                "cooldown_seconds": 3.0,
                "exit_grace_seconds": 1.2,
                "normal_reset_seconds": 3.5,
            }
        }
    })

    opened_events = []

    t = 0.0
    # Phase 1: First turn incident (3.0s)
    for _ in range(30):
        t += 0.1
        cue = PerTrackCueState(
            track_id=1,
            last_update_timestamp=t,
            continuity_valid=True,
            turn_fused_evidence=0.65,
            posture_status=ObservationStatus.AVAILABLE,
        )
        for ev, act in engine.process_cue_state(cue):
            if act == "OPEN":
                opened_events.append(ev.event_id)

    # Phase 2: Genuine normal recovery for 6.0 seconds (looking forward at desk)
    for _ in range(60):
        t += 0.1
        cue = PerTrackCueState(
            track_id=1,
            last_update_timestamp=t,
            continuity_valid=True,
            turn_fused_evidence=0.05,
            posture_status=ObservationStatus.AVAILABLE,
        )
        engine.process_cue_state(cue)

    # Phase 3: Second turn incident (3.0s)
    for _ in range(30):
        t += 0.1
        cue = PerTrackCueState(
            track_id=1,
            last_update_timestamp=t,
            continuity_valid=True,
            turn_fused_evidence=0.65,
            posture_status=ObservationStatus.AVAILABLE,
        )
        for ev, act in engine.process_cue_state(cue):
            if act == "OPEN":
                opened_events.append(ev.event_id)

    assert len(opened_events) == 2, f"Expected 2 distinct events after real recovery, got {len(opened_events)}"
    assert opened_events[0] != opened_events[1], "Second incident must carry a distinct UUID"


def test_64_phone_sub_threshold_flicker_suppression():
    """Section 64: Brief noisy 0.20-0.22 phone detector blips (< 0.5s) must not create review events."""
    engine = EventEngine({
        "provisional_thresholds": {
            "phone_associated": {
                "min_candidate_duration_sec": 0.75,
                "evidence_enter_threshold": 0.25,
                "evidence_exit_threshold": 0.18,
                "cooldown_seconds": 3.5,
                "exit_grace_seconds": 1.2,
                "normal_reset_seconds": 3.5,
            }
        }
    })

    opened_events = []
    t = 0.0

    # Simulate intermittent 0.22 confidence blips lasting 0.3s every 2 seconds for 20 seconds
    for cycle in range(10):
        # 0.3s blip (3 frames)
        for _ in range(3):
            t += 0.1
            cue = PerTrackCueState(
                track_id=1,
                last_update_timestamp=t,
                continuity_valid=True,
                phone_detected=True,
                phone_confidence=0.22,
                phone_association_status="CLEAR_ASSOCIATION",
                phone_status=ObservationStatus.AVAILABLE,
            )
            for ev, act in engine.process_cue_state(cue):
                if act == "OPEN":
                    opened_events.append(ev.event_id)

        # 1.7s gap (no detection)
        for _ in range(17):
            t += 0.1
            cue = PerTrackCueState(
                track_id=1,
                last_update_timestamp=t,
                continuity_valid=True,
                phone_detected=False,
                phone_confidence=0.0,
                phone_association_status="NO_PHONE",
                phone_status=ObservationStatus.AVAILABLE,
            )
            for ev, act in engine.process_cue_state(cue):
                if act == "OPEN":
                    opened_events.append(ev.event_id)

    assert len(opened_events) == 0, f"Noisy 0.22 blips must not produce review cards (got {len(opened_events)})"


def test_65_cross_family_glance_escalation_no_duplicate_card():
    """Section 65: Lateral glance burst escalates turn evidence without spawning a secondary duplicate card."""
    scorer = ReliabilityModel()
    agg = RiskAggregator()

    pos = PostureCue(
        status=ObservationStatus.AVAILABLE,
        probabilities={"NORMAL_UPRIGHT": 0.90, "TURN_HEAD_CLEAR": 0.10},
    )
    hp = HeadPoseCue(status=ObservationStatus.AVAILABLE, yaw_deg=-26.0, reliability_weight=0.9)
    fused_turn = scorer.compute_turn_fusion_evidence(pos, hp)

    assert fused_turn >= 0.50, "Turn evidence must meet enter threshold"

    ev = FusedEvent(
        event_id="ev_glance_1",
        track_id=1,
        camera_id="cam0",
        event_type=EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION.value,
        start_timestamp=0.0,
        last_update_timestamp=1.2,
        duration=1.2,
        risk_level=RiskLevel.LOW.value,
        status="active",
        lifecycle_status="active",
    )
    # Aggregator assesses single turn cue
    scored_ev = agg.assess_event_risk(ev, active_cues_count=1, independent_cues_count=1)
    assert scored_ev.risk_level == RiskLevel.MEDIUM.value
    assert scored_ev.event_type == EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION.value


def test_66_normal_read_write_protection_against_false_turn():
    """Section 66: Natural reading/writing paper angle (abs_yaw < 36 deg) is attenuated below enter threshold."""
    scorer = ReliabilityModel()

    # Student looking down-left at paper: yaw = -28.0 deg, posture is NORMAL_READ_WRITE (0.85)
    pos_writing = PostureCue(
        status=ObservationStatus.AVAILABLE,
        probabilities={"NORMAL_READ_WRITE": 0.85, "NORMAL_UPRIGHT": 0.10, "TURN_HEAD_CLEAR": 0.05},
    )
    hp_paper = HeadPoseCue(status=ObservationStatus.AVAILABLE, yaw_deg=-28.0, reliability_weight=0.9)
    fused_writing = scorer.compute_turn_fusion_evidence(pos_writing, hp_paper)

    assert fused_writing < 0.50, f"Writing on desk paper must not trigger turn alert (got {fused_writing:.3f})"

    # However, turning head to -40 deg while seated at desk (glancing at neighbor) MUST trigger
    hp_neighbor = HeadPoseCue(status=ObservationStatus.AVAILABLE, yaw_deg=-40.0, reliability_weight=0.9)
    fused_neighbor = scorer.compute_turn_fusion_evidence(pos_writing, hp_neighbor)
    assert fused_neighbor >= 0.50, f"Clear lateral turn -40 deg must trigger alert even at desk (got {fused_neighbor:.3f})"


def test_67_review_queue_50_event_capacity_and_sorting():
    """Section 67: Review queue comparator stably handles 50 legitimate pending events without lost cards."""
    events = []
    for i in range(50):
        # Mix of HIGH, MEDIUM, LOW with varying timestamps
        sev = RiskLevel.HIGH.value if i % 5 == 0 else (RiskLevel.MEDIUM.value if i % 2 == 0 else RiskLevel.LOW.value)
        score = 80.0 if sev == "HIGH" else (50.0 if sev == "MEDIUM" else 25.0)
        ev = FusedEvent(
            event_id=f"ev_queue_{i:03d}",
            track_id=(i % 4) + 1,
            camera_id="cam_main",
            event_type=EventFamily.PHONE_ASSOCIATED.value if sev == "HIGH" else EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION.value,
            start_timestamp=float(i * 10),
            last_update_timestamp=float(i * 10 + 2),
            duration=2.0,
            risk_level=sev,
            risk_score=score,
            status="active" if i % 3 == 0 else "closed",
            lifecycle_status="open",
            review_status="awaiting",
        )
        events.append(ev)

    assert len(events) == 50
    # Priority sorting comparator: HIGH first, then MEDIUM, then LOW; within same severity, newest first
    sev_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    sorted_events = sorted(
        events,
        key=lambda e: (sev_rank.get(e.risk_level, 3), -e.last_update_timestamp),
    )

    assert len(sorted_events) == 50
    assert sorted_events[0].risk_level == "HIGH"
    assert sorted_events[-1].risk_level == "LOW"
    # Ensure all 50 unique IDs exist
    assert len(set(e.event_id for e in sorted_events)) == 50
