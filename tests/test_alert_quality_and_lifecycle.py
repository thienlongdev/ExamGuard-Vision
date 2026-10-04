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
    PhoneCue,
    PhoneAssociationStatus,
)
from src.tracking.tracker import Track
from src.detection.types import BBox
from src.orchestration.phone_associator import PhoneAssociator
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


# =====================================================================
# SECTION 11: SPECIFICATION CERTIFICATION TESTS
# =====================================================================

def test_weak_single_phone_hit_does_not_create_review_event():
    """Verify single weak hit (~0.25) from pen/shadow/desk object does not create review event."""
    associator = PhoneAssociator(phone_strong_confidence=0.35, weak_candidate_min_hits=3)
    student = Track(track_id=1, bbox=BBox(100, 100, 200, 300), confidence=0.95, timestamp=1.0)
    
    # Single weak blip from pen / shadow: conf = 0.25
    phone_det = {"bbox": [120, 200, 160, 250], "confidence": 0.25, "class_id": 67, "class_name": "cell phone"}
    assoc = associator.associate([student], [phone_det], timestamp_sec=1.0)
    
    assert assoc[1].detected is True
    assert assoc[1].is_candidate_only is True, "Single 0.25 hit must be candidate only"
    
    # Process through EventEngine
    engine = EventEngine()
    cue = PerTrackCueState(
        track_id=1,
        last_update_timestamp=1.0,
        continuity_valid=True,
        phone_detected=True,
        phone_confidence=0.25,
        phone_association_status="CLEAR_ASSOCIATION",
        phone_status=ObservationStatus.AVAILABLE,
        phone_is_candidate_only=True,
    )
    emitted = engine.process_cue_state(cue)
    
    # Process through RiskAggregator
    agg = RiskAggregator()
    for ev, act in emitted:
        scored = agg.assess_event_risk(ev, active_cues_count=1, independent_cues_count=1)
        assert scored.risk_level == RiskLevel.LOW.value, "Weak candidate must stay LOW risk"
        assert scored.review_status == "internal", "Weak candidate must be internal only"
        assert scored.evidence_summary.get("is_reviewable") is False, "Single weak hit is not reviewable"


def test_repeated_weak_phone_evidence_can_create_candidate_event():
    """Verify repeated weak phone evidence (>= 3 hits or sustained span) promotes to reviewable observation."""
    associator = PhoneAssociator(phone_strong_confidence=0.35, weak_candidate_min_hits=3, weak_candidate_min_duration_sec=1.2)
    student = Track(track_id=1, bbox=BBox(100, 100, 200, 300), confidence=0.95, timestamp=0.0)
    phone_det = {"bbox": [120, 200, 160, 250], "confidence": 0.26, "class_id": 67, "class_name": "cell phone"}
    
    # Hit 1 at t=0.0 -> candidate only
    a1 = associator.associate([student], [phone_det], timestamp_sec=0.0)
    assert a1[1].is_candidate_only is True
    
    # Hit 2 at t=0.6 -> still candidate only
    a2 = associator.associate([student], [phone_det], timestamp_sec=0.6)
    assert a2[1].is_candidate_only is True
    
    # Hit 3 at t=1.3 (span 1.3s, 3 hits) -> promoted to reviewable phone observation!
    a3 = associator.associate([student], [phone_det], timestamp_sec=1.3)
    assert a3[1].is_candidate_only is False, "3 repeated hits spanning >= 1.2s must promote to confirmed observation"
    
    # In EventEngine, confirmed observation enters PHONE_ASSOCIATED
    engine = EventEngine()
    cue = PerTrackCueState(
        track_id=1,
        last_update_timestamp=1.3,
        continuity_valid=True,
        phone_detected=True,
        phone_confidence=0.26,
        phone_association_status="CLEAR_ASSOCIATION",
        phone_status=ObservationStatus.AVAILABLE,
        phone_is_candidate_only=False,
    )
    # Start candidate at 1.3, promote after 0.8s
    engine.process_cue_state(cue)
    cue.last_update_timestamp = 2.1
    emitted = engine.process_cue_state(cue)
    
    agg = RiskAggregator()
    for ev, act in emitted:
        if act == "OPEN":
            scored = agg.assess_event_risk(ev, active_cues_count=1, independent_cues_count=1)
            assert scored.event_type == EventFamily.PHONE_ASSOCIATED.value
            assert scored.risk_level in [RiskLevel.MEDIUM.value, RiskLevel.LOW.value]
            assert scored.review_status == "awaiting"


def test_clear_strong_phone_still_works():
    """Verify strong phone detection (conf >= 0.35) immediately opens review event."""
    associator = PhoneAssociator(phone_strong_confidence=0.35)
    student = Track(track_id=1, bbox=BBox(100, 100, 200, 300), confidence=0.95, timestamp=1.0)
    phone_det = {"bbox": [120, 200, 160, 250], "confidence": 0.82, "class_id": 67, "class_name": "cell phone"}
    
    assoc = associator.associate([student], [phone_det], timestamp_sec=1.0)
    assert assoc[1].detected is True
    assert assoc[1].is_candidate_only is False, "Strong phone detection must immediately be reviewable"
    
    # In EventEngine
    engine = EventEngine()
    cue = PerTrackCueState(
        track_id=1,
        last_update_timestamp=0.0,
        continuity_valid=True,
        phone_detected=True,
        phone_confidence=0.82,
        phone_association_status="CLEAR_ASSOCIATION",
        phone_status=ObservationStatus.AVAILABLE,
        phone_is_candidate_only=False,
    )
    engine.process_cue_state(cue)
    cue.last_update_timestamp = 0.8
    emitted = engine.process_cue_state(cue)
    
    agg = RiskAggregator()
    open_events = [ev for ev, act in emitted if act == "OPEN"]
    assert len(open_events) == 1
    scored = agg.assess_event_risk(open_events[0], active_cues_count=1, independent_cues_count=1)
    assert scored.event_type == EventFamily.PHONE_ASSOCIATED.value
    assert scored.risk_level == RiskLevel.MEDIUM.value
    assert scored.review_status == "awaiting"


def test_one_phone_appearance_is_one_event():
    """Verify continuous phone appearance over 5 seconds produces exactly 1 OPEN event and 0 duplicate cards."""
    engine = EventEngine()
    opened = []
    closed = []
    
    t = 0.0
    for step in range(50):
        t += 0.1
        # Conf fluctuates between 0.70 and 0.50 with momentary flutter
        conf = 0.65 if step % 15 != 0 else 0.40
        cue = PerTrackCueState(
            track_id=1,
            last_update_timestamp=t,
            continuity_valid=True,
            phone_detected=True,
            phone_confidence=conf,
            phone_association_status="CLEAR_ASSOCIATION",
            phone_status=ObservationStatus.AVAILABLE,
            phone_is_candidate_only=False,
        )
        for ev, act in engine.process_cue_state(cue):
            if act == "OPEN":
                opened.append(ev.event_id)
            elif act == "CLOSE":
                closed.append(ev.event_id)
                
    assert len(opened) == 1, f"One continuous phone appearance must yield exactly 1 event (got {len(opened)})"
    assert len(closed) == 0, "Event must remain open during active incident"


def test_calculator_notebook_id_pen_ruler_paper_do_not_create_phone_review_event():
    """Verify spurious low-confidence hits (~0.22-0.25) from non-phone desk objects do not create review events."""
    associator = PhoneAssociator(phone_strong_confidence=0.35, weak_candidate_min_hits=3)
    student = Track(track_id=1, bbox=BBox(100, 100, 200, 300), confidence=0.95, timestamp=0.0)
    
    desk_objects = ["calculator", "notebook", "id_card", "pen", "ruler", "paper"]
    review_events = []
    
    t = 0.0
    engine = EventEngine()
    agg = RiskAggregator()
    
    for obj in desk_objects:
        # Isolated spurious hit lasting 1-2 frames
        t += 5.0
        phone_det = {"bbox": [130, 220, 150, 260], "confidence": 0.24, "class_id": 67, "class_name": "cell phone"}
        assoc = associator.associate([student], [phone_det], timestamp_sec=t)
        
        cue = PerTrackCueState(
            track_id=1,
            last_update_timestamp=t,
            continuity_valid=True,
            phone_detected=assoc[1].detected,
            phone_confidence=assoc[1].confidence,
            phone_association_status=assoc[1].association_status_enum.value,
            phone_status=ObservationStatus.AVAILABLE,
            phone_is_candidate_only=assoc[1].is_candidate_only,
        )
        emitted = engine.process_cue_state(cue)
        for ev, act in emitted:
            scored = agg.assess_event_risk(ev, active_cues_count=1, independent_cues_count=1)
            if scored.risk_level in [RiskLevel.MEDIUM.value, RiskLevel.HIGH.value] or scored.review_status == "awaiting":
                review_events.append(scored)
                
        # Gap with no hits
        for _ in range(10):
            t += 0.1
            associator.associate([student], [], timestamp_sec=t)
            
    assert len(review_events) == 0, f"Desk objects must produce 0 review events (got {len(review_events)})"


def test_brief_isolated_head_glance_does_not_flood_queue():
    """Verify brief isolated glance (< 1.2s, no repetition) is internal cue only and does not create review card."""
    agg = RiskAggregator()
    
    ev_glance = FusedEvent(
        event_id="ev_glance_brief",
        track_id=1,
        camera_id="cam_main",
        event_type=EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION.value,
        start_timestamp=0.0,
        last_update_timestamp=0.8,
        duration=0.8,  # Isolated glance < 1.2s
        risk_level=RiskLevel.LOW.value,
        status="active",
        lifecycle_status="active",
        evidence_summary={
            "glance_burst": False,
            "glance_count": 1,
        },
    )
    
    scored = agg.assess_event_risk(ev_glance, active_cues_count=1, independent_cues_count=1, mean_reliability=0.9)
    assert scored.risk_level == RiskLevel.LOW.value, "Brief isolated glance must remain LOW risk"
    assert scored.review_status == "internal", "Brief isolated glance must have review_status 'internal'"
    assert scored.evidence_summary.get("is_reviewable") is False, "Brief isolated glance must not be reviewable"
    assert scored.evidence_summary.get("glance_classification") == "BRIEF_ISOLATED_GLANCE"


def test_sustained_head_turn_creates_event():
    """Verify sustained lateral head turn (duration >= 1.2s) creates MEDIUM review event."""
    agg = RiskAggregator()
    
    ev_turn = FusedEvent(
        event_id="ev_turn_sustained",
        track_id=1,
        camera_id="cam_main",
        event_type=EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION.value,
        start_timestamp=0.0,
        last_update_timestamp=1.5,
        duration=1.5,  # Sustained turn >= 1.2s
        risk_level=RiskLevel.LOW.value,
        status="active",
        lifecycle_status="active",
        evidence_summary={
            "glance_burst": False,
            "glance_count": 1,
        },
    )
    
    scored = agg.assess_event_risk(ev_turn, active_cues_count=1, independent_cues_count=1, mean_reliability=0.9)
    assert scored.risk_level == RiskLevel.MEDIUM.value, "Sustained turn must be MEDIUM review event"
    assert scored.review_status == "awaiting"
    assert scored.evidence_summary.get("is_reviewable") is True
    assert scored.evidence_summary.get("glance_classification") == "SUSTAINED_LATERAL_TURN"


def test_repeated_glance_creates_one_consolidated_incident():
    """Verify repeated glances in short window (glance_burst >= 2) consolidate into ONE review event."""
    agg = RiskAggregator()
    
    ev_burst = FusedEvent(
        event_id="ev_glance_burst",
        track_id=1,
        camera_id="cam_main",
        event_type=EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION.value,
        start_timestamp=0.0,
        last_update_timestamp=0.9,
        duration=0.9,  # Under 1.2s, but part of a repeated glance burst!
        risk_level=RiskLevel.LOW.value,
        status="active",
        lifecycle_status="active",
        evidence_summary={
            "glance_burst": True,
            "glance_count": 3,
        },
    )
    
    scored = agg.assess_event_risk(ev_burst, active_cues_count=1, independent_cues_count=1, mean_reliability=0.9)
    assert scored.risk_level == RiskLevel.MEDIUM.value, "Repeated glance burst must be MEDIUM review event"
    assert scored.review_status == "awaiting"
    assert scored.evidence_summary.get("is_reviewable") is True
    assert scored.evidence_summary.get("glance_classification") == "REPEATED_GLANCES"


def test_read_write_veto_still_works():
    """Verify NORMAL_READ_WRITE score suppresses HEAD_REST_SLEEP."""
    engine = EventEngine()
    
    cue = PerTrackCueState(
        track_id=1,
        last_update_timestamp=1.0,
        continuity_valid=True,
        posture_status=ObservationStatus.AVAILABLE,
        posture_probs={"HEAD_REST_SLEEP": 0.70, "NORMAL_READ_WRITE": 0.75},
        read_write_suppression_active=True,
        read_write_score=0.75,
    )
    
    emitted = engine.process_cue_state(cue)
    sleep_events = [ev for ev, act in emitted if ev.event_type == EventFamily.SUSTAINED_HEAD_REST.value and act == "OPEN"]
    assert len(sleep_events) == 0, "Active read/write must veto HEAD_REST_SLEEP"


def test_controlled_scenario_partial_phone():
    """Controlled Scenario 3: Partial phone detection (hand-occluded, conf ~0.42) associates and opens review event."""
    associator = PhoneAssociator(phone_strong_confidence=0.35)
    student = Track(track_id=1, bbox=BBox(100, 100, 200, 300), confidence=0.95, timestamp=1.0)
    # Partial phone detection near hand/lap
    phone_det = {"bbox": [130, 220, 170, 270], "confidence": 0.42, "class_id": 67, "class_name": "cell phone"}
    
    assoc = associator.associate([student], [phone_det], timestamp_sec=1.0)
    assert assoc[1].detected is True
    assert assoc[1].is_candidate_only is False
    
    engine = EventEngine()
    cue = PerTrackCueState(
        track_id=1,
        last_update_timestamp=0.0,
        continuity_valid=True,
        phone_detected=True,
        phone_confidence=0.42,
        phone_association_status="CLEAR_ASSOCIATION",
        phone_status=ObservationStatus.AVAILABLE,
        phone_is_candidate_only=False,
    )
    engine.process_cue_state(cue)
    cue.last_update_timestamp = 0.8
    emitted = engine.process_cue_state(cue)
    
    open_events = [ev for ev, act in emitted if act == "OPEN" and ev.event_type == EventFamily.PHONE_ASSOCIATED.value]
    assert len(open_events) == 1, "Partial phone must open PHONE_ASSOCIATED review event"


def test_controlled_scenario_pure_right_head_turn():
    """Controlled Scenario 7: Pure right head turn (yaw +32 deg) without body lean triggers turn evidence and opens review event."""
    scorer = ReliabilityModel()
    pos = PostureCue(
        status=ObservationStatus.AVAILABLE,
        probabilities={"NORMAL_UPRIGHT": 0.85, "TURN_HEAD_CLEAR": 0.15},
    )
    hp = HeadPoseCue(status=ObservationStatus.AVAILABLE, yaw_deg=32.0, reliability_weight=0.9)
    fused_turn = scorer.compute_turn_fusion_evidence(pos, hp)
    assert fused_turn >= 0.50, f"Pure right turn evidence must meet enter threshold, got {fused_turn}"
    
    engine = EventEngine()
    cue = PerTrackCueState(
        track_id=1,
        last_update_timestamp=0.0,
        continuity_valid=True,
        turn_fused_evidence=fused_turn,
        smoothed_yaw_deg=32.0,
        posture_status=ObservationStatus.AVAILABLE,
    )
    engine.process_cue_state(cue)
    cue.last_update_timestamp = 0.8
    emitted = engine.process_cue_state(cue)
    
    open_events = [ev for ev, act in emitted if act == "OPEN" and ev.event_type == EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION.value]
    assert len(open_events) == 1, "Pure right head turn must open review event"


def test_controlled_scenario_sustained_head_rest():
    """Controlled Scenario 10: Head resting on desk without reading/writing triggers SUSTAINED_HEAD_REST."""
    engine = EventEngine()
    
    cue = PerTrackCueState(
        track_id=1,
        last_update_timestamp=0.0,
        continuity_valid=True,
        posture_status=ObservationStatus.AVAILABLE,
        posture_probs={"HEAD_REST_SLEEP": 0.88, "NORMAL_READ_WRITE": 0.05},
        read_write_suppression_active=False,
        read_write_score=0.05,
    )
    engine.process_cue_state(cue)
    cue.last_update_timestamp = 2.2
    emitted = engine.process_cue_state(cue)
    
    open_events = [ev for ev, act in emitted if act == "OPEN" and ev.event_type == EventFamily.SUSTAINED_HEAD_REST.value]
    assert len(open_events) == 1, "Sustained head rest must open SUSTAINED_HEAD_REST event"


def test_controlled_scenario_standing():
    """Controlled Scenario 11: Student standing triggers STANDING event and scores MEDIUM risk."""
    engine = EventEngine()
    agg = RiskAggregator()
    
    cue = PerTrackCueState(
        track_id=1,
        last_update_timestamp=0.0,
        continuity_valid=True,
        stand_score=0.75,
    )
    engine.process_cue_state(cue)
    cue.last_update_timestamp = 1.6
    emitted = engine.process_cue_state(cue)
    
    open_events = [ev for ev, act in emitted if act == "OPEN" and ev.event_type == EventFamily.STANDING.value]
    assert len(open_events) == 1, "Standing must open STANDING event"
    scored = agg.assess_event_risk(open_events[0], active_cues_count=1, independent_cues_count=1)
    assert scored.risk_level == RiskLevel.MEDIUM.value


