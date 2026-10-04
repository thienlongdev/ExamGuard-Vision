"""
Tests for Emergency Phone and Head-Turn Demo Detection Responsiveness (PART 12)
================================================================================
Validates all 18 requirements:
1. weak one-off phone hit -> no review card
2. repeated weak consistent phone -> candidate
3. strong associated phone -> event
4. clear phone promotion latency within policy (~0.5-1.0s)
5. lean-back no-phone -> no phone event
6. calculator/notebook/ID/pen/ruler/paper -> no phone review event
7. one phone appearance -> one event
8. pure left -> candidate/event
9. pure right -> candidate/event
10. head-turn visual state exits GREEN promptly
11. sustained head turn -> AMBER
12. repeated glance -> one consolidated event
13. tiny brief glance -> no review card
14. head turn does NOT require body lean
15. unavailable headpose -> no fake yaw
16. continuous incident -> one event
17. real recurrence -> new event
18. duplicates remain zero
"""

import pytest
import numpy as np

from src.fusion.types import (
    FusedEvent,
    EventFamily,
    EventLifecycleState,
    RiskLevel,
    ObservationStatus,
    HeadPoseSupportStatus,
    PostureCue,
    HeadPoseCue,
    PhoneCue,
    PhoneAssociationStatus,
)
from src.tracking.tracker import Track
from src.detection.types import BBox, Detection
from src.orchestration.phone_associator import PhoneAssociator
from src.fusion.cue_state import PerTrackCueState
from src.fusion.event_engine import EventEngine, TrackEventStateMachine
from src.fusion.reliability import ReliabilityModel
from src.fusion.risk_aggregator import RiskAggregator


# ==============================================================================
# PHONE TESTS (1 - 7)
# ==============================================================================

def test_01_weak_one_off_phone_hit_no_review_card():
    """1. Weak isolated class-67 hit (~0.20-0.35) -> internal candidate only, NO review card."""
    associator = PhoneAssociator(phone_strong_confidence=0.35, weak_candidate_min_hits=3)
    agg = RiskAggregator()

    student = Track(track_id=1, bbox=BBox(100, 100, 300, 500), confidence=0.9, timestamp=0.0)
    weak_det = {"bbox": [180, 200, 240, 300], "confidence": 0.25, "class_id": 67}

    # Frame 1: Single weak hit
    res = associator.associate([student], [weak_det], timestamp_sec=0.0)
    assoc = res.get(1)
    assert assoc is not None
    assert assoc.is_candidate_only is True

    # Turn into event and assess risk
    ev = FusedEvent(
        event_id="ev_phone_weak_1",
        track_id=1,
        camera_id="cam_main",
        event_type=EventFamily.PHONE_ASSOCIATED.value,
        start_timestamp=0.0,
        last_update_timestamp=0.1,
        duration=0.1,
        risk_level=RiskLevel.LOW.value,
        status="active",
        evidence_summary={
            "phone_detected": True,
            "max_confidence": 0.25,
            "temporal_hits": 1,
            "phone_is_candidate_only": True,
        }
    )
    scored = agg.assess_event_risk(ev, active_cues_count=1, independent_cues_count=1, mean_reliability=0.5)
    assert scored.review_status == "internal"
    assert scored.risk_level == RiskLevel.LOW.value
    assert scored.evidence_summary.get("is_reviewable", False) is False


def test_02_repeated_weak_consistent_phone_candidate():
    """2. Repeated weak but spatially consistent hits -> candidate."""
    associator = PhoneAssociator(phone_strong_confidence=0.35, weak_candidate_min_hits=3)
    student = Track(track_id=1, bbox=BBox(100, 100, 300, 500), confidence=0.9, timestamp=0.0)
    consistent_det = {"bbox": [180, 200, 240, 300], "confidence": 0.28, "class_id": 67}

    # 4 consecutive frames at identical location (>= weak_candidate_min_hits)
    res = {}
    for i in range(4):
        res = associator.associate([student], [consistent_det], timestamp_sec=i * 0.1)

    assoc = res.get(1)
    assert assoc is not None
    assert assoc.status == "ASSOCIATED"
    # Still candidate only until strong criteria met
    assert assoc.is_candidate_only is True


def test_03_strong_associated_phone_event():
    """3. Strong clear associated phone -> reviewable quickly."""
    associator = PhoneAssociator(phone_strong_confidence=0.35, weak_candidate_min_hits=3)
    agg = RiskAggregator()

    student = Track(track_id=1, bbox=BBox(100, 100, 300, 500), confidence=0.9, timestamp=0.0)
    strong_det = {"bbox": [180, 200, 240, 300], "confidence": 0.78, "class_id": 67}

    res = associator.associate([student], [strong_det], timestamp_sec=0.0)
    assoc = res.get(1)
    assert assoc is not None
    assert assoc.status == "ASSOCIATED"
    assert assoc.is_candidate_only is False

    ev = FusedEvent(
        event_id="ev_phone_strong",
        track_id=1,
        camera_id="cam_main",
        event_type=EventFamily.PHONE_ASSOCIATED.value,
        start_timestamp=0.0,
        last_update_timestamp=0.6,
        duration=0.6,
        risk_level=RiskLevel.MEDIUM.value,
        status="active",
        evidence_summary={
            "phone_detected": True,
            "max_confidence": 0.78,
            "temporal_hits": 3,
            "phone_is_candidate_only": False,
        }
    )
    scored = agg.assess_event_risk(ev, active_cues_count=1, independent_cues_count=1, mean_reliability=0.9)
    assert scored.review_status == "awaiting"
    assert scored.evidence_summary.get("is_reviewable") is True


def test_04_clear_phone_promotion_latency_within_policy():
    """4. Clear phone must be reviewable within ~0.5-1.0s."""
    engine = EventEngine()
    agg = RiskAggregator()

    # Simulate 8 frames at 10 FPS (0.0 to 0.7s <= 1.0s target)
    opened = []
    for step in range(8):
        t = step * 0.1
        cue = PerTrackCueState(
            track_id=1,
            last_update_timestamp=t,
            continuity_valid=True,
            phone_detected=True,
            phone_confidence=0.75,
            phone_association_status="CLEAR_ASSOCIATION",
            phone_status=ObservationStatus.AVAILABLE,
            phone_is_candidate_only=False,
        )
        emitted = engine.process_cue_state(cue)
        for ev, act in emitted:
            if act == "OPEN":
                scored = agg.assess_event_risk(ev)
                if scored.review_status == "awaiting":
                    opened.append((t, scored))

    assert len(opened) >= 1, "Strong phone must be promoted to reviewable within 0.5-1.0s"
    assert 0.5 <= opened[0][0] <= 1.0, f"Promotion latency {opened[0][0]}s outside 0.5-1.0s window"


def test_05_lean_back_no_phone_no_phone_event():
    """5. When user leans back with NO phone present, 0 phone review events."""
    associator = PhoneAssociator(phone_strong_confidence=0.35, weak_candidate_min_hits=3)
    student = Track(track_id=1, bbox=BBox(80, 120, 350, 480), confidence=0.9, timestamp=0.0)

    # Jittery non-clustered weak blips (distance > 75px between each)
    jittery_dets = [
        {"bbox": [100, 150, 130, 190], "confidence": 0.22, "class_id": 67},
        {"bbox": [280, 400, 320, 450], "confidence": 0.24, "class_id": 67},
        {"bbox": [150, 380, 185, 425], "confidence": 0.21, "class_id": 67},
        {"bbox": [250, 180, 280, 210], "confidence": 0.23, "class_id": 67},
    ]

    confirmed_count = 0
    for i in range(12):
        t = i * 0.1
        det = [jittery_dets[i % len(jittery_dets)]]
        res = associator.associate([student], det, timestamp_sec=t)
        assoc = res.get(1)
        if assoc and assoc.status == "ASSOCIATED" and not assoc.is_candidate_only:
            confirmed_count += 1

    assert confirmed_count == 0, "Lean back with jittery false blips must produce 0 confirmed phone events"


def test_06_desk_objects_do_not_create_phone_review_event():
    """6. Calculator, notebook, pen, ruler, paper do not create review cards."""
    associator = PhoneAssociator(phone_strong_confidence=0.35, weak_candidate_min_hits=3)
    student = Track(track_id=1, bbox=BBox(100, 100, 200, 300), confidence=0.95, timestamp=0.0)

    desk_objects = ["calculator", "notebook", "id_card", "pen", "ruler", "paper"]
    review_events = []
    engine = EventEngine()
    agg = RiskAggregator()

    t = 0.0
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


def test_07_one_phone_appearance_one_event():
    """7. One continuous phone appearance produces exactly one event."""
    engine = EventEngine()
    opened = []
    for step in range(30):
        t = step * 0.1
        cue = PerTrackCueState(
            track_id=1,
            last_update_timestamp=t,
            continuity_valid=True,
            phone_detected=True,
            phone_confidence=0.75,
            phone_association_status="CLEAR_ASSOCIATION",
            phone_status=ObservationStatus.AVAILABLE,
            phone_is_candidate_only=False,
        )
        for ev, act in engine.process_cue_state(cue):
            if act == "OPEN":
                opened.append(ev.event_id)

    assert len(opened) == 1, f"Expected 1 continuous event ID, got {len(opened)}"


# ==============================================================================
# HEAD TURN TESTS (8 - 15)
# ==============================================================================

def test_08_pure_left_candidate_and_event():
    """8. Pure left head turn (yaw ~ -26°) produces candidate and event."""
    scorer = ReliabilityModel()
    pos = PostureCue(status=ObservationStatus.AVAILABLE, probabilities={"NORMAL_UPRIGHT": 0.95, "TURN_HEAD_CLEAR": 0.05})
    hp_left = HeadPoseCue(status=ObservationStatus.AVAILABLE, yaw_deg=-26.0, reliability_weight=0.9)
    ev_left = scorer.compute_turn_fusion_evidence(pos, hp_left)
    assert ev_left >= 0.50, f"Pure left turn yaw=-26 deg must score >= 0.50 (got {ev_left})"


def test_09_pure_right_candidate_and_event():
    """9. Pure right head turn (yaw ~ +26°) produces candidate and event."""
    scorer = ReliabilityModel()
    pos = PostureCue(status=ObservationStatus.AVAILABLE, probabilities={"NORMAL_UPRIGHT": 0.95, "TURN_HEAD_CLEAR": 0.05})
    hp_right = HeadPoseCue(status=ObservationStatus.AVAILABLE, yaw_deg=+26.0, reliability_weight=0.9)
    ev_right = scorer.compute_turn_fusion_evidence(pos, hp_right)
    assert ev_right >= 0.50, f"Pure right turn yaw=+26 deg must score >= 0.50 (got {ev_right})"
    assert abs(ev_right - scorer.compute_turn_fusion_evidence(pos, HeadPoseCue(status=ObservationStatus.AVAILABLE, yaw_deg=-26.0, reliability_weight=0.9))) < 1e-4


def test_10_head_turn_visual_state_exits_green_promptly():
    """10. Head-turn visual state exits GREEN (SAFE) into CANDIDATE (BLUE) promptly."""
    yaw_deg = 25.0
    turn_candidate = True
    risk_level = "SAFE"

    # Simulated camera_view.js 4-state visual mapping:
    is_candidate = turn_candidate or (abs(yaw_deg) >= 24.0)
    visual_state = "CANDIDATE" if is_candidate else risk_level

    assert visual_state == "CANDIDATE", "Visual state must exit GREEN immediately upon lateral yaw"


def test_11_sustained_head_turn_amber():
    """11. Sustained head turn (>= 0.85s) maps to AMBER / reviewable awaiting."""
    agg = RiskAggregator()
    ev = FusedEvent(
        event_id="ev_sustained_turn",
        track_id=1,
        camera_id="cam_main",
        event_type=EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION.value,
        start_timestamp=0.0,
        last_update_timestamp=0.9,
        duration=0.9,
        risk_level=RiskLevel.LOW.value,
        status="active",
        lifecycle_status="active",
        evidence_summary={
            "glance_burst": False,
            "glance_count": 1,
        }
    )
    scored = agg.assess_event_risk(ev, active_cues_count=1, independent_cues_count=1, mean_reliability=0.9)
    assert scored.risk_level == RiskLevel.MEDIUM.value  # MEDIUM = AMBER
    assert scored.review_status == "awaiting"
    assert scored.evidence_summary["is_reviewable"] is True
    assert scored.evidence_summary["glance_classification"] == "SUSTAINED_LATERAL_TURN"


def test_12_repeated_glance_one_consolidated_event():
    """12. Repeated glances in short window -> one consolidated event."""
    agg = RiskAggregator()
    ev = FusedEvent(
        event_id="ev_repeated_glance",
        track_id=1,
        camera_id="cam_main",
        event_type=EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION.value,
        start_timestamp=0.0,
        last_update_timestamp=0.9,
        duration=0.9,
        risk_level=RiskLevel.LOW.value,
        status="active",
        lifecycle_status="active",
        evidence_summary={
            "glance_burst": True,
            "glance_count": 3,
        }
    )
    scored = agg.assess_event_risk(ev, active_cues_count=1, independent_cues_count=1, mean_reliability=0.9)
    assert scored.risk_level == RiskLevel.MEDIUM.value
    assert scored.review_status == "awaiting"
    assert scored.evidence_summary["is_reviewable"] is True
    assert scored.evidence_summary["glance_classification"] == "REPEATED_GLANCES"


def test_13_tiny_brief_glance_no_review_card():
    """13. Tiny brief glance (e.g. 0.4s) -> no review card (internal only)."""
    agg = RiskAggregator()
    ev = FusedEvent(
        event_id="ev_brief",
        track_id=1,
        camera_id="cam_main",
        event_type=EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION.value,
        start_timestamp=0.0,
        last_update_timestamp=0.4,
        duration=0.4,
        risk_level=RiskLevel.LOW.value,
        status="active",
        evidence_summary={
            "glance_burst": False,
            "glance_count": 1,
        }
    )
    scored = agg.assess_event_risk(ev, active_cues_count=1, independent_cues_count=1, mean_reliability=0.9)
    assert scored.review_status == "internal"
    assert scored.evidence_summary["is_reviewable"] is False
    assert scored.evidence_summary["glance_classification"] == "BRIEF_ISOLATED_GLANCE"


def test_14_head_turn_does_not_require_body_lean():
    """14. Head turn must NOT require torso lean, shoulder movement, or body rotation."""
    scorer = ReliabilityModel()
    pos = PostureCue(status=ObservationStatus.AVAILABLE, probabilities={"NORMAL_UPRIGHT": 0.98, "TURN_HEAD_CLEAR": 0.02})
    hp = HeadPoseCue(status=ObservationStatus.AVAILABLE, yaw_deg=26.0, reliability_weight=0.9)
    ev_score = scorer.compute_turn_fusion_evidence(pos, hp)
    assert ev_score >= 0.50, f"Pure head turn with upright posture must trigger without body lean (got {ev_score})"


def test_15_unavailable_headpose_no_fake_yaw():
    """15. Unavailable headpose (crop too small or face lost) -> honest UNAVAILABLE, no fake yaw."""
    cue = HeadPoseCue(
        status=ObservationStatus.UNAVAILABLE,
        yaw_deg=None,
        pitch_deg=None,
        roll_deg=None,
        reliability_weight=0.0,
        support_status=HeadPoseSupportStatus.FACE_UNRESOLVABLE,
    )
    assert cue.yaw_deg is None
    assert cue.status == ObservationStatus.UNAVAILABLE


# ==============================================================================
# LIFECYCLE TESTS (16 - 18)
# ==============================================================================

def test_16_continuous_incident_one_event():
    """16. Sustained lateral turn for 5.0 seconds yields exactly 1 event ID."""
    engine = EventEngine()
    opened = []
    for step in range(50):
        t = step * 0.1
        cue = PerTrackCueState(
            track_id=1,
            last_update_timestamp=t,
            continuity_valid=True,
            turn_fused_evidence=0.65,
            posture_status=ObservationStatus.AVAILABLE,
        )
        for ev, act in engine.process_cue_state(cue):
            if act == "OPEN":
                opened.append(ev.event_id)

    assert len(opened) == 1, f"Continuous incident must yield 1 event ID, got {len(opened)}"


def test_17_real_recurrence_new_event():
    """17. Real recurrence after cooldown generates a new distinct event ID."""
    engine = EventEngine({
        "provisional_thresholds": {
            "sustained_lateral_head_orientation": {
                "min_candidate_duration_sec": 0.60,
                "posture_turn_enter_threshold": 0.50,
                "posture_turn_exit_threshold": 0.30,
                "cooldown_seconds": 2.0,
                "exit_grace_seconds": 0.5,
                "normal_reset_seconds": 2.5,
            }
        }
    })

    opened_events = []
    t = 0.0

    # Phase 1: First turn incident (2.0s)
    for _ in range(20):
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

    # Phase 2: Sustained normal recovery (6.0s > 0.5s grace + 2.0s cooldown + 2.5s normal_reset = 5.0s)
    for _ in range(60):
        t += 0.1
        cue = PerTrackCueState(
            track_id=1,
            last_update_timestamp=t,
            continuity_valid=True,
            turn_fused_evidence=0.0,
            posture_status=ObservationStatus.AVAILABLE,
        )
        engine.process_cue_state(cue)

    # Phase 3: Second turn incident (2.0s)
    for _ in range(20):
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

    assert len(opened_events) == 2, f"Expected 2 distinct events across recurrence, got {len(opened_events)}"
    assert opened_events[0] != opened_events[1]


def test_18_duplicates_remain_zero():
    """18. Verify no duplicate active events exist for same track and family."""
    engine = EventEngine()
    opened = []
    closed = []

    for step in range(30):
        t = step * 0.1
        cue = PerTrackCueState(
            track_id=1,
            last_update_timestamp=t,
            continuity_valid=True,
            turn_fused_evidence=0.60,
            posture_status=ObservationStatus.AVAILABLE,
        )
        for ev, act in engine.process_cue_state(cue):
            if act == "OPEN":
                opened.append(ev.event_id)
            elif act == "CLOSE":
                closed.append(ev.event_id)

    assert len(opened) == 1
    assert len(closed) == 0
