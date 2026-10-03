"""
Tests for Stage 2 Event Integration & Multi-Cue Decision Logic
"""

import pytest
import numpy as np

from src.orchestration.phone_associator import PhoneAssociator
from src.tracking.tracker import Track
from src.detection.types import BBox, Detection
from src.fusion.types import PhoneAssociationStatus, EventFamily
from src.fusion.event_engine import EventEngine
from src.fusion.risk_aggregator import RiskAggregator
from src.fusion.cue_state import PerTrackCueState


def test_phone_association_ambiguity_gating():
    associator = PhoneAssociator(ambiguity_margin=0.20)

    # Two students seated side by side
    tracks = [
        Track(track_id=1, bbox=BBox(100, 100, 200, 300), confidence=0.9, timestamp=1.0),
        Track(track_id=2, bbox=BBox(210, 100, 310, 300), confidence=0.9, timestamp=1.0),
    ]

    # Phone placed right on the borderline between student 1 and student 2
    phone_det = Detection(
        bbox=BBox(200, 150, 220, 200),
        class_id=67,
        class_name="cell phone",
        confidence=0.85,
    )

    associations = associator.associate(tracks, [phone_det])

    # Because phone is near both students with close scores, both must be AMBIGUOUS!
    # Neither student should get an ASSOCIATED status!
    assert associations[1].status == "AMBIGUOUS"
    assert associations[2].status == "AMBIGUOUS"
    assert associations[1].association_status_enum == PhoneAssociationStatus.AMBIGUOUS_ASSOCIATION
    assert associations[2].association_status_enum == PhoneAssociationStatus.AMBIGUOUS_ASSOCIATION
    assert associations[1].detected is False
    assert associations[2].detected is False


def test_phone_association_clear():
    associator = PhoneAssociator(ambiguity_margin=0.20)

    # Two students far apart
    tracks = [
        Track(track_id=1, bbox=BBox(100, 100, 200, 300), confidence=0.9, timestamp=1.0),
        Track(track_id=2, bbox=BBox(600, 100, 700, 300), confidence=0.9, timestamp=1.0),
    ]

    # Phone clearly in lap of student 1
    phone_det = Detection(
        bbox=BBox(140, 200, 170, 250),
        class_id=67,
        class_name="cell phone",
        confidence=0.85,
    )

    associations = associator.associate(tracks, [phone_det])

    assert associations[1].status == "ASSOCIATED"
    assert associations[1].association_status_enum == PhoneAssociationStatus.CLEAR_ASSOCIATION
    assert associations[1].detected is True

    assert associations[2].status == "UNASSOCIATED"
    assert associations[2].association_status_enum == PhoneAssociationStatus.NO_PHONE
    assert associations[2].detected is False


def test_event_deduplication_and_lifecycle():
    engine = EventEngine()
    risk_agg = RiskAggregator()

    # Track 1 exhibits sustained head rest for 3.0 seconds
    timestamps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    opened_events = []
    updated_events = []
    closed_events = []

    for t in timestamps:
        # Sleep score 0.85, no read/write veto
        is_sleeping = (t <= 2.5)
        cue_state = PerTrackCueState(
            track_id=1,
            last_update_timestamp=t,
            posture_probs={"HEAD_REST_SLEEP": 0.85 if is_sleeping else 0.10, "NORMAL_READ_WRITE": 0.05},
            read_write_suppression_active=False,
            read_write_score=0.05,
            posture_reliability=1.0,
            continuity_valid=True,
        )

        emitted = engine.process_cue_state(cue_state)
        for ev, action in emitted:
            scored = risk_agg.assess_event_risk(ev)
            if action == "OPEN":
                opened_events.append(scored)
            elif action == "UPDATE":
                updated_events.append(scored)
            elif action == "CLOSE":
                closed_events.append(scored)

    # Exactly 1 OPEN and 1 CLOSE for the entire 2.5s episode
    assert len(opened_events) == 1
    assert len(closed_events) == 1
    assert opened_events[0].event_id == closed_events[0].event_id
    assert closed_events[0].duration >= 2.0
