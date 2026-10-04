"""
Automated Multi-Track Scalability, Reliability, and Isolation Tests
===================================================================
Enforces:
- Multiple track states evaluated independently
- Phone A assigned ONLY to Track A
- Ambiguous phone not forced onto wrong track
- Two phones / two tracks assigned 1-to-1 without crosstalk
- Standing baseline isolated strictly per track
- One track's behavior does not bleed into another track
- Max severity computed per track
- Event deduplication per (track_id, event_family)
- Track removal cleanly evicts temporal buffer and scheduler states
"""

import pytest
import numpy as np
import time

from src.detection.types import BBox, Detection, BehaviorDetection
from src.tracking.tracker import Track
from src.fusion.types import (
    ObservationStatus,
    PostureCue,
    HeadPoseCue,
    PhoneCue,
    MacroBehaviorCue,
    UnifiedTrackUpdate,
    TrackingState,
    FusedEvent,
    EventFamily,
    RiskLevel,
    PhoneAssociationStatus,
)
from src.fusion.cue_state import PerTrackCueState
from src.fusion.temporal_buffer import TemporalBuffer
from src.fusion.fusion_engine import MultiCueFusionEngine
from src.fusion.event_engine import EventEngine
from src.fusion.risk_aggregator import RiskAggregator
from src.orchestration.phone_associator import PhoneAssociator
from src.orchestration.stage2_pipeline import Stage2Pipeline
from src.video.base import VideoFrame


def test_multi_track_states_independently():
    """Verify that multiple tracks maintain strictly independent temporal buffer and cue states."""
    buf = TemporalBuffer(time_horizon_seconds=10.0)
    engine = MultiCueFusionEngine(temporal_buffer=buf)

    # Track 1: Normal upright
    u1 = UnifiedTrackUpdate(
        track_id=1,
        timestamp_sec=1.0,
        tracking=TrackingState(track_id=1, timestamp_sec=1.0, bbox=(100, 100, 200, 300)),
        posture=PostureCue(
            status=ObservationStatus.AVAILABLE,
            predicted_class="NORMAL_UPRIGHT",
            probabilities={"NORMAL_UPRIGHT": 0.90, "NORMAL_READ_WRITE": 0.10},
            confidence=0.90,
            reliability_weight=1.0,
        ),
        headpose=HeadPoseCue(
            status=ObservationStatus.AVAILABLE,
            yaw_deg=2.0,
            reliability_weight=1.0,
        ),
    )

    # Track 2: Turn head clear
    u2 = UnifiedTrackUpdate(
        track_id=2,
        timestamp_sec=1.0,
        tracking=TrackingState(track_id=2, timestamp_sec=1.0, bbox=(300, 100, 400, 300)),
        posture=PostureCue(
            status=ObservationStatus.AVAILABLE,
            predicted_class="TURN_HEAD_CLEAR",
            probabilities={"TURN_HEAD_CLEAR": 0.85, "NORMAL_UPRIGHT": 0.10},
            confidence=0.85,
            reliability_weight=1.0,
        ),
        headpose=HeadPoseCue(
            status=ObservationStatus.AVAILABLE,
            yaw_deg=35.0,
            reliability_weight=1.0,
        ),
    )

    s1 = engine.update_track(u1)
    s2 = engine.update_track(u2)

    assert s1.track_id == 1
    assert s1.posture_probs["NORMAL_UPRIGHT"] == 0.90
    assert s1.smoothed_yaw_deg == 2.0
    assert s1.turn_multi_cue_agreement is False

    assert s2.track_id == 2
    assert s2.posture_probs["TURN_HEAD_CLEAR"] == 0.85
    assert s2.smoothed_yaw_deg == 35.0
    assert s2.turn_multi_cue_agreement is True


def test_phone_a_assigned_only_track_a():
    """Verify phone placed near Track A is associated ONLY with Track A, not Track B."""
    associator = PhoneAssociator(ambiguity_margin=0.15)

    track_a = Track(track_id=10, bbox=BBox(100, 200, 250, 500), confidence=0.95, timestamp=1.0)
    track_b = Track(track_id=20, bbox=BBox(500, 200, 650, 500), confidence=0.95, timestamp=1.0)

    # Phone inside Track A's desk region
    phone_a = Detection(
        bbox=BBox(140, 350, 180, 420),
        class_id=67,
        class_name="cell phone",
        confidence=0.91,
    )

    results = associator.associate(tracks=[track_a, track_b], detections=[phone_a])

    assert results[10].status == "ASSOCIATED"
    assert results[10].association_status_enum == PhoneAssociationStatus.CLEAR_ASSOCIATION
    assert results[10].detected is True
    assert results[10].confidence == 0.91

    assert results[20].status == "UNASSOCIATED"
    assert results[20].association_status_enum == PhoneAssociationStatus.NO_PHONE
    assert results[20].detected is False


def test_ambiguous_phone_not_forced_onto_wrong_track():
    """Verify phone placed equidistantly between two tracks results in AMBIGUOUS, not forced association."""
    associator = PhoneAssociator(ambiguity_margin=0.20)

    track_a = Track(track_id=1, bbox=BBox(100, 200, 250, 500), confidence=0.9, timestamp=1.0)
    track_b = Track(track_id=2, bbox=BBox(260, 200, 410, 500), confidence=0.9, timestamp=1.0)

    # Phone right at the boundary between track_a and track_b
    border_phone = Detection(
        bbox=BBox(245, 300, 265, 350),
        class_id=67,
        class_name="cell phone",
        confidence=0.88,
    )

    results = associator.associate(tracks=[track_a, track_b], detections=[border_phone])

    assert results[1].status == "AMBIGUOUS"
    assert results[2].status == "AMBIGUOUS"
    assert results[1].detected is False
    assert results[2].detected is False


def test_two_phones_two_tracks_independent():
    """Verify that when 2 students each possess a phone, both are associated cleanly without crosstalk."""
    associator = PhoneAssociator(ambiguity_margin=0.15)

    track_a = Track(track_id=1, bbox=BBox(100, 200, 250, 500), confidence=0.95, timestamp=1.0)
    track_b = Track(track_id=2, bbox=BBox(500, 200, 650, 500), confidence=0.95, timestamp=1.0)

    phone_a = Detection(
        bbox=BBox(130, 320, 170, 390),
        class_id=67,
        class_name="cell phone",
        confidence=0.92,
    )
    phone_b = Detection(
        bbox=BBox(530, 320, 570, 390),
        class_id=67,
        class_name="cell phone",
        confidence=0.89,
    )

    results = associator.associate(tracks=[track_a, track_b], detections=[phone_a, phone_b])

    assert results[1].status == "ASSOCIATED"
    assert results[1].confidence == 0.92
    assert results[1].detected is True

    assert results[2].status == "ASSOCIATED"
    assert results[2].confidence == 0.89
    assert results[2].detected is True


def test_standing_baseline_isolated_by_track():
    """Verify that seated baseline and standing evaluation is tracked per track ID without global pollution."""
    # Test pipeline seated baseline storage isolation
    p = Stage2Pipeline(config_path="configs/stage2_pipeline.yaml")

    # Frame 1 to 15: Establish baselines for track 1 (seated) and track 2 (seated)
    for frame_idx in range(1, 16):
        t1 = Track(track_id=1, bbox=BBox(100, 200, 250, 450), confidence=0.95, timestamp=frame_idx * 0.1)
        t2 = Track(track_id=2, bbox=BBox(500, 200, 650, 450), confidence=0.95, timestamp=frame_idx * 0.1)
        frame_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        vf = VideoFrame(frame=frame_img, timestamp=frame_idx * 0.1, frame_idx=frame_idx, fps=10.0, width=1280, height=720, source_id="test")
        p.process_frame(vf, injected_tracks=[t1, t2])

    meta1 = p._track_metadata.get(1, {})
    meta2 = p._track_metadata.get(2, {})

    assert meta1.get("baseline_y1") is not None
    assert meta2.get("baseline_y1") is not None
    assert abs(meta1["baseline_y1"] - 200.0) < 1.0
    assert abs(meta2["baseline_y1"] - 200.0) < 1.0

    # Frame 16: Track 1 stands (y1 moves up from 200 to 80, height increases from 250 to 370)
    # Track 2 remains seated at y1=200, h=250
    t1_standing = Track(track_id=1, bbox=BBox(100, 80, 250, 450), confidence=0.95, timestamp=1.6)
    t2_seated = Track(track_id=2, bbox=BBox(500, 200, 650, 450), confidence=0.95, timestamp=1.6)

    frame_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    vf = VideoFrame(frame=frame_img, timestamp=1.6, frame_idx=16, fps=10.0, width=1280, height=720, source_id="test")
    p.process_frame(vf, injected_tracks=[t1_standing, t2_seated])

    # Track 2 baseline must remain unaffected by track 1 standing
    assert abs(p._track_metadata[2]["baseline_y1"] - 200.0) < 1.0


def test_one_track_event_does_not_affect_another():
    """Verify that when Track 1 triggers a sustained event, Track 2 remains GREEN/SAFE."""
    engine = EventEngine()
    risk_agg = RiskAggregator()

    # Track 1 turns head sustainedly for 3 seconds
    # Track 2 remains normal upright
    for step in range(25):
        t = step * 0.1

        # Cue state 1: Lateral head orientation
        cs1 = PerTrackCueState(
            track_id=1,
            last_update_timestamp=t,
            posture_status=ObservationStatus.AVAILABLE,
            posture_probs={"TURN_HEAD_CLEAR": 0.85, "NORMAL_UPRIGHT": 0.10},
            headpose_status=ObservationStatus.AVAILABLE,
            smoothed_yaw_deg=35.0,
            yaw_abs_mean=35.0,
            posture_reliability=1.0,
            headpose_reliability=1.0,
            turn_fused_evidence=0.85,
            turn_multi_cue_agreement=True,
            continuity_valid=True,
        )

        # Cue state 2: Normal upright
        cs2 = PerTrackCueState(
            track_id=2,
            last_update_timestamp=t,
            posture_status=ObservationStatus.AVAILABLE,
            posture_probs={"NORMAL_UPRIGHT": 0.95, "NORMAL_READ_WRITE": 0.05},
            headpose_status=ObservationStatus.AVAILABLE,
            smoothed_yaw_deg=2.0,
            yaw_abs_mean=2.0,
            posture_reliability=1.0,
            headpose_reliability=1.0,
            turn_multi_cue_agreement=False,
            continuity_valid=True,
        )

        emitted1 = engine.process_cue_state(cs1)
        emitted2 = engine.process_cue_state(cs2)

        # Track 2 must never emit any turn event
        assert len(emitted2) == 0

    # Track 1 must have an active event
    active1 = engine.get_active_events()
    t1_events = [e for e in active1 if e.track_id == 1]
    t2_events = [e for e in active1 if e.track_id == 2]

    assert len(t1_events) >= 1
    assert t1_events[0].event_type == EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION.value
    assert len(t2_events) == 0


def test_max_severity_computed_per_track():
    """Verify that downstream summary calculates risk severity independently per track."""
    p = Stage2Pipeline(config_path="configs/stage2_pipeline.yaml")

    # Manually register an active HIGH event for track 5
    ev_high = FusedEvent(
        event_id="ev_high_005",
        track_id=5,
        camera_id="cam_0",
        event_type=EventFamily.PHONE_ASSOCIATED.value,
        start_timestamp=1.0,
        last_update_timestamp=2.0,
        risk_level=RiskLevel.HIGH.value,
        evidence_summary={"risk_assessment": {"final_score": 85.0, "risk_level": "HIGH"}},
    )
    p._active_events_map[ev_high.event_id] = ev_high

    t5 = Track(track_id=5, bbox=BBox(100, 100, 200, 300), confidence=0.9, timestamp=2.0)
    t6 = Track(track_id=6, bbox=BBox(400, 100, 500, 300), confidence=0.9, timestamp=2.0)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    p._update_latest_frame_buffer(
        frame=frame,
        tracks=[t5, t6],
        posture_cues={},
        headpose_cues={},
        phone_associations={},
    )

    summary = {item["track_id"]: item for item in p._latest_tracks_summary}
    assert summary[5]["risk_level"] == "HIGH"
    assert summary[6]["risk_level"] == "LOW"


def test_event_dedup_per_track_and_family():
    """Verify that a continuous single incident produces exactly 1 OPEN event and updates, not duplicate OPENs."""
    engine = EventEngine()

    opens = 0
    updates = 0

    for step in range(30):
        t = step * 0.1
        cs = PerTrackCueState(
            track_id=3,
            last_update_timestamp=t,
            posture_status=ObservationStatus.AVAILABLE,
            posture_probs={"HEAD_REST_SLEEP": 0.88, "NORMAL_UPRIGHT": 0.05},
            posture_reliability=1.0,
            continuity_valid=True,
        )
        emitted = engine.process_cue_state(cs)
        for ev, action in emitted:
            if action == "OPEN":
                opens += 1
            elif action == "UPDATE":
                updates += 1

    assert opens == 1, f"Expected exactly 1 OPEN event, got {opens}"
    assert updates >= 5, f"Expected multiple updates for ongoing event, got {updates}"


def test_track_removal_cleans_associated_temporal_state():
    """Verify that when a track expires, its temporal buffer and scheduler states are pruned."""
    p = Stage2Pipeline(config_path="configs/stage2_pipeline.yaml")

    # Ingest track 42 for a few frames
    for i in range(5):
        t = Track(track_id=42, bbox=BBox(100, 100, 200, 300), confidence=0.9, timestamp=10.0 + i * 0.1)
        vf = VideoFrame(frame=np.zeros((720, 1280, 3), dtype=np.uint8), timestamp=10.0 + i * 0.1, frame_idx=i, fps=10.0, width=1280, height=720, source_id="cam_0")
        p.process_frame(vf, injected_tracks=[t])

    assert 42 in p._track_metadata
    assert 42 in p.crop_scheduler._track_states
    assert 42 in p.temporal_buffer.active_track_ids("cam_0")

    # Clean up expired track (simulate track disappearing for > 15s)
    p.crop_scheduler.cleanup_expired_tracks(active_track_ids=[])
    assert 42 not in p.crop_scheduler._track_states

    evicted = p.temporal_buffer.evict_inactive_tracks(current_timestamp=30.0, camera_id="cam_0")
    assert 42 in evicted
    assert 42 not in p.temporal_buffer.active_track_ids("cam_0")
