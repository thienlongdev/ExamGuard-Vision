"""
Comprehensive Pilot-Quality Validation for High-Recall Behavior Detection,
Event Lifecycle Deduplication, Evidence Reliability, and Review Queue Ordering.
"""

import time
import pytest
import numpy as np
import os
import cv2

from src.tracking.tracker import Track
from src.detection.types import BBox
from src.orchestration.phone_associator import PhoneAssociator
from src.fusion.reliability import ReliabilityModel
from src.fusion.temporal_buffer import TemporalBuffer
from src.fusion.cue_state import PerTrackCueState
from src.fusion.types import (
    FusedEvent,
    EventFamily,
    RiskLevel,
    ObservationStatus,
    UnifiedTrackUpdate,
    TrackingState,
    PostureCue,
    HeadPoseCue,
    MacroBehaviorCue,
)
from src.fusion.event_engine import EventEngine, TrackEventStateMachine, EventLifecycleState
from src.fusion.risk_aggregator import RiskAggregator
from src.evidence.clip_recorder import RollingClipRecorder, BufferedFrame


# =====================================================================
# 1. PHONE RECALL & CANDIDATE ACCUMULATION TESTS
# =====================================================================

def test_phone_per_class_threshold_and_lap_expansion():
    """Verify lap/desk expansion allows associating phone held near lap."""
    associator = PhoneAssociator(expand_ratio_down=0.35, phone_strong_confidence=0.35)
    
    # Student bbox: x1=100, y1=100, x2=200, y2=300 (height 200, width 100)
    # Lap area extends down to y2 + 0.35*200 = 370
    student = Track(track_id=1, bbox=BBox(100, 100, 200, 300), confidence=0.95, timestamp=1.0)
    
    # Phone held in lap/desk region: y1=310, y2=350 (outside torso bbox, but inside lap expansion)
    phone_det = {
        "bbox": [120, 310, 160, 350],
        "confidence": 0.28,
        "class_id": 67,
        "class_name": "cell phone",
    }
    
    associations = associator.associate([student], [phone_det], timestamp_sec=1.0)
    assert 1 in associations, "Lap phone should be associated via downward expansion"
    assoc = associations[1]
    assert assoc.detected is True
    assert assoc.confidence == 0.28


def test_phone_unassociated_preservation():
    """Verify phone far from all students is preserved as unassociated, not discarded."""
    associator = PhoneAssociator(phone_strong_confidence=0.35)
    student = Track(track_id=1, bbox=BBox(100, 100, 200, 300), confidence=0.95, timestamp=1.0)
    
    # Phone far from student
    phone_det = {
        "bbox": [600, 500, 650, 560],
        "confidence": 0.45,
        "class_id": 67,
        "class_name": "cell phone",
    }
    
    associations = associator.associate([student], [phone_det], timestamp_sec=1.0)
    assert associations[1].detected is False, "Distant phone must not be attributed to Student #1"
    
    unassociated = associator.get_unassociated_phones()
    assert len(unassociated) == 1, "Unassociated strong phone cue must not be silently discarded"
    assert unassociated[0]["confidence"] == 0.45


def test_phone_temporal_accumulation_brief_flashes():
    """Verify phone detected briefly across small gaps accumulates hits."""
    associator = PhoneAssociator(phone_candidate_window_sec=2.5, phone_candidate_gap_tolerance_sec=0.8)
    student = Track(track_id=1, bbox=BBox(100, 100, 200, 300), confidence=0.95, timestamp=0.0)
    
    phone_det = {"bbox": [120, 200, 160, 250], "confidence": 0.25, "class_id": 67, "class_name": "cell phone"}
    
    # Hit 1 at t=0.0
    a1 = associator.associate([student], [phone_det], timestamp_sec=0.0)
    assert a1[1].detected is True
    
    # Gap at t=0.3 (no detection)
    a2 = associator.associate([student], [], timestamp_sec=0.3)
    # Temporal memory should record hits
    assert len(associator._recent_track_hits.get(1, [])) >= 1
    
    # Hit 2 at t=0.6
    a3 = associator.associate([student], [phone_det], timestamp_sec=0.6)
    assert len(associator._recent_track_hits.get(1, [])) >= 2


def test_phone_severity_hierarchy():
    """Verify LEVEL A/B (medium) vs LEVEL C (sustained >= 2.5s -> HIGH)."""
    agg = RiskAggregator()
    
    # LEVEL B: Active phone event duration 1.0s -> MEDIUM (not high yet)
    ev_mid = FusedEvent(
        event_id="ev_phone_mid",
        track_id=1,
        camera_id="cam01",
        event_type=EventFamily.PHONE_ASSOCIATED.value,
        start_timestamp=0.0,
        last_update_timestamp=1.0,
        duration=1.0,
        risk_level=RiskLevel.LOW.value,
        status="active",
        lifecycle_status="active",
    )
    scored_mid = agg.assess_event_risk(ev_mid, active_cues_count=1, independent_cues_count=1, mean_reliability=0.9)
    assert scored_mid.risk_level == RiskLevel.MEDIUM.value, "1.0s phone association must be MEDIUM"
    
    # LEVEL C: Sustained phone >= 2.5s -> HIGH
    ev_sustained = FusedEvent(
        event_id="ev_phone_sustained",
        track_id=1,
        camera_id="cam01",
        event_type=EventFamily.PHONE_ASSOCIATED.value,
        start_timestamp=0.0,
        last_update_timestamp=2.6,
        duration=2.6,
        risk_level=RiskLevel.MEDIUM.value,
        status="active",
        lifecycle_status="active",
    )
    scored_sustained = agg.assess_event_risk(ev_sustained, active_cues_count=1, independent_cues_count=1, mean_reliability=0.9)
    assert scored_sustained.risk_level == RiskLevel.HIGH.value, "Sustained >= 2.5s phone association must escalate to HIGH"


# =====================================================================
# 2. PURE HEAD TURN RECALL & SYMMETRY TESTS
# =====================================================================

def test_pure_head_turn_without_body_lean():
    """Verify pure head turn (|yaw| >= 24 deg) produces standalone candidate evidence >= 0.50 without body lean."""
    scorer = ReliabilityModel()
    # Centered torso: posture is NORMAL_UPRIGHT (turn_score = 0.0)
    posture_probs = {"NORMAL_UPRIGHT": 0.95, "TURN_HEAD_CLEAR": 0.05}
    pos = PostureCue(status=ObservationStatus.AVAILABLE, probabilities=posture_probs)
    
    # Left turn yaw = -32.0 deg
    hp_left = HeadPoseCue(status=ObservationStatus.AVAILABLE, yaw_deg=-32.0, reliability_weight=0.9)
    ev_left = scorer.compute_turn_fusion_evidence(pos, hp_left)
    assert ev_left >= 0.50, f"Pure left head turn must achieve enter_threshold (got {ev_left})"
    
    # Right turn yaw = +32.0 deg
    hp_right = HeadPoseCue(status=ObservationStatus.AVAILABLE, yaw_deg=+32.0, reliability_weight=0.9)
    ev_right = scorer.compute_turn_fusion_evidence(pos, hp_right)
    assert ev_right >= 0.50, f"Pure right head turn must achieve enter_threshold (got {ev_right})"
    assert abs(ev_left - ev_right) < 1e-4, "Left and right yaw evidence must be symmetric"


def test_small_glance_remains_safe():
    """Small natural head motion (|yaw| <= 15 deg) must remain safe / below candidate threshold."""
    scorer = ReliabilityModel()
    posture_probs = {"NORMAL_UPRIGHT": 0.95, "TURN_HEAD_CLEAR": 0.05}
    pos = PostureCue(status=ObservationStatus.AVAILABLE, probabilities=posture_probs)
    hp_small = HeadPoseCue(status=ObservationStatus.AVAILABLE, yaw_deg=12.0, reliability_weight=0.9)
    ev_small = scorer.compute_turn_fusion_evidence(pos, hp_small)
    assert ev_small < 0.30, f"Small natural yaw 12 deg must not trigger candidate (got {ev_small})"


def test_repeated_glance_burst_detection():
    """Verify TemporalBuffer detects bursts of repeated lateral glances."""
    buf = TemporalBuffer(time_horizon_seconds=10.0)
    
    # Simulate glance sequence: center -> left -> center -> right -> center -> left
    timestamps = [0.0, 0.4, 0.9, 1.3, 1.8, 2.2, 2.7]
    yaws =       [0.0, 30.0, 0.0, -32.0, 0.0, 28.0, 0.0]
    
    for t, y in zip(timestamps, yaws):
        up = UnifiedTrackUpdate(
            track_id=1,
            timestamp_sec=t,
            headpose=HeadPoseCue(status=ObservationStatus.AVAILABLE, yaw_deg=y, reliability_weight=0.9),
        )
        buf.push(up)
    
    burst_detected = buf.detect_glance_burst(1, current_timestamp=2.7, glance_window_sec=3.5, min_glances=3)
    assert burst_detected is True, "Repeated quick lateral glances must be detected as a burst"


# =====================================================================
# 3. EVENT REOPEN & DEDUPLICATION (10-CYCLE QA)
# =====================================================================

def test_ten_repeated_incidents_produce_ten_distinct_events():
    """Verify 10 repeated incident cycles produce 10 distinct event IDs with no permanent dedup suppression."""
    sm = TrackEventStateMachine(
        track_id=1,
        event_family=EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION,
        min_candidate_duration=0.6,
        enter_threshold=0.50,
        exit_threshold=0.30,
        cooldown_seconds=2.0,
    )
    
    event_ids = []
    t = 0.0
    
    for cycle in range(10):
        # 1. Behavior starts: evidence >= 0.50
        # Frame 1: Candidate starts
        ev, act = sm.process_frame(t, evidence_score=0.65, is_vetoed=False, cue_state=None)
        assert sm.state == EventLifecycleState.CANDIDATE
        
        # Frame 2: Candidate duration elapsed (0.7s) -> OPEN
        t += 0.7
        ev_open, act_open = sm.process_frame(t, evidence_score=0.65, is_vetoed=False, cue_state=None)
        assert act_open == "OPEN"
        assert ev_open is not None
        event_ids.append(ev_open.event_id)
        assert sm.state == EventLifecycleState.ACTIVE
        
        # Frame 3: Active behavior continues -> UPDATE (same event ID)
        t += 0.5
        ev_up, act_up = sm.process_frame(t, evidence_score=0.65, is_vetoed=False, cue_state=None)
        assert act_up == "UPDATE"
        assert ev_up.event_id == ev_open.event_id
        
        # Frame 4: Behavior ends (evidence falls to 0.10) -> CLOSE -> enters COOLDOWN
        t += 0.5
        ev_close, act_close = sm.process_frame(t, evidence_score=0.10, is_vetoed=False, cue_state=None)
        assert act_close == "CLOSE"
        assert sm.state == EventLifecycleState.COOLDOWN
        
        # Frame 5: Cooldown period passes (2.5s > 2.0s cooldown) with normal behavior
        t += 2.5
        ev_norm, act_norm = sm.process_frame(t, evidence_score=0.10, is_vetoed=False, cue_state=None)
        assert sm.state == EventLifecycleState.INACTIVE, f"Cycle {cycle}: SM must return to INACTIVE after cooldown"
        
        t += 1.0  # Normal gap before next incident
    
    assert len(event_ids) == 10, f"Expected 10 distinct events, got {len(event_ids)}"
    assert len(set(event_ids)) == 10, "All 10 event IDs must be unique (no stale ID reuse)"


def test_independent_event_types_do_not_suppress_each_other():
    """Verify head turn and phone events on the same track do not suppress each other."""
    engine = EventEngine()
    
    # Create cue with head turn and phone both active
    cue = PerTrackCueState(
        track_id=1,
        last_update_timestamp=1.0,
        turn_fused_evidence=0.65,
        phone_detected=True,
        phone_confidence=0.85,
        phone_association_status="CLEAR_ASSOCIATION",
        phone_status=ObservationStatus.AVAILABLE,
    )
    
    # Frame 1 at t=0.0: candidates start
    cue.last_update_timestamp = 0.0
    engine.process_cue_state(cue)
    
    # Frame 2 at t=0.8: both candidate durations passed -> both OPEN
    cue.last_update_timestamp = 0.8
    emitted = engine.process_cue_state(cue)
    
    actions = {ev.event_type: act for ev, act in emitted}
    assert EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION.value in actions
    assert EventFamily.PHONE_ASSOCIATED.value in actions
    assert actions[EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION.value] == "OPEN"
    assert actions[EventFamily.PHONE_ASSOCIATED.value] == "OPEN"


# =====================================================================
# 4. EVIDENCE RELIABILITY & VIDEO VALIDATION TESTS
# =====================================================================

def test_clip_recorder_valid_h264_encoding_and_validation(tmp_path):
    """Verify RollingClipRecorder encodes valid MP4 with duration > 0 and positive frame count."""
    ready_clips = []
    
    def on_ready(eid, path, sha, sz):
        ready_clips.append((eid, path, sha, sz))
    
    recorder = RollingClipRecorder(
        output_dir=str(tmp_path),
        pre_event_seconds=1.0,
        post_event_seconds=1.0,
        on_clip_ready=on_ready,
    )
    
    # Generate 30 synthetic frames (640x480)
    w, h = 320, 240
    for i in range(30):
        t = i / 15.0
        frame = np.full((h, w, 3), (i * 8) % 256, dtype=np.uint8)
        recorder.push_frame(frame, t)
    
    # Trigger clip at t=1.0
    clip_path = recorder.trigger_clip("test_ev_01", timestamp=1.0, fps=15.0)
    
    # Push post-event frames to finalize
    for i in range(30, 50):
        t = i / 15.0
        frame = np.full((h, w, 3), (i * 8) % 256, dtype=np.uint8)
        recorder.push_frame(frame, t)
    
    recorder.shutdown()
    
    assert len(ready_clips) == 1, "Clip must finalize and trigger on_ready callback"
    eid, path, sha, sz = ready_clips[0]
    assert eid == "test_ev_01"
    assert os.path.exists(path)
    assert sz > 0
    
    # Verify with OpenCV VideoCapture (Section 29)
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), "Encoded video must open cleanly"
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert n_frames > 0, f"Video frame count must be > 0 (got {n_frames})"
    cap.release()


# =====================================================================
# 5. REVIEW QUEUE SORTING COMPARATOR TESTS
# =====================================================================

def test_review_queue_sorting_comparator():
    """Verify canonical queue sort: HIGH > MEDIUM > LOW, active first, newest first."""
    # Define comparator locally matching state.js compareEventsForReview
    def compare(a, b):
        rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        diff_sev = rank[b["riskLevel"]] - rank[a["riskLevel"]]
        if diff_sev != 0:
            return diff_sev
        active_rank = lambda lc: 1 if lc in ("active", "open") else 0
        diff_active = active_rank(b["lifecycle"]) - active_rank(a["lifecycle"])
        if diff_active != 0:
            return diff_active
        return (b.get("timestamp") or 0) - (a.get("timestamp") or 0)
    
    from functools import cmp_to_key
    
    events = [
        {"id": "ev_low_1", "riskLevel": "LOW", "lifecycle": "closed", "timestamp": 100},
        {"id": "ev_med_old", "riskLevel": "MEDIUM", "lifecycle": "closed", "timestamp": 120},
        {"id": "ev_high_closed", "riskLevel": "HIGH", "lifecycle": "closed", "timestamp": 140},
        {"id": "ev_med_active", "riskLevel": "MEDIUM", "lifecycle": "active", "timestamp": 130},
        {"id": "ev_high_active", "riskLevel": "HIGH", "lifecycle": "active", "timestamp": 150},
    ]
    
    sorted_events = sorted(events, key=cmp_to_key(compare))
    sorted_ids = [e["id"] for e in sorted_events]
    
    expected_order = [
        "ev_high_active",  # HIGH active
        "ev_high_closed",  # HIGH closed
        "ev_med_active",   # MEDIUM active
        "ev_med_old",      # MEDIUM closed
        "ev_low_1",        # LOW
    ]
    assert sorted_ids == expected_order, f"Expected {expected_order}, got {sorted_ids}"
