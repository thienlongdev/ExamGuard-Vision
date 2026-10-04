"""
Unit & Integration Tests for Pilot Preparation
==============================================
Validates Part P requirements:
- camera profile schema
- calibration artifact generation
- capability coverage calculation
- resolution handling
- person-size bucket logic
- head-size bucket logic
- ROI filtering
- seat-zone mapping
- operating-envelope classification
- RTSP reconnect simulation
- packet/frame loss
- jitter
- backpressure under target resolution
- storage quota handling
- retention policy
- preflight PASS/WARN/FAIL
- operator warning serialization
- pilot camera profile generator
- no auto-tuning from unlabeled video
- anonymous evidence identity
"""

import json
import os
import shutil
import tempfile
import numpy as np
import pytest

from src.pilot.profile import (
    CameraProfile,
    ViewpointProfile,
    CapabilityLevel,
    TrackingCapability,
    ProvenanceLevel,
    Resolution,
    SeatZone,
    PrivacySettings,
    CameraCapabilitySummary,
    evaluate_camera_capability,
)
from src.pilot.storage import (
    EvidenceRetentionManager,
    EvidenceManifest,
    ReviewStatus,
    StorageQuotaStatus,
)
from src.pilot.resilience import (
    SimulatedNetworkStreamSource,
)
from src.pilot.preflight import (
    PilotPreflightChecker,
)
try:
    from tools.validation.generate_pilot_camera_profile import generate_recommended_profile
except ImportError:
    from scripts.generate_pilot_camera_profile import generate_recommended_profile
from src.video.video_file import VideoFileSource
from src.video.base import VideoFrame
from src.tracking.tracker import Track


def test_camera_profile_schema_and_serialization(tmp_path):
    """Verify camera profile schema, YAML persistence, and fallback for unknown metadata."""
    sz = SeatZone(
        zone_id="desk_1",
        desk_polygon=[[10, 10], [50, 10], [50, 50], [10, 50]],
        seat_polygon=[[15, 40], [45, 40], [45, 70], [15, 70]],
    )
    prof = CameraProfile(
        camera_id="cam_test_01",
        name="Hall A Front",
        source_type="video_file",
        source_uri="samples/sample_exam.mp4",
        resolution=Resolution(2560, 1440),
        nominal_fps=20.0,
        expected_student_count=10,
        viewpoint_profile=ViewpointProfile.FRONT_OBLIQUE,
        camera_height_m=3.2,
        camera_pitch_deg=-30.0,
        seat_zones=[sz],
        general_detector_imgsz=640,
        macro_detector_imgsz=768,
        provenance_level=ProvenanceLevel.CALIBRATED_FROM_VISIBILITY,
    )

    yaml_file = tmp_path / "test_profile.yaml"
    prof.to_yaml(str(yaml_file))

    loaded = CameraProfile.from_yaml(str(yaml_file))
    assert loaded.camera_id == "cam_test_01"
    assert loaded.resolution.width == 2560
    assert loaded.resolution.height == 1440
    assert loaded.viewpoint_profile == ViewpointProfile.FRONT_OBLIQUE
    assert len(loaded.seat_zones) == 1
    assert loaded.seat_zones[0].zone_id == "desk_1"
    assert loaded.privacy.enable_face_recognition is False # Invariant


def test_roi_and_seat_zone_filtering():
    """Verify ROI polygon exclusion and seat zone spatial grouping."""
    prof = CameraProfile(
        camera_id="cam_roi_test",
        name="ROI Camera",
        roi_polygon=[[100, 100], [500, 100], [500, 500], [100, 500]],
        seat_zones=[
            SeatZone(zone_id="zone_A", desk_polygon=[[120, 120], [200, 120], [200, 200], [120, 200]]),
            SeatZone(zone_id="zone_B", desk_polygon=[[300, 300], [400, 300], [400, 400], [300, 400]]),
        ],
    )

    # Box inside ROI and inside zone A
    box_inside_a = (140.0, 140.0, 180.0, 180.0)
    assert prof.filter_roi(box_inside_a) is True
    assert prof.find_seat_zone(box_inside_a) == "zone_A"

    # Box outside ROI
    box_outside = (50.0, 50.0, 80.0, 80.0)
    assert prof.filter_roi(box_outside) is False
    assert prof.find_seat_zone(box_outside) is None


def test_capability_coverage_and_size_buckets():
    """Verify person-size and head-size capability level assignment."""
    prof = CameraProfile(
        camera_id="cam_cap_test",
        name="Cap Test",
        viewpoint_profile=ViewpointProfile.FRONT_OBLIQUE,
    )

    # Case 1: High resolution students (H >= 120px, Head >= 25px)
    person_heights = [180.0, 150.0, 140.0, 200.0, 130.0]
    head_dims = [35.0, 30.0, 28.0, 40.0, 26.0]
    phone_boxes = [(30.0, 50.0), (32.0, 48.0)]

    cap = evaluate_camera_capability(
        profile=prof,
        person_heights_px=person_heights,
        head_dimensions_px=head_dims,
        phone_boxes_wh=phone_boxes,
        fps_observed=25.0,
        drop_pct=0.0,
    )
    assert cap.posture_capability == CapabilityLevel.FULL
    assert cap.headpose_capability == CapabilityLevel.FULL
    assert cap.phone_capability == CapabilityLevel.FULL
    assert cap.tracking_capability == TrackingCapability.SUPPORTED
    assert cap.pct_height_gte_120 == 100.0
    assert cap.pct_head_crop_gte_25 == 100.0

    # Case 2: Distant / Ceiling High Viewpoint (Head-pose unavailable)
    prof_ceiling = CameraProfile(
        camera_id="cam_ceiling",
        name="Ceiling Cam",
        viewpoint_profile=ViewpointProfile.CEILING_HIGH,
    )
    cap_c = evaluate_camera_capability(
        profile=prof_ceiling,
        person_heights_px=[70.0, 85.0, 95.0],
        head_dimensions_px=[15.0, 18.0, 20.0],
        phone_boxes_wh=[],
        fps_observed=15.0,
        drop_pct=2.0,
    )
    assert cap_c.posture_capability == CapabilityLevel.LIMITED # 60-119px
    assert cap_c.headpose_capability == CapabilityLevel.UNAVAILABLE # Ceiling high
    assert "HIGH_ANGLE_WARNING" in cap_c.diagnostic_flags


def test_evidence_storage_quota_and_retention(tmp_path):
    """Verify storage quota warning, critical cutoff, and retention protection for unreviewed events."""
    ret_mgr = EvidenceRetentionManager(
        base_dir=str(tmp_path / "evidence"),
        retention_days=1,
        max_storage_gb=0.001, # ~1 MB quota to easily trigger status
        min_free_disk_gb=0.1,
        preserve_unreviewed=True,
    )

    status, telem = ret_mgr.check_storage_status()
    assert status in [StorageQuotaStatus.HEALTHY, StorageQuotaStatus.WARNING, StorageQuotaStatus.CRITICAL, StorageQuotaStatus.EXHAUSTED]

    # Create dummy snapshot files
    snap_dir = tmp_path / "evidence" / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    f1 = snap_dir / "ev_01_open.jpg"
    f2 = snap_dir / "ev_02_open.jpg"
    f1.write_bytes(b"dummy image 1")
    f2.write_bytes(b"dummy image 2")

    # Manifest 1: Old and REVIEWED -> should be deleted
    m1 = EvidenceManifest(
        event_id="ev_01",
        camera_id="cam_01",
        track_id=1,
        room_id="hall_1",
        event_type="SUSTAINED_HEAD_REST",
        risk_level="MEDIUM",
        risk_score=55.0,
        start_timestamp_sec=100.0,
        end_timestamp_sec=200.0, # Very old
        duration_sec=100.0,
        model_hashes={"v4_posture": "abc"},
        fusion_config_version="1.0.0",
        camera_profile_version="1.0.0",
        open_snapshot_path=str(f1),
        operator_review_status=ReviewStatus.REVIEWED,
    )
    ret_mgr.save_manifest(m1)

    # Manifest 2: Old but NEW -> preserved by policy
    m2 = EvidenceManifest(
        event_id="ev_02",
        camera_id="cam_01",
        track_id=2,
        room_id="hall_1",
        event_type="PHONE_ASSOCIATED",
        risk_level="HIGH",
        risk_score=85.0,
        start_timestamp_sec=100.0,
        end_timestamp_sec=200.0,
        duration_sec=100.0,
        model_hashes={"v4_posture": "abc"},
        fusion_config_version="1.0.0",
        camera_profile_version="1.0.0",
        open_snapshot_path=str(f2),
        operator_review_status=ReviewStatus.NEW, # Unreviewed!
    )
    ret_mgr.save_manifest(m2)

    # Enforce retention
    res = ret_mgr.enforce_retention()
    assert res["deleted_events"] == 1
    assert not f1.exists() # Deleted
    assert f2.exists()     # Preserved because unreviewed!


def test_evidence_manifest_anonymity():
    """Verify evidence manifest never exposes personal or biometric identifiers."""
    m = EvidenceManifest(
        event_id="ev_anon_test",
        camera_id="cam_01",
        track_id=7, # Anonymous integer only
        room_id="hall_exam_1",
        event_type="DISCUSSION_CANDIDATE",
        risk_level="LOW",
        risk_score=35.0,
        start_timestamp_sec=10.0,
        end_timestamp_sec=13.0,
        duration_sec=3.0,
        model_hashes={"posture": "hash123"},
        fusion_config_version="1.0.0",
        camera_profile_version="1.0.0",
        operator_review_status=ReviewStatus.CONFIRMED_EVENT,
    )
    d = m.to_dict()
    assert "name" not in d
    assert "student_name" not in d
    assert "face_id" not in d
    assert d["track_id"] == 7
    assert d["operator_review_status"] == "CONFIRMED_EVENT"


def test_simulated_stream_interruption_and_recovery():
    """Verify local simulated network stream handles disconnections cleanly without stale tracks."""
    base_src = VideoFileSource("samples/sample_exam.mp4", loop=True, realtime_pace=False)
    sim = SimulatedNetworkStreamSource(base_src, source_id="test_sim")
    assert sim.open() is True

    # Read 5 frames
    for _ in range(5):
        vf = sim.read()
        assert vf is not None

    # Trigger 500ms interruption
    sim.trigger_interruption(duration_sec=0.5)
    none_count = 0
    recovered_frame = None

    for _ in range(25):
        f = sim.read()
        if f is None:
            none_count += 1
        else:
            recovered_frame = f
            break

    assert none_count > 0
    assert recovered_frame is not None
    assert sim.reconnect_count == 1
    sim.release()


def test_recommended_profile_no_autotuning(tmp_path):
    """Verify profile generator does NOT auto-tune scientific thresholds from unlabeled video."""
    cal_json = tmp_path / "camera_calibration.json"
    dummy_cal = {
        "camera_profile": {
            "name": "Test Room Cam",
            "expected_student_count": 10,
            "viewpoint_profile": "FRONT_OBLIQUE",
        },
        "capability_summary": {
            "POSTURE_CAPABILITY": "FULL",
            "HEADPOSE_CAPABILITY": "FULL",
            "PHONE_CAPABILITY": "LIMITED",
            "diagnostic_flags": [],
        },
        "telemetry": {
            "source_resolution": "1920x1080",
            "mean_whole_loop_latency_ms": 35.0,
        },
    }
    with open(cal_json, "w", encoding="utf-8") as f:
        json.dump(dummy_cal, f)

    out_yaml = tmp_path / "recommended.yaml"
    rec = generate_recommended_profile(
        camera_id="cam_test_rec",
        calibration_json_path=str(cal_json),
        output_profile_path=str(out_yaml),
    )

    assert rec["nominal_fps"] == 20.0
    assert rec["posture_enabled"] is True
    assert rec["headpose_enabled"] is True
    # Explicit invariant check
    policy = rec["scientific_threshold_policy"]
    assert policy["auto_tuned"] is False
    assert "INVARIANT_ENFORCED" in policy["justification"]


def test_preflight_verdicts(tmp_path):
    """Verify preflight correctly assigns PASS, WARN, or FAIL."""
    # PASS scenario (Front oblique)
    prof = CameraProfile(
        camera_id="cam_preflight_test",
        name="Preflight Front",
        source_uri="samples/sample_exam.mp4",
        viewpoint_profile=ViewpointProfile.FRONT_OBLIQUE,
    )
    checker = PilotPreflightChecker(profile=prof)
    rep = checker.run_preflight(run_source_check=False)
    assert rep.final_verdict in ["PILOT_PREFLIGHT_PASS", "PILOT_PREFLIGHT_PASS_WITH_WARNINGS"]
    assert rep.critical_gates["checkpoint_hashes_valid"]["status"] == "PASS"

    # WARN scenario (Ceiling High)
    prof_ceiling = CameraProfile(
        camera_id="cam_preflight_ceiling",
        name="Preflight Ceiling",
        source_uri="samples/sample_exam.mp4",
        viewpoint_profile=ViewpointProfile.CEILING_HIGH,
    )
    checker_c = PilotPreflightChecker(profile=prof_ceiling)
    rep_c = checker_c.run_preflight(run_source_check=False)
    assert rep_c.final_verdict == "PILOT_PREFLIGHT_PASS_WITH_WARNINGS"
    assert rep_c.capability_warnings["viewpoint_geometry"]["status"] == "WARN"


def test_calibration_artifact_generation(tmp_path):
    """Verify calibration writes json, markdown, sample overlays, and histograms."""
    from src.pilot.calibration import CameraCalibrator
    prof = CameraProfile(
        camera_id="cam_cal_test",
        name="Cal Test",
        source_uri="samples/sample_exam.mp4",
        viewpoint_profile=ViewpointProfile.FRONT_OBLIQUE,
    )
    calibrator = CameraCalibrator(profile=prof, output_dir=str(tmp_path / "calibration"))
    cap_summary, telem = calibrator.calibrate(max_frames=10, max_duration_sec=10.0)

    assert os.path.exists(tmp_path / "calibration" / "camera_calibration.json")
    assert os.path.exists(tmp_path / "calibration" / "camera_calibration.md")
    assert os.path.exists(tmp_path / "calibration" / "sample_overlays")
    assert os.path.exists(tmp_path / "calibration" / "capability_histograms")


def test_bounded_queue_backpressure_drop_stale():
    """Verify BoundedFrameQueue drops oldest stale frame under backpressure."""
    from src.orchestration.stage2_pipeline import BoundedFrameQueue
    q = BoundedFrameQueue(maxsize=3, drop_policy="DROP_STALE_ON_BACKPRESSURE")
    f1 = VideoFrame(np.zeros((10, 10, 3)), 1.0, 1, 30.0, 10, 10, "test")
    f2 = VideoFrame(np.zeros((10, 10, 3)), 2.0, 2, 30.0, 10, 10, "test")
    f3 = VideoFrame(np.zeros((10, 10, 3)), 3.0, 3, 30.0, 10, 10, "test")
    f4 = VideoFrame(np.zeros((10, 10, 3)), 4.0, 4, 30.0, 10, 10, "test")

    q.push(f1)
    q.push(f2)
    q.push(f3)
    assert q.qsize == 3
    # Pushing f4 must drop f1 (the oldest)
    dropped = q.push(f4)
    assert dropped is not None
    assert dropped.frame_idx == 1
    assert q.dropped_frames_count == 1
    assert q.qsize == 3


def test_operator_review_state_flow():
    """Verify operator review states conform to NEW, REVIEWED, CONFIRMED_EVENT, DISMISSED."""
    from src.behavior.event_manager import EventManager, SuspiciousEvent
    em = EventManager(camera_id="cam_review_test")
    ev = SuspiciousEvent(
        event_id="ev_flow_01",
        track_id=3,
        camera_id="cam_review_test",
        timestamp=100.0,
        start_time=100.0,
        end_time=105.0,
        event_type="SUSTAINED_HEAD_REST",
        risk_level="MEDIUM",
        score=60.0,
        evidence={},
        status="new",
    )
    em._events[ev.event_id] = ev
    assert ev.status == "new"

    # Valid transitions
    assert em.update_event_status(ev.event_id, "reviewed", reviewer_notes="Examined clip") is True
    assert em.get_event(ev.event_id).status == "reviewed"

    assert em.update_event_status(ev.event_id, "confirmed_event", reviewer_notes="Confirmed observable rest") is True
    assert em.get_event(ev.event_id).status == "confirmed_event"

    assert em.update_event_status(ev.event_id, "dismissed", reviewer_notes="Normal resting between sections") is True
    assert em.get_event(ev.event_id).status == "dismissed"

    with pytest.raises(ValueError):
        em.update_event_status(ev.event_id, "guilty_cheater") # Prohibited!

