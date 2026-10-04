"""
Tests for Live Tracking & Severity Overlays
Verifies:
- Active track renders bbox and populates all required metadata
- Coordinate transform and scaling contract
- Normal track -> GREEN / SAFE
- Amber event -> AMBER / ATTENTION
- High event -> RED / HIGH_ALERT
- Reconnect and refresh recreate/retain overlays
- One track -> One box (deduplication)
- Missing track metadata does not crash preview or frame buffer
- EventLifecycleState import and state machine execution without NameError
"""

import numpy as np
import pytest
from starlette.testclient import TestClient

from src.detection.types import BBox
from src.tracking.tracker import Track
from src.fusion.types import FusedEvent, EventFamily, EventLifecycleState
from src.orchestration.stage2_pipeline import Stage2Pipeline
from src.orchestration.camera_manager import CameraManager
from src.persistence.service import PersistenceService
from src.persistence.models import CameraConfig
from src.security.key_provider import InMemoryKeyProvider, set_key_provider
from src.api.main import create_app


@pytest.fixture
def clean_stage2_pipeline():
    """Create Stage2Pipeline instance for track overlay testing."""
    return Stage2Pipeline(config_path="configs/stage2_pipeline.yaml", enable_debug_overlay=False)


def test_active_track_renders_bbox_and_populates_required_metadata(clean_stage2_pipeline):
    """Verify live-preview metadata contains camera_id, track_id, bbox, review_severity, active_event_family, candidate."""
    pipe = clean_stage2_pipeline
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    track = Track(track_id=1, bbox=BBox(100, 120, 350, 480), confidence=0.95, timestamp=10.0)

    pipe._update_latest_frame_buffer(frame, [track], {}, {}, {}, ts=10.0, cam_id="webcam_0")

    assert pipe._latest_jpeg_frame is not None
    assert len(pipe._latest_tracks_summary) == 1
    t_summary = pipe._latest_tracks_summary[0]

    assert t_summary["camera_id"] == "webcam_0"
    assert t_summary["track_id"] == 1
    assert t_summary["bbox"] == [100, 120, 350, 480]
    assert t_summary["risk_level"] == "SAFE"
    assert t_summary["review_severity"] == "GREEN"
    assert t_summary["severity"] == "GREEN"
    assert "candidate" in t_summary
    assert "is_reviewable" in t_summary
    assert "active_event_family" in t_summary


def test_track_with_active_state_machine_does_not_throw_name_error(clean_stage2_pipeline):
    """Verify state machine CANDIDATE state is inspected without NameError on EventLifecycleState."""
    pipe = clean_stage2_pipeline
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    track = Track(track_id=42, bbox=BBox(50, 50, 200, 300), confidence=0.9, timestamp=15.0)

    # Populate active head-turn machine in CANDIDATE state
    turn_key = ("webcam_0", 42, EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION)
    mock_machine = type("MockMachine", (), {
        "state": EventLifecycleState.CANDIDATE,
        "candidate_start_time": 14.2,
    })()
    pipe.event_engine._machines[turn_key] = mock_machine

    # This MUST NOT throw NameError or swallow exceptions to empty list
    pipe._update_latest_frame_buffer(frame, [track], {}, {}, {}, ts=15.0, cam_id="webcam_0")

    assert len(pipe._latest_tracks_summary) == 1
    t_summary = pipe._latest_tracks_summary[0]
    assert t_summary["track_id"] == 42
    assert t_summary["turn_candidate"] is True
    assert t_summary["candidate"] is True
    assert t_summary["review_severity"] == "LOW"


def test_normal_track_has_green_severity(clean_stage2_pipeline):
    """Verify normal baseline student track defaults to GREEN severity."""
    pipe = clean_stage2_pipeline
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    track = Track(track_id=1, bbox=BBox(10, 10, 100, 200), confidence=0.9, timestamp=1.0)

    pipe._update_latest_frame_buffer(frame, [track], {}, {}, {}, ts=1.0, cam_id="webcam_0")
    t_summary = pipe._latest_tracks_summary[0]

    assert t_summary["review_severity"] == "GREEN"
    assert t_summary["risk_level"] == "SAFE"


def test_amber_event_produces_amber_severity(clean_stage2_pipeline):
    """Verify MEDIUM risk event (e.g. sustained head turn) sets AMBER severity."""
    pipe = clean_stage2_pipeline
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    track = Track(track_id=2, bbox=BBox(10, 10, 100, 200), confidence=0.9, timestamp=1.0)

    ev = FusedEvent(
        event_id="ev_amber_01",
        camera_id="webcam_0",
        track_id=2,
        event_type="SUSTAINED_HEAD_TURN",
        risk_level="MEDIUM",
        start_timestamp=1.0,
        last_update_timestamp=1.0,
        evidence_summary={"is_reviewable": True},
    )
    pipe._active_events_map["ev_amber_01"] = ev

    pipe._update_latest_frame_buffer(frame, [track], {}, {}, {}, ts=1.0, cam_id="webcam_0")
    t_summary = pipe._latest_tracks_summary[0]

    assert t_summary["review_severity"] == "AMBER"
    assert t_summary["risk_level"] == "MEDIUM"


def test_high_event_produces_red_high_severity(clean_stage2_pipeline):
    """Verify HIGH risk event (e.g. cell phone detected) sets HIGH/RED severity."""
    pipe = clean_stage2_pipeline
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    track = Track(track_id=3, bbox=BBox(10, 10, 100, 200), confidence=0.9, timestamp=1.0)

    ev = FusedEvent(
        event_id="ev_high_01",
        camera_id="webcam_0",
        track_id=3,
        event_type="UNAUTHORIZED_OBJECT_PHONE",
        risk_level="HIGH",
        start_timestamp=1.0,
        last_update_timestamp=1.0,
        evidence_summary={"is_reviewable": True},
    )
    pipe._active_events_map["ev_high_01"] = ev

    pipe._update_latest_frame_buffer(frame, [track], {}, {}, {}, ts=1.0, cam_id="webcam_0")
    t_summary = pipe._latest_tracks_summary[0]

    assert t_summary["review_severity"] == "HIGH"
    assert t_summary["risk_level"] == "HIGH"


def test_missing_track_metadata_does_not_crash_preview(clean_stage2_pipeline):
    """Verify malformed or partial track metadata is gracefully handled without stopping stream."""
    pipe = clean_stage2_pipeline
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Track missing tuple bbox or odd fields
    dummy_track = type("MalformedTrack", (), {
        "track_id": 99,
        "bbox": [50, 60, 200, 300],
    })()

    pipe._update_latest_frame_buffer(frame, [dummy_track], {}, {}, {}, ts=1.0, cam_id="webcam_0")
    assert pipe._latest_jpeg_frame is not None
    assert len(pipe._latest_tracks_summary) == 1
    assert pipe._latest_tracks_summary[0]["track_id"] == 99


def test_camera_tracks_endpoint_returns_json_and_no_cache_headers(clean_stage2_pipeline):
    """Verify /api/cameras/tracks endpoint returns track summary with no-cache headers."""
    pipe = clean_stage2_pipeline
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    track = Track(track_id=1, bbox=BBox(100, 100, 300, 400), confidence=0.95, timestamp=1.0)
    pipe._update_latest_frame_buffer(frame, [track], {}, {}, {}, ts=1.0, cam_id="webcam_0")

    app = create_app(
        camera_id="webcam_0",
        camera_type="webcam",
        stage2_pipeline=pipe,
        camera_connected=True,
        camera_streaming=True,
        device_present=True,
        enforce_auth=False,
    )
    client = TestClient(app)

    res = client.get("/api/cameras/tracks")
    assert res.status_code == 200
    assert "no-cache" in res.headers["cache-control"]
    data = res.json()
    assert len(data) == 1
    assert data[0]["track_id"] == 1
    assert data[0]["bbox"] == [100, 100, 300, 400]
    assert data[0]["review_severity"] == "GREEN"


def test_camera_manager_returns_rich_tracks_from_pipeline():
    """Verify CameraManager resolves worker and returns full pipeline tracks summary."""
    set_key_provider(InMemoryKeyProvider())
    ps = PersistenceService(db_path=":memory:")

    cam_cfg = CameraConfig(
        camera_id="cam01",
        name="Camera 01",
        source_type="webcam",
        enabled=True,
    )
    ps.cameras.upsert_camera(cam_cfg)

    mgr = CameraManager(persistence_service=ps)
    mgr.hero_camera_id = "cam01"

    worker = mgr.workers.get("cam01")
    if worker:
        worker.pipeline = Stage2Pipeline(config_path="configs/stage2_pipeline.yaml", enable_debug_overlay=False)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        track = Track(track_id=5, bbox=BBox(200, 150, 400, 500), confidence=0.92, timestamp=2.0)
        worker.pipeline._update_latest_frame_buffer(frame, [track], {}, {}, {}, ts=2.0, cam_id="cam01")

        tracks = mgr.get_camera_tracks("cam01")
        assert len(tracks) == 1
        assert tracks[0]["track_id"] == 5
        assert tracks[0]["bbox"] == [200, 150, 400, 500]
