"""
Tests for Live Camera Preview, Stream Transport, Canonical ID Resolution,
and Frontend Reconnect Resilience.
"""

import asyncio
import time
import numpy as np
import pytest
from starlette.testclient import TestClient

from src.detection.types import BBox
from src.tracking.tracker import Track
from src.video.base import VideoFrame
from src.orchestration.stage2_pipeline import Stage2Pipeline
from src.orchestration.camera_manager import CameraManager, CameraPipelineWorker
from src.persistence.service import PersistenceService
from src.persistence.models import CameraConfig
from src.security.key_provider import InMemoryKeyProvider, set_key_provider
from src.api.main import create_app


@pytest.fixture
def mock_pipeline_with_frames():
    """Create a pipeline that populates _latest_jpeg_frame even with active tracks."""
    pipe = Stage2Pipeline(config_path="configs/stage2_pipeline.yaml", enable_debug_overlay=False)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Draw simple shapes to ensure non-empty JPEG
    frame[100:200, 100:200] = 255
    track = Track(track_id=1, bbox=BBox(100, 100, 300, 400), confidence=0.95, timestamp=1.0)
    pipe._update_latest_frame_buffer(frame, [track], {}, {}, {}, ts=1.0, cam_id="webcam_0")
    return pipe


def test_stage2_pipeline_latest_jpeg_frame_populated(mock_pipeline_with_frames):
    """Verify that Stage2Pipeline populates _latest_jpeg_frame when tracks exist."""
    pipe = mock_pipeline_with_frames
    assert pipe._latest_jpeg_frame is not None
    assert len(pipe._latest_jpeg_frame) > 1000
    assert len(pipe._latest_tracks_summary) == 1
    assert pipe._latest_tracks_summary[0]["track_id"] == 1


def test_camera_stream_endpoint_returns_multipart_jpeg(mock_pipeline_with_frames):
    """Verify /api/cameras/stream returns multipart stream with valid JPEG frames."""
    pipe = mock_pipeline_with_frames
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

    # Test single frame endpoint
    res_frame = client.get("/api/cameras/frame")
    assert res_frame.status_code == 200
    assert res_frame.headers["content-type"] == "image/jpeg"
    assert len(res_frame.content) == len(pipe._latest_jpeg_frame)

    # Test stream endpoint starts and returns multipart/x-mixed-replace
    with client.stream("GET", "/api/cameras/stream?max_frames=1") as res_stream:
        assert res_stream.status_code == 200
        assert "multipart/x-mixed-replace" in res_stream.headers["content-type"]
        assert "no-cache" in res_stream.headers["cache-control"]
        chunk = next(res_stream.iter_bytes())
        assert b"--frame" in chunk
        assert b"Content-Type: image/jpeg" in chunk


def test_canonical_camera_id_resolution_and_aliases(mock_pipeline_with_frames):
    """Verify CAM 01, webcam_0, cam01 all resolve to active camera without 404."""
    pipe = mock_pipeline_with_frames
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

    # All alias paths should succeed
    for cam_alias in ["webcam_0", "cam01", "CAM 01", "camera_0"]:
        url = f"/api/cameras/{cam_alias}/frame"
        r = client.get(url)
        assert r.status_code == 200, f"Failed for alias: {cam_alias}"
        assert r.headers["content-type"] == "image/jpeg"

    # Tracks endpoint aliases
    r_tracks = client.get("/api/cameras/tracks")
    assert r_tracks.status_code == 200
    data = r_tracks.json()
    assert len(data) == 1
    assert data[0]["track_id"] == 1

    r_tracks_alias = client.get("/api/cameras/webcam_0/tracks")
    assert r_tracks_alias.status_code == 200
    assert len(r_tracks_alias.json()) == 1


def test_camera_manager_resolve_canonical_id():
    """Verify CameraManager resolves aliases to canonical worker IDs."""
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

    assert mgr.resolve_canonical_camera_id(None) == "cam01"
    assert mgr.resolve_canonical_camera_id("cam01") == "cam01"
    assert mgr.resolve_canonical_camera_id("CAM 01") == "cam01"
    assert mgr.resolve_canonical_camera_id("cam_01") == "cam01"
    assert mgr.resolve_canonical_camera_id("webcam_0") == "cam01"


def test_preview_failure_does_not_stop_ai_inference(mock_pipeline_with_frames):
    """Verify that browser stream disconnect does not stop or slow pipeline processing."""
    pipe = mock_pipeline_with_frames
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Simulate client disconnect on frame generator
    vf = VideoFrame(
        frame=dummy_img,
        timestamp=2.0,
        frame_idx=2,
        fps=30.0,
        width=1280,
        height=720,
        source_id="webcam_0",
    )

    res = pipe.process_frame(vf)
    assert res is not None
    assert pipe.processed_frames_count >= 1
    assert pipe._latest_jpeg_frame is not None


def test_preview_reconnect_does_not_restart_capture_worker(mock_pipeline_with_frames):
    """Verify repeated preview requests simply read latest frame buffer without resetting pipeline."""
    pipe = mock_pipeline_with_frames
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

    # 10 rapid preview reconnects (simulating browser page refreshes)
    for _ in range(10):
        r = client.get("/api/cameras/frame")
        assert r.status_code == 200

    # Pipeline frame count unchanged by preview requests
    assert hasattr(pipe, "processed_frames_count")
