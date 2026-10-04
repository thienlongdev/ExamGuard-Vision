"""Tests for multi-camera single-node orchestration, state isolation, and failure resilience."""

import os
import shutil
import tempfile
import time
import pytest
from unittest.mock import MagicMock, patch

from src.security.key_provider import InMemoryKeyProvider, set_key_provider
from src.persistence.service import PersistenceService
from src.persistence.models import CameraConfig
from src.orchestration.model_registry import ModelRegistry
from src.orchestration.camera_manager import CameraManager, CameraPipelineWorker, CameraTelemetry
from src.orchestration.stage2_pipeline import Stage2Pipeline
from src.security.redactor import redact_sensitive_text


@pytest.fixture
def multi_camera_env():
    """Isolated environment for multi-camera testing."""
    temp_dir = tempfile.mkdtemp()
    kp = InMemoryKeyProvider()
    set_key_provider(kp)

    db_path = os.path.join(temp_dir, "test_multicam.db")
    ps = PersistenceService(db_path=db_path)

    # Register two distinct camera configs
    cam1 = CameraConfig(
        camera_id="cam01",
        name="Camera Phòng thi 01 (Trước)",
        source_type="video_file",
        source_uri_ref="tests/fixtures/sample_exam.mp4",
        enabled=True,
        resolution_width=1280,
        resolution_height=720,
        target_capture_fps=30.0,
        room="Phòng 402",
    )
    cam2 = CameraConfig(
        camera_id="cam02",
        name="Camera Phòng thi 02 (Sau)",
        source_type="video_file",
        source_uri_ref="tests/fixtures/sample_exam.mp4",
        enabled=True,
        resolution_width=1280,
        resolution_height=720,
        target_capture_fps=30.0,
        room="Phòng 402",
    )
    ps.cameras.upsert_camera(cam1)
    ps.cameras.upsert_camera(cam2)

    yield ps, temp_dir

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_shared_model_registry_singleton_and_inference_lock():
    """Verify ModelRegistry is a shared singleton with an active RLock across pipelines."""
    reg1 = ModelRegistry.get_instance()
    reg2 = ModelRegistry.get_instance()

    assert reg1 is reg2
    assert hasattr(reg1, "inference_lock")
    assert reg1.inference_lock is not None

    # Test lock reentrancy in same thread
    with reg1.inference_lock:
        with reg2.inference_lock:
            assert True


def test_camera_state_isolation_same_track_id(multi_camera_env):
    """Verify track #1 on cam01 and track #1 on cam02 remain strictly isolated."""
    ps, temp_dir = multi_camera_env
    reg = ModelRegistry.get_instance()

    cfg1 = ps.cameras.get_camera("cam01")
    cfg2 = ps.cameras.get_camera("cam02")

    worker1 = CameraPipelineWorker(config=cfg1, shared_registry=reg, persistence_service=ps)
    worker2 = CameraPipelineWorker(config=cfg2, shared_registry=reg, persistence_service=ps)

    # Initialize isolated Stage2 pipelines
    pipe1 = Stage2Pipeline(camera_id="cam01", model_registry=reg)
    pipe2 = Stage2Pipeline(camera_id="cam02", model_registry=reg)

    assert pipe1.camera_id == "cam01"
    assert pipe2.camera_id == "cam02"

    from src.fusion.types import UnifiedTrackUpdate, PhoneCue

    # Feed distinct track states for track_id = 1 on separate cameras
    up1 = UnifiedTrackUpdate(
        track_id=1,
        camera_id="cam01",
        timestamp_sec=100.0,
        phone=PhoneCue(detected=True, association_confidence=0.95),
    )
    up2 = UnifiedTrackUpdate(
        track_id=1,
        camera_id="cam02",
        timestamp_sec=100.0,
        phone=PhoneCue(detected=False, association_confidence=0.0),
    )

    pipe1.temporal_buffer.push(up1)
    pipe2.temporal_buffer.push(up2)

    # Verify states on pipe 1 do not leak to pipe 2
    hist1 = pipe1.temporal_buffer.get_track_history(track_id=1, window_seconds=10.0, camera_id="cam01")
    hist2 = pipe2.temporal_buffer.get_track_history(track_id=1, window_seconds=10.0, camera_id="cam02")

    assert len(hist1) == 1
    assert hist1[0].phone.detected is True
    assert hist1[0].phone.association_confidence == 0.95

    assert len(hist2) == 1
    assert hist2[0].phone.detected is False


def test_rtsp_secret_redaction():
    """Verify sensitive RTSP credentials are redacted from logs and telemetry."""
    secret_url = "rtsp://admin:VerySecretPass123@192.168.1.100:554/stream1"
    redacted = redact_sensitive_text(secret_url)

    assert "VerySecretPass123" not in redacted
    assert "@192.168.1.100" in redacted
    assert "••••••••" in redacted or "***" in redacted


def test_camera_telemetry_model():
    """Verify CameraTelemetry correctly formats metrics for the multi-camera UI grid."""
    telemetry = CameraTelemetry(
        camera_id="cam01",
        name="Camera 1",
        source_type="webcam",
        configured=True,
        connected=True,
        streaming=True,
        status_display="TRỰC TIẾP",
        capture_fps=29.8,
        processing_fps=28.5,
        active_tracks=4,
    )

    t_dict = telemetry.to_dict()
    assert t_dict["camera_id"] == "cam01"
    assert t_dict["status"] == "STREAMING"
    assert t_dict["status_display"] == "TRỰC TIẾP"
    assert t_dict["fps"] == 28.5
    assert t_dict["active_tracks"] == 4


def test_camera_failure_isolation(multi_camera_env):
    """Verify that stopping or failing one camera worker does not crash the other."""
    ps, temp_dir = multi_camera_env
    reg = ModelRegistry.get_instance()

    cfg1 = ps.cameras.get_camera("cam01")
    cfg2 = ps.cameras.get_camera("cam02")

    worker1 = CameraPipelineWorker(config=cfg1, shared_registry=reg, persistence_service=ps)
    worker2 = CameraPipelineWorker(config=cfg2, shared_registry=reg, persistence_service=ps)

    # Simulate worker1 active and streaming
    worker1.telemetry.connected = True
    worker1.telemetry.streaming = True
    worker1.telemetry.status_display = "TRỰC TIẾP"

    # Simulate worker2 encountering disconnect
    worker2.telemetry.connected = False
    worker2.telemetry.streaming = False
    worker2.telemetry.status_display = "MẤT KẾT NỐI"

    assert worker1.telemetry.streaming is True
    assert worker2.telemetry.streaming is False

    # Stopping worker2 must have zero impact on worker1
    worker2.stop()
    assert worker1.telemetry.streaming is True
    assert worker1.telemetry.status_display == "TRỰC TIẾP"
