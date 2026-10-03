import json
import pytest
from pathlib import Path
from src.api.schemas import (
    CameraInfo,
    CameraCounts,
    SystemStatusResponse,
    ConfiguredRates,
    ObservedRates,
    EventResponse
)
from src.api.static_ui import DASHBOARD_HTML
from src.orchestration.runtime_profile import (
    RuntimeProfileName,
    RuntimeProfileConfig,
    BOOTSTRAP_PROFILE_CANDIDATE,
    evaluate_runtime_profile_from_telemetry
)
from scripts.laptop_preflight import check_git_lfs_pointer

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_camera_counts_separation_in_schemas():
    """Verify configured camera != connected camera and connected != streaming."""
    counts = CameraCounts(
        registered=1,
        configured=1,
        connected=0,
        streaming=0
    )
    assert counts.configured != counts.connected
    assert counts.connected == 0
    assert counts.streaming == 0

    cam = CameraInfo(
        camera_id="laptop_webcam_0",
        name="Integrated Webcam",
        source_type="webcam",
        configured=True,
        device_present=False,
        connected=False,
        streaming=False,
        status="NO_PHYSICAL_CAMERA",
        configured_capture_fps=30.0,
        configured_resolution="1280x720",
        observed_capture_fps=None,
        observed_resolution=None
    )
    assert cam.configured is True
    assert cam.connected is False
    assert cam.streaming is False
    assert cam.status == "NO_PHYSICAL_CAMERA"
    assert cam.observed_capture_fps is None


def test_system_status_no_camera_null_rates():
    """Verify that when active_streams == 0, observed FPS, effective FPS, and drop rate are null."""
    status = SystemStatusResponse(
        active_cameras=0,
        active_students=0,
        queue_depth=0,
        drop_percentage=None,
        effective_fps=None,
        inference_fps=None,
        observed_rates=ObservedRates(capture_fps=None, processed_fps=None, inference_fps=None),
        camera_counts=CameraCounts(registered=1, configured=1, connected=0, streaming=0)
    )
    assert status.active_cameras == 0
    assert status.effective_fps is None
    assert status.inference_fps is None
    assert status.drop_percentage is None
    assert status.observed_rates.capture_fps is None


def test_runtime_reset_independent_of_camera_restart():
    """Verify runtime state reset is decoupled from physical camera hardware release/reopen."""
    reset_artifact = REPO_ROOT / "runs" / "local_live" / "runtime_state_reset_validation.json"
    restart_artifact = REPO_ROOT / "runs" / "local_live" / "camera_restart_validation.json"

    assert reset_artifact.exists(), "runtime_state_reset_validation.json must exist"
    assert restart_artifact.exists(), "camera_restart_validation.json must exist"

    with open(reset_artifact, "r", encoding="utf-8") as f:
        reset_data = json.load(f)
    with open(restart_artifact, "r", encoding="utf-8") as f:
        restart_data = json.load(f)

    # Runtime reset must be YES
    assert reset_data.get("RUNTIME_STATE_RESET_PASS") == "YES"

    # Camera restart must be DEFERRED because no physical hardware exists on desktop
    assert restart_data.get("validation_scope") == "RUNTIME_STATE_RESET_ONLY"
    assert restart_data.get("physical_camera_restart_executed") is False
    assert restart_data.get("camera_restart_pass") is None
    assert restart_data.get("camera_restart_status") == "DEFERRED_NO_CAMERA_HARDWARE"


def test_software_operational_independent_of_live_camera_pass():
    """Verify software operational flags are YES while physical live camera flags are DEFERRED."""
    exec_state_path = REPO_ROOT / "runs" / "local_live" / "LOCAL_LIVE_EXECUTION_STATE.json"
    assert exec_state_path.exists()

    with open(exec_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    flags = state.get("readiness_flags", {})

    # Software branches are certified operational
    operational_branches = [
        "GENERAL_DETECTOR_OPERATIONAL",
        "TRACKER_OPERATIONAL",
        "POSTURE_BRANCH_OPERATIONAL",
        "HEADPOSE_BRANCH_OPERATIONAL",
        "MACRO_BRANCH_OPERATIONAL",
        "PHONE_BRANCH_OPERATIONAL",
        "V4D_FUSION_OPERATIONAL",
        "EVENT_LIFECYCLE_SOFTWARE_PASS",
        "EVIDENCE_PIPELINE_SOFTWARE_PASS",
        "RUNTIME_STATE_RESET_PASS"
    ]
    for b in operational_branches:
        assert flags.get(b) == "YES", f"{b} must be YES"

    # Physical live camera flags must be DEFERRED
    physical_flags = [
        "GENERAL_DETECTOR_LIVE_CAMERA_PASS",
        "BYTETRACK_LIVE_CAMERA_PASS",
        "POSTURE_BRANCH_LIVE_CAMERA_PASS",
        "HEADPOSE_BRANCH_LIVE_CAMERA_PASS",
        "MACRO_BRANCH_LIVE_CAMERA_PASS",
        "PHONE_BRANCH_LIVE_CAMERA_PASS",
        "V4D_FUSION_LIVE_CAMERA_PASS",
        "EVENT_LIFECYCLE_PHYSICAL_CAMERA_PASS",
        "EVIDENCE_FROM_PHYSICAL_CAMERA_PASS",
        "CAMERA_RESTART_PASS"
    ]
    for b in physical_flags:
        assert flags.get(b) == "DEFERRED_NO_CAMERA_HARDWARE", f"{b} must be DEFERRED_NO_CAMERA_HARDWARE"


def test_software_fixture_event_origin_explicit():
    """Verify test fixtures explicitly declare event_origin='SOFTWARE_VALIDATION_FIXTURE'."""
    ev = EventResponse(
        event_id="test-ev-01",
        camera_id="cam_0",
        track_id=1,
        timestamp=100.0,
        start_time=100.0,
        end_time=102.0,
        event_type="ORIENTATION_SUSTAINED_LEFT",
        risk_level="MEDIUM",
        score=68.5,
        evidence={"detail": "test fixture"},
        status="new",
        event_origin="SOFTWARE_VALIDATION_FIXTURE"
    )
    assert ev.event_origin == "SOFTWARE_VALIDATION_FIXTURE"
    assert ev.event_origin != "PHYSICAL_CAMERA_EVENT"


def test_software_fixture_evidence_origin_explicit():
    """Verify evidence validation artifact explicitly documents fixture origin."""
    evidence_val_path = REPO_ROOT / "runs" / "local_live" / "live_evidence_validation.json"
    assert evidence_val_path.exists()

    with open(evidence_val_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("EVIDENCE_PIPELINE_SOFTWARE_PASS") == "YES"
    assert data.get("EVIDENCE_FROM_PHYSICAL_CAMERA_PASS") == "DEFERRED_NO_CAMERA_HARDWARE"
    assert data.get("evidence_origin") == "SOFTWARE_VALIDATION_FIXTURE"


def test_dashboard_no_camera_state():
    """Verify dashboard UI contains semantic markup for no-camera state."""
    html = DASHBOARD_HTML
    assert "CONFIGURED / NOT CONNECTED" in html
    assert "NOT DETECTED" in html
    assert "INACTIVE" in html
    assert "camera-info-bar" in html


def test_human_tests_remain_not_run_without_camera():
    """Verify all interactive human tests remain NOT_RUN when no physical webcam exists."""
    obs_path = REPO_ROOT / "runs" / "local_live" / "live_behavior_observations.json"
    assert obs_path.exists()

    with open(obs_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tests = data.get("tests", [])
    assert len(tests) >= 11, "Must contain all 11 human test definitions"
    for t in tests:
        assert t.get("result") == "NOT_RUN", f"Test {t.get('test_id')} must have result NOT_RUN"
        assert t.get("reason") in ["DEFERRED_NO_CAMERA_HARDWARE", "NO_PHYSICAL_WEBCAM"]


def test_git_lfs_pointer_detection(tmp_path):
    """Verify Git LFS pointer detection identifies text pointers vs binary files."""
    pointer_content = (
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1218128f7c29c4e705da303\n"
        "size 284175765\n"
    )
    pointer_file = tmp_path / "pointer.pt"
    pointer_file.write_text(pointer_content, encoding="utf-8")

    real_file = tmp_path / "real.pt"
    real_file.write_bytes(b"\x80\x02}q\x00(X\x04\x00\x00\x00test" + b"\x00" * 4096)

    assert check_git_lfs_pointer(str(pointer_file)) is True
    assert check_git_lfs_pointer(str(real_file)) is False


def test_runtime_profile_framework():
    """Verify runtime profile candidate properties and scientific threshold invariance."""
    cfg = BOOTSTRAP_PROFILE_CANDIDATE
    assert cfg.profile_name == RuntimeProfileName.BALANCED
    assert cfg.camera_resolution == (1280, 720)
    assert cfg.camera_requested_fps == 30.0
    assert cfg.detector_imgsz == 640
    assert cfg.detector_cadence_hz == 12.0
    assert cfg.bytetrack_every_frame is True
    assert cfg.posture_resolution == (224, 224)
    assert cfg.posture_cadence_hz == 10.0
    assert cfg.headpose_resolution == (224, 224)
    assert cfg.headpose_cadence_hz == 5.0
    assert cfg.macro_imgsz == 768
    assert cfg.macro_cadence_hz == 4.0
    assert cfg.queue_depth == 5
    assert cfg.evidence_mode == "EVENT_SNAPSHOT_ONLY"
    assert cfg.behavior_thresholds_immutable is True
