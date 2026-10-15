"""Controlled capability/failure tests; these do not claim physical hardware coverage."""
import json
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.orchestration import adaptive_runtime as ar
from src.video.webcam import WebcamSource


def test_camera_identity_persists_when_index_changes():
    cams = [{"id": "b", "index": 0}, {"id": "a", "index": 1}]
    assert ar.choose_camera(cams, "a")["index"] == 1


def test_camera_identity_missing_never_switches():
    with pytest.raises(RuntimeError, match="disconnected"):
        ar.choose_camera([{"id": "replacement", "index": 0}], "original")


def test_missing_camera_blocks():
    with pytest.raises(RuntimeError, match="No physical camera"):
        ar.choose_camera([])


def test_invalid_camera_choice():
    with pytest.raises(RuntimeError, match="unavailable"):
        ar.choose_camera([{"id": "a", "index": 0}], index=2)


def test_capability_selection_does_not_depend_on_gpu_name():
    results = {k: {"forward_ms": 10} for k in ("detector", "macro", "posture", "headpose")}
    assert ar.select_cadence(results, {"gpu": "unseen GPU", "vram_mb": 8000}, 30) == ar.select_cadence(results, {"gpu": "other GPU", "vram_mb": 8000}, 30)
    assert ar.select_cadence(results, {"vram_mb": 1000}, 30)["detector_hz"] <= 8


def test_detector_resolution_policy_preserves_low_vram_reference():
    high_a = {"device": "cuda:0", "gpu": "unseen GPU", "vram_mb": 12227, "ram_available_mb": 16000}
    high_b = {"device": "cuda:0", "gpu": "different GPU", "vram_mb": 12227, "ram_available_mb": 16000}
    asus_reference = {"device": "cuda:0", "gpu": "RTX 3050", "vram_mb": 4096, "ram_available_mb": 8000}

    assert ar.select_detector_image_size(high_a) == 960
    assert ar.select_detector_image_size(high_b) == 960
    assert ar.select_detector_image_size(asus_reference) == 640
    assert ar.select_detector_image_size({"device": "cpu", "vram_mb": 16384, "ram_available_mb": 16000}) == 640


def test_measured_throughput_changes_cadence():
    fast = {"model": {"forward_ms": 10}}
    slow = {"model": {"forward_ms": 200}}
    assert ar.select_cadence(slow, {}, 30)["detector_hz"] < ar.select_cadence(fast, {}, 30)["detector_hz"]


def test_cpu_fallback_is_reported(monkeypatch):
    monkeypatch.setattr(ar.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ar.shutil, "which", lambda _: None)
    data = ar.discover_hardware()
    assert data["device"] == "cpu"
    assert data["fallback_reason"]
    assert data["supported_backends"] == ["cpu"]


def test_cuda_kernel_failure_reports_reason(monkeypatch):
    monkeypatch.setattr(ar.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ar.shutil, "which", lambda _: None)
    def fail(*args, **kwargs):
        raise RuntimeError("incompatible CUDA kernel")
    monkeypatch.setattr(ar.torch, "ones", fail)
    data = ar.discover_hardware()
    assert data["device"] == "cpu" and "incompatible" in data["fallback_reason"]


def fake_capture(monkeypatch, accepted="MJPG", shape=(720, 1280, 3)):
    cap = MagicMock()
    cap.isOpened.return_value = True
    props = {cv2.CAP_PROP_FRAME_WIDTH: shape[1], cv2.CAP_PROP_FRAME_HEIGHT: shape[0],
             cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*accepted)}
    cap.get.side_effect = lambda key: props.get(key, 0)
    cap.read.return_value = (True, np.zeros(shape, np.uint8))
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: cap)
    return cap


def test_real_accepted_mode_contract(monkeypatch):
    cap = fake_capture(monkeypatch)
    source = WebcamSource(fourcc="MJPG", preferred_backend="CAP_DSHOW")
    assert source.open() and source.actual_fourcc == "MJPG"
    keys = [call.args[0] for call in cap.set.call_args_list]
    assert keys[:4] == [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS, cv2.CAP_PROP_FOURCC]
    source.release()
    cap.release.assert_called_once()


def test_yuy2_cannot_claim_mjpeg(monkeypatch):
    cap = fake_capture(monkeypatch, accepted="YUY2")
    source = WebcamSource(fourcc="MJPG")
    assert not source.open()
    cap.release.assert_called_once()


def test_wrong_resolution_rejected(monkeypatch):
    cap = fake_capture(monkeypatch, shape=(480, 640, 3))
    source = WebcamSource(fourcc="MJPG")
    assert not source.open()
    cap.release.assert_called_once()


def test_identity_checked_before_acquiring_camera(monkeypatch):
    capture = MagicMock()
    monkeypatch.setattr(cv2, "VideoCapture", capture)
    source = WebcamSource()
    source.identity_check = lambda: False
    assert not source.open()
    capture.assert_not_called()


def test_changed_camera_invalidates_config_cache():
    hardware = {"versions": {"torch": "test"}, "device": "cpu"}
    assert ar.fingerprint(hardware, {"id": "a"}) != ar.fingerprint(hardware, {"id": "b"})


def test_corrupt_checkpoint_is_blocked(monkeypatch, tmp_path):
    from src.pilot import preflight
    monkeypatch.setattr(ar, "ROOT", tmp_path)
    monkeypatch.setattr(ar, "LOCAL", tmp_path)
    (tmp_path / "model.pt").write_bytes(b"bad" * 1000)
    monkeypatch.setattr(preflight, "CERTIFIED_HASHES", {"model.pt": "expected"})
    with pytest.raises(RuntimeError, match="Corrupt checkpoint"):
        ar.validate_checkpoints()


def test_readiness_does_not_trust_health_only():
    from fastapi.testclient import TestClient
    from src.api.main import create_app
    client = TestClient(create_app())
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503


def test_rtsp_default_identity_never_contains_credentials():
    from src.video.factory import create_video_source
    source = create_video_source("rtsp", "rtsp://operator:private-password@example.invalid/live")
    assert "private-password" not in source.source_id
    assert "operator" not in source.source_id


def test_dpapi_camera_secret_round_trip():
    from src.video.source_settings import protect_rtsp_url, resolve_rtsp_url, ROOT
    url = "rtsp://local-test:random-fixture@127.0.0.1:1/not-a-live-camera"
    reference = protect_rtsp_url(url)
    path = ROOT / "storage/security" / reference.removeprefix("dpapi:")
    try:
        assert url.encode() not in path.read_bytes()
        assert resolve_rtsp_url(reference) == url
    finally:
        path.unlink()  # This test created the disposable secret.


def test_dpapi_reference_rejects_traversal():
    from src.video.source_settings import resolve_rtsp_url
    with pytest.raises(ValueError):
        resolve_rtsp_url("dpapi:../../master-key.dpapi")
