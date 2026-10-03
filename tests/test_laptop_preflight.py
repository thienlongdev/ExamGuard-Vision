"""
Regression and validation tests for scripts/laptop_preflight.py.
Verifies:
1. Readiness variable is always deterministically defined.
2. Camera present vs absent paths handled cleanly without exceptions.
3. Successful CAP_DSHOW path is recognized.
4. No desktop hardcoded machine output (e.g. Ryzen 9 9950X, RTX 5070).
5. Clean structured report generation.
"""

import json
import os
import unittest.mock as mock
from pathlib import Path
import pytest

from scripts.laptop_preflight import (
    run_preflight,
    compare_expectation,
    query_windows_system_info,
    CERTIFIED_HASHES,
    EXPECTED_REFERENCE,
)


def test_laptop_preflight_readiness_always_defined():
    """Verify that run_preflight computes readiness without uninitialized variable crashes."""
    report = run_preflight()
    assert "readiness" in report
    readiness = report["readiness"]
    assert "READY_FOR_LAPTOP_LIVE_VALIDATION" in readiness
    assert readiness["READY_FOR_LAPTOP_LIVE_VALIDATION"] in [
        "YES",
        "NO",
    ] or readiness["READY_FOR_LAPTOP_LIVE_VALIDATION"].startswith("DEFERRED_NO_CAMERA_HARDWARE")
    assert "READY_FOR_PRODUCTION" in readiness
    assert "NO" in readiness["READY_FOR_PRODUCTION"]


def test_no_desktop_hardcoded_strings():
    """Verify no hardcoded desktop specs appear in system info or reports."""
    report = run_preflight()
    hw_str = json.dumps(report["hardware"]).upper()
    assert "9950X" not in hw_str
    assert "5070" not in hw_str


def test_camera_absent_path_structured():
    """Verify that when no camera is discovered, preflight returns clean structured verdict without crashing."""
    mock_cameras = ([], False, -1, "NONE", None, None, {
        "pnp_devices": [],
        "configured": True,
        "device_present": False,
        "open_success": False,
        "frame_success": False,
        "connected": False,
        "streaming": False,
        "detailed_modes": {},
    })
    with mock.patch("scripts.laptop_preflight.probe_cameras", return_value=mock_cameras):
        report = run_preflight()
        assert report["camera"]["discovered"] is False
        assert "DEFERRED_NO_CAMERA_HARDWARE" in report["readiness"]["READY_FOR_LAPTOP_LIVE_VALIDATION"]


def test_camera_present_path_structured():
    """Verify that when camera is discovered and produces frames, verdict is YES if all other checks pass."""
    mock_cameras = ([{
        "index": 0,
        "backend": "CAP_DSHOW",
        "configured": True,
        "device_present": True,
        "open_success": True,
        "frame_success": True,
        "connected": True,
        "streaming": False,
        "reported_width": 1280,
        "reported_height": 720,
        "reported_fps": 30.0,
        "actual_width": 1280,
        "actual_height": 720,
    }], True, 0, "CAP_DSHOW", (1280, 720), 30.0, {
        "pnp_devices": [{"name": "USB2.0 HD UVC WebCam", "device_id": "USB\\VID_322E", "status": "OK"}],
        "configured": True,
        "device_present": True,
        "open_success": True,
        "frame_success": True,
        "connected": True,
        "streaming": False,
        "detailed_modes": {"0:CAP_DSHOW": [{"requested": "1280x720@30.0", "delivered": "1280x720@30.0", "success": True}]},
    })
    with mock.patch("scripts.laptop_preflight.probe_cameras", return_value=mock_cameras):
        report = run_preflight()
        assert report["camera"]["discovered"] is True
        assert report["readiness"]["camera_frame_success"] is True
        assert report["readiness"]["READY_FOR_LAPTOP_LIVE_VALIDATION"] == "YES"


def test_camera_semantics_distinction():
    """Verify camera semantics properly distinguish open_success, frame_success, connected, streaming."""
    report = run_preflight()
    semantics = report["camera"]["semantics"]
    assert "configured" in semantics
    assert "device_present" in semantics
    assert "open_success" in semantics
    assert "frame_success" in semantics
    assert "connected" in semantics
    assert "streaming" in semantics
    assert isinstance(semantics["device_present"], bool)
    assert isinstance(semantics["streaming"], bool)
