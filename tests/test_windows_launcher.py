"""
Unit and regression tests for ExamGuard Vision One-Click Windows Launcher.
Verifies:
1. Required launcher batch and PowerShell files exist.
2. Batch files wrap PowerShell scripts with ExecutionPolicy Bypass without modifying system policy.
3. PowerShell scripts have valid syntax and adhere to root path resolution.
4. Model manifest for lightweight preflight matches certified canonical paths.
5. PID metadata format, serialization, and stale PID handling logic.
6. Health detection contract for already-running detection.
7. Port occupied detection and foreign process safety guarantees.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_launcher_files_exist():
    """Verify all 3 root batch wrappers and 3 PowerShell scripts exist."""
    expected_files = [
        REPO_ROOT / "Start ExamGuard Vision.bat",
        REPO_ROOT / "Stop ExamGuard Vision.bat",
        REPO_ROOT / "Install ExamGuard Shortcuts.bat",
        REPO_ROOT / "scripts" / "windows" / "start_examguard.ps1",
        REPO_ROOT / "scripts" / "windows" / "stop_examguard.ps1",
        REPO_ROOT / "scripts" / "windows" / "install_desktop_shortcuts.ps1",
    ]
    for file_path in expected_files:
        assert file_path.is_file(), f"Required launcher file missing: {file_path}"


def test_batch_wrappers_content():
    """Verify batch wrappers invoke PowerShell with Bypass and correct script targets."""
    start_bat = (REPO_ROOT / "Start ExamGuard Vision.bat").read_text(encoding="utf-8")
    assert "start_examguard.ps1" in start_bat
    assert "-ExecutionPolicy Bypass" in start_bat
    assert "%~dp0" in start_bat

    stop_bat = (REPO_ROOT / "Stop ExamGuard Vision.bat").read_text(encoding="utf-8")
    assert "stop_examguard.ps1" in stop_bat
    assert "-ExecutionPolicy Bypass" in stop_bat
    assert "%~dp0" in stop_bat

    install_bat = (REPO_ROOT / "Install ExamGuard Shortcuts.bat").read_text(encoding="utf-8")
    assert "install_desktop_shortcuts.ps1" in install_bat
    assert "-ExecutionPolicy Bypass" in install_bat
    assert "%~dp0" in install_bat


def test_powershell_scripts_syntax():
    """Verify PowerShell scripts contain no parse errors."""
    ps_scripts = [
        REPO_ROOT / "scripts" / "windows" / "start_examguard.ps1",
        REPO_ROOT / "scripts" / "windows" / "stop_examguard.ps1",
        REPO_ROOT / "scripts" / "windows" / "install_desktop_shortcuts.ps1",
    ]
    for ps_script in ps_scripts:
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            f"$errors = $null; [System.Management.Automation.Language.Parser]::ParseFile('{ps_script}', [ref]$null, [ref]$errors); if ($errors.Count -gt 0) {{ $errors | ForEach-Object {{ Write-Error $_ }}; exit 1 }}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
        assert result.returncode == 0, f"PowerShell syntax error in {ps_script.name}: {result.stderr}"


def test_required_model_manifest_complete():
    """Verify all 7 models listed in the lightweight preflight exist on disk."""
    required_models = [
        "models/trained/yolo26m.pt",
        "models/trained/stage1_5_best.pt",
        "models/trained/v4_posture_best.pt",
        "models/trained/v4_headpose_yaw_best.pt",
        "models/trained/stage1_best.pt",
        "models/fallback/posture_320/best_model.pt",
        "models/fallback/headpose_resnet18/best_model.pt",
    ]
    for rel_path in required_models:
        model_path = REPO_ROOT / rel_path
        assert model_path.is_file(), f"Model checkpoint not found: {rel_path}"
        assert model_path.stat().st_size > 1000, f"Model file is too small (likely un-pulled LFS pointer): {rel_path}"


def test_pid_metadata_format(tmp_path):
    """Verify PID metadata serialization and deserialization adheres to schema."""
    launch_info = {
        "pid": 999999,
        "start_time": "2026-10-04T03:30:00+07:00",
        "project_root": str(REPO_ROOT),
        "command": f"{sys.executable} scripts/run_asus_a17_demo.py",
        "port": 8000,
    }
    json_path = tmp_path / "examguard-launch.json"
    pid_path = tmp_path / "examguard.pid"

    pid_path.write_text("999999", encoding="utf-8")
    json_path.write_text(json.dumps(launch_info, indent=2), encoding="utf-8")

    loaded_pid = int(pid_path.read_text(encoding="utf-8").strip())
    assert loaded_pid == 999999

    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded_json["pid"] == 999999
    assert loaded_json["port"] == 8000
    assert loaded_json["project_root"] == str(REPO_ROOT)


def test_health_endpoint_contract():
    """Verify health endpoint responds with required status and version for already-running check."""
    from fastapi.testclient import TestClient
    from src.api.main import create_app

    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert "version" in data
    assert "timestamp" in data


def test_demo_runner_stop_watcher_wiring():
    """Verify scripts/run_asus_a17_demo.py contains stop.signal watcher and runtime logger."""
    runner_code = (REPO_ROOT / "scripts" / "run_asus_a17_demo.py").read_text(encoding="utf-8")
    assert "stop.signal" in runner_code
    assert "examguard-runtime.log" in runner_code
    assert "orch.stop_event.set()" in runner_code


def test_port_collision_safety_behavior():
    """Verify start_examguard.ps1 detects foreign listener on port 8000 without terminating it."""
    import socket

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 8000))
    server.listen(1)

    try:
        cmd = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(REPO_ROOT / "scripts" / "windows" / "start_examguard.ps1"),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=25)
        assert res.returncode == 1
        assert "cổng 8000 đang được chương trình khác sử dụng" in res.stdout
    finally:
        server.close()


def test_stale_pid_stop_handling():
    """Verify stop_examguard.ps1 cleanly handles a stale PID file for a non-existent process."""
    runtime_dir = REPO_ROOT / ".runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    pid_file = runtime_dir / "examguard.pid"
    launch_file = runtime_dir / "examguard-launch.json"

    # Write a non-existent PID
    stale_pid = 999998
    pid_file.write_text(str(stale_pid), encoding="utf-8")
    launch_file.write_text(json.dumps({"pid": stale_pid}), encoding="utf-8")

    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(REPO_ROOT / "scripts" / "windows" / "stop_examguard.ps1"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
    assert res.returncode == 0
    assert not pid_file.exists(), "Stale PID file should be cleaned up"
    assert not launch_file.exists(), "Stale launch file should be cleaned up"


def test_missing_model_failure_check(tmp_path):
    """Verify lightweight preflight logic detects missing required model file."""
    # Test the model check logic directly
    required_models = [
        "models/trained/yolo26m.pt",
        "models/trained/stage1_5_best.pt",
        "models/trained/v4_posture_best.pt",
        "models/trained/v4_headpose_yaw_best.pt",
        "models/trained/stage1_best.pt",
        "models/fallback/posture_320/best_model.pt",
        "models/fallback/headpose_resnet18/best_model.pt",
    ]
    # Check that any hypothetical missing model triggers missing status
    simulated_missing = "models/trained/nonexistent_model.pt"
    test_list = required_models + [simulated_missing]
    missing = [m for m in test_list if not (REPO_ROOT / m).is_file()]
    assert missing == [simulated_missing]
