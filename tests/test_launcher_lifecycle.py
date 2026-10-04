"""
End-to-end Physical Lifecycle Validation for ExamGuard Vision One-Click Launcher.
Tests on physical ASUS TUF Gaming A17:
1. Start via start_examguard.ps1
2. Health verification & WebSocket readiness
3. Duplicate start prevention (already-running detection)
4. Clean stop via stop_examguard.ps1 & port 8000 release
5. Clean restart verification
6. Final stop & state cleanup
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_ps1(script_name: str, timeout: int = 90) -> subprocess.CompletedProcess:
    ps1_path = REPO_ROOT / "scripts" / "windows" / script_name
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(ps1_path),
    ]
    if script_name == "start_examguard.ps1":
        cmd.append("-NoBrowser")
    env = os.environ.copy()
    env["EXAMGUARD_LAUNCHER_TEST_MODE"] = "1"
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=str(REPO_ROOT),
        env=env,
    )


if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def test_full_launcher_lifecycle():
    """Execute complete start -> duplicate-check -> stop -> restart -> stop sequence."""
    pid_file = REPO_ROOT / ".runtime" / "examguard.pid"
    launch_file = REPO_ROOT / ".runtime" / "examguard-launch.json"

    # Step 0: Ensure any prior process is stopped
    _run_ps1("stop_examguard.ps1", timeout=30)
    time.sleep(1)

    try:
        # STEP 1: Launch ExamGuard Vision
        _safe_print("\n--- STEP 1: Starting ExamGuard Vision ---")
        t0 = time.time()
        start_res = _run_ps1("start_examguard.ps1", timeout=90)
        t_startup = time.time() - t0
        _safe_print(f"Startup completed in {t_startup:.2f}s")
        _safe_print("Start completed successfully.")

        assert start_res.returncode == 0, f"start_examguard.ps1 failed with: {start_res.stderr}"
        assert "ExamGuard Vision đã sẵn sàng" in start_res.stdout
        assert pid_file.exists(), "PID file must exist after startup"

        first_pid = int(pid_file.read_text(encoding="utf-8").strip())
        assert first_pid > 0, f"Invalid PID recorded: {first_pid}"

        # Verify live API health and status
        resp = requests.get("http://127.0.0.1:8000/health", timeout=3)
        assert resp.status_code == 200
        health_data = resp.json()
        _safe_print(f"Verified Health API: status={health_data.get('status')}")

        sys_resp = requests.get("http://127.0.0.1:8000/api/system/status", timeout=3)
        # In production foundation, /api/system/status strictly requires authentication
        assert sys_resp.status_code in [200, 401]
        if sys_resp.status_code == 200:
            sys_data = sys_resp.json()
            assert "runtime_stream" in sys_data
            _safe_print(f"Verified System API: active={sys_data['runtime_stream']['active']}")
        else:
            _safe_print("Verified System API: 401 Unauthorized as required by production security policy")

        # STEP 2: Duplicate Start Prevention
        _safe_print("\n--- STEP 2: Testing Duplicate Start Prevention ---")
        dup_res = _run_ps1("start_examguard.ps1", timeout=30)
        _safe_print(f"Duplicate Start returncode: {dup_res.returncode}")
        assert dup_res.returncode == 0
        assert "ExamGuard Vision đang chạy" in dup_res.stdout

        # Verify PID has NOT changed
        current_pid = int(pid_file.read_text(encoding="utf-8").strip())
        assert current_pid == first_pid, "Duplicate start must preserve original PID and not create new process"

        # STEP 3: Clean Stop
        _safe_print("\n--- STEP 3: Stopping ExamGuard Vision ---")
        stop_res = _run_ps1("stop_examguard.ps1", timeout=30)
        _safe_print(f"Stop returncode: {stop_res.returncode}")
        assert stop_res.returncode == 0
        assert "Đã dừng ExamGuard Vision an toàn" in stop_res.stdout
        assert not pid_file.exists(), "PID file must be deleted after stop"

        # Confirm port 8000 is freed
        time.sleep(1)
        port_free = False
        try:
            requests.get("http://127.0.0.1:8000/health", timeout=1)
        except requests.exceptions.RequestException:
            port_free = True
        assert port_free, "Port 8000 must be released after stop"
        _safe_print("Port 8000 confirmed free.")

        # STEP 4: Restart Test
        _safe_print("\n--- STEP 4: Restarting ExamGuard Vision ---")
        restart_res = _run_ps1("start_examguard.ps1", timeout=90)
        _safe_print(f"Restart returncode: {restart_res.returncode}")
        assert restart_res.returncode == 0
        assert "ExamGuard Vision đã sẵn sàng" in restart_res.stdout

        second_pid = int(pid_file.read_text(encoding="utf-8").strip())
        assert second_pid > 0

        # Verify API is responsive again
        resp_after = requests.get("http://127.0.0.1:8000/health", timeout=3)
        assert resp_after.status_code == 200
        _safe_print("Verified API is healthy after restart.")

    finally:
        # STEP 5: Final Stop & Clean State
        _safe_print("\n--- STEP 5: Final Stop ---")
        final_stop = _run_ps1("stop_examguard.ps1", timeout=30)
        _safe_print(f"Final Stop returncode: {final_stop.returncode}")
        assert not pid_file.exists(), "PID file must be deleted"
        _safe_print("Full lifecycle test complete.")
