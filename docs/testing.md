# Testing ExamGuard Vision

Use the focused software selection below for source maintenance. Historical suite totals are recorded in [academic validation](final-academic-validation.md) and [detection-quality validation](final-detection-quality-validation.md); they are not a promise about a new machine or a reason to rerun every test.

## Prepare an isolated checkout

Run Windows setup on a new machine as described in the [README](../README.md). For maintenance while the owner's app is active, create a disposable checkout or source snapshot and a separate environment instead of running tests against the operational folder. API fixtures can create databases, keys, and evidence under the checkout's `storage/` directory even when they do not use a physical camera.

The supported dependency order is PyTorch/torchvision for CPU or CUDA first, then `requirements-runtime.txt`, then `python -m pip check`. `requirements.txt` forwards to that shared definition. The required Git/LFS models and synthetic/metadata fixtures are tracked; real session data is not.

## Software regression checks

From the disposable repository root, in PowerShell:

```powershell
.\.venv\Scripts\python.exe tools/validation/verify_checkpoint_hashes.py
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q tests/test_adaptive_runtime.py tests/test_high_recall_and_reliability.py tests/test_live_track_overlay.py tests/test_multi_track_validation.py tests/test_api.py tests/test_auth_and_rbac.py tests/test_video_source.py tests/test_async_capture_and_latest_frame.py tests/test_repo_secret_audit.py
```

This selection covers capability policy, mocked camera failure/mode checks, phone association, track isolation, source/preview coordinates, event semantics, REST/auth/RBAC, source abstractions, asynchronous capture, and secret-scanner redaction. Tests use synthetic data; passing them does not establish classroom detection accuracy. The DPAPI fixture is Windows-specific and creates only its own disposable credential blob.

Other useful focused modules include `test_backup_encryption.py`, `test_session_bound_capture_and_auth_restart.py`, and `test_windows_launcher.py`. Some launcher tests write PID metadata or invoke stop scripts; execute them only in a disposable checkout, never against an active instance's `.runtime/` directory.

Check changed Python syntax/imports, JavaScript with `node --check` (or `node --input-type=module --check` on stdin), PowerShell with `System.Management.Automation.Language.Parser`, configuration parsing, and `git diff --check`. Parse launchers without executing setup/start/stop during source-only review.

## Physical checks: schedule separately

The following can open a camera, change application lifecycle state, or compete for inference resources:

- `scripts/laptop_preflight.py` and unmocked tests in `test_laptop_preflight.py`.
- `test_launcher_lifecycle.py`: starts, stops, and restarts an application on port 8765.
- Camera benchmarks and `tools/validation/validate_adaptive_workstation.py`.
- Historical soak/session/physical certification utilities.

Stop the intended ExamGuard instance normally before physical checks; never stop another application to free its port. Real hardware, first-run GUI actions, USB disconnect/reconnect, multi-camera use, RTSP, and crowded-room accuracy require their own evidence. A software test is not a substitute.

`tools/validation/observe_live_runtime.py` only samples an already active process's telemetry, but its raw JSON output still remains local. Newly generated physical summary files go under ignored `reports/validation/`.

## Offline research prerequisites

There is no bundled training corpus or `training` Python package in the deployment checkout. Preserve existing datasets and their labels/splits when working in a research environment. In a clean deployment checkout:

- `tests/test_dataset_tooling.py` cannot collect without the external `training.*` modules.
- `test_acquisition_v3.py`, `test_stage1_5_pipeline.py`, and `test_v4b_crop_dataset.py` include checks needing separately acquired/prepared datasets.
- Historical artifact assertions may skip when the original local `runs/` or `reports/` outputs are absent.

Use explicit module selections rather than treating `pytest tests/` as a camera-free deployment gate. Do not manufacture datasets, historical reports, or evaluation results to make missing-prerequisite tests pass. See [research provenance](research.md).

## Before publishing

Stage an explicit reviewed file list, then run:

```powershell
.\.venv\Scripts\python.exe tools/dev/audit_repo_secrets.py --staged
git diff --cached --check
git diff --cached --stat
```

The scanner reads exact index blobs in staged mode and reports path/line/rule only. Any finding needs review, including known synthetic test passwords; it is a pattern scan, not a guarantee that no secret exists. Inspect additions/deletions, credential-bearing URLs, binary artifacts, and ignored source separately. Preserve local-only data with `git rm --cached` when untracking; do not delete the owner's evidence or rewrite history as part of routine cleanup.
