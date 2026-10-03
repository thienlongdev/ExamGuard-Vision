# Local Live Claim Traceability Matrix

**Phase:** CAMERALESS SOFTWARE PREFLIGHT & FINAL DESKTOP INTEGRITY  
**Host Role:** `DESKTOP_ROLE = TRAINING_AND_BENCHMARK_WORKSTATION`  
**Target Demo Machine:** ASUS TUF Gaming A17 (`FA707RC`, Ryzen 7 6800H, RTX 3050 Laptop 4GB VRAM)  
**Traceability Purpose:** Provide exact cryptographic, physical, and artifact references separating software verification from deferred physical hardware validation.

---

| Core Claim | Verified Statement | Authoritative Source Artifact | Verification Method |
|---|---|---|---|
| **Checkpoint Integrity (7/7)** | All 7 certified checkpoints match authoritative SHA-256 hashes exactly | `runs/local_live/checkpoint_hashes.json`, `deployment/laptop/repository_packaging_audit.json` | Python cryptographic audit via `hashlib.sha256` |
| **Localhost-Only Security** | Services strictly bind to `127.0.0.1:8000`, 0.0.0.0 prohibited | `configs/local_live_camera.yaml`, `scripts/run_local_live_validation.py` | Socket binding audit |
| **Clean Absence of Camera** | System handles 0 cameras gracefully without crash; registers=1, configured=1, connected=0, streaming=0 | `runs/local_live/camera_probe.json`, `runs/local_live/camera_actual_mode.json` | Direct probe across indices `[0..3]` and candidate backends |
| **FastAPI Localhost Endpoints** | Endpoints return `200 OK` on `127.0.0.1:8000` (`/health`, `/api/cameras`, `/api/events`, `/api/system/status`, `/api/system/models`, `/`) | `runs/local_live/live_api_validation.json` | Real localhost HTTP requests via `requests` library |
| **WebSocket Lifecycle Ordering** | Client receives `EVENT_OPEN`, `EVENT_UPDATE`, `EVENT_STATUS_UPDATED`, `EVENT_CLOSE` in chronological order with origin `SOFTWARE_VALIDATION_FIXTURE` | `runs/local_live/live_websocket_validation.json`, `runs/local_live/live_event_lifecycle.json` | Asynchronous WebSocket client connection to `/ws/events` |
| **Dashboard UI Integration** | Dashboard serves embedded HTML, displays `Camera: CONFIGURED / NOT CONNECTED`, `Measured FPS: N/A` | `runs/local_live/live_dashboard_validation.json` | Embedded HTML response check & semantic status verification |
| **Evidence Generation & Privacy** | Anonymous snapshot and JSON metadata generated with explicit `SOFTWARE_VALIDATION_FIXTURE` origin | `runs/local_live/live_evidence_validation.json`, `evidence/local_validation/` | Physical file inspection of `evidence/local_validation/` |
| **Runtime State Reset** | `reset_runtime_state()` purges queues, ByteTrack, TemporalBuffer without reloading weights | `runs/local_live/runtime_state_reset_validation.json` | Execution test verifying `RUNTIME_STATE_RESET_PASS = YES` |
| **Physical Camera Restart** | Physical device release/reopen deferred due to absence of camera hardware | `runs/local_live/camera_restart_validation.json` | Verified status `DEFERRED_NO_CAMERA_HARDWARE` |
| **Cameraless Memory Stability** | GPU VRAM remains stable at 130.3 MB allocated / 256 MB peak during software execution | `runs/local_live/memory_soak.json` | `torch.cuda.memory_allocated` instrumentation |
| **Physical Camera Soak** | 10-minute steady-state camera soak deferred until physical webcam connected on ASUS A17 | `runs/local_live/memory_soak.json`, `runs/local_live/live_performance.json` | Verified status `DEFERRED_UNTIL_PHYSICAL_WEBCAM_CONNECTED` |
| **Full Regression Integrity** | Automated test suite passes with zero failures | Pytest terminal output | Execution of `pytest tests/` |
