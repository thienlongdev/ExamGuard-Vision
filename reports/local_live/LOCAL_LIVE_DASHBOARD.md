# Local Live Dashboard Validation Report

**Phase:** CAMERALESS SOFTWARE PREFLIGHT  
**Host Role:** `DESKTOP_ROLE = TRAINING_AND_BENCHMARK_WORKSTATION`  
**Access URL:** `http://127.0.0.1:8000/`  
**Status:** DASHBOARD SOFTWARE INTEGRATION CERTIFIED (`DASHBOARD_SOFTWARE_INTEGRATION_PASS = YES`)

---

## 1. Dashboard Architecture & Integration

The embedded invigilator review dashboard (`src/api/static_ui.py`) is served directly by the FastAPI web server on `127.0.0.1:8000`:
- **Styling:** Premium dark mode (`#0f172a`), Inter typography, CSS grid layout, responsive stat cards, color-coded risk borders (High: Red, Medium: Amber, Low: Blue).
- **Camera Semantic Status Bar:**
  When no physical camera exists, the dashboard displays:
  - **Camera Status:** `CONFIGURED / NOT CONNECTED`
  - **Physical Device:** `NOT DETECTED`
  - **Stream State:** `INACTIVE`
  - **Measured Capture FPS:** `N/A`
  - **Measured Processing FPS:** `N/A`
  - **Inference FPS:** `N/A`
  Configured target frame rates (e.g. 30 FPS) are never displayed as measured rates when the stream is inactive.
- **Communication Channels:**
  1. **HTTP Fetch:** Calls `/api/events` on initial load to populate historic alerts.
  2. **WebSocket Realtime Stream:** Connects to `/ws/events` to handle live pushes (`EVENT_OPEN`, `EVENT_UPDATE`, `EVENT_CLOSE`, `EVENT_STATUS_UPDATED`).
  3. **Periodic System Status Polling:** Polls `/api/system/status` every 2.5 seconds to refresh the header subtitle HUD and camera status bar.

---

## 2. Invigilator Human Review Controls

The UI strictly adheres to scientific governance and legal fairness:
- Events display factual behavior descriptions (e.g., `ORIENTATION_SUSTAINED_LEFT`, `PHONE_ASSOCIATED`) rather than accusatory labels.
- Human review action buttons:
  - **Confirm (`btn-confirm`):** Sends `PATCH /api/events/{id}` with `status: 'confirmed'`. Card visibly turns green (`✓ Confirmed by Human`).
  - **Dismiss (`btn-dismiss`):** Sends `PATCH /api/events/{id}` with `status: 'dismissed'`. Card updates to dismissed status.
- Real-time stat counters: Total Events Flagged, Awaiting Human Review, Confirmed Suspicious, Dismissed / False Positives.
- Explicit origin badge displayed for software fixtures: `Origin: SOFTWARE_VALIDATION_FIXTURE`.

---

## 3. Demo Launch Instructions for ASUS TUF A17

To view the dashboard during the subsequent physical webcam session on the ASUS TUF A17:
1. Ensure the local server is started:
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
   ```
2. Open a modern browser to `http://127.0.0.1:8000/`.
