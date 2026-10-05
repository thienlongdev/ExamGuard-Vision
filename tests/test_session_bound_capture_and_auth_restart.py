"""
Session-bound camera lifecycle, hard event gating, application-restart auth invalidation,
and the UI contract for the session gate / closed-session state.
"""

import secrets
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.fusion.types import FusedEvent
from src.orchestration.session_capture import SessionCaptureController
from src.persistence.service import PersistenceService

STATIC = Path(__file__).resolve().parents[1] / "src" / "api" / "static"
CAM_ID = "webcam_0"


# ---------------------------------------------------------------------------
# Fakes: a physical camera handle and a Stage 2 pipeline surface
# ---------------------------------------------------------------------------


class FakeSource:
    def __init__(self, can_open: bool = True):
        self.can_open = can_open
        self.opened = False
        self.open_count = 0
        self.release_count = 0
        self.width, self.height, self.fps = 1280, 720, 30.0

    def open(self) -> bool:
        if not self.can_open:
            return False
        self.opened = True
        self.open_count += 1
        return True

    def release(self) -> None:
        if self.opened:
            self.release_count += 1
        self.opened = False

    def is_opened(self) -> bool:
        return self.opened


class FakeClipRecorder:
    def __init__(self):
        self.finalize_calls = 0

    def finalize_all_active(self):
        self.finalize_calls += 1


class FakeEvidenceManager:
    def __init__(self):
        self.clip_recorder = FakeClipRecorder()


class FakePipeline:
    def __init__(self, source):
        self.source = source
        self.listeners = []
        self.evidence_listeners = []
        self.evidence_manager = FakeEvidenceManager()
        self._frame_lock = threading.Lock()
        self._latest_jpeg_frame = None
        self._latest_tracks_summary = []
        self._track_metadata = {}
        self._active = {}
        self.reset_calls = 0
        self.processed_frames_count = 0
        self.dropped_frames_count = 0
        self.observed_capture_fps = 29.5
        self.observed_processed_fps = 12.0
        self.observed_inference_fps = 12.0

        class _Q:
            qsize = 0

        self.ingestion_queue = _Q()

    def add_event_listener(self, cb):
        self.listeners.append(cb)

    def add_evidence_listener(self, cb):
        self.evidence_listeners.append(cb)

    def emit(self, event, action):
        if action == "OPEN":
            self._active[event.event_id] = event
        elif action == "CLOSE":
            self._active.pop(event.event_id, None)
        for cb in self.listeners:
            cb(event, action)

    def close_camera_session(self, camera_id):
        for ev in list(self._active.values()):
            ev.lifecycle_status = "closed"
            self.emit(ev, "CLOSE")

    def reset_runtime_state(self):
        self.reset_calls += 1
        self._active.clear()
        self._track_metadata.clear()


def make_event(eid, camera_id=CAM_ID):
    return FusedEvent(
        event_id=eid,
        track_id=1,
        camera_id=camera_id,
        event_type="SUSTAINED_LATERAL_HEAD_ORIENTATION",
        start_timestamp=1000.0,
        last_update_timestamp=1001.0,
        duration=1.0,
        risk_level="MEDIUM",
        risk_score=55.0,
    )


@pytest.fixture
def runtime(tmp_path):
    ps = PersistenceService(db_path=str(tmp_path / "capture.db"))
    source = FakeSource()
    pipeline = FakePipeline(source)
    ctrl = SessionCaptureController(source=source, pipeline=pipeline, camera_id=CAM_ID)
    app = create_app(
        camera_id=CAM_ID,
        camera_type="webcam",
        stage2_pipeline=pipeline,
        device_present=True,
        persistence_service=ps,
        capture_controller=ctrl,
        enforce_auth=False,
    )
    client = TestClient(app)
    yield client, ps, source, pipeline, ctrl
    ps.db.close()


def event_count(ps, session_id=None):
    with ps.db.cursor() as cur:
        if session_id:
            cur.execute("SELECT COUNT(*) FROM events WHERE session_id = ?;", (session_id,))
        else:
            cur.execute("SELECT COUNT(*) FROM events;")
        return int(cur.fetchone()[0])


# ---------------------------------------------------------------------------
# Camera lifecycle bound to MonitoringSession
# ---------------------------------------------------------------------------


def test_no_session_means_no_camera_no_capture_no_events(runtime):
    client, ps, source, pipeline, ctrl = runtime

    assert source.open_count == 0
    assert not ctrl.is_active()

    # Stray detector output before any session is never persisted
    pipeline.emit(make_event("ev_stray_1"), "OPEN")
    assert event_count(ps) == 0

    cams = client.get("/api/cameras").json()
    assert cams[0]["camera_id"] == CAM_ID
    assert cams[0]["streaming"] is False
    assert cams[0]["capture_state"] == "IDLE"
    assert cams[0]["status"] == "READY"
    assert client.get("/api/cameras/frame").status_code == 404
    assert client.get("/api/cameras/tracks").json() == []

    status = client.get("/api/system/status").json()
    assert status["runtime_stream"]["active"] is False
    assert status["runtime_stream"]["monitoring_active"] is False
    assert status["observed_rates"]["capture_fps"] is None
    assert status["effective_fps"] is None

    # Page navigation never starts the camera
    for path in ("/api/sessions/page", "/api/system/status", "/api/cameras/config", "/api/events", "/api/system/security"):
        client.get(path)
    assert source.open_count == 0


def test_start_refresh_end_and_second_session(runtime):
    client, ps, source, pipeline, ctrl = runtime

    r = client.post("/api/sessions/start", json={"name": "UX VERIFY", "room": "A203", "camera_ids": [CAM_ID]})
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]
    assert source.open_count == 1 and source.is_opened()
    assert ctrl.is_active() and ctrl.session_id == sid
    assert [c.camera_id for c in ps.cameras.list_cameras_for_session(sid)] == [CAM_ID]

    cams = client.get("/api/cameras").json()
    assert cams[0]["streaming"] is True and cams[0]["capture_state"] == "ACTIVE"
    assert cams[0]["monitoring_session_id"] == sid

    # Active session: normal event flow
    pipeline.emit(make_event("ev_live_1"), "OPEN")
    assert event_count(ps, sid) == 1

    # Refresh / re-entry never duplicates the camera
    assert client.get("/api/sessions/current").json()["session_id"] == sid
    assert ctrl.start(sid) is True
    assert source.open_count == 1

    # Live preview + tracks present while monitoring
    with pipeline._frame_lock:
        pipeline._latest_jpeg_frame = b"\xff\xd8jpeg"
        pipeline._latest_tracks_summary = [{"track_id": 1, "bbox": [0, 0, 10, 10]}]
    assert client.get("/api/cameras/frame").status_code == 200

    r = client.post(f"/api/sessions/{sid}/end", json={"reason": "COMPLETED"})
    assert r.status_code == 200
    assert r.json()["status"] == "CLOSED"
    assert source.release_count == 1 and not source.is_opened()
    assert not ctrl.is_active() and ctrl.session_id is None
    assert pipeline._latest_jpeg_frame is None
    assert pipeline._latest_tracks_summary == []
    assert pipeline.evidence_manager.clip_recorder.finalize_calls >= 1
    assert client.get("/api/cameras/frame").status_code == 404
    assert client.get("/api/cameras/tracks").json() == []

    # The open event was closed while the session was still ACTIVE
    persisted = ps.events.get_event("ev_live_1")
    assert persisted.lifecycle_status == "closed"

    # Nothing is created after the end
    pipeline.emit(make_event("ev_after_end"), "OPEN")
    assert ps.events.get_event("ev_after_end") is None
    assert event_count(ps) == 1

    # Second session: camera reopens, new id, clean counters
    r2 = client.post("/api/sessions/start", json={"name": "Second", "room": "A204"})
    assert r2.status_code == 200
    sid2 = r2.json()["session_id"]
    assert sid2 != sid
    assert source.open_count == 2 and source.is_opened()
    assert r2.json()["summary"]["total_events"] == 0
    assert event_count(ps, sid2) == 0


def test_camera_open_failure_leaves_no_zombie_session(tmp_path):
    ps = PersistenceService(db_path=str(tmp_path / "fail.db"))
    source = FakeSource(can_open=False)
    pipeline = FakePipeline(source)
    ctrl = SessionCaptureController(source=source, pipeline=pipeline, camera_id=CAM_ID)
    app = create_app(
        camera_id=CAM_ID, stage2_pipeline=pipeline, device_present=True,
        persistence_service=ps, capture_controller=ctrl, enforce_auth=False,
    )
    client = TestClient(app)
    r = client.post("/api/sessions/start", json={"name": "No cam", "room": "B1"})
    assert r.status_code == 503
    assert "Không thể mở camera" in r.json()["detail"]
    assert ps.get_current_monitoring_session() is None
    rows = ps.sessions.list_sessions(limit=10)
    assert rows[0].status == "START_FAILED"
    assert not ctrl.is_active()
    ps.db.close()


def test_camera_config_and_dropdown_share_canonical_registry(tmp_path):
    ps = PersistenceService(db_path=str(tmp_path / "registry.db"))
    password = "Eg-" + secrets.token_urlsafe(18)
    ps.create_initial_admin(username="cfg_admin", password=password, display_name="Admin")
    source = FakeSource()
    pipeline = FakePipeline(source)
    ctrl = SessionCaptureController(source=source, pipeline=pipeline, camera_id=CAM_ID)
    app = create_app(
        camera_id=CAM_ID, camera_type="webcam", stage2_pipeline=pipeline, device_present=True,
        persistence_service=ps, capture_controller=ctrl, enforce_auth=True,
    )
    client = TestClient(app)
    login = client.post("/api/auth/login", json={"username": "cfg_admin", "password": password})
    assert login.status_code == 200
    client.cookies.set("eg_session", login.cookies.get("eg_session"))

    dropdown = client.get("/api/cameras").json()
    config = client.get("/api/cameras/config").json()
    assert dropdown[0]["camera_id"] == CAM_ID
    assert config, "the operational camera must be listed"
    auto = config[0]
    assert auto["camera_id"] == dropdown[0]["camera_id"]
    assert auto["name"] == dropdown[0]["name"] == "CAM 01"
    assert auto["auto_discovered"] is True
    assert auto["device_index"] == 0
    assert auto["capture_state"] == "IDLE"
    assert "Exam Room Camera (WEBCAM)" not in str(dropdown)
    assert source.open_count == 0
    ps.db.close()


# ---------------------------------------------------------------------------
# Hard backend event gate
# ---------------------------------------------------------------------------


def test_persistence_gate_none_closed_active_and_foreign_camera(tmp_path):
    ps = PersistenceService(db_path=str(tmp_path / "gate.db"))

    ps.persist_stage2_event(make_event("ev_none"), "OPEN")
    assert ps.events.get_event("ev_none") is None

    s = ps.start_monitoring_session(name="Gate", room="R", camera_ids=[CAM_ID])
    ps.persist_stage2_event(make_event("ev_active"), "OPEN")
    assert ps.events.get_event("ev_active").session_id == s.session_id

    ps.persist_stage2_event(make_event("ev_foreign", camera_id="webcam_9"), "OPEN")
    assert ps.events.get_event("ev_foreign") is None

    ps.end_monitoring_session(session_id=s.session_id, reason="COMPLETED")
    ps.persist_stage2_event(make_event("ev_closed"), "OPEN")
    assert ps.events.get_event("ev_closed") is None
    ps.db.close()


# ---------------------------------------------------------------------------
# Auth: every application restart requires a fresh login
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_db(tmp_path):
    db_path = str(tmp_path / "auth_restart.db")
    ps = PersistenceService(db_path=db_path)
    password = "Eg-" + secrets.token_urlsafe(18)
    ps.create_initial_admin(username="restart_admin", password=password, display_name="Admin")
    yield db_path, ps, password
    ps.db.close()


def _login(client, password):
    r = client.post("/api/auth/login", json={"username": "restart_admin", "password": password})
    assert r.status_code == 200
    return {"eg_session": r.cookies.get("eg_session")}


def test_restart_invalidates_previous_login(auth_db):
    db_path, ps, password = auth_db
    app1 = create_app(persistence_service=ps, enforce_auth=True)
    with TestClient(app1) as c1:
        old_cookie = _login(c1, password)
        assert c1.get("/api/auth/me", cookies=old_cookie).status_code == 200

    # Simulated application restart: a new process/server boots on the same database
    ps2 = PersistenceService(db_path=db_path)
    app2 = create_app(persistence_service=ps2, enforce_auth=True, fresh_boot=True)
    with TestClient(app2, follow_redirects=False) as c2:
        assert c2.get("/api/auth/me", cookies=old_cookie).status_code == 401
        assert c2.get("/api/sessions/page", cookies=old_cookie).status_code == 401
        dash = c2.get("/", cookies=old_cookie)
        assert dash.status_code == 302 and dash.headers["location"] == "/login"
        # The login page is served (not bounced to the dashboard) for the stale cookie
        assert c2.get("/login", cookies=old_cookie).status_code == 200

        new_cookie = _login(c2, password)
        assert new_cookie["eg_session"] != old_cookie["eg_session"]
        me = c2.get("/api/auth/me", cookies=new_cookie)
        assert me.status_code == 200
        assert "password_hash" not in me.text and password not in me.text
        assert c2.get("/", cookies=new_cookie).headers.get("cache-control", "").startswith("no-store")
    ps2.db.close()


def test_logout_revokes_server_side_session(auth_db):
    db_path, ps, password = auth_db
    app = create_app(persistence_service=ps, enforce_auth=True)
    client = TestClient(app, follow_redirects=False)
    cookie = _login(client, password)
    r = client.post("/api/auth/logout", cookies=cookie)
    assert r.status_code == 200
    assert "eg_session" in r.headers.get("set-cookie", "")
    # The raw old token is rejected server-side even if the browser still sends it
    assert client.get("/api/auth/me", cookies=cookie).status_code == 401
    assert client.get("/api/events", cookies=cookie).status_code == 401
    assert client.get("/", cookies=cookie).status_code == 302


def test_restart_interrupts_active_monitoring_session(auth_db):
    db_path, ps, _ = auth_db
    s = ps.start_monitoring_session(name="Running at crash", room="R")
    ps2 = PersistenceService(db_path=db_path)
    app = create_app(persistence_service=ps2, enforce_auth=True, fresh_boot=True)
    with TestClient(app):
        row = ps2.sessions.get_session(s.session_id)
        assert row.status == "INTERRUPTED"
        assert ps2.get_current_monitoring_session() is None
    ps2.db.close()


def test_application_stop_interrupts_rather_than_closes(tmp_path):
    ps = PersistenceService(db_path=str(tmp_path / "stop.db"))
    app = create_app(persistence_service=ps, enforce_auth=False)
    with TestClient(app) as client:
        sid = client.post("/api/sessions/start", json={"name": "Stop", "room": "R"}).json()["session_id"]
    row = ps.sessions.get_session(sid)
    assert row.status == "INTERRUPTED"
    assert row.close_reason == "APPLICATION_STOPPED"
    ps.db.close()


# ---------------------------------------------------------------------------
# UI contract (static assets)
# ---------------------------------------------------------------------------


def _read(rel):
    return (STATIC / rel).read_text(encoding="utf-8")


def test_ui_camera_view_idle_by_default_and_bound_to_monitoring():
    js = _read("js/components/camera_view.js")
    assert 'src="/api/cameras/stream"' not in js, "no MJPEG connection before a session is monitored"
    assert "MONITORING_ACTIVE_CHANGED" in js
    assert "Chưa có phiên giám sát đang hoạt động" in js
    assert "Bắt đầu phiên để kích hoạt camera và AI." in js
    assert 'removeAttribute("src")' in js


def test_ui_gate_is_page_state_and_end_clears_overlay_first():
    gate = _read("js/components/session_gate.js")
    assert "monitor-gated" in gate
    assert "setMonitoringActive" in gate
    assert "Kết thúc phiên" in gate and "btn-end-session" in gate
    assert "Phiên đã kết thúc" in gate and "Xem lịch sử phiên" in gate and "Bắt đầu phiên mới" in gate
    confirm = gate.index('id="end-session-confirm"')
    handler = gate[confirm:]
    assert handler.index("setMonitoringActive(false)") < handler.index("ApiClient.endSession")

    state = _read("js/state.js")
    assert "setMonitoringActive" in state and "TRACKS_UPDATED" in state

    css = _read("css/dashboard.css")
    assert "#view-monitor.monitor-gated > *" in css


def test_ui_header_and_system_tabs_styled():
    header = _read("js/components/header.js")
    assert "CAMERA · CHƯA HOẠT ĐỘNG" in header
    assert '<span id="live-status-text">TRỰC TIẾP · CAM 01</span>' not in header

    css = _read("css/dashboard.css")
    for sel in (".sys-subtab-btn {", ".sys-subtab-btn:hover", ".sys-subtab-btn.active", ".sys-subtab-btn:focus-visible", ".system-subnav-tabs {"):
        assert sel in css, sel

    sysjs = _read("js/components/system_view.js")
    assert "Tự động phát hiện" in sysjs
    assert "Theo phiên / Chưa gán cố định" in sysjs
    assert "27.6" not in sysjs
