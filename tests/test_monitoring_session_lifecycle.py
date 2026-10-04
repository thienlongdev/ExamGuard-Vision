"""
ExamGuard Vision — Workstream 47: Monitoring Session Lifecycle Test Matrix
Verifies complete separation of AuthSession vs MonitoringSession, explicit state machine transitions,
session start/end semantics, and zero event leakage between monitoring sessions.
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.persistence.service import PersistenceService
from src.persistence.models import PersistedEvent


@pytest.fixture
def session_lifecycle_setup(tmp_path):
    db_file = tmp_path / "test_session_lifecycle.sqlite3"
    ps = PersistenceService(db_path=str(db_file))
    app = create_app(persistence_service=ps, enforce_auth=False)
    client = TestClient(app)
    yield client, ps, tmp_path
    ps.db.close()


def test_no_silent_session_creation_on_startup_or_login(session_lifecycle_setup):
    """1. App startup / login does not automatically create duplicate MonitoringSession."""
    client, ps, _ = session_lifecycle_setup
    res = client.get("/api/sessions/current")
    assert res.status_code == 404
    assert ps.get_current_monitoring_session() is None


def test_explicit_start_session_creates_one_session(session_lifecycle_setup):
    """4. Explicit Start Session creates exactly one session in ACTIVE state."""
    client, ps, _ = session_lifecycle_setup

    res = client.post(
        "/api/sessions/start",
        json={
            "session_name": "Kỳ thi Thử nghiệm 1",
            "room_id": "Phòng A101",
            "class_name": "20CNTT1",
            "subject_code": "CSDL101",
            "notes": "Kiểm tra giữa kỳ",
        },
    )
    assert res.status_code == 200
    sess_data = res.json()
    assert sess_data["status"] == "ACTIVE"
    assert sess_data["name"] == "Kỳ thi Thử nghiệm 1"
    assert sess_data["room"] == "Phòng A101"
    sid = sess_data["session_id"]

    # Reconnect to active session on refresh
    res_cur = client.get("/api/sessions/current")
    assert res_cur.status_code == 200
    assert res_cur.json()["session_id"] == sid


def test_explicit_end_session_closes_session_normally(session_lifecycle_setup):
    """5. Explicit End Session marks session CLOSED, not INTERRUPTED."""
    client, ps, _ = session_lifecycle_setup

    res_start = client.post(
        "/api/sessions/start",
        json={"session_name": "Kỳ thi Cuối kỳ", "room_id": "Phòng B202"},
    )
    sid = res_start.json()["session_id"]

    res_end = client.post(
        f"/api/sessions/{sid}/end",
        json={"reason": "COMPLETED", "summary": {"total_events": 0}},
    )
    assert res_end.status_code == 200
    closed_sess = res_end.json()
    assert closed_sess["status"] == "CLOSED"
    assert closed_sess["close_reason"] == "COMPLETED"

    # Verify no active session remains
    res_cur = client.get("/api/sessions/current")
    assert res_cur.status_code == 404


def test_start_new_session_safely_closes_previous(session_lifecycle_setup):
    """6 & 7. Start New Session closes current first, yields fresh session_id."""
    client, ps, _ = session_lifecycle_setup

    res1 = client.post(
        "/api/sessions/start",
        json={"session_name": "Phiên Sáng", "room_id": "Phòng 101"},
    )
    sid1 = res1.json()["session_id"]

    # Starting a new session while session 1 is active
    res2 = client.post(
        "/api/sessions/start",
        json={"session_name": "Phiên Chiều", "room_id": "Phòng 102"},
    )
    sid2 = res2.json()["session_id"]

    assert sid1 != sid2
    # Verify session 1 is now CLOSED
    s1_detail = client.get(f"/api/sessions/{sid1}").json()
    assert s1_detail["status"] == "CLOSED"
    assert s1_detail["close_reason"] in ["SESSION_SUPERSEDED", "SWITCHED_TO_NEW_SESSION"]

    # Verify session 2 is ACTIVE
    cur = client.get("/api/sessions/current").json()
    assert cur["session_id"] == sid2
    assert cur["status"] == "ACTIVE"


def test_session_isolation_and_no_event_leakage(session_lifecycle_setup):
    """8, 9, 10, 14. Old events remain in history, never leak into new session."""
    client, ps, _ = session_lifecycle_setup

    # Session A
    res_a = client.post(
        "/api/sessions/start",
        json={"session_name": "Session A", "room_id": "Room A"},
    )
    sid_a = res_a.json()["session_id"]

    # Insert event for Session A
    ev_a = PersistedEvent(
        event_id="ev_a_001",
        session_id=sid_a,
        camera_id="cam_01",
        track_id=1,
        event_type="PHONE_SUSPECTED",
        opened_at="2026-10-04T08:00:00",
        severity="HIGH",
        score=85.0,
        lifecycle_status="closed",
        review_status="awaiting",
        source_origin="ai_detector",
    )
    ps.events.upsert_event(ev_a)

    # End Session A
    client.post(f"/api/sessions/{sid_a}/end", json={"reason": "COMPLETED"})

    # Session B
    res_b = client.post(
        "/api/sessions/start",
        json={"session_name": "Session B", "room_id": "Room B"},
    )
    sid_b = res_b.json()["session_id"]

    # Query events for current session (Session B)
    res_b_events = client.get(f"/api/events?session_id={sid_b}")
    assert res_b_events.status_code == 200
    assert len(res_b_events.json()) == 0

    # Query events for Session A in History
    res_a_events = client.get(f"/api/sessions/{sid_a}/events")
    assert res_a_events.status_code == 200
    a_events = res_a_events.json()
    assert len(a_events) == 1
    assert a_events[0]["event_id"] == "ev_a_001"
    # Unreviewed pending event survived session close
    assert a_events[0]["review_status"] == "awaiting"
