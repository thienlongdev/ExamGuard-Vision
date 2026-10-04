"""
Tests for ExamGuard Session API, History Retrieval, Evidence Integrity Verification, and Backup Endpoints.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.persistence.service import PersistenceService
from src.persistence.models import PersistedEvent, ExamSession


@pytest.fixture
def api_test_setup(tmp_path):
    """Isolated PersistenceService and FastAPI test client."""
    db_file = tmp_path / "test_api_examguard.sqlite3"
    ps = PersistenceService(db_path=str(db_file))
    app = create_app(persistence_service=ps, enforce_auth=False)
    client = TestClient(app)
    yield client, ps, tmp_path
    ps.db.close()


def test_session_endpoints(api_test_setup):
    client, ps, tmp_path = api_test_setup

    # 1. GET /api/sessions/current
    res_cur = client.get("/api/sessions/current")
    assert res_cur.status_code == 200
    cur_data = res_cur.json()
    assert cur_data["session_id"].startswith("sess_")
    assert cur_data["status"] == "ACTIVE"
    assert "summary" in cur_data
    sid = cur_data["session_id"]

    # 2. PATCH /api/sessions/{session_id} - update metadata
    res_patch = client.patch(
        f"/api/sessions/{sid}",
        json={"name": "Kỳ thi Đánh giá Năng lực", "room": "Phòng C301", "invigilator_name": "TS. Trần Văn B"},
    )
    assert res_patch.status_code == 200
    patched_data = res_patch.json()
    assert patched_data["name"] == "Kỳ thi Đánh giá Năng lực"
    assert patched_data["room"] == "Phòng C301"
    assert patched_data["invigilator_name"] == "TS. Trần Văn B"

    # 3. GET /api/sessions
    res_list = client.get("/api/sessions")
    assert res_list.status_code == 200
    sessions = res_list.json()
    assert len(sessions) >= 1
    assert any(s["session_id"] == sid for s in sessions)

    # 4. GET /api/sessions/{session_id}
    res_detail = client.get(f"/api/sessions/{sid}")
    assert res_detail.status_code == 200
    detail = res_detail.json()
    assert detail["name"] == "Kỳ thi Đánh giá Năng lực"


def test_session_events_and_audit_endpoints(api_test_setup):
    client, ps, tmp_path = api_test_setup

    active_sess = ps.active_session
    sid = active_sess.session_id

    # Create event in DB
    ev = PersistedEvent(
        event_id="ev_api_01",
        session_id=sid,
        camera_id="laptop_webcam_0",
        track_id=5,
        event_type="PHONE_ASSOCIATED",
        opened_at="2026-10-04T12:00:00",
        severity="HIGH",
        score=89.0,
        lifecycle_status="active",
        review_status="awaiting",
    )
    ps.events.upsert_event(ev)

    # 1. GET /api/sessions/{session_id}/events
    res_events = client.get(f"/api/sessions/{sid}/events")
    assert res_events.status_code == 200
    events_data = res_events.json()
    assert len(events_data) == 1
    assert events_data[0]["event_id"] == "ev_api_01"
    assert events_data[0]["risk_level"] == "HIGH"

    # 2. PATCH /api/events/{event_id} - confirm event
    res_review = client.patch(
        "/api/events/ev_api_01",
        json={"status": "confirmed", "reviewer_notes": "Xác nhận sử dụng tài liệu trái phép"},
    )
    assert res_review.status_code == 200
    reviewed_ev = res_review.json()
    assert reviewed_ev["review_status"] == "confirmed"
    assert reviewed_ev["reviewer_notes"] == "Xác nhận sử dụng tài liệu trái phép"

    # 3. GET /api/sessions/{session_id}/audit
    res_audit = client.get(f"/api/sessions/{sid}/audit")
    assert res_audit.status_code == 200
    audit_data = res_audit.json()
    actions = [a["action"] for a in audit_data]
    assert "SESSION_CREATED" in actions
    assert "EVENT_CONFIRMED" in actions


def test_evidence_verify_and_backup_endpoints(api_test_setup):
    client, ps, tmp_path = api_test_setup

    sid = ps.active_session.session_id
    # Create event and synthetic evidence
    ev = PersistedEvent(
        event_id="ev_backup_api",
        session_id=sid,
        camera_id="laptop_webcam_0",
        track_id=1,
        event_type="STANDING",
        opened_at="2026-10-04T12:00:00",
        severity="LOW",
        score=35.0,
        lifecycle_status="closed",
        review_status="awaiting",
    )
    ps.events.upsert_event(ev)

    fake_file = tmp_path / "fake_snap.jpg"
    fake_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x11" * 100)
    ev_rec = ps.record_evidence_file("ev_backup_api", "SNAPSHOT", str(fake_file))

    # 1. GET /api/evidence/verify/{evidence_id}
    res_ver = client.get(f"/api/evidence/verify/{ev_rec.evidence_id}")
    assert res_ver.status_code == 200
    ver_data = res_ver.json()
    assert ver_data["valid"] is True
    assert ver_data["status"] == "VALID"

    # 2. POST /api/backup
    res_backup = client.post("/api/backup")
    assert res_backup.status_code == 200
    b_data = res_backup.json()
    assert b_data["success"] is True
    assert "backup_id" in b_data
    assert "backup_path" in b_data

    # 3. GET /api/backup/status
    res_b_status = client.get("/api/backup/status")
    assert res_b_status.status_code == 200
    assert res_b_status.json()["total_backups"] >= 1

    # 4. GET /api/retention/preview
    res_ret = client.get("/api/retention/preview")
    assert res_ret.status_code == 200
    assert res_ret.json()["policy"]["retention_enabled_default"] is False
