"""Tests for FastAPI backend endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import create_app
from src.behavior.event_manager import EventManager
from src.behavior.rules import RuleMatch
from src.behavior.scorer import RiskScorer


@pytest.fixture
def client_with_event():
    em = EventManager(camera_id="cam-test-101")
    scorer = RiskScorer()

    # Pre-populate one event
    rm = RuleMatch(
        rule_id="prolonged_phone_use",
        rule_name="Prolonged Phone Usage",
        track_id=3,
        severity="high",
        score=85.0,
        evidence={"duration": 4.2},
        timestamp=100.0,
    )
    ass = scorer.score_matches(3, [rm])
    ev = em.process_assessment(ass, timestamp=100.0)

    app = create_app(event_manager=em, camera_id="cam-test-101", camera_type="webcam")
    return TestClient(app), ev.event_id


def test_health_endpoint():
    app = create_app()
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_dashboard_endpoint():
    app = create_app()
    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    assert "Exam Suspicious Behavior Monitoring" in res.text


def test_cameras_endpoint():
    app = create_app(camera_id="cam-room-1", camera_type="rtsp")
    client = TestClient(app)
    res = client.get("/api/cameras")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    assert data[0]["camera_id"] == "cam-room-1"
    assert data[0]["source_type"] == "rtsp"


def test_events_list_and_get(client_with_event):
    client, event_id = client_with_event

    # List events
    res = client.get("/api/events")
    assert res.status_code == 200
    events = res.json()
    assert len(events) >= 1
    assert events[0]["event_id"] == event_id
    assert events[0]["risk_level"] == "HIGH"
    assert events[0]["status"] == "new"

    # Get single event
    res_single = client.get(f"/api/events/{event_id}")
    assert res_single.status_code == 200
    assert res_single.json()["event_id"] == event_id


def test_human_review_status_update(client_with_event):
    client, event_id = client_with_event

    # Invigilator confirms event
    patch_res = client.patch(
        f"/api/events/{event_id}",
        json={"status": "confirmed", "reviewer_notes": "Confirmed on CCTV camera angle"},
    )
    assert patch_res.status_code == 200
    updated = patch_res.json()
    assert updated["status"] == "confirmed"
    assert updated["reviewer_notes"] == "Confirmed on CCTV camera angle"

    # Verify reflected in list
    res_list = client.get("/api/events")
    assert res_list.json()[0]["status"] == "confirmed"


def test_system_status(client_with_event):
    client, _ = client_with_event
    res = client.get("/api/system/status")
    assert res.status_code == 200
    status = res.json()
    assert status["total_events"] >= 1
    assert "effective_fps" in status
