"""
Tests for Stage 2 FastAPI & WebSocket Contract Verification
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import create_app
from src.api.websocket import ConnectionManager
from src.behavior.event_manager import EventManager, SuspiciousEvent


@pytest.fixture
def client():
    ev_mgr = EventManager(camera_id="test_cam")
    app = create_app(event_manager=ev_mgr, enforce_auth=False)
    app.extra = {"event_manager": ev_mgr}
    return TestClient(app)


def test_api_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "timestamp" in data


def test_api_cameras_endpoint(client):
    res = client.get("/api/cameras")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "camera_id" in data[0]


def test_api_system_status_and_models(client):
    res = client.get("/api/system/status")
    assert res.status_code == 200
    data = res.json()
    assert "active_cameras" in data
    assert "total_events" in data
    assert "pipeline_version" in data

    # Test /api/system/models
    res_models = client.get("/api/system/models")
    assert res_models.status_code == 200
    models_data = res_models.json()
    assert "detector" in models_data
    assert "posture" in models_data
    assert "headpose" in models_data


def test_api_events_observable_language(client):
    # Verify no cheating words in event listings
    res = client.get("/api/events")
    assert res.status_code == 200
    events = res.json()
    for ev in events:
        for forbidden in ["CHEATING", "CHEATER", "GUILTY", "FRAUD"]:
            assert forbidden not in ev["event_type"].upper()
            assert forbidden not in ev["risk_level"].upper()


from src.fusion.types import FusedEvent

def test_api_patch_event_endpoint(client):
    # Create a SuspiciousEvent in manager to patch
    ev = SuspiciousEvent(
        event_id="test_patch_01",
        track_id=1,
        camera_id="test_cam",
        timestamp=100.0,
        start_time=98.0,
        end_time=100.0,
        event_type="turn_head",
        risk_level="MEDIUM",
        score=65.0,
        evidence={"turn_angle": 35.0},
        status="new",
    )
    client.app.extra["event_manager"]._events[ev.event_id] = ev
    event_id = ev.event_id

    # Test PATCH to update status
    patch_res = client.patch(
        f"/api/events/{event_id}",
        json={"status": "confirmed", "reviewer_notes": "Invigilator confirmed turning head"},
    )
    assert patch_res.status_code == 200
    data = patch_res.json()
    assert data["status"] == "confirmed"
    assert data["reviewer_notes"] == "Invigilator confirmed turning head"
    assert "score" in data, "Score must be present as configured engineering evidence score"
    assert "probability_of_cheating" not in data, "Must NEVER expose probability_of_cheating!"


def test_websocket_lifecycle_order_verification():
    """Verify exact EVENT_OPEN -> EVENT_UPDATE -> EVENT_CLOSE sequence without duplicate spam."""
    ev_mgr = EventManager(camera_id="test_cam")
    ws_mgr = ConnectionManager()

    class MockPipeline:
        def __init__(self):
            self.listeners = []
        def add_event_listener(self, l):
            self.listeners.append(l)

    mock_pipeline = MockPipeline()
    app = create_app(event_manager=ev_mgr, connection_manager=ws_mgr, stage2_pipeline=mock_pipeline, enforce_auth=False)
    app.extra["event_manager"] = ev_mgr

    with TestClient(app) as client:
        with client.websocket_connect("/ws/events") as ws:
            # Create synthetic FusedEvent
            synthetic_event = FusedEvent(
                event_id="test_lifecycle_001",
                track_id=7,
                camera_id="test_cam",
                event_type="SUSTAINED_LATERAL_HEAD_ORIENTATION",
                start_timestamp=100.0,
                last_update_timestamp=101.5,
                duration=1.5,
                risk_level="MEDIUM",
                risk_score=55.0,
            )

            # Trigger lifecycle in sequence
            for listener in mock_pipeline.listeners:
                listener(synthetic_event, "OPEN")
                listener(synthetic_event, "UPDATE")
                listener(synthetic_event, "CLOSE")

            msg1 = ws.receive_json()
            assert msg1["type"] == "EVENT_OPEN"
            assert msg1["action"] == "OPEN"
            assert msg1["event"]["event_id"] == "test_lifecycle_001"
            assert msg1["event"]["track_id"] == 7
            assert msg1["event"]["risk_score"] == 55.0

            msg2 = ws.receive_json()
            assert msg2["type"] == "EVENT_UPDATE"
            assert msg2["action"] == "UPDATE"

            msg3 = ws.receive_json()
            assert msg3["type"] == "EVENT_CLOSE"
            assert msg3["action"] == "CLOSE"


def test_websocket_connection_and_ping(client):
    with client.websocket_connect("/ws/events") as ws:
        ws.send_text("PING")
        assert True

