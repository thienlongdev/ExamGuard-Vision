"""
Tests for ExamGuard Vision P0/P1 Repairs:
- WebSocket connection & lifecycle event handling
- Safe evidence serving with subdirectories & path-traversal prevention
- Event-time immutable observation snapshot persistence
- Event lifecycle separate from human review status
- Review status updates & reviewer notes persistence
- Clean product camera frame vs debug HUD separation
- KPI count consistency & terminology mapping
"""

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.behavior.event_manager import EventManager, SuspiciousEvent
from src.fusion.types import FusedEvent
from src.fusion.event_engine import TrackEventStateMachine


@pytest.fixture
def session_evidence():
    """Create a temporary test evidence image inside a session subfolder."""
    ev_dir = Path("evidence/asus_a17_demo/snapshots")
    ev_dir.mkdir(parents=True, exist_ok=True)
    test_img = ev_dir / "test_session_snap_001.jpg"
    test_img.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00")
    yield test_img
    if test_img.exists():
        test_img.unlink()


def test_evidence_subpath_and_path_traversal(session_evidence):
    """Verify evidence serving handles session subfolders and blocks path traversal."""
    app = create_app(enforce_auth=False)
    client = TestClient(app)

    # 1. Direct subpath retrieval
    res_direct = client.get("/api/evidence/asus_a17_demo/snapshots/test_session_snap_001.jpg")
    assert res_direct.status_code == 200
    assert res_direct.headers["content-type"].startswith("image/jpeg")

    # 2. Fallback search by filename
    res_fallback = client.get("/api/evidence/snapshots/test_session_snap_001.jpg")
    assert res_fallback.status_code == 200
    assert res_fallback.headers["content-type"].startswith("image/jpeg")

    # 3. Path traversal attacks must return 403 Forbidden
    traversal_attacks = [
        "/api/evidence/../main.py",
        "/api/evidence/../../src/api/main.py",
        "/api/evidence/%2e%2e/main.py",
        "/api/evidence/%2e%2e%2fmain.py",
        "/api/evidence/..\\..\\main.py",
    ]
    for url in traversal_attacks:
        res = client.get(url)
        assert res.status_code in [403, 404], f"Attack {url} was not rejected with 403/404!"

    # 4. Legitimate missing file returns 404
    res_missing = client.get("/api/evidence/snapshots/non_existent_file_12345.jpg")
    assert res_missing.status_code == 404


def test_event_time_observation_snapshot():
    """Verify event-time observation snapshot preserves exact cues at event creation time."""
    em = EventManager(camera_id="cam-obs-test")
    
    # Create an event with an immutable observation snapshot
    obs_data = {
        "posture": {
            "class": "HEAD_REST_SLEEP",
            "confidence": 0.942,
            "availability": "available",
            "probabilities": {"HEAD_REST_SLEEP": 0.942, "NORMAL_UPRIGHT": 0.058},
        },
        "headpose": {
            "available": True,
            "yaw_deg": -35.4,
            "status": "VALID",
            "reliability": 0.91,
        },
        "phone": {
            "detected": False,
            "status": "NONE",
            "confidence": 0.0,
        },
        "macro_behavior": {
            "cue": "POSTURE_HEAD_REST_SLEEP",
            "confidence": 0.942,
        },
        "risk": {
            "score": 68.0,
            "level": "HIGH",
        },
        "timestamp": 105.0,
        "origin": "PHYSICAL_LIVE_CAMERA",
    }

    event = SuspiciousEvent(
        event_id="ev_obs_001",
        track_id=1,
        camera_id="cam-obs-test",
        timestamp=105.0,
        start_time=100.0,
        end_time=105.0,
        event_type="SUSTAINED_HEAD_REST",
        risk_level="HIGH",
        score=68.0,
        evidence={"posture": "HEAD_REST_SLEEP", "yaw_deg": -35.4},
        snapshot_path="evidence/asus_a17_demo/snapshots/snap_head_rest.jpg",
        status="new",
        lifecycle_status="active",
        review_status="awaiting",
        observation_snapshot=obs_data,
    )
    em._events[event.event_id] = event

    app = create_app(event_manager=em, enforce_auth=False)
    client = TestClient(app)

    # Fetch event via REST API
    res = client.get(f"/api/events/{event.event_id}")
    assert res.status_code == 200
    data = res.json()

    # Verify lifecycle and review status separation
    assert data["lifecycle_status"] == "active"
    assert data["review_status"] == "awaiting"

    # Verify immutable snapshot is preserved
    obs = data.get("observation_snapshot", {})
    assert obs["posture"]["class"] == "HEAD_REST_SLEEP"
    assert obs["posture"]["confidence"] == pytest.approx(0.942)
    assert obs["headpose"]["yaw_deg"] == pytest.approx(-35.4)
    assert obs["origin"] == "PHYSICAL_LIVE_CAMERA"

    # Verify browser-safe snapshot URL
    assert data["snapshot_path"] == "/api/evidence/asus_a17_demo/snapshots/snap_head_rest.jpg"
    assert "E:\\" not in data["snapshot_path"]


def test_lifecycle_vs_human_review_status_separation():
    """Verify closing an event does not automatically confirm or dismiss it."""
    em = EventManager(camera_id="cam-life-test")
    event = SuspiciousEvent(
        event_id="ev_life_001",
        track_id=1,
        camera_id="cam-life-test",
        timestamp=150.0,
        start_time=145.0,
        end_time=150.0,
        event_type="PHONE_ASSOCIATED",
        risk_level="HIGH",
        score=75.0,
        evidence={"phone_status": "ASSOCIATED"},
        snapshot_path="evidence/asus_a17_demo/snapshots/snap_phone.jpg",
        status="new",
        lifecycle_status="active",
        review_status="awaiting",
    )
    em._events[event.event_id] = event

    # Now simulate event closure by pipeline (e.g. phone put away)
    event.lifecycle_status = "closed"
    # Notice: review_status must remain 'awaiting'
    assert event.review_status == "awaiting"

    app = create_app(event_manager=em, enforce_auth=False)
    client = TestClient(app)

    res = client.get(f"/api/events/{event.event_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["lifecycle_status"] == "closed"
    assert data["review_status"] == "awaiting", "Closed physical event must still be awaiting human review!"


def test_human_review_actions_and_notes_persistence():
    """Verify PATCH /api/events/{id} updates review status and persists reviewer notes."""
    em = EventManager(camera_id="cam-review-test")
    event = SuspiciousEvent(
        event_id="ev_rev_001",
        track_id=1,
        camera_id="cam-review-test",
        timestamp=200.0,
        start_time=195.0,
        end_time=200.0,
        event_type="PHONE_ASSOCIATED",
        risk_level="HIGH",
        score=85.0,
        evidence={"phone_status": "ASSOCIATED"},
        snapshot_path="",
        status="new",
        lifecycle_status="closed",
        review_status="awaiting",
    )
    em._events[event.event_id] = event

    app = create_app(event_manager=em, enforce_auth=False)
    client = TestClient(app)

    # 1. Confirm the event with invigilator notes
    patch_payload = {
        "status": "confirmed",
        "reviewer_notes": "Confirmed mobile device in right hand during exam.",
    }
    res_patch = client.patch(f"/api/events/{event.event_id}", json=patch_payload)
    assert res_patch.status_code == 200
    data_patch = res_patch.json()
    assert data_patch["review_status"] == "confirmed"
    assert data_patch["reviewer_notes"] == "Confirmed mobile device in right hand during exam."
    assert data_patch["lifecycle_status"] == "closed"

    # 2. Fetch event to confirm persistence
    res_get = client.get(f"/api/events/{event.event_id}")
    assert res_get.status_code == 200
    data_get = res_get.json()
    assert data_get["review_status"] == "confirmed"
    assert data_get["reviewer_notes"] == "Confirmed mobile device in right hand during exam."

    # 3. Dismiss another event
    event2 = SuspiciousEvent(
        event_id="ev_rev_002",
        track_id=2,
        camera_id="cam-review-test",
        timestamp=210.0,
        start_time=205.0,
        end_time=210.0,
        event_type="SUSTAINED_HEAD_REST",
        risk_level="MEDIUM",
        score=45.0,
        evidence={"posture": "HEAD_REST_SLEEP"},
        snapshot_path="",
        status="new",
        lifecycle_status="closed",
        review_status="awaiting",
    )
    em._events[event2.event_id] = event2

    res_dismiss = client.patch(
        f"/api/events/{event2.event_id}",
        json={"status": "dismissed", "reviewer_notes": "Student dropped pencil on desk."},
    )
    assert res_dismiss.status_code == 200
    assert res_dismiss.json()["review_status"] == "dismissed"


def test_clean_product_camera_stream_frame():
    """Verify camera frame endpoint returns clean JPEG frame without burned-in debug text."""
    class MockStage2Pipeline:
        def __init__(self):
            # Clean raw frame (valid JPEG mock)
            self._latest_jpeg_frame = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01clean_frame"
            self.source = type("MockSource", (), {"is_opened": lambda self: True})()

    mock_pipe = MockStage2Pipeline()
    app = create_app(stage2_pipeline=mock_pipe, enforce_auth=False)
    client = TestClient(app)

    res = client.get("/api/cameras/frame")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert res.content == mock_pipe._latest_jpeg_frame


def test_websocket_events_endpoint_and_broadcast():
    """Verify WebSocket /ws/events connects cleanly and broadcasts live events."""
    em = EventManager(camera_id="cam-ws-test")
    app = create_app(event_manager=em, enforce_auth=False)
    client = TestClient(app)

    with client.websocket_connect("/ws/events") as websocket:
        # 1. Connection established
        assert websocket is not None

        # 2. Trigger an event status update via REST
        event = SuspiciousEvent(
            event_id="ev_ws_001",
            track_id=1,
            camera_id="cam-ws-test",
            timestamp=300.0,
            start_time=295.0,
            end_time=300.0,
            event_type="PHONE_ASSOCIATED",
            risk_level="HIGH",
            score=90.0,
            evidence={"phone_status": "ASSOCIATED"},
            snapshot_path="",
            status="new",
            lifecycle_status="active",
            review_status="awaiting",
        )
        em._events[event.event_id] = event

        # Update event status to trigger broadcast
        client.patch(
            f"/api/events/{event.event_id}",
            json={"status": "confirmed", "reviewer_notes": "Verified via live monitoring."},
        )

        # Receive WS message
        data = websocket.receive_json()
        assert data["type"] == "EVENT_STATUS_UPDATED"
        assert data["event_id"] == "ev_ws_001"
        assert data["status"] == "confirmed"
        assert data["reviewer_notes"] == "Verified via live monitoring."


def test_track_state_machine_snapshot_immutability():
    """Verify TrackEventStateMachine produces an immutable observation snapshot that resists later changes."""
    from src.fusion.types import EventFamily, FusedEvent, ObservationStatus
    from src.fusion.cue_state import PerTrackCueState

    sm = TrackEventStateMachine(
        track_id=1,
        event_family=EventFamily.SUSTAINED_HEAD_REST,
        min_candidate_duration=0.5,
        enter_threshold=0.6,
        exit_threshold=0.3,
        cooldown_seconds=1.0,
    )

    # Frame 1: Candidate start (time = 10.0s)
    cue1 = PerTrackCueState(
        track_id=1,
        last_update_timestamp=10.0,
        posture_status=ObservationStatus.AVAILABLE,
        posture_probs={"HEAD_REST_SLEEP": 0.95, "NORMAL_UPRIGHT": 0.05},
        headpose_status=ObservationStatus.AVAILABLE,
        smoothed_yaw_deg=-32.0,
    )
    ev1, action1 = sm.process_frame(
        timestamp=10.0,
        evidence_score=0.8,
        is_vetoed=False,
        cue_state=cue1,
        camera_id="cam-immut-test",
    )
    assert action1 == "NONE"

    # Frame 2: Candidate duration met (time = 10.6s) -> OPENS event
    cue2 = PerTrackCueState(
        track_id=1,
        last_update_timestamp=10.6,
        posture_status=ObservationStatus.AVAILABLE,
        posture_probs={"HEAD_REST_SLEEP": 0.96, "NORMAL_UPRIGHT": 0.04},
        headpose_status=ObservationStatus.AVAILABLE,
        smoothed_yaw_deg=-32.0,
    )
    ev_open, action_open = sm.process_frame(
        timestamp=10.6,
        evidence_score=0.85,
        is_vetoed=False,
        cue_state=cue2,
        camera_id="cam-immut-test",
    )
    assert action_open == "OPEN"
    assert ev_open is not None
    assert ev_open.lifecycle_status == "open"
    assert ev_open.review_status == "awaiting"

    snap1 = ev_open.observation_snapshot
    assert snap1["posture"]["class"] == "HEAD_REST_SLEEP"
    assert snap1["posture"]["confidence"] == pytest.approx(0.96)
    assert snap1["headpose"]["yaw_deg"] == pytest.approx(-32.0)

    # Frame 3: Student returns upright. Score drops below exit threshold (time = 12.0s) -> CLOSES event
    cue3 = PerTrackCueState(
        track_id=1,
        last_update_timestamp=12.0,
        posture_status=ObservationStatus.AVAILABLE,
        posture_probs={"HEAD_REST_SLEEP": 0.01, "NORMAL_UPRIGHT": 0.99},
        headpose_status=ObservationStatus.AVAILABLE,
        smoothed_yaw_deg=0.0,
    )
    ev_close, action_close = sm.process_frame(
        timestamp=12.0,
        evidence_score=0.1,
        is_vetoed=False,
        cue_state=cue3,
        camera_id="cam-immut-test",
    )
    assert action_close == "CLOSE"
    assert ev_close is not None
    assert ev_close.lifecycle_status == "closed"
    assert ev_close.review_status == "awaiting", "Closed event must remain awaiting human review!"

    # CRITICAL: Snapshot on the closed event must STILL contain the original HEAD_REST_SLEEP cue, NOT the upright posture!
    snap2 = ev_close.observation_snapshot
    assert snap2["posture"]["class"] == "HEAD_REST_SLEEP"
    assert snap2["posture"]["confidence"] == pytest.approx(0.96)
    assert snap2["headpose"]["yaw_deg"] == pytest.approx(-32.0)


def test_kpi_counting_semantics_and_consistency():
    """Verify KPI counting logic matches exact definitions across states."""
    em = EventManager(camera_id="cam-kpi-test")
    
    events = [
        # Event 1: Active, awaiting
        SuspiciousEvent(
            event_id="ev_kpi_1",
            track_id=1,
            camera_id="cam-kpi-test",
            timestamp=100.0,
            start_time=95.0,
            end_time=100.0,
            event_type="PHONE_ASSOCIATED",
            risk_level="HIGH",
            score=85.0,
            evidence={},
            snapshot_path="",
            status="new",
            lifecycle_status="active",
            review_status="awaiting",
        ),
        # Event 2: Closed, awaiting
        SuspiciousEvent(
            event_id="ev_kpi_2",
            track_id=2,
            camera_id="cam-kpi-test",
            timestamp=110.0,
            start_time=105.0,
            end_time=110.0,
            event_type="SUSTAINED_HEAD_REST",
            risk_level="MEDIUM",
            score=50.0,
            evidence={},
            snapshot_path="",
            status="new",
            lifecycle_status="closed",
            review_status="awaiting",
        ),
        # Event 3: Closed, confirmed
        SuspiciousEvent(
            event_id="ev_kpi_3",
            track_id=1,
            camera_id="cam-kpi-test",
            timestamp=120.0,
            start_time=115.0,
            end_time=120.0,
            event_type="PHONE_ASSOCIATED",
            risk_level="HIGH",
            score=90.0,
            evidence={},
            snapshot_path="",
            status="confirmed",
            lifecycle_status="closed",
            review_status="confirmed",
        ),
        # Event 4: Closed, dismissed
        SuspiciousEvent(
            event_id="ev_kpi_4",
            track_id=3,
            camera_id="cam-kpi-test",
            timestamp=130.0,
            start_time=125.0,
            end_time=130.0,
            event_type="STANDING",
            risk_level="LOW",
            score=30.0,
            evidence={},
            snapshot_path="",
            status="dismissed",
            lifecycle_status="closed",
            review_status="dismissed",
        ),
    ]

    for ev in events:
        em._events[ev.event_id] = ev

    app = create_app(event_manager=em, enforce_auth=False)
    client = TestClient(app)

    res = client.get("/api/events")
    assert res.status_code == 200
    all_events = res.json()
    assert len(all_events) == 4

    # Calculate KPIs
    kpi_total = len(all_events)
    kpi_awaiting = sum(1 for e in all_events if e["review_status"] == "awaiting")
    kpi_confirmed = sum(1 for e in all_events if e["review_status"] == "confirmed")
    kpi_dismissed = sum(1 for e in all_events if e["review_status"] == "dismissed")

    assert kpi_total == 4
    assert kpi_awaiting == 2  # Both active & closed events awaiting review
    assert kpi_confirmed == 1
    assert kpi_dismissed == 1
    assert kpi_total == (kpi_awaiting + kpi_confirmed + kpi_dismissed)

