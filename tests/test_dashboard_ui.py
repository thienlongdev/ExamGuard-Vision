"""
Tests for ExamGuard Vision Dashboard UI, Static Assets, Camera Stream, and Safe Evidence
"""

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.behavior.event_manager import EventManager, SuspiciousEvent


@pytest.fixture
def dummy_evidence():
    """Create a temporary test evidence image inside storage/evidence/snapshots."""
    ev_dir = Path("storage/evidence/snapshots")
    ev_dir.mkdir(parents=True, exist_ok=True)
    test_img = ev_dir / "test_eval_snap.jpg"
    # Write a minimal valid JPEG header
    test_img.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb")
    yield test_img
    if test_img.exists():
        test_img.unlink()


def test_dashboard_routes():
    app = create_app()
    client = TestClient(app)

    # 1. Root route
    res = client.get("/")
    assert res.status_code == 200
    assert "ExamGuard Vision" in res.text
    assert "Exam Suspicious Behavior Monitoring" in res.text
    assert "connectWebSocket" in res.text
    assert "/static/css/tokens.css" in res.text
    assert "/static/css/dashboard.css" in res.text
    assert "/static/js/dashboard.js" in res.text

    # 2. Alias route /dashboard
    res_alias = client.get("/dashboard")
    assert res_alias.status_code == 200
    assert "ExamGuard Vision" in res_alias.text


def test_static_assets_serving():
    app = create_app()
    client = TestClient(app)

    # Tokens CSS
    res_tokens = client.get("/static/css/tokens.css")
    assert res_tokens.status_code == 200
    assert "--bg-primary" in res_tokens.text

    # Dashboard CSS
    res_dash_css = client.get("/static/css/dashboard.css")
    assert res_dash_css.status_code == 200
    assert ".camera-viewport-card" in res_dash_css.text

    # Dashboard JS
    res_js = client.get("/static/js/dashboard.js")
    assert res_js.status_code == 200
    assert "DashboardApp" in res_js.text

    # Adapter JS
    res_adapter = client.get("/static/js/adapter.js")
    assert res_adapter.status_code == 200
    assert "normalizeEvent" in res_adapter.text


def test_safe_evidence_serving(dummy_evidence):
    app = create_app()
    client = TestClient(app)

    # 1. Valid snapshot retrieval
    res = client.get("/api/evidence/snapshots/test_eval_snap.jpg")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("image/jpeg")
    assert len(res.content) > 0

    # 2. Directory traversal attempt must be rejected
    res_traversal = client.get("/api/evidence/../../src/api/main.py")
    assert res_traversal.status_code in [403, 404]

    # 3. Non-existent file
    res_404 = client.get("/api/evidence/snapshots/does_not_exist_9999.jpg")
    assert res_404.status_code == 404


def test_camera_stream_and_tracks():
    class MockPipeline:
        def __init__(self):
            self._latest_jpeg_frame = b"\xff\xd8\xff\xe0mock_jpeg_bytes"
            self._latest_tracks_summary = [
                {
                    "track_id": 1,
                    "bbox": [100, 150, 400, 600],
                    "posture": "Normal / Upright",
                    "posture_conf": 0.95,
                    "yaw_deg": 12.4,
                    "phone_status": "NONE",
                    "active_events": [],
                    "risk_level": "LOW",
                }
            ]
            self.source = type("MockSource", (), {"is_opened": lambda self: True})()

    mock_pipe = MockPipeline()
    app = create_app(stage2_pipeline=mock_pipe)
    client = TestClient(app)

    # Tracks endpoint
    res_tracks = client.get("/api/cameras/tracks")
    assert res_tracks.status_code == 200
    tracks = res_tracks.json()
    assert len(tracks) == 1
    assert tracks[0]["track_id"] == 1
    assert tracks[0]["risk_level"] == "LOW"

    # Frame endpoint
    res_frame = client.get("/api/cameras/frame")
    assert res_frame.status_code == 200
    assert res_frame.headers["content-type"] == "image/jpeg"
    assert res_frame.content == mock_pipe._latest_jpeg_frame


def test_event_snapshot_url_normalization():
    em = EventManager(camera_id="cam-norm-test")
    raw_ev = SuspiciousEvent(
        event_id="ev_norm_01",
        track_id=2,
        camera_id="cam-norm-test",
        timestamp=200.0,
        start_time=198.0,
        end_time=200.0,
        event_type="SUSTAINED_HEAD_REST",
        risk_level="MEDIUM",
        score=58.0,
        evidence={},
        snapshot_path="E:\\WorkingSpace\\ExamGuard-Vision\\storage\\evidence\\snapshots\\snap_test_01.jpg",
        status="new",
    )
    em._events[raw_ev.event_id] = raw_ev

    app = create_app(event_manager=em)
    client = TestClient(app)

    res = client.get("/api/events/ev_norm_01")
    assert res.status_code == 200
    data = res.json()
    assert data["snapshot_path"] == "/api/evidence/snapshots/snap_test_01.jpg"
    assert "E:\\" not in data["snapshot_path"], "Must never leak host filesystem paths!"


def test_history_tab_and_view_rendered():
    """Verify History tab, view section, and components are integrated."""
    app = create_app()
    client = TestClient(app)

    res = client.get("/")
    assert res.status_code == 200
    assert 'id="view-history"' in res.text

    # Verify history_view.js serves properly
    res_hv = client.get("/static/js/components/history_view.js")
    assert res_hv.status_code == 200
    assert "HistoryViewComponent" in res_hv.text

    # Verify header tab
    res_hdr = client.get("/static/js/components/header.js")
    assert res_hdr.status_code == 200
    assert "LỊCH SỬ" in res_hdr.text
    assert 'id="tab-history"' in res_hdr.text
    assert 'header-session-pill' in res_hdr.text

    # Verify event drawer has video player and integrity badge
    res_drw = client.get("/static/js/components/event_drawer.js")
    assert res_drw.status_code == 200
    assert "<video" in res_drw.text
    assert "drawer-integrity-badge" in res_drw.text
    assert "Mã theo dõi tạm thời của camera" in res_drw.text


def test_review_queue_scroll_and_card_stability_css():
    """Verify Review Queue CSS rules guarantee independent scrolling, non-shrinking cards, and stable thumbnails."""
    app = create_app()
    client = TestClient(app)

    res = client.get("/static/css/dashboard.css")
    assert res.status_code == 200
    css = res.text

    # 1. Queue scroll area must have flex and overflow-y: auto
    assert ".queue-scroll-area" in css
    assert "overflow-y: auto" in css

    # 2. Event card must have flex-shrink: 0 and min-height so cards never compress vertically
    assert ".event-card" in css
    assert "flex-shrink: 0" in css
    assert "min-height: 105px" in css

    # 3. Thumbnail wrapper must have fixed/min dimensions and flex-shrink: 0
    assert ".card-thumbnail-wrapper" in css
    assert "min-width: 96px" in css
    assert "min-height: 64px" in css
    assert ("aspect-ratio: 16 / 10" in css) or ("aspect-ratio: 16/10" in css)

    # 4. History view styles
    assert ".history-view-container" in css
    assert ".drawer-evidence-video" in css
    assert ".integrity-pill" in css

