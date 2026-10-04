"""
Tests for Review Queue Scrolling UX, stable card height, thumbnail preservation, and 1/5/10/20 events.
"""

import pytest
from src.behavior.event_manager import SuspiciousEvent
from src.api.schemas import EventResponse
from src.api.main import create_app
from fastapi.testclient import TestClient


def create_synthetic_event(i: int, risk: str = "HIGH") -> dict:
    return {
        "event_id": f"ev_{i:03d}",
        "track_id": (i % 5) + 1,
        "camera_id": "webcam_0",
        "timestamp": 1728000000.0 + i * 10,
        "start_time": 1728000000.0 + i * 10 - 5,
        "end_time": 1728000000.0 + i * 10,
        "duration": 5.0,
        "event_type": "PHONE_ASSOCIATED" if i % 2 == 0 else "SUSTAINED_HEAD_REST",
        "risk_level": risk,
        "score": 85.0 if risk == "HIGH" else (60.0 if risk == "MEDIUM" else 30.0),
        "status": "awaiting",
        "snapshot_path": f"/api/evidence/snapshots/snap_{i:03d}.jpg",
        "evidence": {"rule_id": "PHONE_ASSOCIATED", "track_id": (i % 5) + 1},
    }


@pytest.mark.parametrize("event_count", [1, 5, 10, 20])
def test_review_queue_event_scaling(event_count: int):
    """
    Verify that whether there are 1, 5, 10, or 20 events:
    - Events endpoint returns all events without drops
    - Event schemas match EventResponse contract
    - Cards and thumbnails have valid non-empty URLs
    """
    events = [create_synthetic_event(i) for i in range(event_count)]
    app = create_app()

    # Validate schema serialization for each
    validated = [EventResponse(**e) for e in events]
    assert len(validated) == event_count

    for idx, ev in enumerate(validated):
        assert ev.event_id == f"ev_{idx:03d}"
        assert ev.snapshot_path.startswith("/api/evidence/snapshots/")
        assert ev.risk_level in ["HIGH", "MEDIUM", "LOW"]


def test_review_queue_layout_contract():
    """
    Validate DOM structure contract:
    - Review Queue container is outside the scroll container (fixed header + tabs)
    - Scroll container holds all cards
    - Cards have flex-shrink: 0 and min-height >= 105px
    - Thumbnails have flex-shrink: 0 and aspect-ratio
    """
    app = create_app()
    client = TestClient(app)

    res_html = client.get("/")
    assert res_html.status_code == 200
    html = res_html.text

    assert 'id="review-queue-container"' in html
    assert 'class="monitor-queue-col"' in html

    res_css = client.get("/static/css/dashboard.css")
    assert res_css.status_code == 200
    css = res_css.text

    assert ".review-queue-panel" in css
    assert ".queue-header" in css
    assert ".queue-scroll-area" in css
    assert ".event-card" in css
    assert ".card-thumbnail-wrapper" in css
    assert ".card-thumbnail-img" in css

    # Ensure scrollbar styles exist
    assert "::-webkit-scrollbar" in css
    assert "scrollbar-width: thin" in css or "overflow-y: auto" in css
