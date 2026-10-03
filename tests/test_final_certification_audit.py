"""
Final Certification Audit Test Suite
====================================
Validates:
1. Source Origin Semantics:
   - Physical runtime source -> PHYSICAL_LIVE_CAMERA
   - Software validation fixture -> SOFTWARE_VALIDATION_FIXTURE
   - Replay source -> REPLAY_STREAM
   - Video file source -> VIDEO_FILE
   - Unknown fallback -> UNKNOWN
   - Event history preserves source origin
   - Evidence metadata preserves source origin
   - API serialization preserves source origin
2. Evidence Route Addressing & Ambiguity Prevention:
   - Direct canonical relative subpath addressing (/api/evidence/<session>/snapshots/<file>.jpg)
   - Unique filename fallback resolution
   - Multi-match ambiguous filename resolution returns HTTP 409 Conflict (no arbitrary selection)
   - 0 matches returns HTTP 404 Not Found
   - Path traversal security: ../, ..\\, %2e%2e, %252e%252e, drive-letters (C:), UNC paths (\\\\)
   - Physical event ID -> evidence metadata -> relative path -> HTTP route -> JPEG byte integrity
"""

import json
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.fusion.types import (
    SourceOrigin,
    UnifiedTrackUpdate,
    TrackingState,
    PostureCue,
    HeadPoseCue,
    PhoneCue,
    MacroBehaviorCue,
    ObservationStatus,
    FusedEvent,
    EventFamily,
)
from src.fusion.fusion_engine import MultiCueFusionEngine
from src.fusion.event_engine import EventEngine, TrackEventStateMachine
from src.behavior.event_manager import EventManager, SuspiciousEvent
from src.video.webcam import WebcamSource
from src.video.video_file import VideoFileSource
from src.video.rtsp import RTSPSource
from src.api.main import create_app


# ==============================================================================
# 1. SOURCE ORIGIN SEMANTICS TESTS
# ==============================================================================

def test_source_origin_enum_and_defaults():
    """Verify SourceOrigin enum definitions and safe fallback defaults."""
    assert SourceOrigin.PHYSICAL_LIVE_CAMERA.value == "PHYSICAL_LIVE_CAMERA"
    assert SourceOrigin.SOFTWARE_VALIDATION_FIXTURE.value == "SOFTWARE_VALIDATION_FIXTURE"
    assert SourceOrigin.REPLAY_STREAM.value == "REPLAY_STREAM"
    assert SourceOrigin.VIDEO_FILE.value == "VIDEO_FILE"
    assert SourceOrigin.RTSP_STREAM.value == "RTSP_STREAM"
    assert SourceOrigin.UNKNOWN.value == "UNKNOWN"

    # Default UnifiedTrackUpdate must be UNKNOWN, NEVER falsely elevated to physical
    update_default = UnifiedTrackUpdate(track_id=1, timestamp_sec=1.0)
    assert update_default.source_origin == "UNKNOWN"
    assert update_default.origin == "UNKNOWN"

    # Default FusedEvent must be UNKNOWN
    ev_default = FusedEvent(
        event_id="ev_def",
        track_id=1,
        camera_id="cam_0",
        event_type="SUSTAINED_HEAD_REST",
        start_timestamp=1.0,
        last_update_timestamp=2.0,
    )
    assert ev_default.event_origin == "UNKNOWN"


def test_video_source_classes_declare_correct_origin():
    """Verify video source abstraction classes carry truthful provenance origin."""
    webcam = WebcamSource(source=0, source_id="test-cam")
    assert getattr(webcam, "source_origin", None) == "PHYSICAL_LIVE_CAMERA"

    vfile = VideoFileSource(file_path="samples/test.mp4")
    assert getattr(vfile, "source_origin", None) == "VIDEO_FILE"

    rtsp = RTSPSource(rtsp_url="rtsp://localhost:554/live")
    assert getattr(rtsp, "source_origin", None) == "RTSP_STREAM"


def test_origin_propagation_pipeline_software_fixture():
    """Verify software validation fixture origin propagates cleanly through fusion to FusedEvent."""
    fusion_engine = MultiCueFusionEngine()
    event_engine = EventEngine()

    # Create synthetic track update tagged as SOFTWARE_VALIDATION_FIXTURE
    update = UnifiedTrackUpdate(
        track_id=42,
        timestamp_sec=10.0,
        camera_id="cam_fixture",
        source_origin=SourceOrigin.SOFTWARE_VALIDATION_FIXTURE.value,
        posture=PostureCue(
            status=ObservationStatus.AVAILABLE,
            probabilities={"HEAD_REST_SLEEP": 0.95, "NORMAL_UPRIGHT": 0.05},
        ),
        headpose=HeadPoseCue(
            status=ObservationStatus.AVAILABLE,
            yaw_deg=-30.0,
        ),
    )

    cue_state = fusion_engine.update_track(update)
    assert cue_state is not None
    assert cue_state.source_origin == "SOFTWARE_VALIDATION_FIXTURE"

    # Process through event engine
    emitted = event_engine.process_cue_state(cue_state, camera_id="cam_fixture")
    # Feed enough frames to transition CANDIDATE -> ACTIVE
    active_ev = None
    for sec in range(11, 16):
        update.timestamp_sec = float(sec)
        cs = fusion_engine.update_track(update)
        em = event_engine.process_cue_state(cs, camera_id="cam_fixture")
        for ev, action in em:
            if action in ["OPEN", "UPDATE"]:
                active_ev = ev

    assert active_ev is not None
    assert active_ev.event_origin == "SOFTWARE_VALIDATION_FIXTURE"
    assert active_ev.observation_snapshot.get("origin") == "SOFTWARE_VALIDATION_FIXTURE"
    assert active_ev.evidence_summary.get("event_origin") == "SOFTWARE_VALIDATION_FIXTURE"
    assert active_ev.event_origin != "PHYSICAL_LIVE_CAMERA", "Software fixture must NEVER claim physical live camera!"


def test_origin_propagation_physical_camera():
    """Verify physical camera update propagates PHYSICAL_LIVE_CAMERA origin."""
    fusion_engine = MultiCueFusionEngine()
    event_engine = EventEngine()

    update = UnifiedTrackUpdate(
        track_id=99,
        timestamp_sec=100.0,
        camera_id="cam_physical_0",
        source_origin=SourceOrigin.PHYSICAL_LIVE_CAMERA.value,
        posture=PostureCue(
            status=ObservationStatus.AVAILABLE,
            probabilities={"HEAD_REST_SLEEP": 0.96, "NORMAL_UPRIGHT": 0.04},
        ),
    )

    cue_state = fusion_engine.update_track(update)
    assert cue_state is not None
    assert cue_state.source_origin == "PHYSICAL_LIVE_CAMERA"

    for sec in range(101, 106):
        update.timestamp_sec = float(sec)
        cs = fusion_engine.update_track(update)
        em = event_engine.process_cue_state(cs, camera_id="cam_physical_0")
        for ev, action in em:
            if action == "OPEN":
                assert ev.event_origin == "PHYSICAL_LIVE_CAMERA"
                assert ev.observation_snapshot.get("origin") == "PHYSICAL_LIVE_CAMERA"


def test_origin_api_serialization():
    """Verify origin is preserved in REST API response serialization."""
    em = EventManager(camera_id="cam-origin-test")
    ev1 = SuspiciousEvent(
        event_id="ev_phys_001",
        track_id=1,
        camera_id="cam-0",
        timestamp=10.0,
        start_time=5.0,
        end_time=10.0,
        event_type="PHONE_ASSOCIATED",
        risk_level="HIGH",
        score=85.0,
        evidence={"event_origin": "PHYSICAL_LIVE_CAMERA"},
        event_origin="PHYSICAL_LIVE_CAMERA",
        observation_snapshot={"origin": "PHYSICAL_LIVE_CAMERA"},
    )
    ev2 = SuspiciousEvent(
        event_id="ev_soft_002",
        track_id=2,
        camera_id="cam-0",
        timestamp=15.0,
        start_time=10.0,
        end_time=15.0,
        event_type="SUSTAINED_HEAD_REST",
        risk_level="MEDIUM",
        score=60.0,
        evidence={"event_origin": "SOFTWARE_VALIDATION_FIXTURE"},
        event_origin="SOFTWARE_VALIDATION_FIXTURE",
        observation_snapshot={"origin": "SOFTWARE_VALIDATION_FIXTURE"},
    )
    em._events[ev1.event_id] = ev1
    em._events[ev2.event_id] = ev2

    app = create_app(event_manager=em)
    client = TestClient(app)

    r1 = client.get("/api/events/ev_phys_001")
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["event_origin"] == "PHYSICAL_LIVE_CAMERA"
    assert d1["observation_snapshot"]["origin"] == "PHYSICAL_LIVE_CAMERA"

    r2 = client.get("/api/events/ev_soft_002")
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["event_origin"] == "SOFTWARE_VALIDATION_FIXTURE"
    assert d2["observation_snapshot"]["origin"] == "SOFTWARE_VALIDATION_FIXTURE"


# ==============================================================================
# 2. EVIDENCE ROUTE AMBIGUITY & HARDENING TESTS
# ==============================================================================

@pytest.fixture
def evidence_test_dirs(tmp_path):
    """Create test session evidence structure."""
    session_a = Path("evidence/session_test_a/snapshots")
    session_b = Path("evidence/session_test_b/snapshots")
    session_a.mkdir(parents=True, exist_ok=True)
    session_b.mkdir(parents=True, exist_ok=True)

    # 1. Unique file in session A
    unique_file = session_a / "unique_audit_snap_001.jpg"
    unique_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF_UNIQUE_A")

    # 2. Duplicate file across session A and session B
    dup_file_a = session_a / "conflict_dup_snap_001.jpg"
    dup_file_b = session_b / "conflict_dup_snap_001.jpg"
    dup_file_a.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF_DUP_A")
    dup_file_b.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF_DUP_B")

    yield {
        "session_a": session_a,
        "session_b": session_b,
        "unique": unique_file,
        "dup_a": dup_file_a,
        "dup_b": dup_file_b,
    }

    # Cleanup
    for f in [unique_file, dup_file_a, dup_file_b]:
        if f.exists():
            f.unlink()
    try:
        session_a.rmdir()
        session_a.parent.rmdir()
        session_b.rmdir()
        session_b.parent.rmdir()
    except OSError:
        pass


def test_evidence_canonical_direct_resolution(evidence_test_dirs):
    """Verify canonical root-relative paths resolve directly without ambiguity."""
    app = create_app()
    client = TestClient(app)

    # Direct canonical lookup for session A unique file
    res = client.get("/api/evidence/session_test_a/snapshots/unique_audit_snap_001.jpg")
    assert res.status_code == 200
    assert b"JFIF_UNIQUE_A" in res.content

    # Direct canonical lookup for duplicated filename disambiguated by session
    res_a = client.get("/api/evidence/session_test_a/snapshots/conflict_dup_snap_001.jpg")
    assert res_a.status_code == 200
    assert b"JFIF_DUP_A" in res_a.content

    res_b = client.get("/api/evidence/session_test_b/snapshots/conflict_dup_snap_001.jpg")
    assert res_b.status_code == 200
    assert b"JFIF_DUP_B" in res_b.content


def test_evidence_ambiguity_returns_409_conflict(evidence_test_dirs):
    """Verify fallback lookup with duplicate filename across sessions returns 409 Conflict."""
    app = create_app()
    client = TestClient(app)

    # Legacy filename-only lookup for duplicate filename present in both session A and B
    res = client.get("/api/evidence/conflict_dup_snap_001.jpg")
    assert res.status_code == 409, f"Expected 409 Conflict for ambiguous file, got {res.status_code}"
    data = res.json()
    assert "Ambiguous evidence reference" in data["detail"]
    assert "session_test_a" not in data["detail"], "Internal absolute paths must not leak to client!"


def test_evidence_unique_fallback_returns_200(evidence_test_dirs):
    """Verify fallback lookup for unique filename returns 200."""
    app = create_app()
    client = TestClient(app)

    res = client.get("/api/evidence/unique_audit_snap_001.jpg")
    assert res.status_code == 200
    assert b"JFIF_UNIQUE_A" in res.content


def test_evidence_missing_returns_404():
    """Verify missing evidence file returns 404 Not Found."""
    app = create_app()
    client = TestClient(app)

    res = client.get("/api/evidence/completely_non_existent_file_98765.jpg")
    assert res.status_code == 404


def test_evidence_security_traversal_rejections():
    """Verify path traversal, drive-letter, UNC, and double-encoding attacks return 403 Forbidden."""
    app = create_app()
    client = TestClient(app)

    attacks = [
        "/api/evidence/../main.py",
        "/api/evidence/..\\main.py",
        "/api/evidence/%2e%2e/main.py",
        "/api/evidence/%252e%252e/main.py",
        "/api/evidence/C:/Windows/System32/calc.exe",
        "/api/evidence/c:test.jpg",
        "/api/evidence/\\\\server\\share\\test.jpg",
        "/api/evidence//server/share/test.jpg",
    ]

    for attack in attacks:
        res = client.get(attack)
        assert res.status_code in [403, 404], f"Attack '{attack}' unexpectedly succeeded with {res.status_code}"


def test_physical_event_to_jpeg_content_correctness():
    """Verify event ID -> evidence metadata -> relative path -> HTTP route -> JPEG byte integrity."""
    ev_dir = Path("evidence/asus_a17_demo/snapshots")
    meta_dir = Path("evidence/asus_a17_demo/metadata")
    ev_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    event_id = "test-cert-ev-001"
    snap_file = ev_dir / f"{event_id}_open.jpg"
    meta_file = meta_dir / f"{event_id}_metadata.json"

    # Write synthetic test jpeg bytes with identifiable marker
    marker_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF_EXAMGUARD_CERT_VERIFY"
    snap_file.write_bytes(marker_bytes)

    meta_content = {
        "event_id": event_id,
        "track_id": 1,
        "camera_id": "cam_0",
        "event_origin": "PHYSICAL_LIVE_CAMERA",
        "snapshot_path": f"asus_a17_demo/snapshots/{event_id}_open.jpg",
    }
    meta_file.write_text(json.dumps(meta_content), encoding="utf-8")

    try:
        em = EventManager(camera_id="cam_0")
        event = SuspiciousEvent(
            event_id=event_id,
            track_id=1,
            camera_id="cam_0",
            timestamp=100.0,
            start_time=95.0,
            end_time=100.0,
            event_type="SUSTAINED_HEAD_REST",
            risk_level="MEDIUM",
            score=50.0,
            evidence={"event_origin": "PHYSICAL_LIVE_CAMERA"},
            snapshot_path=f"asus_a17_demo/snapshots/{event_id}_open.jpg",
            event_origin="PHYSICAL_LIVE_CAMERA",
        )
        em._events[event_id] = event

        app = create_app(event_manager=em)
        client = TestClient(app)

        # 1. Fetch event from API
        res_ev = client.get(f"/api/events/{event_id}")
        assert res_ev.status_code == 200
        ev_data = res_ev.json()
        assert ev_data["event_id"] == event_id
        assert ev_data["event_origin"] == "PHYSICAL_LIVE_CAMERA"
        route_url = ev_data["snapshot_path"]
        assert route_url.startswith("/api/evidence/")

        # 2. Fetch JPEG bytes via HTTP route
        res_img = client.get(route_url)
        assert res_img.status_code == 200
        assert res_img.headers["content-type"].startswith("image/jpeg")
        assert marker_bytes in res_img.content, "HTTP route served incorrect image bytes!"
    finally:
        if snap_file.exists():
            snap_file.unlink()
        if meta_file.exists():
            meta_file.unlink()
