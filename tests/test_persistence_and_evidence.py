"""
Comprehensive test suite for ExamGuard Persistence, Sessions, Evidence Ring Buffer, and Backups.
Enforces:
- No real webcam images in automated fixtures (SYNTHETIC ONLY)
- Foreign-key integrity
- Stale session recovery -> INTERRUPTED
- Tamper-evident audit chain verification
- Atomic backup verification and corruption detection
"""

import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import numpy as np
import pytest

from src.persistence.database import DatabaseManager
from src.persistence.migrations import run_migrations, get_current_version, CURRENT_SCHEMA_VERSION
from src.persistence.models import (
    ExamSession,
    PersistedEvent,
    EventEvidence,
    ReviewRecord,
    AuditLogEntry,
)
from src.persistence.service import PersistenceService, compute_file_sha256
from src.persistence.backup import (
    create_local_backup,
    verify_backup_directory,
    preview_retention_candidates,
)
from src.evidence.clip_recorder import RollingClipRecorder
from src.evidence.snapshot import SnapshotCapture


@pytest.fixture
def temp_db_service(tmp_path):
    """Create an isolated temporary SQLite database and PersistenceService."""
    db_file = tmp_path / "test_examguard.sqlite3"
    svc = PersistenceService(db_path=str(db_file))
    yield svc
    svc.db.close()


def test_schema_migration(tmp_path):
    """Verify that schema migrations initialize all tables and schema_meta version 2."""
    db_file = tmp_path / "test_mig.sqlite3"
    db = DatabaseManager(db_path=str(db_file))
    
    assert get_current_version(db) == 0
    ver = run_migrations(db)
    assert ver == CURRENT_SCHEMA_VERSION
    assert get_current_version(db) == CURRENT_SCHEMA_VERSION

    # Verify tables exist
    with db.cursor() as cur:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cur.fetchall()}
        assert "exam_sessions" in tables
        assert "events" in tables
        assert "event_evidence" in tables
        assert "reviews" in tables
        assert "audit_logs" in tables
        assert "users" in tables
        assert "auth_sessions" in tables
        assert "cameras" in tables
        assert "session_cameras" in tables
        assert "schema_meta" in tables
    db.close()


def test_session_lifecycle(temp_db_service):
    """Verify session creation, metadata update, heartbeat, and graceful closure."""
    svc = temp_db_service

    # 1. Initialize session
    sess = svc.initialize_runtime_session(name="Kỳ thi Thử nghiệm 1", room="Phòng B102")
    assert sess.session_id.startswith("sess_")
    assert sess.status == "ACTIVE"
    assert sess.name == "Kỳ thi Thử nghiệm 1"
    assert sess.room == "Phòng B102"
    assert svc.active_session is not None

    # 2. Heartbeat update
    old_hb = sess.last_heartbeat_at
    time.sleep(0.01)
    svc.heartbeat()
    refreshed = svc.sessions.get_session(sess.session_id)
    assert refreshed.last_heartbeat_at >= old_hb

    # 3. Metadata update
    success = svc.update_session_metadata(
        session_id=sess.session_id,
        name="Kỳ thi Chính thức",
        invigilator_name="Thầy Nguyễn Văn A",
    )
    assert success is True
    updated = svc.sessions.get_session(sess.session_id)
    assert updated.name == "Kỳ thi Chính thức"
    assert updated.invigilator_name == "Thầy Nguyễn Văn A"

    # 4. Graceful stop
    closed = svc.close_active_session(reason="GRACEFUL_STOP")
    assert closed is not None
    assert closed.status == "CLOSED"
    assert closed.close_reason == "GRACEFUL_STOP"
    assert closed.ended_at is not None
    assert svc.active_session is None


def test_stale_session_interrupted_recovery(temp_db_service):
    """Simulate abnormal runtime crash and ensure next startup marks stale session as INTERRUPTED."""
    svc = temp_db_service

    # Session 1 starts but never gracefully closes (simulates crash)
    sess1 = svc.initialize_runtime_session(name="Phiên bị ngắt đột ngột", room="Phòng Lab")
    s1_id = sess1.session_id
    svc.heartbeat()

    # Next startup creates new session and detects stale session
    svc2 = PersistenceService(db_path=svc.db.db_path)
    sess2 = svc2.initialize_runtime_session(name="Phiên mới kế tiếp", room="Phòng Lab")

    # Verify session 1 is marked INTERRUPTED
    s1_recovered = svc2.sessions.get_session(s1_id)
    assert s1_recovered.status == "INTERRUPTED"
    assert s1_recovered.close_reason == "UNEXPECTED_TERMINATION"
    assert s1_recovered.ended_at is not None

    # Verify session 2 is active
    assert sess2.status == "ACTIVE"
    assert sess2.session_id != s1_id


def test_event_persistence_and_deduplication(temp_db_service):
    """Verify event INSERT, UPDATE (without duplicates), and CLOSE lifecycle."""
    svc = temp_db_service
    sess = svc.initialize_runtime_session()

    class MockFusedEvent:
        def __init__(self, eid, risk="LOW", score=25.0, duration=1.0, end_ts=None):
            self.event_id = eid
            self.track_id = 7
            self.camera_id = "laptop_webcam_0"
            self.event_type = "SUSTAINED_HEAD_REST"
            self.start_timestamp = 1000.0
            self.last_update_timestamp = 1001.0
            self.end_timestamp = end_ts
            self.duration = duration
            self.risk_level = risk
            self.risk_score = score
            self.lifecycle_status = "active"
            self.observation_snapshot = {"posture": "HEAD_REST_SLEEP"}
            self.evidence_summary = {}
            self.event_origin = "PHYSICAL_LIVE_CAMERA"

    # 1. Event OPEN
    ev = MockFusedEvent("ev_test_101", risk="LOW", score=30.0, duration=1.5)
    svc.persist_stage2_event(ev, "OPEN")

    stored = svc.events.get_event("ev_test_101")
    assert stored is not None
    assert stored.event_id == "ev_test_101"
    assert stored.session_id == sess.session_id
    assert stored.track_id == 7
    assert stored.severity == "LOW"
    assert stored.score == 30.0

    # 2. Event UPDATE (escalation to HIGH)
    ev.risk_level = "HIGH"
    ev.risk_score = 88.0
    ev.duration = 4.2
    # Ensure throttle window passes
    svc._last_event_update_times.pop("ev_test_101", None)
    svc.persist_stage2_event(ev, "UPDATE")

    stored2 = svc.events.get_event("ev_test_101")
    assert stored2.severity == "HIGH"
    assert stored2.score == 88.0
    assert stored2.duration_sec == 4.2

    # 3. Event CLOSE
    ev.end_timestamp = 1005.0
    ev.duration = 5.0
    svc.persist_stage2_event(ev, "CLOSE")

    stored3 = svc.events.get_event("ev_test_101")
    assert stored3.lifecycle_status == "closed"
    assert stored3.closed_at is not None
    assert stored3.duration_sec == 5.0

    # Ensure no duplicates: only 1 event in DB
    events_list = svc.events.list_events_by_session(sess.session_id)
    assert len(events_list) == 1


def test_review_persistence_and_history(temp_db_service):
    """Verify human review persistence and append-only decision audit history."""
    svc = temp_db_service
    sess = svc.initialize_runtime_session()

    # Create base event
    ev = PersistedEvent(
        event_id="ev_review_test",
        session_id=sess.session_id,
        camera_id="cam_0",
        track_id=2,
        event_type="PHONE_ASSOCIATED",
        opened_at="2026-10-04T12:00:00",
        severity="HIGH",
        score=95.0,
        lifecycle_status="active",
        review_status="awaiting",
    )
    svc.events.upsert_event(ev)

    # 1. First review: CONFIRMED
    svc.record_review_decision(
        event_id="ev_review_test",
        decision="CONFIRMED",
        note="Phát hiện điện thoại trên bàn",
        reviewer_name="Giám thị 1",
    )

    ev_after1 = svc.events.get_event("ev_review_test")
    assert ev_after1.review_status == "confirmed"

    # 2. Second review: DISMISSED (e.g. overturned after closer inspection)
    svc.record_review_decision(
        event_id="ev_review_test",
        decision="DISMISSED",
        note="Nhầm lẫn với hộp bút học sinh",
        reviewer_name="Hội đồng thi",
    )

    ev_after2 = svc.events.get_event("ev_review_test")
    assert ev_after2.review_status == "dismissed"

    # 3. Verify complete historical audit trail retained
    rev_history = svc.reviews.get_review_history("ev_review_test")
    assert len(rev_history) == 2
    assert rev_history[0].decision == "CONFIRMED"
    assert rev_history[0].note == "Phát hiện điện thoại trên bàn"
    assert rev_history[1].decision == "DISMISSED"
    assert rev_history[1].note == "Nhầm lẫn với hộp bút học sinh"


def test_tamper_evident_audit_chain(temp_db_service):
    """Verify that audit logs maintain a valid SHA-256 hash chain and detect tampering."""
    svc = temp_db_service
    sess = svc.initialize_runtime_session()

    # Log several actions
    svc.audit.log_action("a1", "SESSION_CREATED", session_id=sess.session_id)
    svc.audit.log_action("a2", "EVENT_OPENED", session_id=sess.session_id, event_id="ev_1")
    svc.audit.log_action("a3", "EVENT_CONFIRMED", session_id=sess.session_id, event_id="ev_1")

    # Verification passes
    is_valid, err = svc.audit.verify_audit_chain()
    assert is_valid is True
    assert err is None

    # Simulate tampering: manually alter action in SQLite directly
    with svc.db.transaction() as cur:
        cur.execute("UPDATE audit_logs SET action = 'TAMPERED_ACTION' WHERE audit_id = 'a2';")

    # Verification must now fail
    is_tampered, tamper_err = svc.audit.verify_audit_chain()
    assert is_tampered is False
    assert "mismatch" in tamper_err.lower() or "break" in tamper_err.lower()


def test_synthetic_evidence_recording_and_sha256(temp_db_service, tmp_path):
    """Verify evidence file recording, SHA-256 calculation, and integrity checks."""
    svc = temp_db_service
    sess = svc.initialize_runtime_session()

    # Create synthetic parent event
    parent_ev = PersistedEvent(
        event_id="ev_synth_01",
        session_id=sess.session_id,
        camera_id="laptop_webcam_0",
        track_id=3,
        event_type="PHONE_ASSOCIATED",
        opened_at="2026-10-04T12:00:00",
        severity="HIGH",
        score=92.0,
        lifecycle_status="active",
        review_status="awaiting",
    )
    svc.events.upsert_event(parent_ev)

    # Create synthetic frame and image file
    fake_img = tmp_path / "synthetic_snapshot.jpg"
    synthetic_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 200 + b"\xff\xd9"
    fake_img.write_bytes(synthetic_bytes)

    # Record evidence
    ev_rec = svc.record_evidence_file(
        event_id="ev_synth_01",
        evidence_type="SNAPSHOT",
        file_path=str(fake_img),
    )
    assert ev_rec is not None
    assert os.path.exists(ev_rec.file_path)
    assert ev_rec.sha256 == compute_file_sha256(ev_rec.file_path)

    # Verify integrity check returns VALID
    res = svc.verify_evidence_integrity(ev_rec.evidence_id)
    assert res["status"] == "VALID"
    assert res["valid"] is True

    # Tamper with file
    Path(ev_rec.file_path).write_bytes(b"\xff\xd8" + b"\x99" * 300)
    res_tampered = svc.verify_evidence_integrity(ev_rec.evidence_id)
    assert res_tampered["status"] == "HASH_MISMATCH"
    assert res_tampered["valid"] is False


def test_backup_and_verification_with_corruption_detection(temp_db_service, tmp_path):
    """Verify atomic local backup creation, manifest verification, and corruption detection."""
    svc = temp_db_service
    sess = svc.initialize_runtime_session(name="Phiên cần sao lưu")

    # Add an event
    ev = PersistedEvent(
        event_id="ev_backup_01",
        session_id=sess.session_id,
        camera_id="laptop_webcam_0",
        track_id=1,
        event_type="STANDING",
        opened_at="2026-10-04T12:00:00",
        severity="MEDIUM",
        score=65.0,
        lifecycle_status="closed",
        review_status="awaiting",
    )
    svc.events.upsert_event(ev)

    # Add synthetic evidence file
    ev_dir = tmp_path / "evidence_src"
    ev_dir.mkdir(parents=True, exist_ok=True)
    ev_file = ev_dir / "test_evidence.jpg"
    ev_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x42" * 150)
    svc.record_evidence_file("ev_backup_01", "SNAPSHOT", str(ev_file))

    # 1. Create backup
    backup_base = tmp_path / "test_backups"
    res = create_local_backup(svc, backup_base_dir=str(backup_base))
    assert res["success"] is True
    assert "backup_path" in res

    bpath = res["backup_path"]
    backup_full_path = Path(bpath) if os.path.isabs(bpath) else (Path(__file__).resolve().parent.parent / bpath)
    assert backup_full_path.exists()
    assert (backup_full_path / "backup-manifest.json").exists()
    assert (backup_full_path / "examguard.sqlite3").exists()

    # 2. Verify backup
    valid, msg, details = verify_backup_directory(str(backup_full_path))
    assert valid is True

    # 3. Corrupt one backup file -> verification must fail
    manifest_file = backup_full_path / "backup-manifest.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Corrupt the database file
    db_backup = backup_full_path / manifest["database"]["filename"]
    db_backup.write_bytes(b"CORRUPTED_DATABASE_CONTENT")

    corrupted_valid, corrupted_msg, _ = verify_backup_directory(str(backup_full_path))
    assert corrupted_valid is False
    assert "không khớp" in corrupted_msg.lower() or "lỗi" in corrupted_msg.lower()


def test_retention_policy_preview(temp_db_service):
    """Verify retention policy preview lists old events without modifying database."""
    svc = temp_db_service
    sess = svc.initialize_runtime_session()

    # Event 1: Recent awaiting
    ev1 = PersistedEvent(
        event_id="ev_ret_recent",
        session_id=sess.session_id,
        camera_id="cam_0",
        track_id=1,
        event_type="PHONE_ASSOCIATED",
        opened_at="2026-10-04T12:00:00",
        severity="HIGH",
        score=90.0,
        lifecycle_status="active",
        review_status="awaiting",
    )
    svc.events.upsert_event(ev1)

    # Event 2: Old dismissed (60 days ago)
    ev2 = PersistedEvent(
        event_id="ev_ret_old_dismissed",
        session_id=sess.session_id,
        camera_id="cam_0",
        track_id=2,
        event_type="SUSTAINED_HEAD_REST",
        opened_at="2026-08-01T12:00:00",
        severity="LOW",
        score=20.0,
        lifecycle_status="closed",
        review_status="dismissed",
    )
    svc.events.upsert_event(ev2)

    preview = preview_retention_candidates(svc, dismissed_days=30, confirmed_days=90)
    assert preview["policy"]["retention_enabled_default"] is False
    assert preview["preview_summary"]["dismissed_candidates_count"] == 1
    assert preview["candidates"][0]["event_id"] == "ev_ret_old_dismissed"

    # Database records must still exist completely untouched
    assert svc.events.get_event("ev_ret_old_dismissed") is not None


def test_synthetic_ring_buffer_and_clip_recorder(tmp_path):
    """Verify bounded ring buffer, pre-roll/post-roll frames, atomic MP4 write, and failure isolation."""
    out_dir = tmp_path / "clips"
    clip_ready = []
    clip_failed = []

    recorder = RollingClipRecorder(
        output_dir=str(out_dir),
        pre_event_seconds=1.0,
        post_event_seconds=1.0,
        max_fps=10.0,
        max_buffer_mb=2.0,
        on_clip_ready=lambda eid, path, sha, sz: clip_ready.append((eid, path, sha, sz)),
        on_clip_failed=lambda eid, path, err: clip_failed.append((eid, path, err)),
    )

    # 1. Push synthetic frames into pre-roll buffer
    t0 = 100.0
    dummy_frame = np.zeros((120, 160, 3), dtype=np.uint8)
    for i in range(15):
        recorder.push_frame(dummy_frame, timestamp=t0 + i * 0.1)

    # Pre-roll buffer has frames
    assert len(recorder._rolling_buffer) > 0

    # 2. Trigger clip capture at t = 101.5
    clip_path = recorder.trigger_clip(
        event_id="ev_clip_test",
        timestamp=101.5,
        fps=10.0,
    )
    assert clip_path.endswith("ev_clip_test_evidence.mp4")

    # 3. Push post-roll frames until past end_timestamp (102.5)
    for i in range(16, 30):
        recorder.push_frame(dummy_frame, timestamp=t0 + i * 0.1)

    # 4. Wait for background worker to finalize MP4
    recorder.shutdown()

    assert len(clip_ready) == 1
    eid, path, sha, sz = clip_ready[0]
    assert eid == "ev_clip_test"
    assert os.path.exists(path)
    assert sz > 0
    assert len(sha) == 64  # valid SHA-256 length

    # 5. Verify failure isolation: invalid frame decode or corrupted write does not raise
    recorder._dispatch_encode("ev_fail_test", str(out_dir / "nonexistent" / "bad.mp4"), [], 10.0)
    # Pipeline remains healthy


def test_integrated_evidence_lifecycle_with_manifest(temp_db_service, tmp_path):
    """Verify IntegratedEvidenceManager coordination of snapshots, clip, manifest and DB records."""
    from src.orchestration.evidence_manager import IntegratedEvidenceManager
    from src.fusion.types import FusedEvent

    svc = temp_db_service
    sess = svc.initialize_runtime_session(name="Phiên kiểm thử bằng chứng")

    ev_dir = tmp_path / "integrated_evidence"
    mgr = IntegratedEvidenceManager(
        output_dir=str(ev_dir),
        pre_event_seconds=1.0,
        post_event_seconds=1.0,
        persistence_service=svc,
    )

    fev = FusedEvent(
        event_id="fev_test_99",
        track_id=4,
        camera_id="cam_main",
        event_type="PHONE_ASSOCIATED",
        start_timestamp=200.0,
        last_update_timestamp=200.5,
        risk_level="HIGH",
        risk_score=91.0,
    )
    # Pre-insert event into DB to satisfy foreign keys
    svc.persist_stage2_event(fev, "OPEN")

    dummy_frame = np.ones((120, 160, 3), dtype=np.uint8) * 128

    # 1. Push frames to rolling buffer
    for i in range(10):
        mgr.push_frame(dummy_frame, timestamp_sec=199.0 + i * 0.1)

    # 2. OPEN lifecycle
    mgr.handle_event_lifecycle(fev, "OPEN", dummy_frame, timestamp_sec=200.0, fps=10.0)
    assert "snapshot_path" in fev.evidence_summary

    # 3. CLOSE lifecycle
    fev.end_timestamp = 202.0
    for i in range(10):
        mgr.push_frame(dummy_frame, timestamp_sec=201.0 + i * 0.1)

    mgr.handle_event_lifecycle(fev, "CLOSE", dummy_frame, timestamp_sec=202.0, fps=10.0)
    mgr.shutdown()

    # Verify manifest file exists
    assert "manifest_path" in fev.evidence_summary
    manifest_p = fev.evidence_summary["manifest_path"]
    assert os.path.exists(manifest_p)

    raw_manifest, _ = svc.load_and_decrypt_evidence(manifest_p)
    mdata = json.loads(raw_manifest.decode("utf-8"))
    assert mdata["event_id"] == "fev_test_99"
    assert mdata["severity"] == "HIGH"
    assert mdata["session_id"] == sess.session_id
