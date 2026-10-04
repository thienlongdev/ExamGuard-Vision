"""
Database migration management and schema definitions for ExamGuard.
Maintains versioned, atomic, non-destructive schema migrations.
"""

from datetime import datetime
import logging
import sqlite3
from typing import List, Tuple, Optional
from src.persistence.database import DatabaseManager

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 3

MIGRATION_V1_SQL = """
-- Version 1: Core persistence schema for ExamGuard Vision

CREATE TABLE IF NOT EXISTS schema_meta (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS exam_sessions (
    session_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    room TEXT NOT NULL DEFAULT 'Phòng thi chính',
    invigilator_name TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL,
    camera_count INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_heartbeat_at TEXT,
    close_reason TEXT
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    track_id INTEGER,
    seat_id TEXT,
    event_type TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    duration_sec REAL NOT NULL DEFAULT 0.0,
    severity TEXT NOT NULL,
    score REAL NOT NULL,
    lifecycle_status TEXT NOT NULL,
    review_status TEXT NOT NULL,
    source_origin TEXT NOT NULL,
    observation_snapshot_json TEXT,
    evidence_summary_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES exam_sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS event_evidence (
    evidence_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    evidence_type TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    captured_at TEXT,
    clip_start_at TEXT,
    clip_end_at TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    reviewer_id TEXT,
    reviewer_name TEXT,
    note TEXT,
    reviewed_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS audit_logs (
    audit_id TEXT PRIMARY KEY,
    session_id TEXT,
    event_id TEXT,
    actor_type TEXT NOT NULL,
    actor_id TEXT,
    action TEXT NOT NULL,
    details_json TEXT,
    previous_entry_hash TEXT,
    entry_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Performance and retrieval indexes
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_opened_at ON events(opened_at);
CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity);
CREATE INDEX IF NOT EXISTS idx_events_review_status ON events(review_status);
CREATE INDEX IF NOT EXISTS idx_reviews_event ON reviews(event_id);
CREATE INDEX IF NOT EXISTS idx_audit_session_created ON audit_logs(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_evidence_event ON event_evidence(event_id);
"""

MIGRATION_V2_SQL = """
-- Version 2: Authentication, RBAC, Multi-Camera Orchestration, Evidence Encryption

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    must_change_password INTEGER NOT NULL DEFAULT 0,
    failed_login_count INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT,
    last_login_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    created_by TEXT
);

CREATE TABLE IF NOT EXISTS auth_sessions (
    auth_session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    token_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    revoked_at TEXT,
    user_agent_hash TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS cameras (
    camera_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_type TEXT NOT NULL,
    device_index INTEGER,
    source_uri_ref TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    resolution_width INTEGER DEFAULT 1280,
    resolution_height INTEGER DEFAULT 720,
    target_capture_fps REAL DEFAULT 30.0,
    room TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_cameras (
    session_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    PRIMARY KEY (session_id, camera_id),
    FOREIGN KEY (session_id) REFERENCES exam_sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_token ON auth_sessions(token_hash);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_cameras_enabled ON cameras(enabled);
CREATE INDEX IF NOT EXISTS idx_session_cameras_session ON session_cameras(session_id);
"""


def get_current_version(db: DatabaseManager) -> int:
    """Return latest schema version recorded in schema_meta, or 0 if uninitialized."""
    try:
        with db.cursor() as cur:
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_meta';")
            if not cur.fetchone():
                return 0
            cur.execute("SELECT MAX(version) FROM schema_meta;")
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else 0
    except Exception as e:
        logger.error(f"Failed to query schema version: {e}")
        return 0


def _apply_v2_column_migrations(cur: sqlite3.Cursor) -> None:
    """Safely apply column additions to existing tables for schema version 2."""
    # Check event_evidence columns
    cur.execute("PRAGMA table_info(event_evidence);")
    ev_cols = {row["name"] for row in cur.fetchall()}
    if "encryption_state" not in ev_cols:
        cur.execute("ALTER TABLE event_evidence ADD COLUMN encryption_state TEXT NOT NULL DEFAULT 'LEGACY_PLAINTEXT';")
    if "key_id" not in ev_cols:
        cur.execute("ALTER TABLE event_evidence ADD COLUMN key_id TEXT;")
    if "aad_json" not in ev_cols:
        cur.execute("ALTER TABLE event_evidence ADD COLUMN aad_json TEXT;")

    # Check audit_logs columns
    cur.execute("PRAGMA table_info(audit_logs);")
    audit_cols = {row["name"] for row in cur.fetchall()}
    if "actor_display_name" not in audit_cols:
        cur.execute("ALTER TABLE audit_logs ADD COLUMN actor_display_name TEXT;")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_evidence_encryption ON event_evidence(encryption_state);")


def _apply_v3_column_migrations(cur: sqlite3.Cursor) -> None:
    """Safely apply column additions to existing tables for schema version 3 (Session Lifecycle & Evidence Reliability)."""
    # Check exam_sessions columns
    cur.execute("PRAGMA table_info(exam_sessions);")
    sess_cols = {row["name"] for row in cur.fetchall()}
    if "class_name" not in sess_cols:
        cur.execute("ALTER TABLE exam_sessions ADD COLUMN class_name TEXT;")
    if "subject_code" not in sess_cols:
        cur.execute("ALTER TABLE exam_sessions ADD COLUMN subject_code TEXT;")
    if "notes" not in sess_cols:
        cur.execute("ALTER TABLE exam_sessions ADD COLUMN notes TEXT;")
    if "evidence_failure_count" not in sess_cols:
        cur.execute("ALTER TABLE exam_sessions ADD COLUMN evidence_failure_count INTEGER NOT NULL DEFAULT 0;")
    if "summary_json" not in sess_cols:
        cur.execute("ALTER TABLE exam_sessions ADD COLUMN summary_json TEXT;")

    # Check event_evidence columns
    cur.execute("PRAGMA table_info(event_evidence);")
    ev_cols = {row["name"] for row in cur.fetchall()}
    if "artifact_state" not in ev_cols:
        cur.execute("ALTER TABLE event_evidence ADD COLUMN artifact_state TEXT NOT NULL DEFAULT 'READY';")
    if "codec" not in ev_cols:
        cur.execute("ALTER TABLE event_evidence ADD COLUMN codec TEXT;")
    if "container" not in ev_cols:
        cur.execute("ALTER TABLE event_evidence ADD COLUMN container TEXT;")
    if "error_message" not in ev_cols:
        cur.execute("ALTER TABLE event_evidence ADD COLUMN error_message TEXT;")
    if "duration_sec" not in ev_cols:
        cur.execute("ALTER TABLE event_evidence ADD COLUMN duration_sec REAL;")
    if "frame_count" not in ev_cols:
        cur.execute("ALTER TABLE event_evidence ADD COLUMN frame_count INTEGER;")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_evidence_artifact_state ON event_evidence(artifact_state);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON exam_sessions(status);")


def run_migrations(db: DatabaseManager, target_version: Optional[int] = None) -> int:
    """Execute pending migrations in order up to target_version (default CURRENT_SCHEMA_VERSION)."""
    target = target_version if target_version is not None else CURRENT_SCHEMA_VERSION
    current_ver = get_current_version(db)
    if current_ver >= target:
        logger.debug(f"Database schema is up to date (version {current_ver}).")
        return current_ver

    logger.info(f"Migrating database schema from version {current_ver} to {target}...")

    # Step 1: Migration V1 if database is brand new (version 0)
    if current_ver < 1 and target >= 1:
        with db.transaction() as cur:
            cur.executescript(MIGRATION_V1_SQL)
            now_iso = datetime.now().isoformat()
            cur.execute(
                "INSERT OR REPLACE INTO schema_meta (version, applied_at) VALUES (?, ?);",
                (1, now_iso),
            )
        current_ver = 1
        logger.info("Database schema migration to version 1 complete.")

    # Step 2: Migration V2
    if current_ver < 2 and target >= 2:
        with db.transaction() as cur:
            cur.executescript(MIGRATION_V2_SQL)
            _apply_v2_column_migrations(cur)
            now_iso = datetime.now().isoformat()
            cur.execute(
                "INSERT OR REPLACE INTO schema_meta (version, applied_at) VALUES (?, ?);",
                (2, now_iso),
            )
        current_ver = 2
        logger.info("Database schema migration to version 2 complete.")

    # Step 3: Migration V3 (Session lifecycle & Evidence reliability)
    if current_ver < 3 and target >= 3:
        with db.transaction() as cur:
            _apply_v3_column_migrations(cur)
            now_iso = datetime.now().isoformat()
            cur.execute(
                "INSERT OR REPLACE INTO schema_meta (version, applied_at) VALUES (?, ?);",
                (3, now_iso),
            )
        current_ver = 3
        logger.info("Database schema migration to version 3 complete.")

    return current_ver
