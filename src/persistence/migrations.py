"""
Database migration management and schema definitions for ExamGuard.
"""

from datetime import datetime
import logging
import sqlite3
from typing import List, Tuple
from src.persistence.database import DatabaseManager

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 1

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


def run_migrations(db: DatabaseManager) -> int:
    """Execute all pending database migrations in an atomic transaction."""
    current_ver = get_current_version(db)
    if current_ver >= CURRENT_SCHEMA_VERSION:
        logger.debug(f"Database schema is up to date (version {current_ver}).")
        return current_ver

    logger.info(f"Migrating database schema from version {current_ver} to {CURRENT_SCHEMA_VERSION}...")
    with db.transaction() as cur:
        # Executes script statements
        cur.executescript(MIGRATION_V1_SQL)
        now_iso = datetime.now().isoformat()
        cur.execute(
            "INSERT OR REPLACE INTO schema_meta (version, applied_at) VALUES (?, ?);",
            (CURRENT_SCHEMA_VERSION, now_iso),
        )
    logger.info(f"Database schema migration to version {CURRENT_SCHEMA_VERSION} complete.")
    return CURRENT_SCHEMA_VERSION
