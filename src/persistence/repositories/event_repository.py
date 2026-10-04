"""
Repository for events table operations.
"""

from datetime import datetime
import json
import logging
from typing import Optional, List, Dict, Any
from src.persistence.database import DatabaseManager
from src.persistence.models import PersistedEvent

logger = logging.getLogger(__name__)


class EventRepository:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def upsert_event(self, event: PersistedEvent) -> None:
        """Insert a newly opened event or update an existing event without creating duplicates."""
        sql = """
        INSERT INTO events (
            event_id, session_id, camera_id, track_id, seat_id, event_type,
            opened_at, closed_at, duration_sec, severity, score, lifecycle_status,
            review_status, source_origin, observation_snapshot_json,
            evidence_summary_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id) DO UPDATE SET
            duration_sec = excluded.duration_sec,
            severity = excluded.severity,
            score = excluded.score,
            lifecycle_status = excluded.lifecycle_status,
            closed_at = CASE WHEN excluded.closed_at IS NOT NULL THEN excluded.closed_at ELSE events.closed_at END,
            observation_snapshot_json = CASE WHEN excluded.observation_snapshot_json IS NOT NULL THEN excluded.observation_snapshot_json ELSE events.observation_snapshot_json END,
            evidence_summary_json = CASE WHEN excluded.evidence_summary_json IS NOT NULL THEN excluded.evidence_summary_json ELSE events.evidence_summary_json END,
            updated_at = excluded.updated_at;
        """
        now_iso = datetime.now().isoformat()
        with self.db.transaction() as cur:
            cur.execute(
                sql,
                (
                    event.event_id,
                    event.session_id,
                    event.camera_id,
                    event.track_id,
                    event.seat_id,
                    event.event_type,
                    event.opened_at,
                    event.closed_at,
                    event.duration_sec,
                    event.severity,
                    event.score,
                    event.lifecycle_status,
                    event.review_status,
                    event.source_origin,
                    event.observation_snapshot_json,
                    event.evidence_summary_json,
                    event.created_at,
                    now_iso,
                ),
            )

    def update_review_status(self, event_id: str, new_status: str) -> bool:
        sql = "UPDATE events SET review_status = ?, updated_at = ? WHERE event_id = ?;"
        now_iso = datetime.now().isoformat()
        with self.db.transaction() as cur:
            cur.execute(sql, (new_status, now_iso, event_id))
            return cur.rowcount > 0

    def get_event(self, event_id: str) -> Optional[PersistedEvent]:
        sql = "SELECT * FROM events WHERE event_id = ?;"
        with self.db.cursor() as cur:
            cur.execute(sql, (event_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_event(row)

    def list_events_by_session(
        self,
        session_id: str,
        severity: Optional[str] = None,
        review_status: Optional[str] = None,
        limit: int = 500,
    ) -> List[PersistedEvent]:
        conditions = ["session_id = ?"]
        params: List[Any] = [session_id]

        if severity:
            conditions.append("severity = ?")
            params.append(severity.upper())
        if review_status:
            conditions.append("review_status = ?")
            params.append(review_status.lower())

        sql = f"""
        SELECT * FROM events
        WHERE {' AND '.join(conditions)}
        ORDER BY
            CASE severity WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 ELSE 3 END ASC,
            opened_at DESC
        LIMIT ?;
        """
        params.append(limit)

        events = []
        with self.db.cursor() as cur:
            cur.execute(sql, tuple(params))
            for row in cur.fetchall():
                events.append(self._row_to_event(row))
        return events

    def list_recent_events(self, limit: int = 100) -> List[PersistedEvent]:
        sql = """
        SELECT * FROM events
        ORDER BY opened_at DESC
        LIMIT ?;
        """
        events = []
        with self.db.cursor() as cur:
            cur.execute(sql, (limit,))
            for row in cur.fetchall():
                events.append(self._row_to_event(row))
        return events

    def _row_to_event(self, row: Any) -> PersistedEvent:
        return PersistedEvent(
            event_id=row["event_id"],
            session_id=row["session_id"],
            camera_id=row["camera_id"],
            track_id=row["track_id"],
            seat_id=row["seat_id"],
            event_type=row["event_type"],
            opened_at=row["opened_at"],
            closed_at=row["closed_at"],
            duration_sec=row["duration_sec"],
            severity=row["severity"],
            score=row["score"],
            lifecycle_status=row["lifecycle_status"],
            review_status=row["review_status"],
            source_origin=row["source_origin"],
            observation_snapshot_json=row["observation_snapshot_json"],
            evidence_summary_json=row["evidence_summary_json"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
