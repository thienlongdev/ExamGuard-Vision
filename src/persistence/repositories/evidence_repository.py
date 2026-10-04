"""
Repository for event_evidence table operations.
"""

import logging
from typing import Optional, List, Any
from src.persistence.database import DatabaseManager
from src.persistence.models import EventEvidence

logger = logging.getLogger(__name__)


class EvidenceRepository:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def add_evidence(self, ev: EventEvidence) -> None:
        sql = """
        INSERT INTO event_evidence (
            evidence_id, event_id, evidence_type, relative_path,
            mime_type, sha256, size_bytes, captured_at, clip_start_at,
            clip_end_at, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(evidence_id) DO UPDATE SET
            relative_path = excluded.relative_path,
            sha256 = excluded.sha256,
            size_bytes = excluded.size_bytes;
        """
        with self.db.transaction() as cur:
            cur.execute(
                sql,
                (
                    ev.evidence_id,
                    ev.event_id,
                    ev.evidence_type,
                    ev.relative_path,
                    ev.mime_type,
                    ev.sha256,
                    ev.size_bytes,
                    ev.captured_at,
                    ev.clip_start_at,
                    ev.clip_end_at,
                    ev.created_at,
                ),
            )

    def get_evidence_by_id(self, evidence_id: str) -> Optional[EventEvidence]:
        sql = "SELECT * FROM event_evidence WHERE evidence_id = ?;"
        with self.db.cursor() as cur:
            cur.execute(sql, (evidence_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_evidence(row)

    def list_evidence_for_event(self, event_id: str) -> List[EventEvidence]:
        sql = "SELECT * FROM event_evidence WHERE event_id = ? ORDER BY created_at ASC;"
        results = []
        with self.db.cursor() as cur:
            cur.execute(sql, (event_id,))
            for row in cur.fetchall():
                results.append(self._row_to_evidence(row))
        return results

    def list_all_evidence(self, limit: int = 2000) -> List[EventEvidence]:
        sql = "SELECT * FROM event_evidence ORDER BY created_at DESC LIMIT ?;"
        results = []
        with self.db.cursor() as cur:
            cur.execute(sql, (limit,))
            for row in cur.fetchall():
                results.append(self._row_to_evidence(row))
        return results

    def _row_to_evidence(self, row: Any) -> EventEvidence:
        return EventEvidence(
            evidence_id=row["evidence_id"],
            event_id=row["event_id"],
            evidence_type=row["evidence_type"],
            relative_path=row["relative_path"],
            mime_type=row["mime_type"],
            sha256=row["sha256"],
            size_bytes=row["size_bytes"],
            captured_at=row["captured_at"],
            clip_start_at=row["clip_start_at"],
            clip_end_at=row["clip_end_at"],
            created_at=row["created_at"],
        )
