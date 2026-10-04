"""
Repository for event_evidence table operations.
Supports plaintext legacy evidence as well as AES-256-GCM encrypted artifacts.
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
            clip_end_at, encryption_state, key_id, aad_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(evidence_id) DO UPDATE SET
            relative_path = excluded.relative_path,
            sha256 = excluded.sha256,
            size_bytes = excluded.size_bytes,
            encryption_state = excluded.encryption_state,
            key_id = excluded.key_id,
            aad_json = excluded.aad_json;
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
                    ev.encryption_state,
                    ev.key_id,
                    ev.aad_json,
                    ev.created_at,
                ),
            )

    def update_encryption_metadata(
        self,
        evidence_id: str,
        relative_path: str,
        sha256: str,
        size_bytes: int,
        encryption_state: str,
        key_id: Optional[str] = None,
        aad_json: Optional[str] = None,
    ) -> bool:
        """Update file path, ciphertext hash, size, and encryption fields after migration."""
        sql = """
        UPDATE event_evidence SET
            relative_path = ?,
            sha256 = ?,
            size_bytes = ?,
            encryption_state = ?,
            key_id = ?,
            aad_json = ?
        WHERE evidence_id = ?;
        """
        with self.db.transaction() as cur:
            cur.execute(
                sql,
                (
                    relative_path,
                    sha256,
                    size_bytes,
                    encryption_state,
                    key_id,
                    aad_json,
                    evidence_id,
                ),
            )
            return cur.rowcount > 0

    def get_evidence_by_id(self, evidence_id: str) -> Optional[EventEvidence]:
        sql = "SELECT * FROM event_evidence WHERE evidence_id = ?;"
        with self.db.cursor() as cur:
            cur.execute(sql, (evidence_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_evidence(row)

    def get_evidence_by_path(self, path_or_rel: str) -> Optional[EventEvidence]:
        """Find evidence row by exact or normalized relative path or filename."""
        from pathlib import Path
        norm = path_or_rel.replace("\\", "/")
        # Try exact, then stripped prefix, then by filename
        sql = "SELECT * FROM event_evidence WHERE relative_path = ? OR relative_path = ?;"
        rel_only = norm
        if "storage/" in norm:
            rel_only = norm[norm.index("storage/"):]
        with self.db.cursor() as cur:
            cur.execute(sql, (norm, rel_only))
            row = cur.fetchone()
            if row:
                return self._row_to_evidence(row)
            # Fallback by filename match if unique
            fname = Path(norm).name
            cur.execute("SELECT * FROM event_evidence WHERE relative_path LIKE ? ORDER BY created_at DESC LIMIT 1;", (f"%/{fname}",))
            row2 = cur.fetchone()
            if row2:
                return self._row_to_evidence(row2)
        return None

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
        keys = row.keys() if hasattr(row, "keys") else []
        enc_state = row["encryption_state"] if "encryption_state" in keys else "LEGACY_PLAINTEXT"
        key_id = row["key_id"] if "key_id" in keys else None
        aad_json = row["aad_json"] if "aad_json" in keys else None

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
            encryption_state=enc_state,
            key_id=key_id,
            aad_json=aad_json,
            created_at=row["created_at"],
        )
