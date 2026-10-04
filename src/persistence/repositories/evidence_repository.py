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
            clip_end_at, encryption_state, key_id, aad_json, created_at,
            artifact_state, codec, container, error_message, duration_sec, frame_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(evidence_id) DO UPDATE SET
            relative_path = excluded.relative_path,
            mime_type = excluded.mime_type,
            sha256 = excluded.sha256,
            size_bytes = excluded.size_bytes,
            encryption_state = excluded.encryption_state,
            key_id = excluded.key_id,
            aad_json = excluded.aad_json,
            artifact_state = excluded.artifact_state,
            codec = excluded.codec,
            container = excluded.container,
            error_message = excluded.error_message,
            duration_sec = excluded.duration_sec,
            frame_count = excluded.frame_count;
        """
        with self.db.transaction() as cur:
            # Check if an existing row for (event_id, evidence_type) exists with a different evidence_id
            cur.execute(
                "SELECT evidence_id FROM event_evidence WHERE event_id = ? AND evidence_type = ? AND evidence_id != ?;",
                (ev.event_id, ev.evidence_type, ev.evidence_id),
            )
            old_rows = cur.fetchall()
            for old in old_rows:
                cur.execute("DELETE FROM event_evidence WHERE evidence_id = ?;", (old["evidence_id"],))

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
                    ev.artifact_state,
                    ev.codec,
                    ev.container,
                    ev.error_message,
                    ev.duration_sec,
                    ev.frame_count,
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

    def get_evidence_for_event_and_type(self, event_id: str, evidence_type: str) -> Optional[EventEvidence]:
        sql = "SELECT * FROM event_evidence WHERE event_id = ? AND evidence_type = ? ORDER BY created_at DESC LIMIT 1;"
        with self.db.cursor() as cur:
            cur.execute(sql, (event_id, evidence_type.upper()))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_evidence(row)

    def get_evidence_by_path(self, path_or_rel: str) -> Optional[EventEvidence]:
        """Find evidence row by exact or normalized relative path, event_id + filename, or filename."""
        from pathlib import Path
        norm = path_or_rel.replace("\\", "/").strip("/")

        # 1. Exact match across common normalized prefixes
        candidates = [norm]
        if not norm.startswith("storage/"):
            candidates.append(f"storage/{norm}")
        if not norm.endswith(".enc"):
            candidates.append(f"{norm}.enc")
            if not norm.startswith("storage/"):
                candidates.append(f"storage/{norm}.enc")
        else:
            unenc = norm[:-4]
            candidates.append(unenc)
            if not unenc.startswith("storage/"):
                candidates.append(f"storage/{unenc}")

        placeholders = ", ".join(["?"] * len(candidates))
        sql = f"SELECT * FROM event_evidence WHERE relative_path IN ({placeholders}) ORDER BY created_at DESC LIMIT 1;"
        with self.db.cursor() as cur:
            cur.execute(sql, tuple(candidates))
            row = cur.fetchone()
            if row:
                return self._row_to_evidence(row)

        # 2. Extract event_id if path has structure like .../evidence/<event_id>/<filename>
        parts = [p for p in norm.split("/") if p]
        if len(parts) >= 2:
            fname = parts[-1]
            base_fname = fname[:-4] if fname.endswith(".enc") else fname
            possible_event_id = parts[-2]
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM event_evidence 
                    WHERE event_id = ? AND (
                        relative_path LIKE ? OR relative_path LIKE ? OR relative_path LIKE ?
                    )
                    ORDER BY created_at DESC LIMIT 1;
                    """,
                    (possible_event_id, f"%/{base_fname}", f"%/{base_fname}.enc", f"%/{fname}")
                )
                row = cur.fetchone()
                if row:
                    return self._row_to_evidence(row)

        # 3. Fallback by filename match if unique
        fname = Path(norm).name
        base_fname = fname[:-4] if fname.endswith(".enc") else fname
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM event_evidence 
                WHERE relative_path LIKE ? OR relative_path LIKE ? 
                ORDER BY created_at DESC LIMIT 1;
                """,
                (f"%/{base_fname}", f"%/{base_fname}.enc")
            )
            row = cur.fetchone()
            if row:
                return self._row_to_evidence(row)

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
        artifact_state = row["artifact_state"] if "artifact_state" in keys else "READY"
        codec = row["codec"] if "codec" in keys else None
        container = row["container"] if "container" in keys else None
        error_message = row["error_message"] if "error_message" in keys else None
        duration_sec = row["duration_sec"] if "duration_sec" in keys else None
        frame_count = row["frame_count"] if "frame_count" in keys else None

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
            artifact_state=artifact_state,
            codec=codec,
            container=container,
            error_message=error_message,
            duration_sec=duration_sec,
            frame_count=frame_count,
            created_at=row["created_at"],
        )
