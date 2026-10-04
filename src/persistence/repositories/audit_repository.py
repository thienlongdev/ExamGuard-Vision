"""
Repository for audit_logs table operations with tamper-evident SHA-256 hash chaining.
"""

from datetime import datetime
import hashlib
import json
import logging
from typing import Optional, List, Tuple, Any
from src.persistence.database import DatabaseManager
from src.persistence.models import AuditLogEntry

logger = logging.getLogger(__name__)


def compute_entry_hash(
    audit_id: str,
    action: str,
    created_at: str,
    session_id: Optional[str] = None,
    event_id: Optional[str] = None,
    actor_type: str = "SYSTEM",
    actor_id: Optional[str] = None,
    details_json: Optional[str] = None,
    previous_entry_hash: Optional[str] = None,
) -> str:
    canonical = (
        f"{previous_entry_hash or ''}|{audit_id}|{session_id or ''}|{event_id or ''}|"
        f"{actor_type}|{actor_id or ''}|{action}|{created_at}|{details_json or ''}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuditRepository:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def log_action(
        self,
        audit_id: str,
        action: str,
        session_id: Optional[str] = None,
        event_id: Optional[str] = None,
        actor_type: str = "SYSTEM",
        actor_id: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> AuditLogEntry:
        """Append an audit log record, calculating previous hash and entry hash."""
        now_iso = datetime.now().isoformat()
        details_str = json.dumps(details, ensure_ascii=False) if details else None

        sql_last = "SELECT entry_hash FROM audit_logs ORDER BY rowid DESC LIMIT 1;"
        sql_insert = """
        INSERT INTO audit_logs (
            audit_id, session_id, event_id, actor_type, actor_id,
            action, details_json, previous_entry_hash, entry_hash, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        with self.db.transaction() as cur:
            cur.execute(sql_last)
            last_row = cur.fetchone()
            prev_hash = last_row["entry_hash"] if last_row and last_row["entry_hash"] else None

            curr_hash = compute_entry_hash(
                audit_id=audit_id,
                action=action,
                created_at=now_iso,
                session_id=session_id,
                event_id=event_id,
                actor_type=actor_type,
                actor_id=actor_id,
                details_json=details_str,
                previous_entry_hash=prev_hash,
            )

            cur.execute(
                sql_insert,
                (
                    audit_id,
                    session_id,
                    event_id,
                    actor_type,
                    actor_id,
                    action,
                    details_str,
                    prev_hash,
                    curr_hash,
                    now_iso,
                ),
            )

        return AuditLogEntry(
            audit_id=audit_id,
            session_id=session_id,
            event_id=event_id,
            actor_type=actor_type,
            actor_id=actor_id,
            action=action,
            details_json=details_str,
            previous_entry_hash=prev_hash,
            entry_hash=curr_hash,
            created_at=now_iso,
        )

    def list_logs_for_session(self, session_id: str, limit: int = 200) -> List[AuditLogEntry]:
        sql = """
        SELECT * FROM audit_logs
        WHERE session_id = ?
        ORDER BY rowid ASC
        LIMIT ?;
        """
        results = []
        with self.db.cursor() as cur:
            cur.execute(sql, (session_id, limit))
            for row in cur.fetchall():
                results.append(self._row_to_entry(row))
        return results

    def list_all_logs(self, limit: int = 500) -> List[AuditLogEntry]:
        sql = "SELECT * FROM audit_logs ORDER BY rowid ASC LIMIT ?;"
        results = []
        with self.db.cursor() as cur:
            cur.execute(sql, (limit,))
            for row in cur.fetchall():
                results.append(self._row_to_entry(row))
        return results

    def verify_audit_chain(self) -> Tuple[bool, Optional[str]]:
        """Verify the integrity of the audit hash chain from the very first entry."""
        sql = "SELECT * FROM audit_logs ORDER BY rowid ASC;"
        with self.db.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

        expected_prev_hash: Optional[str] = None
        for i, row in enumerate(rows):
            stored_prev = row["previous_entry_hash"]
            stored_curr = row["entry_hash"]

            if stored_prev != expected_prev_hash:
                return (
                    False,
                    f"Chain break at record #{row['audit_id']}: expected prev {expected_prev_hash}, found {stored_prev}",
                )

            recalc = compute_entry_hash(
                audit_id=row["audit_id"],
                action=row["action"],
                created_at=row["created_at"],
                session_id=row["session_id"],
                event_id=row["event_id"],
                actor_type=row["actor_type"],
                actor_id=row["actor_id"],
                details_json=row["details_json"],
                previous_entry_hash=stored_prev,
            )
            if recalc != stored_curr:
                return (
                    False,
                    f"Hash mismatch at record #{row['audit_id']}: recalculated {recalc}, stored {stored_curr}",
                )

            expected_prev_hash = stored_curr

        return True, None

    def _row_to_entry(self, row: Any) -> AuditLogEntry:
        return AuditLogEntry(
            audit_id=row["audit_id"],
            session_id=row["session_id"],
            event_id=row["event_id"],
            actor_type=row["actor_type"],
            actor_id=row["actor_id"],
            action=row["action"],
            details_json=row["details_json"],
            previous_entry_hash=row["previous_entry_hash"],
            entry_hash=row["entry_hash"],
            created_at=row["created_at"],
        )
