"""
Repository for exam_sessions table operations.
"""

from datetime import datetime
import logging
import re
import unicodedata
from typing import Optional, List, Dict, Any
from src.persistence.database import DatabaseManager
from src.persistence.models import ExamSession

logger = logging.getLogger(__name__)

# Display-only classification of development sessions; no stored record is altered.
_TEST_NAME_RE = re.compile(r"\bTEST SESSION\b|^TEST\b|\bPYTEST\b")
_VALIDATION_NAME_RE = re.compile(
    r"\bBASELINE\b|\bSOAK\b|\bCARDINALITY\b|\bCERTIFICATION\b|\bSCALE SUITE\b|"
    r"\bBENCHMARK\b|\bVALIDATION\b|\bKIEM THU\b"
)


def classify_session_kind(name: Optional[str]) -> str:
    """Return OPERATIONAL, VALIDATION or TEST for a session name."""
    if not name:
        return "OPERATIONAL"
    folded = unicodedata.normalize("NFD", name.replace("đ", "d").replace("Đ", "D"))
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    folded = re.sub(r"[_\-]+", " ", folded).upper()
    if _TEST_NAME_RE.search(folded):
        return "TEST"
    if _VALIDATION_NAME_RE.search(folded):
        return "VALIDATION"
    return "OPERATIONAL"


class SessionRepository:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def _row_to_session(self, row: Any) -> ExamSession:
        keys = row.keys() if hasattr(row, "keys") else []
        return ExamSession(
            session_id=row["session_id"],
            name=row["name"],
            room=row["room"],
            invigilator_name=row["invigilator_name"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            status=row["status"],
            camera_count=row["camera_count"],
            class_name=row["class_name"] if "class_name" in keys else None,
            subject_code=row["subject_code"] if "subject_code" in keys else None,
            notes=row["notes"] if "notes" in keys else None,
            evidence_failure_count=row["evidence_failure_count"] if "evidence_failure_count" in keys else 0,
            summary_json=row["summary_json"] if "summary_json" in keys else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_heartbeat_at=row["last_heartbeat_at"],
            close_reason=row["close_reason"],
        )

    def create_session(self, session: ExamSession) -> ExamSession:
        sql = """
        INSERT INTO exam_sessions (
            session_id, name, room, invigilator_name, started_at, ended_at,
            status, camera_count, class_name, subject_code, notes,
            evidence_failure_count, summary_json,
            created_at, updated_at, last_heartbeat_at, close_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.db.transaction() as cur:
            cur.execute(
                sql,
                (
                    session.session_id,
                    session.name,
                    session.room,
                    session.invigilator_name,
                    session.started_at,
                    session.ended_at,
                    session.status,
                    session.camera_count,
                    session.class_name,
                    session.subject_code,
                    session.notes,
                    session.evidence_failure_count,
                    session.summary_json,
                    session.created_at,
                    session.updated_at,
                    session.last_heartbeat_at,
                    session.close_reason,
                ),
            )
        return session

    def get_session(self, session_id: str) -> Optional[ExamSession]:
        sql = "SELECT * FROM exam_sessions WHERE session_id = ?;"
        with self.db.cursor() as cur:
            cur.execute(sql, (session_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_session(row)

    def get_active_session(self) -> Optional[ExamSession]:
        sql = "SELECT * FROM exam_sessions WHERE status = 'ACTIVE' ORDER BY started_at DESC LIMIT 1;"
        with self.db.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_session(row)

    def update_heartbeat(self, session_id: str, ts: Optional[str] = None) -> None:
        now_iso = ts or datetime.now().isoformat()
        sql = "UPDATE exam_sessions SET last_heartbeat_at = ?, updated_at = ? WHERE session_id = ?;"
        with self.db.transaction() as cur:
            cur.execute(sql, (now_iso, now_iso, session_id))

    def update_status(self, session_id: str, status: str) -> bool:
        now_iso = datetime.now().isoformat()
        sql = "UPDATE exam_sessions SET status = ?, updated_at = ? WHERE session_id = ?;"
        with self.db.transaction() as cur:
            cur.execute(sql, (status, now_iso, session_id))
            return cur.rowcount > 0

    def update_metadata(
        self,
        session_id: str,
        name: Optional[str] = None,
        room: Optional[str] = None,
        invigilator_name: Optional[str] = None,
        class_name: Optional[str] = None,
        subject_code: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> bool:
        updates = []
        params = []
        if name is not None:
            updates.append("name = ?")
            params.append(name.strip())
        if room is not None:
            updates.append("room = ?")
            params.append(room.strip())
        if invigilator_name is not None:
            updates.append("invigilator_name = ?")
            params.append(invigilator_name.strip())
        if class_name is not None:
            updates.append("class_name = ?")
            params.append(class_name.strip())
        if subject_code is not None:
            updates.append("subject_code = ?")
            params.append(subject_code.strip())
        if notes is not None:
            updates.append("notes = ?")
            params.append(notes.strip())

        if not updates:
            return True

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(session_id)

        sql = f"UPDATE exam_sessions SET {', '.join(updates)} WHERE session_id = ?;"
        with self.db.transaction() as cur:
            cur.execute(sql, tuple(params))
            return cur.rowcount > 0

    def close_session(
        self,
        session_id: str,
        ended_at: Optional[str] = None,
        reason: str = "GRACEFUL_STOP",
        summary_json: Optional[str] = None,
        evidence_failure_count: int = 0,
        failure_count: Optional[int] = None,
    ) -> bool:
        now_iso = ended_at or datetime.now().isoformat()
        fc = failure_count if failure_count is not None else evidence_failure_count
        sql = """
        UPDATE exam_sessions
        SET status = 'CLOSED', ended_at = ?, close_reason = ?, summary_json = ?, evidence_failure_count = ?, updated_at = ?
        WHERE session_id = ?;
        """
        with self.db.transaction() as cur:
            cur.execute(sql, (now_iso, reason, summary_json, fc, now_iso, session_id))
            return cur.rowcount > 0

    def mark_interrupted(
        self, session_id: str, ended_at: Optional[str] = None, reason: str = "UNEXPECTED_TERMINATION"
    ) -> bool:
        now_iso = ended_at or datetime.now().isoformat()
        sql = """
        UPDATE exam_sessions
        SET status = 'INTERRUPTED', ended_at = ?, close_reason = ?, updated_at = ?
        WHERE session_id = ? AND status = 'ACTIVE';
        """
        with self.db.transaction() as cur:
            cur.execute(sql, (now_iso, reason, now_iso, session_id))
            return cur.rowcount > 0

    def mark_start_failed(self, session_id: str, reason: str) -> bool:
        """A session whose camera never opened: explicit terminal state, never a hidden ACTIVE zombie."""
        now_iso = datetime.now().isoformat()
        sql = """
        UPDATE exam_sessions
        SET status = 'START_FAILED', ended_at = ?, close_reason = ?, updated_at = ?
        WHERE session_id = ? AND status = 'ACTIVE';
        """
        with self.db.transaction() as cur:
            cur.execute(sql, (now_iso, reason, now_iso, session_id))
            return cur.rowcount > 0

    def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0,
        status: Optional[str] = None,
        search: Optional[str] = None,
        started_after: Optional[str] = None,
    ) -> List[ExamSession]:
        conditions = []
        params: List[Any] = []
        if status:
            conditions.append("status = ?")
            params.append(status.upper())
        if started_after:
            conditions.append("started_at >= ?")
            params.append(started_after)
        if search:
            conditions.append("(name LIKE ? OR room LIKE ? OR invigilator_name LIKE ? OR class_name LIKE ? OR subject_code LIKE ?)")
            pattern = f"%{search.strip()}%"
            params.extend([pattern, pattern, pattern, pattern, pattern])

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
        SELECT * FROM exam_sessions
        {where_clause}
        ORDER BY started_at DESC
        LIMIT ? OFFSET ?;
        """
        params.extend([limit, offset])

        sessions = []
        with self.db.cursor() as cur:
            cur.execute(sql, tuple(params))
            for row in cur.fetchall():
                sessions.append(self._row_to_session(row))
        return sessions

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Compute aggregate KPIs for a session."""
        summary = {
            "total_events": 0,
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "low_risk_count": 0,
            "awaiting_count": 0,
            "confirmed_count": 0,
            "dismissed_count": 0,
            "evidence_count": 0,
            "evidence_failure_count": 0,
            "duration_sec": 0.0,
        }
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN severity = 'HIGH' THEN 1 ELSE 0 END) as high_cnt,
                    SUM(CASE WHEN severity = 'MEDIUM' THEN 1 ELSE 0 END) as med_cnt,
                    SUM(CASE WHEN severity = 'LOW' THEN 1 ELSE 0 END) as low_cnt,
                    SUM(CASE WHEN review_status = 'awaiting' THEN 1 ELSE 0 END) as await_cnt,
                    SUM(CASE WHEN review_status = 'confirmed' THEN 1 ELSE 0 END) as conf_cnt,
                    SUM(CASE WHEN review_status = 'dismissed' THEN 1 ELSE 0 END) as dism_cnt
                FROM events WHERE session_id = ?;
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if row:
                summary["total_events"] = row["total"] or 0
                summary["high_risk_count"] = row["high_cnt"] or 0
                summary["medium_risk_count"] = row["med_cnt"] or 0
                summary["low_risk_count"] = row["low_cnt"] or 0
                summary["awaiting_count"] = row["await_cnt"] or 0
                summary["confirmed_count"] = row["conf_cnt"] or 0
                summary["dismissed_count"] = row["dism_cnt"] or 0

            # Count evidence artifacts
            cur.execute(
                """
                SELECT 
                    COUNT(*) as ev_cnt,
                    SUM(CASE WHEN artifact_state = 'FAILED' THEN 1 ELSE 0 END) as fail_cnt
                FROM event_evidence ee
                JOIN events ev ON ee.event_id = ev.event_id
                WHERE ev.session_id = ?;
                """,
                (session_id,),
            )
            ev_row = cur.fetchone()
            if ev_row:
                summary["evidence_count"] = ev_row["ev_cnt"] or 0
                summary["evidence_failure_count"] = ev_row["fail_cnt"] or 0

            # Calculate session duration
            cur.execute("SELECT started_at, ended_at, evidence_failure_count FROM exam_sessions WHERE session_id = ?;", (session_id,))
            s_row = cur.fetchone()
            if s_row:
                if s_row["evidence_failure_count"]:
                    summary["evidence_failure_count"] = max(summary["evidence_failure_count"], s_row["evidence_failure_count"])
                if s_row["started_at"]:
                    st = datetime.fromisoformat(s_row["started_at"])
                    et = datetime.fromisoformat(s_row["ended_at"]) if s_row["ended_at"] else datetime.now()
                    summary["duration_sec"] = max(0.0, (et - st).total_seconds())

        return summary
