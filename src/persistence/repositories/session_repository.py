"""
Repository for exam_sessions table operations.
"""

from datetime import datetime
import logging
from typing import Optional, List, Dict, Any
from src.persistence.database import DatabaseManager
from src.persistence.models import ExamSession

logger = logging.getLogger(__name__)


class SessionRepository:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def create_session(self, session: ExamSession) -> ExamSession:
        sql = """
        INSERT INTO exam_sessions (
            session_id, name, room, invigilator_name, started_at, ended_at,
            status, camera_count, created_at, updated_at, last_heartbeat_at, close_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
            return ExamSession(
                session_id=row["session_id"],
                name=row["name"],
                room=row["room"],
                invigilator_name=row["invigilator_name"],
                started_at=row["started_at"],
                ended_at=row["ended_at"],
                status=row["status"],
                camera_count=row["camera_count"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                last_heartbeat_at=row["last_heartbeat_at"],
                close_reason=row["close_reason"],
            )

    def get_active_session(self) -> Optional[ExamSession]:
        sql = "SELECT * FROM exam_sessions WHERE status = 'ACTIVE' ORDER BY started_at DESC LIMIT 1;"
        with self.db.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
            if not row:
                return None
            return ExamSession(
                session_id=row["session_id"],
                name=row["name"],
                room=row["room"],
                invigilator_name=row["invigilator_name"],
                started_at=row["started_at"],
                ended_at=row["ended_at"],
                status=row["status"],
                camera_count=row["camera_count"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                last_heartbeat_at=row["last_heartbeat_at"],
                close_reason=row["close_reason"],
            )

    def update_heartbeat(self, session_id: str, ts: Optional[str] = None) -> None:
        now_iso = ts or datetime.now().isoformat()
        sql = "UPDATE exam_sessions SET last_heartbeat_at = ?, updated_at = ? WHERE session_id = ?;"
        with self.db.transaction() as cur:
            cur.execute(sql, (now_iso, now_iso, session_id))

    def update_metadata(
        self,
        session_id: str,
        name: Optional[str] = None,
        room: Optional[str] = None,
        invigilator_name: Optional[str] = None,
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

        if not updates:
            return True

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(session_id)

        sql = f"UPDATE exam_sessions SET {', '.join(updates)} WHERE session_id = ?;"
        with self.db.transaction() as cur:
            cur.execute(sql, tuple(params))
            return cur.rowcount > 0

    def close_session(self, session_id: str, ended_at: Optional[str] = None, reason: str = "GRACEFUL_STOP") -> bool:
        now_iso = ended_at or datetime.now().isoformat()
        sql = """
        UPDATE exam_sessions
        SET status = 'CLOSED', ended_at = ?, close_reason = ?, updated_at = ?
        WHERE session_id = ? AND status = 'ACTIVE';
        """
        with self.db.transaction() as cur:
            cur.execute(sql, (now_iso, reason, now_iso, session_id))
            return cur.rowcount > 0

    def mark_interrupted(self, session_id: str, ended_at: Optional[str] = None) -> bool:
        now_iso = ended_at or datetime.now().isoformat()
        sql = """
        UPDATE exam_sessions
        SET status = 'INTERRUPTED', ended_at = ?, close_reason = 'UNEXPECTED_TERMINATION', updated_at = ?
        WHERE session_id = ? AND status = 'ACTIVE';
        """
        with self.db.transaction() as cur:
            cur.execute(sql, (now_iso, now_iso, session_id))
            return cur.rowcount > 0

    def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0,
        status: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[ExamSession]:
        conditions = []
        params: List[Any] = []
        if status:
            conditions.append("status = ?")
            params.append(status.upper())
        if search:
            conditions.append("(name LIKE ? OR room LIKE ? OR invigilator_name LIKE ?)")
            pattern = f"%{search.strip()}%"
            params.extend([pattern, pattern, pattern])

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
                sessions.append(
                    ExamSession(
                        session_id=row["session_id"],
                        name=row["name"],
                        room=row["room"],
                        invigilator_name=row["invigilator_name"],
                        started_at=row["started_at"],
                        ended_at=row["ended_at"],
                        status=row["status"],
                        camera_count=row["camera_count"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        last_heartbeat_at=row["last_heartbeat_at"],
                        close_reason=row["close_reason"],
                    )
                )
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

            # Calculate session duration
            cur.execute("SELECT started_at, ended_at FROM exam_sessions WHERE session_id = ?;", (session_id,))
            s_row = cur.fetchone()
            if s_row and s_row["started_at"]:
                st = datetime.fromisoformat(s_row["started_at"])
                et = datetime.fromisoformat(s_row["ended_at"]) if s_row["ended_at"] else datetime.now()
                summary["duration_sec"] = max(0.0, (et - st).total_seconds())

        return summary
