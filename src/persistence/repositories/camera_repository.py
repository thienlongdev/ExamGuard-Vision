"""
Repository for cameras and session_cameras database operations.
"""

from datetime import datetime
import logging
from typing import Optional, List, Any
from src.persistence.database import DatabaseManager
from src.persistence.models import CameraConfig, SessionCamera

logger = logging.getLogger(__name__)


class CameraRepository:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def get_camera(self, camera_id: str) -> Optional[CameraConfig]:
        """Fetch camera configuration by camera_id."""
        sql = "SELECT * FROM cameras WHERE camera_id = ?;"
        with self.db.cursor() as cur:
            cur.execute(sql, (camera_id,))
            row = cur.fetchone()
            return self._row_to_camera(row) if row else None

    def list_cameras(self) -> List[CameraConfig]:
        """List all configured cameras."""
        sql = "SELECT * FROM cameras ORDER BY camera_id ASC;"
        results = []
        with self.db.cursor() as cur:
            cur.execute(sql)
            for row in cur.fetchall():
                results.append(self._row_to_camera(row))
        return results

    def list_enabled_cameras(self) -> List[CameraConfig]:
        """List enabled cameras only."""
        sql = "SELECT * FROM cameras WHERE enabled = 1 ORDER BY camera_id ASC;"
        results = []
        with self.db.cursor() as cur:
            cur.execute(sql)
            for row in cur.fetchall():
                results.append(self._row_to_camera(row))
        return results

    def add_or_update_camera(self, cam: CameraConfig) -> None:
        """Insert or update camera configuration."""
        now_iso = datetime.now().isoformat()
        sql = """
        INSERT INTO cameras (
            camera_id, name, source_type, device_index, source_uri_ref,
            enabled, resolution_width, resolution_height, target_capture_fps,
            room, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(camera_id) DO UPDATE SET
            name = excluded.name,
            source_type = excluded.source_type,
            device_index = excluded.device_index,
            source_uri_ref = excluded.source_uri_ref,
            enabled = excluded.enabled,
            resolution_width = excluded.resolution_width,
            resolution_height = excluded.resolution_height,
            target_capture_fps = excluded.target_capture_fps,
            room = excluded.room,
            updated_at = excluded.updated_at;
        """
        with self.db.transaction() as cur:
            cur.execute(
                sql,
                (
                    cam.camera_id,
                    cam.name.strip(),
                    cam.source_type.lower().strip(),
                    cam.device_index,
                    cam.source_uri_ref,
                    int(cam.enabled),
                    cam.resolution_width,
                    cam.resolution_height,
                    cam.target_capture_fps,
                    cam.room,
                    cam.created_at,
                    now_iso,
                ),
            )

    upsert_camera = add_or_update_camera

    def delete_camera(self, camera_id: str) -> bool:
        """Remove a camera configuration."""
        sql = "DELETE FROM cameras WHERE camera_id = ?;"
        with self.db.transaction() as cur:
            cur.execute(sql, (camera_id,))
            return cur.rowcount > 0

    # =========================================================================
    # Session Cameras
    # =========================================================================

    def attach_camera_to_session(self, session_id: str, camera_id: str) -> SessionCamera:
        """Record camera association to monitoring session."""
        now_iso = datetime.now().isoformat()
        sql = """
        INSERT INTO session_cameras (session_id, camera_id, started_at, status)
        VALUES (?, ?, ?, 'ACTIVE')
        ON CONFLICT(session_id, camera_id) DO UPDATE SET
            status = 'ACTIVE';
        """
        with self.db.transaction() as cur:
            cur.execute(sql, (session_id, camera_id, now_iso))
        return SessionCamera(session_id=session_id, camera_id=camera_id, started_at=now_iso, status="ACTIVE")

    def detach_camera_from_session(self, session_id: str, camera_id: str, status: str = "ENDED") -> None:
        """Mark camera participation in session as ended or interrupted."""
        now_iso = datetime.now().isoformat()
        sql = "UPDATE session_cameras SET ended_at = ?, status = ? WHERE session_id = ? AND camera_id = ?;"
        with self.db.transaction() as cur:
            cur.execute(sql, (now_iso, status, session_id, camera_id))

    def list_cameras_for_session(self, session_id: str) -> List[SessionCamera]:
        """List all cameras that participated in a monitoring session."""
        sql = "SELECT * FROM session_cameras WHERE session_id = ? ORDER BY started_at ASC;"
        results = []
        with self.db.cursor() as cur:
            cur.execute(sql, (session_id,))
            for row in cur.fetchall():
                results.append(
                    SessionCamera(
                        session_id=row["session_id"],
                        camera_id=row["camera_id"],
                        started_at=row["started_at"],
                        ended_at=row["ended_at"],
                        status=row["status"],
                    )
                )
        return results

    def _row_to_camera(self, row: Any) -> CameraConfig:
        return CameraConfig(
            camera_id=row["camera_id"],
            name=row["name"],
            source_type=row["source_type"],
            device_index=row["device_index"],
            source_uri_ref=row["source_uri_ref"],
            enabled=int(row["enabled"]),
            resolution_width=row["resolution_width"],
            resolution_height=row["resolution_height"],
            target_capture_fps=row["target_capture_fps"],
            room=row["room"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
