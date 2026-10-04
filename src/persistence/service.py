"""
High-level Persistence Service for ExamGuard.
Glues DatabaseManager, migrations, repositories, session lifecycles, and audit logging.
"""

from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Optional, List, Dict, Any, Tuple
import uuid

from src.persistence.database import DatabaseManager, DEFAULT_DB_PATH
from src.persistence.migrations import run_migrations
from src.persistence.models import (
    ExamSession,
    PersistedEvent,
    EventEvidence,
    ReviewRecord,
    AuditLogEntry,
)
from src.persistence.repositories import (
    SessionRepository,
    EventRepository,
    EvidenceRepository,
    ReviewRepository,
    AuditRepository,
)

logger = logging.getLogger(__name__)


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 checksum of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class PersistenceService:
    """Central service managing database persistence, session recovery, event storage, and audits."""

    _instance: Optional["PersistenceService"] = None

    @classmethod
    def get_instance(cls, db_path: Optional[str] = None) -> "PersistenceService":
        if cls._instance is None:
            cls._instance = cls(db_path=db_path)
        return cls._instance

    def __init__(self, db_path: Optional[str] = None):
        self.db = DatabaseManager(db_path=db_path)
        # Run migrations on initialization
        run_migrations(self.db)

        self.sessions = SessionRepository(self.db)
        self.events = EventRepository(self.db)
        self.evidence = EvidenceRepository(self.db)
        self.reviews = ReviewRepository(self.db)
        self.audit = AuditRepository(self.db)

        self.active_session: Optional[ExamSession] = None
        self._last_event_update_times: Dict[str, float] = {}

    def initialize_runtime_session(
        self,
        name: Optional[str] = None,
        room: str = "Phòng thi chính",
        invigilator_name: Optional[str] = None,
    ) -> ExamSession:
        """
        Startup sequence:
        1. Recover any stale ACTIVE session left by previous abnormal exit -> mark INTERRUPTED.
        2. Create a fresh ACTIVE session for this process run.
        """
        # 1. Stale session detection
        existing_active = self.sessions.get_active_session()
        if existing_active:
            ended_ts = existing_active.last_heartbeat_at or datetime.now().isoformat()
            self.sessions.mark_interrupted(existing_active.session_id, ended_at=ended_ts)
            self.audit.log_action(
                audit_id=f"aud_{uuid.uuid4().hex[:12]}",
                action="SESSION_INTERRUPTED",
                session_id=existing_active.session_id,
                details={
                    "reason": "UNEXPECTED_TERMINATION",
                    "recovered_at": datetime.now().isoformat(),
                    "last_heartbeat": existing_active.last_heartbeat_at,
                },
            )
            logger.warning(
                f"Previous stale session '{existing_active.session_id}' marked as INTERRUPTED (ended: {ended_ts})."
            )

        # 2. Create fresh active session with formatted timestamp
        now = datetime.now()
        session_id = f"sess_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        session_name = name or f"Phiên giám sát {now.strftime('%d/%m/%Y %H:%M')}"
        now_iso = now.isoformat()

        session = ExamSession(
            session_id=session_id,
            name=session_name,
            room=room,
            invigilator_name=invigilator_name,
            started_at=now_iso,
            ended_at=None,
            status="ACTIVE",
            camera_count=1,
            created_at=now_iso,
            updated_at=now_iso,
            last_heartbeat_at=now_iso,
            close_reason=None,
        )
        self.sessions.create_session(session)
        self.active_session = session

        self.audit.log_action(
            audit_id=f"aud_{uuid.uuid4().hex[:12]}",
            action="SESSION_CREATED",
            session_id=session_id,
            details={"name": session_name, "room": room},
        )
        logger.info(f"Active monitoring session initialized: '{session_id}' ({session_name}).")
        return session

    def heartbeat(self) -> None:
        """Periodic heartbeat for active session."""
        if self.active_session:
            self.sessions.update_heartbeat(self.active_session.session_id)

    def close_active_session(self, reason: str = "GRACEFUL_STOP") -> Optional[ExamSession]:
        """Gracefully close the currently active monitoring session."""
        if not self.active_session:
            return None

        sid = self.active_session.session_id
        now_iso = datetime.now().isoformat()
        self.sessions.close_session(sid, ended_at=now_iso, reason=reason)
        self.audit.log_action(
            audit_id=f"aud_{uuid.uuid4().hex[:12]}",
            action="SESSION_CLOSED",
            session_id=sid,
            details={"reason": reason, "ended_at": now_iso},
        )
        closed_session = self.sessions.get_session(sid)
        self.active_session = None
        logger.info(f"Session '{sid}' gracefully closed (reason: {reason}).")
        return closed_session

    def update_session_metadata(
        self,
        session_id: str,
        name: Optional[str] = None,
        room: Optional[str] = None,
        invigilator_name: Optional[str] = None,
    ) -> bool:
        """Edit session information."""
        success = self.sessions.update_metadata(
            session_id=session_id,
            name=name,
            room=room,
            invigilator_name=invigilator_name,
        )
        if success:
            if self.active_session and self.active_session.session_id == session_id:
                if name:
                    self.active_session.name = name
                if room:
                    self.active_session.room = room
                if invigilator_name:
                    self.active_session.invigilator_name = invigilator_name

            self.audit.log_action(
                audit_id=f"aud_{uuid.uuid4().hex[:12]}",
                action="SESSION_METADATA_UPDATED",
                session_id=session_id,
                details={"name": name, "room": room, "invigilator": invigilator_name},
            )
        return success

    # --- Event Persistence Handlers ---

    def persist_stage2_event(self, event: Any, action: str) -> None:
        """Synchronize Stage 2 FusedEvent lifecycle transition with SQLite."""
        if not hasattr(event, "event_id"):
            return

        sid = self.active_session.session_id if self.active_session else "sess_default"
        eid = event.event_id

        # Throttle UPDATE events to avoid SQLite thrashing (max once every 1.5s per event)
        now_ts = time.time()
        if action == "UPDATE":
            last_ts = self._last_event_update_times.get(eid, 0.0)
            if now_ts - last_ts < 1.5:
                return
            self._last_event_update_times[eid] = now_ts
        elif action == "CLOSE":
            self._last_event_update_times.pop(eid, None)

        st_sec = getattr(event, "start_timestamp", now_ts)
        opened_iso = datetime.fromtimestamp(st_sec).isoformat()
        closed_iso = datetime.fromtimestamp(event.end_timestamp).isoformat() if getattr(event, "end_timestamp", None) else None

        obs_json = json.dumps(getattr(event, "observation_snapshot", {}), ensure_ascii=False)
        ev_sum_json = json.dumps(getattr(event, "evidence_summary", {}), ensure_ascii=False)

        # Check existing review status if already adjudicated
        existing_ev = self.events.get_event(eid)
        review_st = existing_ev.review_status if existing_ev else getattr(event, "review_status", "awaiting")
        created_at_iso = existing_ev.created_at if existing_ev else opened_iso

        persisted = PersistedEvent(
            event_id=eid,
            session_id=sid,
            camera_id=getattr(event, "camera_id", "laptop_webcam_0"),
            track_id=getattr(event, "track_id", None),
            seat_id=None,
            event_type=getattr(event, "event_type", "OBSERVABLE_EVENT"),
            opened_at=opened_iso,
            closed_at=closed_iso,
            duration_sec=float(getattr(event, "duration", 0.0)),
            severity=getattr(event, "risk_level", "LOW"),
            score=float(getattr(event, "risk_score", 0.0)),
            lifecycle_status="closed" if action == "CLOSE" else getattr(event, "lifecycle_status", "active"),
            review_status=review_st,
            source_origin=getattr(event, "event_origin", "UNKNOWN"),
            observation_snapshot_json=obs_json,
            evidence_summary_json=ev_sum_json,
            created_at=created_at_iso,
            updated_at=datetime.now().isoformat(),
        )
        self.events.upsert_event(persisted)

        if action == "OPEN":
            self.audit.log_action(
                audit_id=f"aud_{uuid.uuid4().hex[:12]}",
                action="EVENT_OPENED",
                session_id=sid,
                event_id=eid,
                details={"event_type": persisted.event_type, "severity": persisted.severity, "score": persisted.score},
            )
        elif action == "CLOSE":
            self.audit.log_action(
                audit_id=f"aud_{uuid.uuid4().hex[:12]}",
                action="EVENT_CLOSED",
                session_id=sid,
                event_id=eid,
                details={"duration_sec": persisted.duration_sec, "severity": persisted.severity},
            )

    def record_evidence_file(
        self,
        event_id: str,
        evidence_type: str,
        file_path: str,
        mime_type: Optional[str] = None,
        captured_at: Optional[str] = None,
        clip_start: Optional[str] = None,
        clip_end: Optional[str] = None,
    ) -> Optional[EventEvidence]:
        """Record evidence file metadata and calculate SHA-256."""
        if not os.path.exists(file_path):
            logger.warning(f"Cannot record evidence: file '{file_path}' does not exist.")
            return None

        # Convert to relative path from repo root
        repo_root = Path(__file__).resolve().parent.parent.parent
        try:
            rel_path = str(Path(file_path).resolve().relative_to(repo_root)).replace("\\", "/")
        except ValueError:
            rel_path = str(file_path).replace("\\", "/")

        size_bytes = os.path.getsize(file_path)
        sha256 = compute_file_sha256(file_path)

        if not mime_type:
            ext = Path(file_path).suffix.lower()
            mime_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".mp4": "video/mp4",
                ".json": "application/json",
            }
            mime_type = mime_map.get(ext, "application/octet-stream")

        evidence_id = f"evd_{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now().isoformat()

        record = EventEvidence(
            evidence_id=evidence_id,
            event_id=event_id,
            evidence_type=evidence_type.upper(),
            relative_path=rel_path,
            mime_type=mime_type,
            sha256=sha256,
            size_bytes=size_bytes,
            captured_at=captured_at or now_iso,
            clip_start_at=clip_start,
            clip_end_at=clip_end,
            created_at=now_iso,
        )
        self.evidence.add_evidence(record)

        sid = self.active_session.session_id if self.active_session else None
        action_name = "EVIDENCE_SNAPSHOT_WRITTEN" if evidence_type == "SNAPSHOT" else "EVIDENCE_CLIP_WRITTEN"
        self.audit.log_action(
            audit_id=f"aud_{uuid.uuid4().hex[:12]}",
            action=action_name,
            session_id=sid,
            event_id=event_id,
            details={"path": rel_path, "sha256": sha256, "size": size_bytes},
        )
        return record

    def record_review_decision(
        self,
        event_id: str,
        decision: str,
        note: Optional[str] = None,
        reviewer_id: Optional[str] = None,
        reviewer_name: Optional[str] = "Giám thị phòng thi",
    ) -> bool:
        """
        Record a human invigilator review decision (CONFIRMED or DISMISSED).
        Appends to reviews history, updates current event review_status, and audits.
        """
        norm_decision = decision.upper()
        if norm_decision not in ["CONFIRMED", "DISMISSED"]:
            raise ValueError(f"Invalid review decision: {decision}")

        review_id = f"rev_{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now().isoformat()

        # 1. Append review record (historical decision retained)
        rev = ReviewRecord(
            review_id=review_id,
            event_id=event_id,
            decision=norm_decision,
            reviewer_id=reviewer_id,
            reviewer_name=reviewer_name,
            note=note,
            reviewed_at=now_iso,
            created_at=now_iso,
        )
        self.reviews.add_review(rev)

        # 2. Update status on event
        new_status = norm_decision.lower()
        self.events.update_review_status(event_id, new_status)

        # 3. Audit log
        action = "EVENT_CONFIRMED" if norm_decision == "CONFIRMED" else "EVENT_DISMISSED"
        sid = self.active_session.session_id if self.active_session else None
        self.audit.log_action(
            audit_id=f"aud_{uuid.uuid4().hex[:12]}",
            action=action,
            session_id=sid,
            event_id=event_id,
            actor_type="INVIGILATOR",
            actor_id=reviewer_id,
            details={"decision": norm_decision, "note": note, "reviewer": reviewer_name},
        )
        return True

    def verify_evidence_integrity(self, evidence_id: str) -> Dict[str, Any]:
        """Verify an evidence file on disk against its database SHA-256."""
        ev = self.evidence.get_evidence_by_id(evidence_id)
        if not ev:
            return {"status": "NOT_FOUND", "valid": False, "message": "Bản ghi bằng chứng không tồn tại trong CSDL."}

        repo_root = Path(__file__).resolve().parent.parent.parent
        full_path = (repo_root / ev.relative_path).resolve()

        if not full_path.is_file():
            return {
                "status": "FILE_MISSING",
                "valid": False,
                "message": "Tệp bằng chứng không còn tồn tại trên ổ đĩa.",
                "relative_path": ev.relative_path,
            }

        curr_sha256 = compute_file_sha256(str(full_path))
        if curr_sha256.lower() != ev.sha256.lower():
            return {
                "status": "HASH_MISMATCH",
                "valid": False,
                "message": "Bằng chứng đã thay đổi so với thời điểm ghi nhận (Mã băm không khớp).",
                "expected_sha256": ev.sha256,
                "actual_sha256": curr_sha256,
            }

        return {
            "status": "VALID",
            "valid": True,
            "message": "Bằng chứng đã xác minh — Toàn vẹn hợp lệ.",
            "sha256": ev.sha256,
            "relative_path": ev.relative_path,
        }
