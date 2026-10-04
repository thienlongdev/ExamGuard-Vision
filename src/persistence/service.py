"""
High-level Persistence Service for ExamGuard.
Glues DatabaseManager, migrations, repositories, session lifecycles, user authentication,
evidence encryption at rest, and audit logging.
"""

from datetime import datetime, timedelta
import hashlib
import json
import logging
import os
from pathlib import Path
import secrets
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
    CameraConfig,
    SessionCamera,
)
from src.persistence.repositories import (
    SessionRepository,
    EventRepository,
    EvidenceRepository,
    ReviewRepository,
    AuditRepository,
    UserRepository,
    CameraRepository,
)
from src.security.models import User, AuthSession, Role, Permission
from src.security.password import (
    hash_password,
    verify_password,
    validate_password_policy,
)
from src.security.key_provider import get_key_provider
from src.security.crypto import (
    encrypt_evidence_bytes,
    decrypt_evidence_bytes,
    canonicalize_aad,
)

logger = logging.getLogger(__name__)


class AmbiguousEvidenceError(Exception):
    """Raised when an un-scoped filename matches multiple evidence artifacts."""
    pass


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
        # Run migrations on initialization to ensure schema v2 is applied
        run_migrations(self.db)

        self.sessions = SessionRepository(self.db)
        self.events = EventRepository(self.db)
        self.evidence = EvidenceRepository(self.db)
        self.reviews = ReviewRepository(self.db)
        self.audit = AuditRepository(self.db)
        self.users = UserRepository(self.db)
        self.cameras = CameraRepository(self.db)

        self.active_session: Optional[ExamSession] = None
        self._last_event_update_times: Dict[str, float] = {}

    def get_current_monitoring_session(self) -> Optional[ExamSession]:
        """Return currently active monitoring session, reconnecting to DB state if needed."""
        if self.active_session and self.active_session.status == "ACTIVE":
            return self.active_session
        active = self.sessions.get_active_session()
        if active and active.status == "ACTIVE":
            self.active_session = active
            return active
        return None

    def start_monitoring_session(
        self,
        name: Optional[str] = None,
        room: str = "Phòng thi chính",
        class_name: Optional[str] = None,
        subject_code: Optional[str] = None,
        invigilator_name: Optional[str] = None,
        notes: Optional[str] = None,
        camera_ids: Optional[List[str]] = None,
        actor_id: Optional[str] = None,
    ) -> ExamSession:
        """
        Explicitly begin a new monitoring session for an exam room.
        If an active session already exists, it is closed gracefully first.
        """
        if self.active_session and self.active_session.status == "ACTIVE":
            logger.info(f"Closing previous active session '{self.active_session.session_id}' before starting new session.")
            self.end_monitoring_session(
                session_id=self.active_session.session_id,
                reason="SWITCHED_TO_NEW_SESSION",
                actor_id=actor_id,
            )

        now = datetime.now()
        session_id = f"sess_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        session_name = name or f"Phiên giám sát {now.strftime('%d/%m/%Y %H:%M')}"
        now_iso = now.isoformat()

        cam_count = len(camera_ids) if camera_ids else 1
        session = ExamSession(
            session_id=session_id,
            name=session_name,
            room=room,
            class_name=class_name,
            subject_code=subject_code,
            invigilator_name=invigilator_name,
            notes=notes,
            started_at=now_iso,
            ended_at=None,
            status="ACTIVE",
            camera_count=cam_count,
            created_at=now_iso,
            updated_at=now_iso,
            last_heartbeat_at=now_iso,
            close_reason=None,
            evidence_failure_count=0,
            summary_json=None,
        )
        self.sessions.create_session(session)
        self.active_session = session

        # Associate cameras if provided
        if camera_ids:
            for cid in camera_ids:
                try:
                    self.sessions.add_camera_to_session(session_id, cid)
                except Exception as e:
                    logger.debug(f"Could not bind camera {cid} to session {session_id}: {e}")

        self.audit.log_action(
            audit_id=f"aud_{uuid.uuid4().hex[:12]}",
            action="SESSION_CREATED",
            session_id=session_id,
            actor_type="USER" if actor_id else "SYSTEM",
            actor_id=actor_id,
            details={
                "name": session_name,
                "room": room,
                "class_name": class_name,
                "subject_code": subject_code,
                "invigilator": invigilator_name,
            },
        )
        logger.info(f"Active monitoring session started: '{session_id}' ({session_name}, Room: {room}).")
        return session

    def end_monitoring_session(
        self,
        session_id: Optional[str] = None,
        reason: str = "GRACEFUL_STOP",
        summary: Optional[Dict[str, Any]] = None,
        evidence_failure_count: int = 0,
        actor_id: Optional[str] = None,
    ) -> Optional[ExamSession]:
        """
        Explicitly close an active monitoring session, persisting end timestamp,
        summary statistics, and failure counts.
        """
        sid = session_id or (self.active_session.session_id if self.active_session else None)
        if not sid:
            return None

        # Transition through CLOSING state if updating active session
        now_iso = datetime.now().isoformat()
        summary_str = json.dumps(summary, ensure_ascii=False) if summary else None

        self.sessions.close_session(
            session_id=sid,
            ended_at=now_iso,
            reason=reason,
            summary_json=summary_str,
            evidence_failure_count=evidence_failure_count,
        )

        self.audit.log_action(
            audit_id=f"aud_{uuid.uuid4().hex[:12]}",
            action="SESSION_CLOSED",
            session_id=sid,
            actor_type="USER" if actor_id else "SYSTEM",
            actor_id=actor_id,
            details={
                "reason": reason,
                "ended_at": now_iso,
                "evidence_failure_count": evidence_failure_count,
            },
        )

        closed_session = self.sessions.get_session(sid)
        if self.active_session and self.active_session.session_id == sid:
            self.active_session = None
        logger.info(f"Monitoring session '{sid}' closed (reason: {reason}, failures: {evidence_failure_count}).")
        return closed_session

    def initialize_runtime_session(
        self,
        name: Optional[str] = None,
        room: str = "Phòng thi chính",
        invigilator_name: Optional[str] = None,
    ) -> ExamSession:
        """
        Backwards-compatible runtime session initializer (used in tests or explicit setup).
        1. Stale session detection (>15m without heartbeat or unexpected shutdown).
        2. Creates a fresh active session.
        """
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

        # Cleanup any stale backup temp artifacts
        try:
            from src.persistence.backup import cleanup_stale_backup_temp_artifacts
            cleanup_stale_backup_temp_artifacts(persistence_service=self)
        except Exception as e:
            logger.warning(f"Could not clean stale backup temp artifacts: {e}")

        return self.start_monitoring_session(
            name=name,
            room=room,
            invigilator_name=invigilator_name,
        )

    def heartbeat(self) -> None:
        """Periodic heartbeat for active session."""
        if self.active_session:
            self.sessions.update_heartbeat(self.active_session.session_id)

    def close_active_session(self, reason: str = "GRACEFUL_STOP") -> Optional[ExamSession]:
        """Gracefully close the currently active monitoring session."""
        return self.end_monitoring_session(reason=reason)

    def update_session_metadata(
        self,
        session_id: str,
        name: Optional[str] = None,
        room: Optional[str] = None,
        invigilator_name: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> bool:
        """Edit session information with optional actor attribution."""
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
                actor_type="USER" if actor_id else "SYSTEM",
                actor_id=actor_id,
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

        ev_type = getattr(event, "event_type", "OBSERVABLE_EVENT")
        existing_ev = self.events.get_event(eid)
        if existing_ev and existing_ev.review_status in ("confirmed", "dismissed"):
            review_st = existing_ev.review_status
        elif ev_type in ("PHONE_VISUAL_CANDIDATE", "MULTI_CUE_ATTENTION_SHIFT") or getattr(event, "review_status", "awaiting") == "internal":
            review_st = "internal"
        else:
            review_st = getattr(event, "review_status", "awaiting")
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
        encrypt: bool = True,
        artifact_state: str = "READY",
        codec: Optional[str] = None,
        container: Optional[str] = None,
        error_message: Optional[str] = None,
        duration_sec: float = 0.0,
        frame_count: int = 0,
    ) -> Optional[EventEvidence]:
        """
        Record evidence file metadata, encrypting with AES-256-GCM at rest by default.
        Calculates SHA-256 of the stored file (ciphertext if encrypted).
        Updates evidence repository and event evidence summary.
        """
        if not os.path.exists(file_path):
            logger.warning(f"Cannot record evidence: file '{file_path}' does not exist.")
            return None

        repo_root = Path(__file__).resolve().parent.parent.parent
        sid = self.active_session.session_id if self.active_session else "sess_default"
        evidence_id = f"evd_{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now().isoformat()

        if not mime_type:
            ext = Path(file_path).suffix.lower()
            mime_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".mp4": "video/mp4",
                ".webm": "video/webm",
                ".json": "application/json",
            }
            mime_type = mime_map.get(ext, "application/octet-stream")

        stored_file_path = file_path
        encryption_state = "LEGACY_PLAINTEXT"
        key_id = None
        aad_json = None

        if encrypt:
            try:
                # Read plaintext bytes
                with open(file_path, "rb") as f:
                    raw_data = f.read()

                aad_dict = {
                    "session_id": sid,
                    "event_id": event_id,
                    "evidence_id": evidence_id,
                    "evidence_type": evidence_type.upper(),
                }
                aad_json = json.dumps(aad_dict, ensure_ascii=False)

                kp = get_key_provider()
                kid, key = kp.get_current_key()
                envelope, ct_sha256 = encrypt_evidence_bytes(raw_data, aad_dict, key, kid)

                enc_path = file_path + ".enc"
                with open(enc_path, "wb") as f:
                    f.write(envelope)

                # Safely delete plaintext file
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Could not remove plaintext file {file_path} after encryption: {e}")

                stored_file_path = enc_path
                encryption_state = "ENCRYPTED_V1"
                key_id = kid
                sha256 = ct_sha256
                size_bytes = len(envelope)

            except Exception as e:
                logger.error(f"Evidence encryption failed for {file_path}: {e}. Falling back to plaintext.")
                encryption_state = "LEGACY_PLAINTEXT"
                stored_file_path = file_path
                sha256 = compute_file_sha256(file_path)
                size_bytes = os.path.getsize(file_path)
        else:
            sha256 = compute_file_sha256(file_path)
            size_bytes = os.path.getsize(file_path)

        # Convert stored path to relative from repo root
        try:
            rel_path = str(Path(stored_file_path).resolve().relative_to(repo_root)).replace("\\", "/")
        except ValueError:
            # Outside the repo: keep an absolute path so the artifact stays resolvable regardless of cwd
            rel_path = str(Path(stored_file_path).resolve()).replace("\\", "/")

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
            encryption_state=encryption_state,
            key_id=key_id,
            aad_json=aad_json,
            artifact_state=artifact_state,
            codec=codec,
            container=container,
            error_message=error_message,
            duration_sec=duration_sec if duration_sec > 0 else None,
            frame_count=frame_count if frame_count > 0 else None,
            created_at=now_iso,
        )
        self.evidence.add_evidence(record)

        # Update event evidence summary in sqlite
        summary_update: Dict[str, Any] = {}
        ev_type_upper = evidence_type.upper()
        if ev_type_upper == "SNAPSHOT":
            summary_update["snapshot_path"] = rel_path
            summary_update["snapshot_state"] = artifact_state
            if error_message:
                summary_update["snapshot_error"] = error_message
        elif ev_type_upper in ("CLIP", "VIDEO", "VIDEO_CLIP"):
            summary_update["clip_path"] = rel_path
            summary_update["clip_state"] = artifact_state
            if duration_sec > 0:
                summary_update["duration_sec"] = duration_sec
            if error_message:
                summary_update["clip_error"] = error_message
        elif ev_type_upper == "MANIFEST":
            summary_update["manifest_path"] = rel_path

        if summary_update:
            try:
                self.events.update_evidence_summary(event_id, summary_update)
            except Exception as e:
                logger.debug(f"Could not update event evidence summary: {e}")

        action_name = "EVIDENCE_SNAPSHOT_WRITTEN" if evidence_type == "SNAPSHOT" else "EVIDENCE_CLIP_WRITTEN"
        self.audit.log_action(
            audit_id=f"aud_{uuid.uuid4().hex[:12]}",
            action=action_name,
            session_id=sid,
            event_id=event_id,
            details={
                "path": rel_path,
                "sha256": sha256,
                "size": size_bytes,
                "encryption": encryption_state,
                "state": artifact_state,
            },
        )
        return record

    def load_and_decrypt_evidence(self, identifier_or_path: str) -> Tuple[bytes, str]:
        """
        Load an evidence artifact by evidence_id or relative path, decrypting in-memory if encrypted.
        Guarantees retrieval of exact AAD metadata so AES-GCM decryption never fails on valid artifacts.
        Returns (decrypted_bytes, mime_type).
        """
        # Defense in depth: block attempts to access credentials or security directory
        cleaned_lookup = identifier_or_path.replace("\\", "/").lower()
        if "security" in cleaned_lookup or cleaned_lookup.endswith(".dpapi") or "master-key" in cleaned_lookup:
            raise FileNotFoundError("Security credentials cannot be accessed via evidence loader.")

        repo_root = Path(__file__).resolve().parent.parent.parent
        ev = self.evidence.get_evidence_by_id(identifier_or_path)
        if not ev:
            ev = self.evidence.get_evidence_by_path(identifier_or_path)

        # If not found yet and contains path structure, parse parts to match by event and type
        if not ev:
            norm_parts = [p for p in cleaned_lookup.split("/") if p]
            if len(norm_parts) >= 2:
                last_name = norm_parts[-1]
                ev_type = "SNAPSHOT" if "snapshot" in last_name else ("VIDEO_CLIP" if ("clip" in last_name or "video" in last_name) else None)
                possible_eid = norm_parts[-2]
                if ev_type and possible_eid:
                    ev = self.evidence.get_evidence_for_event_and_type(possible_eid, ev_type)

        if ev:
            rel_path = ev.relative_path
            mime = ev.mime_type
            enc_state = ev.encryption_state
            aad_dict = json.loads(ev.aad_json) if ev.aad_json else {
                "session_id": "",
                "event_id": ev.event_id,
                "evidence_id": ev.evidence_id,
                "evidence_type": ev.evidence_type,
            }
        else:
            rel_path = identifier_or_path
            ext = Path(rel_path).suffix.lower()
            if ext == ".enc":
                real_ext = Path(rel_path[:-4]).suffix.lower()
            else:
                real_ext = ext
            mime_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".mp4": "video/mp4",
                ".webm": "video/webm",
                ".json": "application/json",
            }
            mime = mime_map.get(real_ext, "application/octet-stream")
            enc_state = "ENCRYPTED_V1" if rel_path.endswith(".enc") else "LEGACY_PLAINTEXT"
            aad_dict = None

        candidate_paths = [
            Path(rel_path) if os.path.isabs(rel_path) else (repo_root / rel_path),
            repo_root / "evidence" / rel_path,
            repo_root / "storage" / "evidence" / rel_path,
            repo_root / "storage" / rel_path,
        ]
        full_path = None
        for cand in candidate_paths:
            if cand.is_file():
                full_path = cand
                break
            enc_cand = cand.parent / (cand.name + ".enc")
            if enc_cand.is_file():
                full_path = enc_cand
                enc_state = "ENCRYPTED_V1"
                break

        # Fallback search by filename across evidence folders (with strict ambiguity detection)
        if not full_path or not full_path.is_file():
            fname = Path(rel_path).name
            matches = []
            for base_dir in [repo_root / "evidence", repo_root / "storage" / "evidence", repo_root / "storage" / "sessions"]:
                if base_dir.is_dir():
                    for match in base_dir.rglob(fname):
                        if match.is_file():
                            matches.append(match.resolve())
                    for match in base_dir.rglob(fname + ".enc"):
                        if match.is_file():
                            matches.append(match.resolve())
            unique_matches = list(dict.fromkeys(matches))
            if len(unique_matches) > 1:
                raise AmbiguousEvidenceError(
                    f"Ambiguous evidence reference '{fname}'. Multiple conflicting evidence records exist."
                )
            elif len(unique_matches) == 1:
                full_path = unique_matches[0]
                if full_path.name.endswith(".enc"):
                    enc_state = "ENCRYPTED_V1"

        if not full_path or not full_path.is_file():
            raise FileNotFoundError(f"Tệp bằng chứng không tồn tại: {rel_path}")

        # Post-path-resolution second chance for DB row if ev was not initially found
        if not ev and full_path:
            try:
                rel_cand = str(full_path.resolve().relative_to(repo_root)).replace("\\", "/")
                ev = self.evidence.get_evidence_by_path(rel_cand)
            except Exception:
                pass
            if not ev:
                ev = self.evidence.get_evidence_by_path(str(full_path))
            if ev:
                mime = ev.mime_type or mime
                if ev.aad_json:
                    aad_dict = json.loads(ev.aad_json)

        file_bytes = full_path.read_bytes()

        # Check if file has EGE1 binary envelope
        if file_bytes.startswith(b"EGE1") or enc_state == "ENCRYPTED_V1":
            if not aad_dict:
                # If AAD not pre-known, infer from path components
                path_parts = str(full_path).replace("\\", "/").split("/")
                inferred_sid = ""
                inferred_eid = ""
                if "sessions" in path_parts:
                    idx = path_parts.index("sessions")
                    if len(path_parts) > idx + 1:
                        inferred_sid = path_parts[idx + 1]
                if "evidence" in path_parts:
                    idx = path_parts.index("evidence")
                    if len(path_parts) > idx + 1:
                        inferred_eid = path_parts[idx + 1]

                ev_type = "SNAPSHOT" if "snapshot" in str(full_path).lower() else "VIDEO_CLIP"
                aad_dict = {
                    "session_id": inferred_sid,
                    "event_id": inferred_eid,
                    "evidence_id": "",
                    "evidence_type": ev_type,
                }
            kp = get_key_provider()
            decrypted = decrypt_evidence_bytes(file_bytes, aad_dict, key_provider=kp)
            return decrypted, mime
        else:
            return file_bytes, mime

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
            actor_type="USER" if reviewer_id else "SYSTEM",
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
            "encryption_state": ev.encryption_state,
        }

    # =========================================================================
    # User Authentication & Session Management
    # =========================================================================

    def has_users(self) -> bool:
        """Return True if at least one user exists in system."""
        return self.users.count_users() > 0

    def create_initial_admin(
        self,
        username: str,
        password: str,
        display_name: str,
    ) -> User:
        """First-run setup for initial ADMIN account. Fails if any user already exists."""
        if self.has_users():
            raise ValueError("Tài khoản quản trị đã tồn tại. Không thể tạo thêm qua thiết lập ban đầu.")

        is_valid, err = validate_password_policy(password)
        if not is_valid:
            raise ValueError(err)

        pwd_hash = hash_password(password)
        now_iso = datetime.now().isoformat()
        user_id = f"usr_{uuid.uuid4().hex[:12]}"

        user = User(
            user_id=user_id,
            username=username.strip(),
            password_hash=pwd_hash,
            display_name=display_name.strip(),
            role=Role.ADMIN.value,
            is_active=1,
            must_change_password=0,
            failed_login_count=0,
            created_at=now_iso,
            updated_at=now_iso,
            created_by="FIRST_RUN_SETUP",
        )
        self.users.create_user(user)

        self.audit.log_action(
            audit_id=f"aud_{uuid.uuid4().hex[:12]}",
            action="USER_CREATED",
            actor_type="SYSTEM",
            actor_id=user.user_id,
            details={"username": user.username, "role": user.role, "mode": "FIRST_RUN_ADMIN"},
        )
        logger.info(f"Initial Administrator account created: {user.username} ({user.user_id})")
        return user

    def create_user(
        self,
        username: str,
        password: str,
        display_name: str,
        role: Any = Role.INVIGILATOR,
        created_by: Optional[str] = None,
    ) -> User:
        """Create regular or admin user."""
        is_valid, err = validate_password_policy(password)
        if not is_valid:
            raise ValueError(err)

        if self.users.get_user_by_username(username.strip()):
            raise ValueError(f"Tên đăng nhập '{username}' đã tồn tại.")

        pwd_hash = hash_password(password)
        now_iso = datetime.now().isoformat()
        user_id = f"usr_{uuid.uuid4().hex[:12]}"

        user = User(
            user_id=user_id,
            username=username.strip(),
            password_hash=pwd_hash,
            display_name=display_name.strip(),
            role=role if isinstance(role, Role) else Role(role),
            is_active=True,
            must_change_password=False,
            failed_login_count=0,
            created_at=now_iso,
            updated_at=now_iso,
            created_by=created_by,
        )
        self.users.create_user(user)
        self.audit.log_action(
            audit_id=f"aud_{uuid.uuid4().hex[:12]}",
            action="USER_CREATED",
            actor_type="USER" if created_by else "SYSTEM",
            actor_id=created_by or user_id,
            details={"username": user.username, "role": user.role.value},
        )
        return user

    def update_user(
        self,
        user_id: str,
        role: Optional[Any] = None,
        is_active: Optional[bool] = None,
        display_name: Optional[str] = None,
        admin_user_id: Optional[str] = None,
    ) -> Optional[User]:
        """Update mutable user fields and audit action."""
        user = self.users.get_user_by_id(user_id)
        if not user:
            return None

        if role is not None:
            user.role = role if isinstance(role, Role) else Role(role)
            self.audit.log_action(
                audit_id=f"aud_{uuid.uuid4().hex[:12]}",
                action="USER_ROLE_CHANGED",
                actor_type="USER",
                actor_id=admin_user_id,
                details={"target_user": user.username, "new_role": user.role.value},
            )
        if is_active is not None:
            user.is_active = is_active
            action = "USER_ENABLED" if is_active else "USER_DISABLED"
            self.audit.log_action(
                audit_id=f"aud_{uuid.uuid4().hex[:12]}",
                action=action,
                actor_type="USER",
                actor_id=admin_user_id,
                details={"target_user": user.username},
            )
        if display_name is not None:
            user.display_name = display_name.strip()

        user.updated_at = datetime.now().isoformat()
        self.users.update_user(user)
        return user

    def reset_password(
        self,
        user_id: str,
        new_password: str,
        admin_user_id: Optional[str] = None,
        must_change_password: bool = False,
    ) -> bool:
        """Reset a user's password with validation, lockout clearing, session revocation, and audit."""
        is_valid, err = validate_password_policy(new_password)
        if not is_valid:
            raise ValueError(err)

        user = self.users.get_user_by_id(user_id)
        if not user:
            return False

        pwd_hash = hash_password(new_password)
        self.users.update_password(
            user.user_id,
            pwd_hash,
            must_change_password=1 if must_change_password else 0,
        )
        self.users.revoke_all_user_sessions(user.user_id)

        self.audit.log_action(
            audit_id=f"aud_{uuid.uuid4().hex[:12]}",
            action="PASSWORD_CHANGED",
            actor_type="USER",
            actor_id=admin_user_id or user_id,
            details={"target_user": user.username, "must_change_password": must_change_password},
        )
        return True

    def unlock_user(
        self,
        user_id: str,
        admin_user_id: Optional[str] = None,
    ) -> bool:
        """Clear temporary lockout and reset failed login count for a user."""
        user = self.users.get_user_by_id(user_id)
        if not user:
            return False

        ok = self.users.unlock_user(user.user_id)
        if ok:
            self.audit.log_action(
                audit_id=f"aud_{uuid.uuid4().hex[:12]}",
                action="USER_UNLOCKED",
                actor_type="USER",
                actor_id=admin_user_id or user_id,
                details={"target_user": user.username},
            )
        return ok

    def authenticate_user(
        self,
        username: str,
        password: str,
    ) -> Tuple[Optional[User], Optional[str]]:
        """
        Authenticate credentials.
        Returns (user, None) on success or (None, localized_error) on failure.
        """
        user = self.users.get_user_by_username(username)
        if not user:
            # Constant-time dummy verify to thwart timing side-channels
            _ = verify_password("$argon2id$v=19$m=65536,t=2,p=2$c29tZXNhbHQ$P1d98r8m22lQ", "dummy")
            self.audit.log_action(
                audit_id=f"aud_{uuid.uuid4().hex[:12]}",
                action="LOGIN_FAILED",
                actor_type="USER",
                details={"username": username, "reason": "USER_NOT_FOUND"},
            )
            return None, "Tên đăng nhập hoặc mật khẩu không đúng."

        # Check account lockout
        if user.locked_until:
            try:
                locked_dt = datetime.fromisoformat(user.locked_until)
                remaining = (locked_dt - datetime.now()).total_seconds()
                if remaining > 0:
                    remaining_mins = max(1, int((remaining + 59) // 60))
                    return None, f"USER_LOCKED:{remaining_mins}"
                else:
                    self.users.unlock_user(user.user_id)
            except Exception:
                pass

        if not user.is_active:
            return None, "USER_DISABLED"

        matched = verify_password(user.password_hash, password)
        if not matched:
            count, locked_until = self.users.record_login_failure(user.username, max_attempts=5, lockout_minutes=15)
            action = "LOGIN_LOCKED" if locked_until else "LOGIN_FAILED"
            self.audit.log_action(
                audit_id=f"aud_{uuid.uuid4().hex[:12]}",
                action=action,
                actor_type="USER",
                actor_id=user.user_id,
                details={"username": user.username, "failed_count": count},
            )
            if locked_until:
                return None, "USER_LOCKED:15"
            return None, "INVALID_CREDENTIALS"

        # Success
        self.users.record_login_success(user.user_id)
        self.audit.log_action(
            audit_id=f"aud_{uuid.uuid4().hex[:12]}",
            action="LOGIN_SUCCESS",
            actor_type="USER",
            actor_id=user.user_id,
            details={"username": user.username, "role": user.role},
        )
        return user, None

    def create_session_for_user(
        self,
        user: User,
        user_agent: Optional[str] = None,
        duration_hours: int = 8,
    ) -> Tuple[AuthSession, str]:
        """Create server-side authenticated session and return (session_record, raw_token)."""
        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

        now_dt = datetime.now()
        now_iso = now_dt.isoformat()
        exp_iso = (now_dt + timedelta(hours=duration_hours)).isoformat()
        ua_hash = hashlib.sha256(user_agent.encode("utf-8")).hexdigest()[:16] if user_agent else None

        sess = AuthSession(
            auth_session_id=f"as_{uuid.uuid4().hex[:12]}",
            user_id=user.user_id,
            token_hash=token_hash,
            created_at=now_iso,
            expires_at=exp_iso,
            last_seen_at=now_iso,
            user_agent_hash=ua_hash,
        )
        self.users.create_auth_session(sess)
        return sess, raw_token

    def create_user_session(
        self,
        user_or_id: Any,
        user_agent: Optional[str] = None,
        duration_hours: int = 8,
    ) -> Tuple[AuthSession, str]:
        """Create session for user instance or user_id."""
        if isinstance(user_or_id, str):
            user = self.users.get_user_by_id(user_or_id)
            if not user:
                raise ValueError(f"User '{user_or_id}' not found.")
        else:
            user = user_or_id
        return self.create_session_for_user(user, user_agent=user_agent, duration_hours=duration_hours)

    def validate_session_token(self, raw_token: Optional[str]) -> Tuple[Optional[User], Optional[AuthSession]]:
        """Validate token from cookie, returning (user, session) or (None, None)."""
        if not raw_token:
            return None, None
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        sess = self.users.get_auth_session_by_token_hash(token_hash)
        if not sess or not sess.is_valid:
            return None, None

        user = self.users.get_user_by_id(sess.user_id)
        if not user or not user.is_active:
            return None, None

        self.users.touch_auth_session(sess.auth_session_id)
        return user, sess

    def revoke_session_token(self, raw_token: Optional[str]) -> None:
        """Revoke active session on user logout."""
        if not raw_token:
            return
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        sess = self.users.get_auth_session_by_token_hash(token_hash)
        if sess:
            self.users.revoke_auth_session(sess.auth_session_id)
            self.audit.log_action(
                audit_id=f"aud_{uuid.uuid4().hex[:12]}",
                action="LOGOUT",
                actor_type="USER",
                actor_id=sess.user_id,
                details={"session_id": sess.auth_session_id},
            )
