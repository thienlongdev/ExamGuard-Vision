"""
Stage 2 Integrated Evidence Manager
===================================
Coordinates bounded JPEG snapshots, MP4 rolling clip recordings, and integrity manifests
for FusedEvent lifecycle transitions (OPEN, UPDATE, CLOSE).
Enforces:
- Privacy: Anonymous track IDs and event IDs only (no facial recognition, no names)
- Rate-limiting: Strictly bounded snapshots per event (max 3)
- Disk safety: Bounded directories, atomic asynchronous writes, graceful failure handling
- SHA-256 calculation and manifest generation
- Integration with PersistenceService
"""

from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Any, List
import numpy as np

from src.fusion.types import FusedEvent
from src.evidence.snapshot import SnapshotCapture
from src.evidence.clip_recorder import RollingClipRecorder
from src.persistence.service import PersistenceService

logger = logging.getLogger(__name__)


class IntegratedEvidenceManager:
    """Manages evidence capture lifecycle tied to V4D FusedEvents with session isolation and SHA-256 manifests."""

    def __init__(
        self,
        output_dir: str = "storage/evidence",
        max_snapshots_per_event: int = 3,
        pre_event_seconds: float = 4.0,
        post_event_seconds: float = 4.0,
        enabled: bool = True,
        persistence_service: Optional[PersistenceService] = None,
    ):
        self.output_dir = output_dir
        self.max_snapshots_per_event = max_snapshots_per_event
        self.enabled = enabled
        self.persistence_service = persistence_service or PersistenceService.get_instance()

        self.snapshots_dir = os.path.join(output_dir, "snapshots")
        self.clips_dir = os.path.join(output_dir, "clips")
        self.metadata_dir = os.path.join(output_dir, "metadata")

        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(self.clips_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        self.snapshot_capture = (
            SnapshotCapture(
                output_dir=self.snapshots_dir,
                on_snapshot_ready=self._on_snapshot_ready,
            )
            if enabled
            else None
        )
        self.clip_recorder = (
            RollingClipRecorder(
                output_dir=self.clips_dir,
                pre_event_seconds=pre_event_seconds,
                post_event_seconds=post_event_seconds,
                on_clip_ready=self._on_clip_ready,
                on_clip_failed=self._on_clip_failed,
            )
            if enabled
            else None
        )

        # Tracking snapshot counts and active paths per event_id
        self._event_snapshot_counts: Dict[str, int] = {}
        self._event_clip_paths: Dict[str, str] = {}
        self._event_hashes: Dict[str, Dict[str, str]] = {}

    def _get_event_dir(self, event_id: str) -> str:
        """Resolve session-isolated evidence folder for an event."""
        sid = (
            self.persistence_service.active_session.session_id
            if self.persistence_service and self.persistence_service.active_session
            else "session_default"
        )
        edir = os.path.join("storage", "sessions", sid, "evidence", event_id)
        os.makedirs(edir, exist_ok=True)
        return edir

    def _on_snapshot_ready(self, event_id: str, filepath: str, sha256: str, size: int) -> None:
        """Callback when a snapshot has been successfully written to disk."""
        if event_id not in self._event_hashes:
            self._event_hashes[event_id] = {}
        self._event_hashes[event_id]["snapshot_sha256"] = sha256
        self._event_hashes[event_id]["snapshot_path"] = filepath

        if self.persistence_service:
            try:
                rec = self.persistence_service.record_evidence_file(
                    event_id=event_id,
                    evidence_type="SNAPSHOT",
                    file_path=filepath,
                )
                if rec and rec.file_path:
                    self._event_hashes[event_id]["snapshot_path"] = rec.file_path
            except Exception as e:
                logger.debug(f"Could not record snapshot evidence in DB: {e}")

    def _on_clip_ready(self, event_id: str, filepath: str, sha256: str, size: int) -> None:
        """Callback when an MP4 clip has been successfully encoded to disk."""
        if event_id not in self._event_hashes:
            self._event_hashes[event_id] = {}
        self._event_hashes[event_id]["clip_sha256"] = sha256
        self._event_hashes[event_id]["clip_path"] = filepath

        if self.persistence_service:
            try:
                rec = self.persistence_service.record_evidence_file(
                    event_id=event_id,
                    evidence_type="VIDEO_CLIP",
                    file_path=filepath,
                )
                if rec and rec.file_path:
                    self._event_hashes[event_id]["clip_path"] = rec.file_path
            except Exception as e:
                logger.debug(f"Could not record video clip evidence in DB: {e}")

    def _on_clip_failed(self, event_id: str, filepath: str, err: Exception) -> None:
        """Callback when MP4 clip encoding fails (pipeline continues safely)."""
        logger.warning(f"Video clip encoding failed for {event_id}: {err}")
        if self.persistence_service:
            try:
                sid = (
                    self.persistence_service.active_session.session_id
                    if self.persistence_service.active_session
                    else None
                )
                self.persistence_service.audit.log_action(
                    audit_id=f"aud_fail_{event_id[:8]}",
                    action="EVIDENCE_WRITE_FAILED",
                    session_id=sid,
                    event_id=event_id,
                    details={"filepath": filepath, "error": str(err)},
                )
            except Exception:
                pass

    def push_frame(self, frame: np.ndarray, timestamp_sec: float) -> None:
        """Feed every incoming video frame to rolling clip buffer."""
        if self.enabled and self.clip_recorder is not None:
            self.clip_recorder.push_frame(frame, timestamp_sec)

    def handle_event_lifecycle(
        self,
        event: FusedEvent,
        action: str,
        frame: np.ndarray,
        timestamp_sec: float,
        fps: float = 30.0,
    ) -> None:
        """Handle lifecycle transitions (OPEN, UPDATE, CLOSE)."""
        if not self.enabled:
            return

        ev_id = event.event_id
        ev_dir = self._get_event_dir(ev_id)

        # 1. OPEN Action
        if action == "OPEN":
            count = self._event_snapshot_counts.get(ev_id, 0)
            if count < self.max_snapshots_per_event and self.snapshot_capture is not None:
                try:
                    custom_snap = os.path.join(ev_dir, "snapshot.jpg")
                    snap_path = self.snapshot_capture.capture_async(
                        frame, ev_id, suffix="open", custom_filepath=custom_snap
                    )
                    event.evidence_summary["open_snapshot_path"] = snap_path
                    event.evidence_summary["snapshot_path"] = snap_path
                    self._event_snapshot_counts[ev_id] = count + 1
                except Exception as e:
                    logger.warning(f"Snapshot capture failed for {ev_id}: {e}")

            if self.clip_recorder is not None:
                try:
                    custom_clip = os.path.join(ev_dir, "clip.mp4")
                    clip_path = self.clip_recorder.trigger_clip(
                        ev_id, timestamp_sec, fps=fps, custom_filepath=custom_clip
                    )
                    self._event_clip_paths[ev_id] = clip_path
                    event.evidence_summary["clip_path"] = clip_path
                except Exception as e:
                    logger.warning(f"Clip trigger failed for {ev_id}: {e}")

        # 2. UPDATE Action (capture escalation snapshot if newly HIGH risk)
        elif action == "UPDATE":
            count = self._event_snapshot_counts.get(ev_id, 0)
            if event.risk_level == "HIGH" and count < self.max_snapshots_per_event and self.snapshot_capture is not None:
                if "escalation_snapshot_path" not in event.evidence_summary:
                    try:
                        custom_esc = os.path.join(ev_dir, "snapshot_escalation.jpg")
                        snap_path = self.snapshot_capture.capture_async(
                            frame, ev_id, suffix="escalation", custom_filepath=custom_esc
                        )
                        event.evidence_summary["escalation_snapshot_path"] = snap_path
                        self._event_snapshot_counts[ev_id] = count + 1
                    except Exception as e:
                        logger.warning(f"Escalation snapshot capture failed for {ev_id}: {e}")

        # 3. CLOSE Action
        elif action == "CLOSE":
            # Generate and save event evidence manifest
            try:
                manifest_path = os.path.join(ev_dir, "manifest.json")
                hashes = self._event_hashes.get(ev_id, {})

                # Retrieve model provenance if available
                model_meta = None
                try:
                    from src.orchestration.model_registry import ModelRegistry
                    reg = ModelRegistry.get_instance()
                    model_meta = reg.get_metadata()
                except Exception:
                    pass

                sid = (
                    self.persistence_service.active_session.session_id
                    if self.persistence_service and self.persistence_service.active_session
                    else "session_default"
                )

                st_iso = datetime.fromtimestamp(event.start_timestamp).isoformat() if event.start_timestamp else None
                et_iso = datetime.fromtimestamp(event.end_timestamp).isoformat() if event.end_timestamp else None

                manifest_data = {
                    "event_id": ev_id,
                    "session_id": sid,
                    "camera_id": getattr(event, "camera_id", "cam_0"),
                    "track_id": getattr(event, "track_id", None),
                    "event_type": getattr(event, "event_type", "OBSERVABLE_EVENT"),
                    "opened_at": st_iso,
                    "closed_at": et_iso,
                    "duration_sec": float(getattr(event, "duration", 0.0)),
                    "severity": getattr(event, "risk_level", "LOW"),
                    "score": float(getattr(event, "risk_score", 0.0)),
                    "source_origin": getattr(event, "event_origin", "UNKNOWN"),
                    "snapshot_path": event.evidence_summary.get("snapshot_path") or event.evidence_summary.get("open_snapshot_path"),
                    "snapshot_sha256": hashes.get("snapshot_sha256"),
                    "clip_path": event.evidence_summary.get("clip_path"),
                    "clip_sha256": hashes.get("clip_sha256"),
                    "model_provenance": model_meta,
                    "created_at": datetime.now().isoformat(),
                }

                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest_data, f, indent=2, ensure_ascii=False)

                event.evidence_summary["manifest_path"] = manifest_path
                event.evidence_summary["metadata_path"] = manifest_path

                if self.persistence_service:
                    rec = self.persistence_service.record_evidence_file(
                        event_id=ev_id,
                        evidence_type="MANIFEST",
                        file_path=manifest_path,
                    )
                    if rec and rec.file_path:
                        event.evidence_summary["manifest_path"] = rec.file_path
                        event.evidence_summary["metadata_path"] = rec.file_path
            except Exception as e:
                logger.warning(f"Failed to write event evidence manifest for {ev_id}: {e}")

            # Cleanup tracking counters to prevent dictionary leak
            self._event_snapshot_counts.pop(ev_id, None)
            self._event_clip_paths.pop(ev_id, None)
            self._event_hashes.pop(ev_id, None)

    def reset(self) -> None:
        """Reset internal tracking and in-memory clip buffer."""
        self._event_snapshot_counts.clear()
        self._event_clip_paths.clear()
        self._event_hashes.clear()
        if self.clip_recorder is not None:
            self.clip_recorder._rolling_buffer.clear()
            self.clip_recorder._current_buffer_bytes = 0

    def shutdown(self) -> None:
        """Flush workers and close handlers."""
        if self.snapshot_capture is not None:
            self.snapshot_capture.shutdown()
        if self.clip_recorder is not None:
            self.clip_recorder.shutdown()
