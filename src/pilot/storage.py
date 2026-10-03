"""
Pilot Storage & Evidence Governance Module
==========================================
Implements Part I requirements:
- Privacy-first evidence retention and storage quota safety
- Non-crashing graceful degradation under disk pressure
- Comprehensive evidence manifests with model provenance and operator review status
- Automatic retention policy enforcement (honoring review protection)
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import logging
import os
import shutil
import time
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class RecordingMode(str, Enum):
    DISABLED = "DISABLED"
    EVENT_SNAPSHOT_ONLY = "EVENT_SNAPSHOT_ONLY"
    EVENT_CLIP = "EVENT_CLIP"
    EVENT_CLIP_WITH_PREBUFFER = "EVENT_CLIP_WITH_PREBUFFER"


class StorageQuotaStatus(str, Enum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EXHAUSTED = "EXHAUSTED"


class ReviewStatus(str, Enum):
    NEW = "NEW"
    REVIEWED = "REVIEWED"
    CONFIRMED_EVENT = "CONFIRMED_EVENT"
    DISMISSED = "DISMISSED"


@dataclass
class EvidenceManifest:
    """Audit manifest for an observable event evidence record."""
    event_id: str
    camera_id: str
    track_id: int
    room_id: str
    event_type: str
    risk_level: str
    risk_score: float
    start_timestamp_sec: float
    end_timestamp_sec: float
    duration_sec: float

    # Model hashes & configurations for strict reproducibility
    model_hashes: Dict[str, str]
    fusion_config_version: str
    camera_profile_version: str

    # Artifact references (relative or absolute)
    open_snapshot_path: Optional[str] = None
    escalation_snapshot_path: Optional[str] = None
    clip_path: Optional[str] = None
    metadata_path: Optional[str] = None

    # Review status
    operator_review_status: ReviewStatus = ReviewStatus.NEW
    reviewer_notes: Optional[str] = None
    reviewed_at: Optional[float] = None

    # Safe audit representation (never prints biometric or personal identity)
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "room_id": self.room_id,
            "event_type": self.event_type,
            "risk_level": self.risk_level,
            "risk_score": round(self.risk_score, 2),
            "start_timestamp_sec": round(self.start_timestamp_sec, 3),
            "end_timestamp_sec": round(self.end_timestamp_sec, 3),
            "duration_sec": round(self.duration_sec, 3),
            "model_hashes": self.model_hashes,
            "fusion_config_version": self.fusion_config_version,
            "camera_profile_version": self.camera_profile_version,
            "open_snapshot_path": self.open_snapshot_path,
            "escalation_snapshot_path": self.escalation_snapshot_path,
            "clip_path": self.clip_path,
            "metadata_path": self.metadata_path,
            "operator_review_status": self.operator_review_status.value,
            "reviewer_notes": self.reviewer_notes,
            "reviewed_at": self.reviewed_at,
        }


class EvidenceRetentionManager:
    """
    Manages evidence directory quotas, retention lifecycles, and graceful degradation.
    Ensures the pipeline NEVER crashes if storage is full.
    """

    def __init__(
        self,
        base_dir: str = "storage/evidence",
        retention_days: int = 7,
        max_storage_gb: float = 50.0,
        min_free_disk_gb: float = 5.0,
        warning_threshold_pct: float = 80.0,
        critical_threshold_pct: float = 95.0,
        preserve_unreviewed: bool = True,
        auto_delete_expired: bool = True,
    ):
        self.base_dir = base_dir
        self.retention_days = retention_days
        self.max_storage_gb = max_storage_gb
        self.min_free_disk_gb = min_free_disk_gb
        self.warning_threshold_pct = warning_threshold_pct
        self.critical_threshold_pct = critical_threshold_pct
        self.preserve_unreviewed = preserve_unreviewed
        self.auto_delete_expired = auto_delete_expired

        self.snapshots_dir = os.path.join(base_dir, "snapshots")
        self.clips_dir = os.path.join(base_dir, "clips")
        self.metadata_dir = os.path.join(base_dir, "metadata")
        self.manifests_dir = os.path.join(base_dir, "manifests")

        for d in [self.snapshots_dir, self.clips_dir, self.metadata_dir, self.manifests_dir]:
            os.makedirs(d, exist_ok=True)

        self._manifests: Dict[str, EvidenceManifest] = {}
        self._load_existing_manifests()

    def _load_existing_manifests(self) -> None:
        """Scan existing manifests on startup."""
        if not os.path.exists(self.manifests_dir):
            return
        for fname in os.listdir(self.manifests_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(self.manifests_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    rev_status = ReviewStatus(data.get("operator_review_status", "NEW"))
                    manifest = EvidenceManifest(
                        event_id=data["event_id"],
                        camera_id=data["camera_id"],
                        track_id=int(data["track_id"]),
                        room_id=data.get("room_id", "unknown"),
                        event_type=data["event_type"],
                        risk_level=data["risk_level"],
                        risk_score=float(data["risk_score"]),
                        start_timestamp_sec=float(data["start_timestamp_sec"]),
                        end_timestamp_sec=float(data["end_timestamp_sec"]),
                        duration_sec=float(data["duration_sec"]),
                        model_hashes=data.get("model_hashes", {}),
                        fusion_config_version=data.get("fusion_config_version", "1.0.0"),
                        camera_profile_version=data.get("camera_profile_version", "1.0.0"),
                        open_snapshot_path=data.get("open_snapshot_path"),
                        escalation_snapshot_path=data.get("escalation_snapshot_path"),
                        clip_path=data.get("clip_path"),
                        metadata_path=data.get("metadata_path"),
                        operator_review_status=rev_status,
                        reviewer_notes=data.get("reviewer_notes"),
                        reviewed_at=data.get("reviewed_at"),
                    )
                    self._manifests[manifest.event_id] = manifest
                except Exception as e:
                    logger.debug(f"Could not load manifest {fpath}: {e}")

    def check_storage_status(self) -> Tuple[StorageQuotaStatus, Dict[str, Any]]:
        """
        Check physical disk usage and configured directory quota.
        Returns status (HEALTHY, WARNING, CRITICAL, EXHAUSTED) and telemetry.
        """
        # 1. Check directory size
        total_dir_bytes = 0
        try:
            for root, _, files in os.walk(self.base_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        total_dir_bytes += os.path.getsize(fp)
                    except OSError:
                        pass
        except Exception:
            pass

        used_gb = total_dir_bytes / (1024.0 ** 3)
        quota_pct = (used_gb / max(0.1, self.max_storage_gb)) * 100.0

        # 2. Check host free disk space
        try:
            total, used, free = shutil.disk_usage(self.base_dir)
            free_disk_gb = free / (1024.0 ** 3)
        except Exception:
            free_disk_gb = 100.0 # Fallback

        if free_disk_gb < self.min_free_disk_gb:
            status = StorageQuotaStatus.EXHAUSTED
        elif quota_pct >= self.critical_threshold_pct:
            status = StorageQuotaStatus.CRITICAL
        elif quota_pct >= self.warning_threshold_pct:
            status = StorageQuotaStatus.WARNING
        else:
            status = StorageQuotaStatus.HEALTHY

        telemetry = {
            "evidence_dir": self.base_dir,
            "used_gb": round(used_gb, 3),
            "max_storage_gb": self.max_storage_gb,
            "quota_usage_pct": round(quota_pct, 1),
            "free_disk_gb": round(free_disk_gb, 2),
            "min_free_disk_gb": self.min_free_disk_gb,
            "status": status.value,
            "allow_media_recording": status in [StorageQuotaStatus.HEALTHY, StorageQuotaStatus.WARNING],
            "allow_metadata_recording": status != StorageQuotaStatus.EXHAUSTED,
        }
        return status, telemetry

    def save_manifest(self, manifest: EvidenceManifest) -> str:
        """Persist evidence manifest to disk."""
        self._manifests[manifest.event_id] = manifest
        fpath = os.path.join(self.manifests_dir, f"{manifest.event_id}_manifest.json")
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(manifest.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write manifest {fpath}: {e}")
        return fpath

    def update_review_status(
        self,
        event_id: str,
        new_status: ReviewStatus,
        reviewer_notes: Optional[str] = None,
    ) -> bool:
        """Update operator review state on manifest."""
        manifest = self._manifests.get(event_id)
        if not manifest:
            return False
        manifest.operator_review_status = new_status
        manifest.reviewed_at = time.time()
        if reviewer_notes is not None:
            manifest.reviewer_notes = reviewer_notes
        self.save_manifest(manifest)
        return True

    def enforce_retention(self) -> Dict[str, Any]:
        """
        Enforce retention window.
        Deletes media files for events older than retention_days, unless preserve_unreviewed is true
        and event status is NEW.
        """
        if not self.auto_delete_expired:
            return {"deleted_events": 0, "freed_bytes": 0}

        now = time.time()
        cutoff_sec = now - (self.retention_days * 86400.0)
        deleted_count = 0
        freed_bytes = 0

        for ev_id, manifest in list(self._manifests.items()):
            if manifest.end_timestamp_sec < cutoff_sec:
                # Check protection of unreviewed events
                if self.preserve_unreviewed and manifest.operator_review_status == ReviewStatus.NEW:
                    logger.info(f"Skipping expired event {ev_id} because it is unreviewed.")
                    continue

                # Delete snapshots & clip
                for p in [manifest.open_snapshot_path, manifest.escalation_snapshot_path, manifest.clip_path]:
                    if p and os.path.exists(p):
                        try:
                            sz = os.path.getsize(p)
                            os.remove(p)
                            freed_bytes += sz
                        except OSError as e:
                            logger.warning(f"Error removing {p}: {e}")

                deleted_count += 1

        return {
            "deleted_events": deleted_count,
            "freed_bytes": freed_bytes,
            "freed_mb": round(freed_bytes / (1024.0 ** 2), 2),
        }
