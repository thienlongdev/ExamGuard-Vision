"""
Domain models and dataclasses for ExamGuard persistence.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List


@dataclass
class ExamSession:
    session_id: str
    name: str
    room: str = "Phòng thi chính"
    invigilator_name: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ended_at: Optional[str] = None
    status: str = "ACTIVE"  # DRAFT, ACTIVE, CLOSING, CLOSED, INTERRUPTED
    camera_count: int = 1
    class_name: Optional[str] = None
    subject_code: Optional[str] = None
    notes: Optional[str] = None
    evidence_failure_count: int = 0
    summary_json: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_heartbeat_at: Optional[str] = None
    close_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PersistedEvent:
    event_id: str
    session_id: str
    camera_id: str
    track_id: Optional[int]
    event_type: str
    opened_at: str
    severity: str
    score: float
    lifecycle_status: str = "active"  # active, closed
    review_status: str = "awaiting"  # awaiting, confirmed, dismissed
    source_origin: str = "UNKNOWN"
    seat_id: Optional[str] = None
    closed_at: Optional[str] = None
    duration_sec: float = 0.0
    observation_snapshot_json: Optional[str] = None
    evidence_summary_json: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EventEvidence:
    evidence_id: str
    event_id: str
    evidence_type: str  # SNAPSHOT, VIDEO_CLIP, MANIFEST
    relative_path: str
    mime_type: str
    sha256: str
    size_bytes: int
    captured_at: Optional[str] = None
    clip_start_at: Optional[str] = None
    clip_end_at: Optional[str] = None
    encryption_state: str = "LEGACY_PLAINTEXT"  # LEGACY_PLAINTEXT, ENCRYPTED_V1
    key_id: Optional[str] = None
    aad_json: Optional[str] = None
    artifact_state: str = "READY"  # PENDING, READY, FAILED, MISSING_LEGACY
    codec: Optional[str] = None
    container: Optional[str] = None
    error_message: Optional[str] = None
    duration_sec: Optional[float] = None
    frame_count: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def file_path(self) -> str:
        return self.relative_path

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewRecord:
    review_id: str
    event_id: str
    decision: str  # CONFIRMED, DISMISSED
    reviewer_id: Optional[str] = None
    reviewer_name: Optional[str] = "Giám thị phòng thi"
    note: Optional[str] = None
    reviewed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditLogEntry:
    audit_id: str
    action: str
    session_id: Optional[str] = None
    event_id: Optional[str] = None
    actor_type: str = "SYSTEM"  # SYSTEM, USER
    actor_id: Optional[str] = None
    actor_display_name: Optional[str] = None
    details_json: Optional[str] = None
    previous_entry_hash: Optional[str] = None
    entry_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CameraConfig:
    camera_id: str
    name: str
    source_type: str = "webcam"  # webcam, rtsp, video_file
    device_index: Optional[int] = 0
    source_uri_ref: Optional[str] = None
    enabled: int = 1
    resolution_width: Optional[int] = 1280
    resolution_height: Optional[int] = 720
    target_capture_fps: Optional[float] = 30.0
    room: Optional[str] = "Phòng thi chính"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionCamera:
    session_id: str
    camera_id: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ended_at: Optional[str] = None
    status: str = "ACTIVE"  # ACTIVE, ENDED, INTERRUPTED

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
