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
    status: str = "ACTIVE"  # ACTIVE, CLOSED, INTERRUPTED
    camera_count: int = 1
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
    lifecycle_status: str  # active, closed
    review_status: str  # awaiting, confirmed, dismissed
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
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

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
    actor_type: str = "SYSTEM"  # SYSTEM, INVIGILATOR, OPERATOR
    actor_id: Optional[str] = None
    details_json: Optional[str] = None
    previous_entry_hash: Optional[str] = None
    entry_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
