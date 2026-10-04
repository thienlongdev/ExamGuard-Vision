"""Pydantic schemas for the FastAPI endpoints."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    timestamp: float


class CameraInfo(BaseModel):
    camera_id: str
    name: str
    source_type: str
    configured: bool = True
    device_present: bool = False
    connected: bool = False
    streaming: bool = False
    status: str = "NO_PHYSICAL_CAMERA"
    configured_capture_fps: float = 30.0
    configured_resolution: str = "1280x720"
    observed_capture_fps: Optional[float] = None
    observed_resolution: Optional[str] = None
    # Backward compatibility fields
    is_active: bool = False
    fps: Optional[float] = None
    resolution: str = "1280x720"


class CameraCounts(BaseModel):
    registered: int = 1
    configured: int = 1
    connected: int = 0
    streaming: int = 0


class RuntimeStreamStatus(BaseModel):
    active: bool = False
    source_type: str = "WEBCAM"
    device_present: bool = False


class ConfiguredRates(BaseModel):
    capture_fps: float = 30.0
    inference_fps: float = 12.0


class ObservedRates(BaseModel):
    capture_fps: Optional[float] = None
    processed_fps: Optional[float] = None
    inference_fps: Optional[float] = None


class EvidenceInfo(BaseModel):
    snapshot_path: Optional[str] = None
    clip_path: Optional[str] = None


class EventResponse(BaseModel):
    event_id: str
    track_id: int
    camera_id: str
    timestamp: float
    start_time: float
    end_time: float
    event_type: str
    risk_level: str
    score: float
    evidence: Dict[str, Any]
    snapshot_path: Optional[str] = None
    clip_path: Optional[str] = None
    status: str
    lifecycle_status: Optional[str] = "active"
    review_status: Optional[str] = "awaiting"
    observation_snapshot: Optional[Dict[str, Any]] = None
    reviewer_notes: Optional[str] = None
    configured_evidence_score: Optional[float] = None
    lifecycle_action: Optional[str] = None
    event_origin: Optional[str] = None


class EventStatusUpdateRequest(BaseModel):
    status: str = Field(..., description="Allowed: 'new', 'reviewed', 'confirmed', 'confirmed_event', 'dismissed'")
    reviewer_notes: Optional[str] = None


class SystemStatusResponse(BaseModel):
    # Semantic camera and rate separation
    camera_counts: Optional[CameraCounts] = None
    runtime_stream: Optional[RuntimeStreamStatus] = None
    configured_rates: Optional[ConfiguredRates] = None
    observed_rates: Optional[ObservedRates] = None

    # Backward compatibility fields
    active_cameras: int = 0
    active_students: int = 0
    total_events: int = 0
    new_events: int = 0
    confirmed_events: int = 0
    dismissed_events: int = 0
    effective_fps: Optional[float] = None  # None when active_streams == 0
    inference_fps: Optional[float] = None  # None when active_streams == 0
    head_pose_enabled: bool = False
    pipeline_version: Optional[str] = "2.0.0-orchestration"
    model_registry: Optional[Dict[str, Any]] = None
    drop_percentage: Optional[float] = None  # None when no camera frames captured
    queue_depth: Optional[int] = 0
    posture_eligible_count: Optional[int] = 0
    headpose_eligible_count: Optional[int] = 0
    gpu_vram_allocated_mb: Optional[float] = 0.0
    evidence_storage_status: Optional[str] = "HEALTHY"
    operator_warnings: Optional[List[str]] = None


class SessionSummary(BaseModel):
    total_events: int = 0
    high_risk_count: int = 0
    medium_risk_count: int = 0
    low_risk_count: int = 0
    awaiting_count: int = 0
    confirmed_count: int = 0
    dismissed_count: int = 0
    duration_sec: float = 0.0


class SessionResponse(BaseModel):
    session_id: str
    name: str
    room: str
    class_name: Optional[str] = None
    subject_code: Optional[str] = None
    invigilator_name: Optional[str] = None
    notes: Optional[str] = None
    started_at: str
    ended_at: Optional[str] = None
    status: str
    camera_count: int = 1
    created_at: str
    updated_at: str
    last_heartbeat_at: Optional[str] = None
    close_reason: Optional[str] = None
    evidence_failure_count: int = 0
    summary: Optional[SessionSummary] = None


class SessionStartRequest(BaseModel):
    name: Optional[str] = None
    session_name: Optional[str] = None
    room: Optional[str] = None
    room_id: Optional[str] = None
    class_name: Optional[str] = None
    subject_code: Optional[str] = None
    invigilator_name: Optional[str] = None
    notes: Optional[str] = None
    camera_ids: Optional[List[str]] = None

    def get_name(self) -> Optional[str]:
        return self.name or self.session_name

    def get_room(self) -> str:
        return self.room or self.room_id or "Phòng thi chính"


class SessionEndRequest(BaseModel):
    reason: str = "GRACEFUL_STOP"
    notes: Optional[str] = None


class SessionUpdateRequest(BaseModel):
    name: Optional[str] = None
    room: Optional[str] = None
    class_name: Optional[str] = None
    subject_code: Optional[str] = None
    invigilator_name: Optional[str] = None
    notes: Optional[str] = None


class EvidenceVerifyResponse(BaseModel):
    status: str
    valid: bool
    message: str
    sha256: Optional[str] = None
    expected_sha256: Optional[str] = None
    actual_sha256: Optional[str] = None
    relative_path: Optional[str] = None


class BackupResponse(BaseModel):
    success: bool
    backup_id: str
    backup_path: str
    database_sha256: str
    evidence_count: int
    message: str


# --- PRODUCTION FOUNDATION SCHEMAS: AUTH, RBAC, USERS, CAMERAS, SECURITY ---

class LoginRequest(BaseModel):
    username: str
    password: str


class SetupRequest(BaseModel):
    username: str
    password: str
    display_name: str


class UserResponse(BaseModel):
    user_id: str
    username: str
    display_name: str
    role: str
    role_display: str
    is_active: bool
    last_login_at: Optional[str] = None
    created_at: str


class AuthMeResponse(BaseModel):
    user: UserResponse
    csrf_token: str
    role_display: str


class UserCreateRequest(BaseModel):
    username: str
    password: str
    display_name: str
    role: str = "INVIGILATOR"


class UserUpdateRequest(BaseModel):
    role: Optional[str] = None
    is_active: Optional[bool] = None
    display_name: Optional[str] = None


class UserResetPasswordRequest(BaseModel):
    new_password: str


class CameraConfigItem(BaseModel):
    camera_id: str
    name: str
    source_type: str
    device_index: Optional[int] = None
    enabled: bool = True
    resolution_width: int = 1280
    resolution_height: int = 720
    target_capture_fps: float = 30.0
    room: Optional[str] = None


class CameraConfigUpdate(BaseModel):
    name: Optional[str] = None
    enabled: Optional[bool] = None
    device_index: Optional[int] = None
    resolution_width: Optional[int] = None
    resolution_height: Optional[int] = None
    target_capture_fps: Optional[float] = None
    room: Optional[str] = None


class BackupCreateRequest(BaseModel):
    recovery_passphrase: Optional[str] = None


class BackupVerifyRequest(BaseModel):
    backup_id_or_path: str
    recovery_passphrase: Optional[str] = None


class SecurityStatusResponse(BaseModel):
    auth_status: str
    evidence_encryption: str
    key_storage: str
    audit_chain_valid: bool
    total_users: int
    active_sessions: int


