"""
V4D Multi-Cue Temporal Fusion - Core Types & Enums
=================================================
Strict Ontology Compliance:
- Models detect observable facts only.
- AI NEVER declares CHEATING, CHEATER, FRAUD, or GUILTY.
- Posture classes remain strictly:
    0: NORMAL_UPRIGHT
    1: NORMAL_READ_WRITE
    2: HEAD_REST_SLEEP
    3: TURN_HEAD_CLEAR
- Risk levels: LOW, MEDIUM, HIGH (degree/persistence/combination of observable suspicious evidence, NOT guilt).
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple


class ObservationStatus(str, Enum):
    """Explicit cue availability status (missing cues are NEVER negative evidence)."""
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_EVALUATED = "NOT_EVALUATED"


class SourceOrigin(str, Enum):
    """Semantic provenance of observations and events (never falsely elevated)."""
    PHYSICAL_LIVE_CAMERA = "PHYSICAL_LIVE_CAMERA"
    SOFTWARE_VALIDATION_FIXTURE = "SOFTWARE_VALIDATION_FIXTURE"
    REPLAY_STREAM = "REPLAY_STREAM"
    VIDEO_FILE = "VIDEO_FILE"
    RTSP_STREAM = "RTSP_STREAM"
    SYNTHETIC_TEST = "SYNTHETIC_TEST"
    UNKNOWN = "UNKNOWN"



class PostureClass(int, Enum):
    """Frozen 4-Class Posture Ontology."""
    NORMAL_UPRIGHT = 0
    NORMAL_READ_WRITE = 1
    HEAD_REST_SLEEP = 2
    TURN_HEAD_CLEAR = 3

    @classmethod
    def from_str(cls, name: str) -> "PostureClass":
        mapping = {
            "NORMAL_UPRIGHT": cls.NORMAL_UPRIGHT,
            "NORMAL_READ_WRITE": cls.NORMAL_READ_WRITE,
            "HEAD_REST_SLEEP": cls.HEAD_REST_SLEEP,
            "TURN_HEAD_CLEAR": cls.TURN_HEAD_CLEAR,
        }
        if name not in mapping:
            raise ValueError(f"Unknown posture class string: {name}")
        return mapping[name]


class EventFamily(str, Enum):
    """Observable fact-based event families."""
    SUSTAINED_HEAD_REST = "SUSTAINED_HEAD_REST"
    SUSTAINED_LATERAL_HEAD_ORIENTATION = "SUSTAINED_LATERAL_HEAD_ORIENTATION"
    PHONE_ASSOCIATED = "PHONE_ASSOCIATED"
    PHONE_VISIBLE_UNASSOCIATED = "PHONE_VISIBLE_UNASSOCIATED"
    PHONE_VISUAL_CANDIDATE = "PHONE_VISUAL_CANDIDATE"
    DISCUSSION_CANDIDATE = "DISCUSSION_CANDIDATE"
    STANDING = "STANDING"
    MULTI_CUE_ATTENTION_SHIFT = "MULTI_CUE_ATTENTION_SHIFT"


class RiskLevel(str, Enum):
    """Risk represents degree/persistence of observable evidence, NOT guilt."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class EventLifecycleState(str, Enum):
    """State machine lifecycle for temporal events."""
    INACTIVE = "INACTIVE"
    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"
    COOLDOWN = "COOLDOWN"


class HeadPoseSupportStatus(str, Enum):
    """Capability status for head-pose measurement."""
    WITHIN_PRIMARY_SUPPORT = "WITHIN_PRIMARY_SUPPORT"    # [-99, +99) HopeNet
    FALLBACK_CIRCULAR = "FALLBACK_CIRCULAR"              # [-180, +180) ResNet18
    OUT_OF_SUPPORT = "OUT_OF_SUPPORT"                    # Out of model domain
    FACE_UNRESOLVABLE = "FACE_UNRESOLVABLE"              # Face too small/occluded


class PhoneAssociationStatus(str, Enum):
    """Spatial association certainty with student track."""
    CLEAR_ASSOCIATION = "CLEAR_ASSOCIATION"
    AMBIGUOUS_ASSOCIATION = "AMBIGUOUS_ASSOCIATION"
    NO_PHONE = "NO_PHONE"


class CameraProfile(str, Enum):
    """Camera viewpoint capability profiles."""
    HIGH_ANGLE_CCTV = "HIGH_ANGLE_CCTV"
    FRONT_OBLIQUE_CCTV = "FRONT_OBLIQUE_CCTV"
    FRONTAL_DESK_LEVEL = "FRONTAL_DESK_LEVEL"
    LOW_RESOLUTION = "LOW_RESOLUTION"
    REAR_OBLIQUE = "REAR_OBLIQUE"


@dataclass
class PostureCue:
    """Posture classification cue for a person crop."""
    status: ObservationStatus = ObservationStatus.NOT_EVALUATED
    probabilities: Dict[str, float] = field(default_factory=lambda: {
        "NORMAL_UPRIGHT": 0.0,
        "NORMAL_READ_WRITE": 0.0,
        "HEAD_REST_SLEEP": 0.0,
        "TURN_HEAD_CLEAR": 0.0,
    })
    predicted_class: Optional[str] = None
    confidence: float = 0.0
    crop_quality: str = "GOOD"
    resolution_used: int = 224
    reliability_weight: float = 1.0


@dataclass
class HeadPoseCue:
    """Continuous head-pose horizontal orientation cue."""
    status: ObservationStatus = ObservationStatus.NOT_EVALUATED
    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    reliability_weight: float = 1.0
    source_model: str = "HopeNet-Yaw"
    support_status: HeadPoseSupportStatus = HeadPoseSupportStatus.WITHIN_PRIMARY_SUPPORT


@dataclass
class PhoneCue:
    """Phone / contraband spatial association cue."""
    status: ObservationStatus = ObservationStatus.NOT_EVALUATED
    detected: bool = False
    association_confidence: float = 0.0
    association_status: PhoneAssociationStatus = PhoneAssociationStatus.NO_PHONE
    spatial_relation: str = "NONE"   # DIRECT_CONTACT, DESK_PROXIMITY, NONE
    phone_bbox: Optional[Tuple[float, float, float, float]] = None
    reliability_weight: float = 1.0
    is_candidate_only: bool = False


@dataclass
class MacroBehaviorCue:
    """Full-frame macro behavior detector cue (Stage 1.5)."""
    status: ObservationStatus = ObservationStatus.NOT_EVALUATED
    stand_score: float = 0.0
    discuss_score: float = 0.0
    paired_peer_id: Optional[int] = None
    reliability_weight: float = 1.0


@dataclass
class TrackingState:
    """ByteTrack track metadata for identity continuity."""
    track_id: int
    timestamp_sec: float
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    track_age_frames: int = 1
    time_since_seen_sec: float = 0.0
    visibility_score: float = 1.0
    occluded: bool = False


@dataclass
class UnifiedTrackUpdate:
    """Unified observation update for a single track at an explicit timestamp."""
    track_id: int
    timestamp_sec: float
    camera_id: str = "cam_0"
    tracking: Optional[TrackingState] = None
    posture: PostureCue = field(default_factory=PostureCue)
    headpose: HeadPoseCue = field(default_factory=HeadPoseCue)
    phone: PhoneCue = field(default_factory=PhoneCue)
    macro_behavior: MacroBehaviorCue = field(default_factory=MacroBehaviorCue)
    source_origin: str = SourceOrigin.UNKNOWN.value

    @property
    def origin(self) -> str:
        return self.source_origin

    def is_valid(self) -> bool:
        """Validate timestamp and track ID."""
        import math
        if math.isnan(self.timestamp_sec) or math.isinf(self.timestamp_sec):
            return False
        if self.track_id < 0:
            return False
        return True


@dataclass
class FusedEvent:
    """Observable fact-based event record."""
    event_id: str
    track_id: int
    camera_id: str
    event_type: str                         # EventFamily value
    start_timestamp: float
    last_update_timestamp: float
    end_timestamp: Optional[float] = None
    duration: float = 0.0
    risk_level: str = "LOW"                 # RiskLevel value
    risk_score: float = 0.0                 # 0.0 - 100.0
    evidence_summary: Dict[str, Any] = field(default_factory=dict)
    cue_availability: Dict[str, str] = field(default_factory=dict)
    cue_reliability: Dict[str, float] = field(default_factory=dict)
    status: str = "active"                  # Legacy field kept for compatibility
    lifecycle_status: str = "active"        # 'open', 'active', 'closed'
    review_status: str = "awaiting"         # 'awaiting', 'confirmed', 'dismissed'
    observation_snapshot: Dict[str, Any] = field(default_factory=dict)
    reviewer_notes: Optional[str] = None
    model_versions: Dict[str, str] = field(default_factory=dict)
    fusion_config_version: str = "4.0.0-v4d"
    event_origin: str = "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
