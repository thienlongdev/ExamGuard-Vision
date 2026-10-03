"""
V4D Per-Track Cue State Representation
======================================
Encapsulates instantaneous and temporally-smoothed cue representations per track.
Maintains explicit availability flags, competing negative evidence,
and identity continuity status.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple

from src.fusion.types import (
    ObservationStatus,
    PostureCue,
    HeadPoseCue,
    PhoneCue,
    MacroBehaviorCue,
    TrackingState,
)


@dataclass
class PerTrackCueState:
    """Current integrated cue state for a single tracked student."""
    track_id: int
    last_update_timestamp: float
    time_since_seen: float = 0.0
    continuity_valid: bool = True

    # Smoothed Posture Evidence
    posture_status: ObservationStatus = ObservationStatus.NOT_EVALUATED
    posture_probs: Dict[str, float] = field(default_factory=lambda: {
        "NORMAL_UPRIGHT": 0.0,
        "NORMAL_READ_WRITE": 0.0,
        "HEAD_REST_SLEEP": 0.0,
        "TURN_HEAD_CLEAR": 0.0,
    })
    posture_reliability: float = 1.0

    # Continuous Head-Pose Orientation Evidence
    headpose_status: ObservationStatus = ObservationStatus.NOT_EVALUATED
    smoothed_yaw_deg: Optional[float] = None
    yaw_abs_mean: float = 0.0
    headpose_reliability: float = 1.0
    headpose_source_model: str = "NONE"

    # Phone Association Evidence
    phone_status: ObservationStatus = ObservationStatus.NOT_EVALUATED
    phone_detected: bool = False
    phone_confidence: float = 0.0
    phone_association_status: str = "NO_PHONE"
    phone_reliability: float = 1.0

    # Macro Behavior Evidence (Stage 1.5)
    macro_status: ObservationStatus = ObservationStatus.NOT_EVALUATED
    stand_score: float = 0.0
    discuss_score: float = 0.0
    paired_peer_id: Optional[int] = None

    # Competing Evidence Metrics
    # (NORMAL_READ_WRITE directly competes against and suppresses HEAD_REST_SLEEP)
    read_write_suppression_active: bool = False
    read_write_score: float = 0.0

    # Turn Multi-Cue Agreement Metric
    turn_fused_evidence: float = 0.0
    turn_multi_cue_agreement: bool = False
