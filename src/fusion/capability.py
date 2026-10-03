"""
V4D Capability & Scale Gating Engine
===================================
Enforces camera viewpoint profile constraints, scale/resolution eligibility,
and head-pose / posture dispatch policies without fabricating missing data.
"""

from typing import Dict, Any, Optional, Tuple
from src.fusion.types import (
    ObservationStatus,
    HeadPoseSupportStatus,
    CameraProfile,
    PostureCue,
    HeadPoseCue,
)


class CapabilityGate:
    """Manages viewpoint capability profiles and resolution eligibility gates."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.camera_profiles = self.config.get("camera_profiles", {})
        self.scale_gating = self.config.get("scale_gating", {})

        # Default scale thresholds
        posture_scale = self.scale_gating.get("posture", {})
        self.min_person_h = float(posture_scale.get("min_person_height_px", 120.0))
        self.min_person_w = float(posture_scale.get("min_person_width_px", 40.0))
        self.high_res_fallback_available = bool(posture_scale.get("high_res_fallback_available", True))
        self.use_320_fallback = bool(posture_scale.get("use_320_fallback", False))

        hp_scale = self.scale_gating.get("head_pose", {})
        self.min_head_w = float(hp_scale.get("min_head_width_px", 25.0))
        self.min_head_h = float(hp_scale.get("min_head_height_px", 25.0))
        self.primary_yaw_range = hp_scale.get("primary_yaw_range_deg", [-99.0, 99.0])
        self.enable_circular_fallback = bool(hp_scale.get("enable_circular_fallback", True))

    def is_posture_eligible(
        self,
        bbox: Tuple[float, float, float, float],
        camera_profile: str = "FRONT_OBLIQUE_CCTV",
    ) -> bool:
        """Check if person bounding box meets posture classification resolution gate."""
        profile_cfg = self.camera_profiles.get(camera_profile, {})
        if profile_cfg and not profile_cfg.get("posture_supported", True):
            return False

        x1, y1, x2, y2 = bbox
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        min_h = float(profile_cfg.get("min_person_height_px", self.min_person_h))

        return (h >= min_h) and (w >= self.min_person_w)

    def select_posture_resolution(
        self,
        bbox: Tuple[float, float, float, float],
        blur_score: Optional[float] = None,
    ) -> int:
        """Select resolution for posture crop classification.
        
        Strict V4D Policy: Defaults to PRIMARY 224x224.
        320x320 high-res is available as an interface capability but only activated
        when explicitly enabled via configuration (use_320_fallback = True).
        """
        if not self.use_320_fallback:
            return 224

        x1, y1, x2, y2 = bbox
        h = max(0.0, y2 - y1)
        # If enabled in config and crop is difficult / small:
        if h < 160.0 or (blur_score is not None and blur_score < 40.0):
            return 320
        return 224

    def is_head_pose_eligible(
        self,
        person_bbox: Tuple[float, float, float, float],
        head_bbox: Optional[Tuple[float, float, float, float]] = None,
        camera_profile: str = "FRONT_OBLIQUE_CCTV",
        face_detected: bool = True,
    ) -> bool:
        """Check if head/face region is resolvable for head-pose orientation estimation."""
        profile_cfg = self.camera_profiles.get(camera_profile, {})
        if profile_cfg and not profile_cfg.get("headpose_supported", True):
            return False

        if not face_detected:
            return False

        px1, py1, px2, py2 = person_bbox
        ph = max(0.0, py2 - py1)
        if ph < self.min_person_h:
            return False

        if head_bbox is not None:
            hx1, hy1, hx2, hy2 = head_bbox
            hw = max(0.0, hx2 - hx1)
            hh = max(0.0, hy2 - hy1)
            min_head_dim = float(profile_cfg.get("min_head_dimension_px", self.min_head_w))
            if hw < min_head_dim or hh < min_head_dim:
                return False

        return True

    def dispatch_head_pose(
        self,
        raw_yaw_candidate: Optional[float],
        is_eligible: bool,
        face_confidence: float = 1.0,
    ) -> HeadPoseCue:
        """Capability-based dispatch between HopeNet and ResNet18-Circular.
        
        DO NOT fabricate a reliable yaw result when the face is not physically resolvable.
        """
        if not is_eligible or raw_yaw_candidate is None or face_confidence < 0.25:
            return HeadPoseCue(
                status=ObservationStatus.UNAVAILABLE,
                yaw_deg=None,
                pitch_deg=None,
                roll_deg=None,
                reliability_weight=0.0,
                source_model="NONE",
                support_status=HeadPoseSupportStatus.FACE_UNRESOLVABLE,
            )

        min_yaw, max_yaw = self.primary_yaw_range
        # Check if within primary HopeNet support [-99, +99)
        if min_yaw <= raw_yaw_candidate < max_yaw:
            return HeadPoseCue(
                status=ObservationStatus.AVAILABLE,
                yaw_deg=float(raw_yaw_candidate),
                reliability_weight=1.0,
                source_model="HopeNet-Yaw",
                support_status=HeadPoseSupportStatus.WITHIN_PRIMARY_SUPPORT,
            )

        # Candidate is outside [-99, +99)
        if self.enable_circular_fallback:
            return HeadPoseCue(
                status=ObservationStatus.AVAILABLE,
                yaw_deg=float(raw_yaw_candidate),
                reliability_weight=0.50,  # Lower weight due to higher profile error
                source_model="ResNet18-Circular",
                support_status=HeadPoseSupportStatus.FALLBACK_CIRCULAR,
            )

        return HeadPoseCue(
            status=ObservationStatus.UNAVAILABLE,
            yaw_deg=None,
            reliability_weight=0.0,
            source_model="HopeNet-Yaw",
            support_status=HeadPoseSupportStatus.OUT_OF_SUPPORT,
        )
