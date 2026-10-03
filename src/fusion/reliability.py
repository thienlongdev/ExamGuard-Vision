"""
V4D Cue Reliability & Evidence Weighting Model
=============================================
Calculates auditable evidence strength and reliability weights for observable cues.
Strictly avoids false precision and Bayesian cheating priors.
Prevents double-counting correlated cues (e.g. Posture Turn + Head-Pose Yaw).
"""

from typing import Dict, Any, Optional
from src.fusion.types import (
    PostureCue,
    HeadPoseCue,
    PhoneCue,
    ObservationStatus,
    HeadPoseSupportStatus,
)


class ReliabilityModel:
    """Computes evidence strength and reliability weights per cue.
    
    All weights are designated PROVISIONAL_FUSION_WEIGHT unless calibrated on
    physical ground truth.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        rw_cfg = self.config.get("reliability_weights", {})
        
        posture_cfg = rw_cfg.get("posture", {})
        self.posture_base = float(posture_cfg.get("base_weight", 1.0))
        self.posture_small_penalty = float(posture_cfg.get("small_crop_penalty", 0.25))
        self.posture_blur_penalty = float(posture_cfg.get("blurry_crop_penalty", 0.20))

        hp_cfg = rw_cfg.get("head_pose", {})
        self.hp_base = float(hp_cfg.get("base_weight", 0.60))
        self.hp_turn_corr_penalty = float(hp_cfg.get("correlated_with_turn_penalty", 0.30))

        phone_cfg = rw_cfg.get("phone", {})
        self.phone_base = float(phone_cfg.get("base_weight", 1.0))
        self.phone_ambiguous_penalty = float(phone_cfg.get("ambiguous_association_penalty", 0.50))

    def evaluate_posture_reliability(
        self,
        posture: PostureCue,
        person_height: float = 200.0,
        blur_score: Optional[float] = None,
    ) -> float:
        """Compute auditable reliability weight for posture prediction."""
        if posture.status != ObservationStatus.AVAILABLE:
            return 0.0

        weight = self.posture_base
        if person_height < 150.0:
            weight -= self.posture_small_penalty
        if blur_score is not None and blur_score < 40.0:
            weight -= self.posture_blur_penalty

        # Scale by top prediction confidence margin
        conf = max(0.0, min(1.0, posture.confidence))
        return max(0.1, min(1.0, weight * conf))

    def evaluate_headpose_reliability(
        self,
        headpose: HeadPoseCue,
        posture_turn_present: bool = False,
    ) -> float:
        """Compute auditable reliability weight for continuous yaw orientation.
        
        CORRELATED-CUE PROTECTION:
        Posture TURN_HEAD_CLEAR and Yaw both derive from head rotation.
        If posture turn is already active, yaw reliability weight is discounted
        to prevent artificial evidence inflation.
        """
        if headpose.status != ObservationStatus.AVAILABLE or headpose.yaw_deg is None:
            return 0.0

        weight = self.hp_base

        # Secondary penalty if circular fallback was used (higher error in extreme profile)
        if headpose.support_status == HeadPoseSupportStatus.FALLBACK_CIRCULAR:
            weight *= 0.50

        # Correlated cue protection
        if posture_turn_present:
            weight -= self.hp_turn_corr_penalty

        return max(0.1, min(1.0, weight))

    def evaluate_phone_reliability(self, phone: PhoneCue) -> float:
        """Compute reliability weight for phone/contraband association.
        
        Strict V4D Rule:
        A clearly associated phone object (CLEAR_ASSOCIATION) carries high evidential weight.
        Ambiguous associations are strongly discounted.
        """
        if phone.status != ObservationStatus.AVAILABLE or not phone.detected:
            return 0.0

        status_val = phone.association_status.value if hasattr(phone.association_status, "value") else str(phone.association_status)
        weight = self.phone_base
        if status_val == "AMBIGUOUS_ASSOCIATION":
            weight -= self.phone_ambiguous_penalty
            conf = max(0.0, min(1.0, phone.association_confidence))
            return max(0.1, min(1.0, weight * conf))

        # Clear association: high reliable evidence (bounded between 0.85 and 1.0)
        conf = max(0.0, min(1.0, phone.association_confidence))
        return max(0.85, min(1.0, weight * (0.80 + 0.20 * conf)))

    def compute_turn_fusion_evidence(
        self,
        posture: PostureCue,
        headpose: HeadPoseCue,
    ) -> float:
        """Fused evidence strength for lateral head orientation.
        
        Strict V4C Rule:
        Classroom yaw alone has weak standalone separation (overlap 87.64%).
        Therefore, yaw CANNOT independently trigger turn without posture support,
        and yaw + posture cannot be naively summed as independent evidence.
        """
        posture_turn_score = 0.0
        if posture.status == ObservationStatus.AVAILABLE:
            posture_turn_score = posture.probabilities.get("TURN_HEAD_CLEAR", 0.0)

        # Baseline: posture alone
        if headpose.status != ObservationStatus.AVAILABLE or headpose.yaw_deg is None:
            return posture_turn_score

        # Supporting yaw cue
        abs_yaw = abs(headpose.yaw_deg)
        # Consistent lateral rotation if |yaw| >= 25 deg
        if abs_yaw >= 25.0:
            # Multi-cue reinforcement: if posture also indicates turn, boost slightly
            yaw_support = min(1.0, (abs_yaw - 20.0) / 40.0)
            if posture_turn_score >= 0.40:
                fused = 0.70 * posture_turn_score + 0.30 * yaw_support
            else:
                # Weak posture cannot be overridden by yaw alone
                fused = 0.50 * posture_turn_score + 0.20 * yaw_support
        else:
            # Head-pose indicates frontal face (|yaw| < 20 deg); slightly attenuates false posture turn
            fused = posture_turn_score * 0.85

        return max(0.0, min(1.0, fused))
