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
        
        Strict V4D Rule:
        A clear lateral head turn (|yaw| >= 24 deg) provides sufficient evidence
        to enter candidate state on its own without requiring torso leaning.
        Torso lean and posture turn reinforce multi-cue evidence.
        Frontal head orientation (|yaw| < 20 deg) attenuates posture turn.
        """
        posture_turn_score = 0.0
        posture_rw_score = 0.0
        if posture.status == ObservationStatus.AVAILABLE:
            posture_turn_score = posture.probabilities.get("TURN_HEAD_CLEAR", 0.0)
            posture_rw_score = posture.probabilities.get("NORMAL_READ_WRITE", 0.0)

        # Baseline: posture alone if headpose unavailable
        if headpose.status != ObservationStatus.AVAILABLE or headpose.yaw_deg is None:
            return posture_turn_score

        # Supporting continuous yaw cue (left and right symmetric)
        abs_yaw = abs(headpose.yaw_deg)

        if abs_yaw >= 24.0:
            # Standalone yaw evidence: scales from 0.52 at 24 deg to 1.0 at 50+ deg
            yaw_evidence = 0.52 + 0.48 * min(1.0, max(0.0, (abs_yaw - 24.0) / 26.0))
            if posture_turn_score >= 0.40:
                # Multi-cue reinforcement: both headpose and posture indicate turn
                fused = max(yaw_evidence, posture_turn_score, 0.65 * yaw_evidence + 0.35 * posture_turn_score)
            else:
                # Pure lateral head turn with upright torso: body lean NOT required!
                fused = max(yaw_evidence * 0.90, 0.52)

            # Normal read/write paper reading protection:
            # When a student is engaged in normal reading/writing on their desk (rw_score >= 0.35),
            # natural downward/diagonal paper glance (abs_yaw < 36 deg) is part of reading paper,
            # NOT a suspicious lateral head turn away from desk.
            # However, if abs_yaw >= 36 deg, it is a clear lateral look towards another seat.
            if posture_rw_score >= 0.35 and abs_yaw < 36.0:
                rw_attenuation = max(0.30, 1.0 - (posture_rw_score - 0.25) * 1.4)
                fused = min(fused * rw_attenuation, 0.40)
        elif abs_yaw < 20.0:
            # Head-pose indicates frontal face (|yaw| < 20 deg); face is facing desk/forward.
            # Attenuates false posture turn below candidate enter threshold (0.50).
            fused = min(0.35, posture_turn_score * 0.40)
        else:
            # Transition zone [20, 24) deg: scale from attenuated to standalone
            yaw_frac = (abs_yaw - 20.0) / 4.0
            base_attenuated = min(0.35, posture_turn_score * 0.40)
            fused = (1.0 - yaw_frac) * base_attenuated + yaw_frac * max(posture_turn_score * 0.85, 0.52)

        return max(0.0, min(1.0, fused))
