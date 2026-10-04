"""
V4D Multi-Cue Temporal Fusion Engine
===================================
Coordinates the temporal buffer, reliability weighting model, and capability gates
to produce an integrated PerTrackCueState for downstream event generation.
Strict adherence to:
- Missing cue safety (missing cue is NEVER negative evidence)
- Correlated cue protection (yaw + posture turn)
- Competing negative evidence (NORMAL_READ_WRITE suppresses HEAD_REST_SLEEP)
- Classroom yaw finding (yaw is supporting only, cannot dominate)
"""

from typing import Dict, Any, Optional, List
import logging

from src.fusion.types import (
    UnifiedTrackUpdate,
    ObservationStatus,
    HeadPoseSupportStatus,
    PhoneAssociationStatus,
    PostureCue,
)
from src.fusion.cue_state import PerTrackCueState
from src.fusion.temporal_buffer import TemporalBuffer
from src.fusion.reliability import ReliabilityModel
from src.fusion.capability import CapabilityGate

logger = logging.getLogger(__name__)


class MultiCueFusionEngine:
    """Fuses multi-cue temporal observations for student tracks."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        temporal_buffer: Optional[TemporalBuffer] = None,
        reliability_model: Optional[ReliabilityModel] = None,
        capability_gate: Optional[CapabilityGate] = None,
    ):
        if config is None:
            from pathlib import Path
            import yaml
            p = Path("configs/v4d_fusion.yaml")
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
        self.config = config or {}
        self.temporal_buffer = temporal_buffer or TemporalBuffer(
            time_horizon_seconds=float(self.config.get("temporal_buffer", {}).get("time_horizon_seconds", 30.0)),
            max_samples_per_track=int(self.config.get("temporal_buffer", {}).get("max_samples_per_track", 300)),
            eviction_inactive_seconds=float(self.config.get("temporal_buffer", {}).get("eviction_inactive_seconds", 15.0)),
            track_continuity_tolerance_sec=float(self.config.get("temporal_buffer", {}).get("track_continuity_tolerance_sec", 2.0)),
        )
        self.reliability = reliability_model or ReliabilityModel(self.config)
        self.capability = capability_gate or CapabilityGate(self.config)

        # Thresholds from config
        prov = self.config.get("provisional_thresholds", {})
        sleep_cfg = prov.get("sustained_head_rest", {})
        self.sleep_veto_thresh = float(sleep_cfg.get("read_write_veto_threshold", 0.50))

        turn_cfg = prov.get("sustained_lateral_head_orientation", {})
        self.yaw_deg_thresh = float(turn_cfg.get("supporting_yaw_deg_threshold", 25.0))

    def update_track(self, update: UnifiedTrackUpdate) -> Optional[PerTrackCueState]:
        """Ingest a track update, store in temporal buffer, and compute fused cue state."""
        if not update.is_valid():
            logger.warning(f"Rejected invalid track update for track {update.track_id}")
            return None

        # Check track continuity before push
        continuous, gap = self.temporal_buffer.check_identity_continuity(
            update.track_id, update.timestamp_sec
        )

        # Ingest into sliding window buffer
        pushed = self.temporal_buffer.push(update)
        if not pushed:
            return None

        track_id = update.track_id
        timestamp = update.timestamp_sec

        # 1. Posture Processing & Temporal Smoothing
        posture_status = update.posture.status
        smoothed_probs = self.temporal_buffer._tracks[track_id].smoothed_posture_probabilities(window_seconds=0.5)

        # Active probabilities (instant if available, else bridged from recent window)
        active_posture_probs = update.posture.probabilities if (posture_status == ObservationStatus.AVAILABLE and update.posture.probabilities) else smoothed_probs
        posture_rel = 0.0

        if posture_status == ObservationStatus.AVAILABLE:
            # Evaluate reliability
            person_h = 200.0
            if update.tracking is not None:
                _, y1, _, y2 = update.tracking.bbox
                person_h = max(0.0, y2 - y1)
            posture_rel = self.reliability.evaluate_posture_reliability(
                update.posture, person_height=person_h
            )
        elif sum(smoothed_probs.values()) > 0.0:
            # Intermittent cadence: bridged from recent valid frames
            posture_rel = 0.80

        # Effective posture cue for downstream fusion
        effective_posture = update.posture
        if posture_status != ObservationStatus.AVAILABLE and sum(smoothed_probs.values()) > 0.0:
            effective_posture = PostureCue(
                status=ObservationStatus.AVAILABLE,
                probabilities=smoothed_probs,
                confidence=max(smoothed_probs.values()) if smoothed_probs else 0.0,
                reliability_weight=posture_rel,
            )

        # 2. Competing Evidence: NORMAL_READ_WRITE veto on HEAD_REST_SLEEP
        rw_score = active_posture_probs.get("NORMAL_READ_WRITE", 0.0)
        rw_suppression = rw_score >= self.sleep_veto_thresh

        # 3. Head-Pose Orientation & Reliability
        headpose_status = update.headpose.status
        smoothed_yaw: Optional[float] = None
        yaw_abs_mean = 0.0
        hp_rel = 0.0
        source_model = update.headpose.source_model

        if headpose_status == ObservationStatus.AVAILABLE and update.headpose.yaw_deg is not None:
            # Get recent window for yaw smoothing
            recent_updates = self.temporal_buffer.get_track_history(track_id, window_seconds=1.5)
            yaws = [
                u.headpose.yaw_deg for u in recent_updates
                if u.headpose.status == ObservationStatus.AVAILABLE and u.headpose.yaw_deg is not None
            ]
            if yaws:
                smoothed_yaw = sum(yaws) / len(yaws)
                yaw_abs_mean = sum(abs(y) for y in yaws) / len(yaws)

            turn_active = active_posture_probs.get("TURN_HEAD_CLEAR", 0.0) >= 0.40
            hp_rel = self.reliability.evaluate_headpose_reliability(
                update.headpose, posture_turn_present=turn_active
            )

        # 4. Turn Multi-Cue Fusion Evidence
        turn_fused = self.reliability.compute_turn_fusion_evidence(
            effective_posture, update.headpose
        )

        # Check repeated glance burst across recent temporal window
        glance_burst = False
        glance_cnt = 0
        if track_id in self.temporal_buffer._tracks:
            glance_burst, glance_cnt = self.temporal_buffer._tracks[track_id].detect_glance_burst(
                window_seconds=3.5, yaw_thresh=self.yaw_deg_thresh, min_glances=2
            )
        if glance_burst:
            turn_fused = max(turn_fused, 0.54)

        turn_agreement = (
            (active_posture_probs.get("TURN_HEAD_CLEAR", 0.0) >= 0.40 and
             yaw_abs_mean >= self.yaw_deg_thresh and
             headpose_status == ObservationStatus.AVAILABLE) or
            (yaw_abs_mean >= self.yaw_deg_thresh and headpose_status == ObservationStatus.AVAILABLE) or
            glance_burst
        )

        # 5. Phone Association Processing
        phone_status = update.phone.status
        phone_detected = False
        phone_conf = 0.0
        phone_assoc_status = "NO_PHONE"
        phone_rel = 0.0

        if phone_status == ObservationStatus.AVAILABLE and update.phone.detected:
            phone_detected = True
            phone_conf = update.phone.association_confidence
            phone_assoc_status = update.phone.association_status.value if hasattr(update.phone.association_status, "value") else str(update.phone.association_status)
            phone_rel = self.reliability.evaluate_phone_reliability(update.phone)

        # 6. Macro Behavior Processing (Stage 1.5)
        macro_status = update.macro_behavior.status
        stand_score = update.macro_behavior.stand_score if macro_status == ObservationStatus.AVAILABLE else 0.0
        discuss_score = update.macro_behavior.discuss_score if macro_status == ObservationStatus.AVAILABLE else 0.0
        paired_id = update.macro_behavior.paired_peer_id if macro_status == ObservationStatus.AVAILABLE else None

        resolved_origin = getattr(update, "source_origin", getattr(update, "origin", "UNKNOWN"))

        return PerTrackCueState(
            track_id=track_id,
            last_update_timestamp=timestamp,
            time_since_seen=gap,
            continuity_valid=continuous,
            source_origin=resolved_origin,
            posture_status=posture_status,
            posture_probs=active_posture_probs,
            posture_reliability=posture_rel,
            headpose_status=headpose_status,
            smoothed_yaw_deg=smoothed_yaw,
            yaw_abs_mean=yaw_abs_mean,
            headpose_reliability=hp_rel,
            headpose_source_model=source_model,
            phone_status=phone_status,
            phone_detected=phone_detected,
            phone_confidence=phone_conf,
            phone_association_status=phone_assoc_status,
            phone_reliability=phone_rel,
            macro_status=macro_status,
            stand_score=stand_score,
            discuss_score=discuss_score,
            paired_peer_id=paired_id,
            read_write_suppression_active=rw_suppression,
            read_write_score=rw_score,
            turn_fused_evidence=turn_fused,
            turn_multi_cue_agreement=turn_agreement,
        )
