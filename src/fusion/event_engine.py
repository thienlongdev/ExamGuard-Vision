"""
V4D Temporal Event State Machine & Deduplication Engine
======================================================
Manages event lifecycles across observable behavior families:
INACTIVE -> CANDIDATE -> ACTIVE -> COOLDOWN -> INACTIVE

Enforces:
- Hysteresis (asymmetric enter/exit criteria)
- Event deduplication (sustained episodes yield 1 OPEN/UPDATE/CLOSE lifecycle, not 300 events)
- Track identity isolation (events never leak across track IDs)
- Explainable observable facts only (no subjective guilt statements)
"""

import uuid
import logging
from typing import Dict, List, Optional, Any, Tuple

from src.fusion.types import (
    FusedEvent,
    EventFamily,
    EventLifecycleState,
    RiskLevel,
    ObservationStatus,
)
from src.fusion.cue_state import PerTrackCueState

logger = logging.getLogger(__name__)


class TrackEventStateMachine:
    """Manages event lifecycle for a single (track_id, event_family) pair."""

    def __init__(
        self,
        track_id: int,
        event_family: EventFamily,
        min_candidate_duration: float,
        enter_threshold: float,
        exit_threshold: float,
        cooldown_seconds: float,
        config_version: str = "4.0.0-v4d",
    ):
        self.track_id = track_id
        self.event_family = event_family
        self.min_candidate_duration = min_candidate_duration
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.cooldown_seconds = cooldown_seconds
        self.config_version = config_version

        self.state: EventLifecycleState = EventLifecycleState.INACTIVE
        self.candidate_start_time: Optional[float] = None
        self.active_event: Optional[FusedEvent] = None
        self.cooldown_start_time: Optional[float] = None
        self.last_update_time: Optional[float] = None

    def process_frame(
        self,
        timestamp: float,
        evidence_score: float,
        is_vetoed: bool,
        cue_state: PerTrackCueState,
        camera_id: str = "cam_0",
        supporting_details: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[FusedEvent], str]:
        """Update state machine and return (event, action) where action in ['OPEN', 'UPDATE', 'CLOSE', 'NONE']."""
        self.last_update_time = timestamp
        details = supporting_details or {}

        # 1. State: INACTIVE
        if self.state == EventLifecycleState.INACTIVE:
            if evidence_score >= self.enter_threshold and not is_vetoed:
                self.state = EventLifecycleState.CANDIDATE
                self.candidate_start_time = timestamp
            return None, "NONE"

        # 2. State: CANDIDATE
        if self.state == EventLifecycleState.CANDIDATE:
            if is_vetoed or evidence_score < self.exit_threshold:
                # Evidence fell below exit threshold or was vetoed
                self.state = EventLifecycleState.INACTIVE
                self.candidate_start_time = None
                return None, "NONE"

            # Check if duration in candidate state meets threshold
            candidate_start = self.candidate_start_time if self.candidate_start_time is not None else timestamp
            candidate_duration = timestamp - candidate_start
            if candidate_duration >= self.min_candidate_duration:
                # Promote to ACTIVE: Open new event
                self.state = EventLifecycleState.ACTIVE
                event_id = str(uuid.uuid4())
                self.active_event = FusedEvent(
                    event_id=event_id,
                    track_id=self.track_id,
                    camera_id=camera_id,
                    event_type=self.event_family.value,
                    start_timestamp=candidate_start,
                    last_update_timestamp=timestamp,
                    duration=candidate_duration,
                    risk_level=RiskLevel.LOW.value,
                    evidence_summary={
                        "initial_evidence_score": round(evidence_score, 4),
                        "candidate_duration_sec": round(candidate_duration, 2),
                        **details,
                    },
                    cue_availability={
                        "posture": cue_state.posture_status.value,
                        "headpose": cue_state.headpose_status.value,
                        "phone": cue_state.phone_status.value,
                        "macro": cue_state.macro_status.value,
                    },
                    cue_reliability={
                        "posture_reliability": round(cue_state.posture_reliability, 4),
                        "headpose_reliability": round(cue_state.headpose_reliability, 4),
                        "phone_reliability": round(cue_state.phone_reliability, 4),
                    },
                    status="active",
                    fusion_config_version=self.config_version,
                )
                return self.active_event, "OPEN"

            return None, "NONE"

        # 3. State: ACTIVE
        if self.state == EventLifecycleState.ACTIVE:
            if not is_vetoed and evidence_score >= self.exit_threshold:
                # Still active: update event duration and last timestamp (DEDUPLICATION: do not create new event!)
                if self.active_event is not None:
                    self.active_event.last_update_timestamp = timestamp
                    self.active_event.duration = timestamp - self.active_event.start_timestamp
                    self.active_event.evidence_summary.update(details)
                    self.active_event.evidence_summary["latest_evidence_score"] = round(evidence_score, 4)
                return self.active_event, "UPDATE"

            # Evidence dropped below exit threshold or was vetoed: Close event and enter COOLDOWN
            self.state = EventLifecycleState.COOLDOWN
            self.cooldown_start_time = timestamp
            closed_event = self.active_event
            if closed_event is not None:
                closed_event.end_timestamp = timestamp
                closed_event.duration = timestamp - closed_event.start_timestamp
                closed_event.status = "closed"
            self.active_event = None
            self.candidate_start_time = None
            return closed_event, "CLOSE"

        # 4. State: COOLDOWN
        if self.state == EventLifecycleState.COOLDOWN:
            cooldown_start = self.cooldown_start_time if self.cooldown_start_time is not None else timestamp
            elapsed = timestamp - cooldown_start
            if elapsed >= self.cooldown_seconds:
                self.state = EventLifecycleState.INACTIVE
                self.cooldown_start_time = None
            return None, "NONE"

        return None, "NONE"

    def force_close(self, timestamp: float, reason: str = "TRACK_LOST") -> Optional[FusedEvent]:
        """Force close active event due to track expiration or continuity break."""
        if self.state == EventLifecycleState.ACTIVE and self.active_event is not None:
            closed_event = self.active_event
            closed_event.end_timestamp = timestamp
            closed_event.duration = timestamp - closed_event.start_timestamp
            closed_event.status = "closed"
            closed_event.evidence_summary["closure_reason"] = reason
            self.active_event = None
            self.state = EventLifecycleState.INACTIVE
            self.candidate_start_time = None
            return closed_event
        self.state = EventLifecycleState.INACTIVE
        self.candidate_start_time = None
        return None


class EventEngine:
    """Multi-track event aggregation engine managing all behavior state machines."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            from pathlib import Path
            import yaml
            p = Path("configs/v4d_fusion.yaml")
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
        self.config = config or {}
        self.config_version = str(self.config.get("version", "4.0.0-v4d"))
        self.provisional = self.config.get("provisional_thresholds", {})

        # Mapping: (camera_id, track_id, event_family) -> TrackEventStateMachine
        self._machines: Dict[Tuple[str, int, EventFamily], TrackEventStateMachine] = {}

        # Historical closed and active events
        self._all_events: List[FusedEvent] = []

    def _get_machine(self, track_id: int, family: EventFamily, camera_id: str = "cam_0") -> TrackEventStateMachine:
        key = (camera_id, track_id, family)
        if key not in self._machines:
            # Load family specific thresholds
            fam_key = family.value.lower()
            cfg = self.provisional.get(fam_key, {})
            min_dur = float(cfg.get("min_candidate_duration_sec", 1.5))
            enter_th = float(cfg.get("evidence_enter_threshold", cfg.get("posture_turn_enter_threshold", 0.50)))
            exit_th = float(cfg.get("evidence_exit_threshold", cfg.get("posture_turn_exit_threshold", 0.30)))
            cd = float(cfg.get("cooldown_seconds", 4.0))

            self._machines[key] = TrackEventStateMachine(
                track_id=track_id,
                event_family=family,
                min_candidate_duration=min_dur,
                enter_threshold=enter_th,
                exit_threshold=exit_th,
                cooldown_seconds=cd,
                config_version=self.config_version,
            )
        return self._machines[key]

    def process_cue_state(
        self,
        cue_state: PerTrackCueState,
        camera_id: str = "cam_0",
    ) -> List[Tuple[FusedEvent, str]]:
        """Process updated per-track cue state across all supported event families.
        
        Returns list of (FusedEvent, action).
        """
        emitted: List[Tuple[FusedEvent, str]] = []
        track_id = cue_state.track_id
        t = cue_state.last_update_timestamp

        # Check track continuity: if continuity broken, force close existing active events
        if not cue_state.continuity_valid:
            for family in EventFamily:
                key = (camera_id, track_id, family)
                if key in self._machines:
                    ev = self._machines[key].force_close(t, reason="TRACK_CONTINUITY_BREAK")
                    if ev is not None:
                        emitted.append((ev, "CLOSE"))
            return emitted

        # -------------------------------------------------------------
        # 1. Event: SUSTAINED_HEAD_REST
        # -------------------------------------------------------------
        sleep_prob = cue_state.posture_probs.get("HEAD_REST_SLEEP", 0.0)
        # Competing evidence: NORMAL_READ_WRITE veto
        sleep_vetoed = cue_state.read_write_suppression_active
        sleep_machine = self._get_machine(track_id, EventFamily.SUSTAINED_HEAD_REST, camera_id=camera_id)
        ev_sleep, act_sleep = sleep_machine.process_frame(
            timestamp=t,
            evidence_score=sleep_prob,
            is_vetoed=sleep_vetoed,
            cue_state=cue_state,
            camera_id=camera_id,
            supporting_details={
                "posture_sleep_prob": round(sleep_prob, 4),
                "competing_read_write_score": round(cue_state.read_write_score, 4),
                "read_write_suppression": sleep_vetoed,
                "posture_reliability": round(cue_state.posture_reliability, 4),
            },
        )
        if ev_sleep is not None and act_sleep != "NONE":
            emitted.append((ev_sleep, act_sleep))

        # -------------------------------------------------------------
        # 2. Event: SUSTAINED_LATERAL_HEAD_ORIENTATION
        # -------------------------------------------------------------
        turn_evidence = cue_state.turn_fused_evidence
        turn_machine = self._get_machine(track_id, EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION, camera_id=camera_id)
        ev_turn, act_turn = turn_machine.process_frame(
            timestamp=t,
            evidence_score=turn_evidence,
            is_vetoed=False,
            cue_state=cue_state,
            camera_id=camera_id,
            supporting_details={
                "posture_turn_score": round(cue_state.posture_probs.get("TURN_HEAD_CLEAR", 0.0), 4),
                "smoothed_yaw_deg": round(cue_state.smoothed_yaw_deg, 2) if cue_state.smoothed_yaw_deg is not None else None,
                "multi_cue_agreement": cue_state.turn_multi_cue_agreement,
                "headpose_source": cue_state.headpose_source_model,
            },
        )
        if ev_turn is not None and act_turn != "NONE":
            emitted.append((ev_turn, act_turn))

        # -------------------------------------------------------------
        # 3. Event: PHONE_ASSOCIATED
        # -------------------------------------------------------------
        phone_ev = cue_state.phone_confidence if cue_state.phone_detected else 0.0
        phone_ambiguous = (cue_state.phone_association_status == "AMBIGUOUS_ASSOCIATION")
        # Ambiguous association is NOT positive
        phone_machine = self._get_machine(track_id, EventFamily.PHONE_ASSOCIATED, camera_id=camera_id)
        ev_phone, act_phone = phone_machine.process_frame(
            timestamp=t,
            evidence_score=phone_ev,
            is_vetoed=phone_ambiguous,
            cue_state=cue_state,
            camera_id=camera_id,
            supporting_details={
                "phone_confidence": round(cue_state.phone_confidence, 4),
                "association_status": cue_state.phone_association_status,
                "ambiguous_association": phone_ambiguous,
            },
        )
        if ev_phone is not None and act_phone != "NONE":
            emitted.append((ev_phone, act_phone))

        # -------------------------------------------------------------
        # 4. Event: DISCUSSION_CANDIDATE
        # -------------------------------------------------------------
        discuss_score = cue_state.discuss_score
        discuss_machine = self._get_machine(track_id, EventFamily.DISCUSSION_CANDIDATE, camera_id=camera_id)
        ev_disc, act_disc = discuss_machine.process_frame(
            timestamp=t,
            evidence_score=discuss_score,
            is_vetoed=False,
            cue_state=cue_state,
            camera_id=camera_id,
            supporting_details={
                "discuss_macro_score": round(discuss_score, 4),
                "paired_peer_id": cue_state.paired_peer_id,
            },
        )
        if ev_disc is not None and act_disc != "NONE":
            emitted.append((ev_disc, act_disc))

        # -------------------------------------------------------------
        # 5. Event: STANDING
        # -------------------------------------------------------------
        stand_score = cue_state.stand_score
        stand_machine = self._get_machine(track_id, EventFamily.STANDING, camera_id=camera_id)
        ev_stand, act_stand = stand_machine.process_frame(
            timestamp=t,
            evidence_score=stand_score,
            is_vetoed=False,
            cue_state=cue_state,
            camera_id=camera_id,
            supporting_details={
                "stand_macro_score": round(stand_score, 4),
            },
        )
        if ev_stand is not None and act_stand != "NONE":
            emitted.append((ev_stand, act_stand))

        return emitted

    def handle_track_expiration(
        self,
        expired_track_ids: List[int],
        timestamp: float,
        camera_id: Optional[str] = None,
    ) -> List[FusedEvent]:
        """Close any open events for expired tracks."""
        closed_events = []
        for t_id in expired_track_ids:
            for family in EventFamily:
                matched_keys = []
                for k in list(self._machines.keys()):
                    if len(k) == 3:
                        cam, tid, fam = k
                        if tid == t_id and fam == family and (camera_id is None or cam == camera_id):
                            matched_keys.append(k)
                    elif len(k) == 2:
                        tid, fam = k
                        if tid == t_id and fam == family:
                            matched_keys.append(k)

                for key in matched_keys:
                    ev = self._machines[key].force_close(timestamp, reason="TRACK_EXPIRED")
                    if ev is not None:
                        closed_events.append(ev)
                    self._machines.pop(key, None)
        return closed_events

    def reset(self, camera_id: Optional[str] = None) -> None:
        """Clear all or camera-specific event state machines and histories."""
        if camera_id is None:
            self._machines.clear()
            self._all_events.clear()
        else:
            to_remove = [k for k in list(self._machines.keys()) if (len(k) == 3 and k[0] == camera_id)]
            for k in to_remove:
                self._machines.pop(k, None)
