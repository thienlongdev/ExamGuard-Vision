"""
V4D Deterministic Replay Engine
===============================
Consumes offline prediction traces without rerunning neural inference.
Enables rapid, deterministic temporal threshold sweeps, ablation studies,
and regression testing.
"""

import json
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path

from src.fusion.types import (
    UnifiedTrackUpdate,
    FusedEvent,
    ObservationStatus,
    PostureCue,
    HeadPoseCue,
    PhoneCue,
    MacroBehaviorCue,
    TrackingState,
)
from src.fusion.fusion_engine import MultiCueFusionEngine
from src.fusion.event_engine import EventEngine
from src.fusion.risk_aggregator import RiskAggregator


class ReplayEngine:
    """Executes deterministic offline replay of multi-cue temporal traces."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        fusion_engine: Optional[MultiCueFusionEngine] = None,
        event_engine: Optional[EventEngine] = None,
        risk_aggregator: Optional[RiskAggregator] = None,
    ):
        if config is None:
            config_path = Path("configs/v4d_fusion.yaml")
            if config_path.exists():
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
        self.config = config or {}
        self.fusion = fusion_engine or MultiCueFusionEngine(self.config)
        self.event_engine = event_engine or EventEngine(self.config)
        self.risk_aggregator = risk_aggregator or RiskAggregator(self.config)

    def replay_trace(
        self,
        updates: List[UnifiedTrackUpdate],
        camera_id: str = "cam_0",
    ) -> List[FusedEvent]:
        """Process an ordered sequence of updates and collect all generated events."""
        events: List[FusedEvent] = []
        # Sort updates by timestamp to guarantee strict chronological replay
        sorted_updates = sorted(updates, key=lambda u: u.timestamp_sec)

        for update in sorted_updates:
            cue_state = self.fusion.update_track(update)
            if cue_state is None:
                continue

            event_actions = self.event_engine.process_cue_state(cue_state, camera_id=camera_id)
            for event, action in event_actions:
                # Assess risk
                active_cues = 1
                if cue_state.headpose_status == ObservationStatus.AVAILABLE:
                    active_cues += 1
                if cue_state.phone_status == ObservationStatus.AVAILABLE and cue_state.phone_detected:
                    active_cues += 1

                assessed_event = self.risk_aggregator.assess_event_risk(
                    event=event,
                    active_cues_count=active_cues,
                    independent_cues_count=active_cues,
                    mean_reliability=cue_state.posture_reliability,
                )
                events.append(assessed_event)

        return events

    def replay_jsonl(self, jsonl_path: Path, camera_id: str = "cam_0") -> List[FusedEvent]:
        """Load and replay serialized traces from a JSONL file."""
        updates: List[UnifiedTrackUpdate] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                update = self._parse_json_update(data)
                if update:
                    updates.append(update)
        return self.replay_trace(updates, camera_id=camera_id)

    def _parse_json_update(self, data: Dict[str, Any]) -> Optional[UnifiedTrackUpdate]:
        """Convert dictionary to UnifiedTrackUpdate."""
        try:
            track_id = int(data.get("track_id", 1))
            t = float(data.get("timestamp_sec", data.get("timestamp", 0.0)))
            cam_id = str(data.get("camera_id", "cam_0"))

            # Posture
            posture_data = data.get("posture", {})
            p_status = ObservationStatus(posture_data.get("status", "AVAILABLE")) if "status" in posture_data else ObservationStatus.AVAILABLE
            posture_cue = PostureCue(
                status=p_status,
                probabilities=posture_data.get("probabilities", {
                    "NORMAL_UPRIGHT": 0.0,
                    "NORMAL_READ_WRITE": 0.0,
                    "HEAD_REST_SLEEP": 0.0,
                    "TURN_HEAD_CLEAR": 0.0,
                }),
                predicted_class=posture_data.get("predicted_class"),
                confidence=float(posture_data.get("confidence", 0.0)),
                crop_quality=posture_data.get("crop_quality", "GOOD"),
                resolution_used=int(posture_data.get("resolution_used", 224)),
                reliability_weight=float(posture_data.get("reliability_weight", 1.0)),
            )

            # HeadPose
            hp_data = data.get("headpose", {})
            hp_status = ObservationStatus(hp_data.get("status", "UNAVAILABLE")) if "status" in hp_data else ObservationStatus.UNAVAILABLE
            headpose_cue = HeadPoseCue(
                status=hp_status,
                yaw_deg=float(hp_data["yaw_deg"]) if hp_data.get("yaw_deg") is not None else None,
                pitch_deg=float(hp_data["pitch_deg"]) if hp_data.get("pitch_deg") is not None else None,
                roll_deg=float(hp_data["roll_deg"]) if hp_data.get("roll_deg") is not None else None,
                reliability_weight=float(hp_data.get("reliability_weight", 1.0)),
                source_model=hp_data.get("source_model", "HopeNet-Yaw"),
            )

            # Phone
            phone_data = data.get("phone", {})
            ph_status = ObservationStatus(phone_data.get("status", "UNAVAILABLE")) if "status" in phone_data else ObservationStatus.UNAVAILABLE
            phone_cue = PhoneCue(
                status=ph_status,
                detected=bool(phone_data.get("detected", False)),
                association_confidence=float(phone_data.get("association_confidence", 0.0)),
                association_status=phone_data.get("association_status", "NO_PHONE"),
                spatial_relation=phone_data.get("spatial_relation", "NONE"),
                reliability_weight=float(phone_data.get("reliability_weight", 1.0)),
            )

            # Macro
            macro_data = data.get("macro_behavior", {})
            m_status = ObservationStatus(macro_data.get("status", "UNAVAILABLE")) if "status" in macro_data else ObservationStatus.UNAVAILABLE
            macro_cue = MacroBehaviorCue(
                status=m_status,
                stand_score=float(macro_data.get("stand_score", 0.0)),
                discuss_score=float(macro_data.get("discuss_score", 0.0)),
                paired_peer_id=macro_data.get("paired_peer_id"),
            )

            # Tracking
            trk_data = data.get("tracking")
            tracking_state = None
            if trk_data:
                tracking_state = TrackingState(
                    track_id=track_id,
                    timestamp_sec=t,
                    bbox=tuple(trk_data.get("bbox", [0, 0, 100, 100])),
                    track_age_frames=int(trk_data.get("track_age_frames", 1)),
                    time_since_seen_sec=float(trk_data.get("time_since_seen_sec", 0.0)),
                    visibility_score=float(trk_data.get("visibility_score", 1.0)),
                )

            replay_origin = str(data.get("source_origin") or data.get("origin") or "REPLAY_STREAM")

            return UnifiedTrackUpdate(
                track_id=track_id,
                timestamp_sec=t,
                camera_id=cam_id,
                source_origin=replay_origin,
                tracking=tracking_state,
                posture=posture_cue,
                headpose=headpose_cue,
                phone=phone_cue,
                macro_behavior=macro_cue,
            )
        except Exception as e:
            return None
