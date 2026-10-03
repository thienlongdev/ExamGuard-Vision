"""Suspicious event manager and alert lifecycle coordinator.

Handles event creation, alert debouncing/cooldown, deduplication, and notification dispatch.
Crucial Rule:
Events are flagged for HUMAN review with initial status 'new'.
'confirmed' or 'dismissed' status is strictly assigned by a human invigilator.
"""

from dataclasses import asdict, dataclass, field
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid
import yaml

from src.behavior.rules import RuleMatch
from src.behavior.scorer import RiskAssessment

logger = logging.getLogger(__name__)


@dataclass
class SuspiciousEvent:
    """Represents a suspicious behavioral event requiring invigilator review."""
    event_id: str
    track_id: int
    camera_id: str
    timestamp: float
    start_time: float
    end_time: float
    event_type: str
    risk_level: str          # 'LOW', 'MEDIUM', 'HIGH'
    score: float             # 0 - 100
    evidence: Dict[str, Any]
    snapshot_path: Optional[str] = None
    clip_path: Optional[str] = None
    status: str = "new"      # 'new', 'reviewed', 'confirmed', 'dismissed'
    reviewer_notes: Optional[str] = None
    event_origin: Optional[str] = "LIVE_OBSERVATION"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EventManager:
    """Manages the creation, debounce cooldown, and dispatch of suspicious behavior events."""

    def __init__(
        self,
        camera_id: str = "cam-0",
        cooldown_seconds: float = 10.0,
        config_path: Optional[str] = None,
    ):
        self.camera_id = camera_id
        self.cooldown_seconds = cooldown_seconds

        # In-memory store of events: event_id -> SuspiciousEvent
        self._events: Dict[str, SuspiciousEvent] = {}

        # Debounce tracking: (track_id, event_type) -> last_alert_time (seconds)
        self._last_alert_time: Dict[Tuple[int, str], float] = {}

        # Registered listeners for real-time dispatch (e.g., WebSocket, EvidenceCapture)
        self._listeners: List[Callable[[SuspiciousEvent], None]] = []

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> None:
        """Load debounce parameters from YAML configuration."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            debounce_cfg = data.get("debounce", {})
            self.cooldown_seconds = float(debounce_cfg.get("cooldown_seconds", self.cooldown_seconds))
            logger.info(f"EventManager alert cooldown set to {self.cooldown_seconds}s")
        except Exception as e:
            logger.warning(f"Could not load debounce config from {config_path}: {e}")

    def add_listener(self, callback: Callable[[SuspiciousEvent], None]) -> None:
        """Register a callback for newly created events."""
        self._listeners.append(callback)

    def process_assessment(
        self,
        assessment: RiskAssessment,
        timestamp: float,
        camera_id: Optional[str] = None,
    ) -> Optional[SuspiciousEvent]:
        """Check if an assessment warrants a new suspicious event, respecting cooldown."""
        if assessment.risk_level in ("NORMAL", "LOW"):
            # Only trigger formal suspicious events for MEDIUM and HIGH risk levels
            # (LOW level can still be logged or viewed in real-time stats)
            return None

        primary = assessment.primary_rule
        if primary is None:
            return None

        track_id = assessment.track_id
        event_type = primary.rule_id
        cam_id = camera_id or self.camera_id

        # Cooldown check: prevent duplicate spam for the same student & behavior
        key = (track_id, event_type)
        last_time = self._last_alert_time.get(key, 0.0)
        if (timestamp - last_time) < self.cooldown_seconds:
            # Within debounce window, suppress duplicate alert
            return None

        # Create new SuspiciousEvent
        event_id = str(uuid.uuid4())
        event = SuspiciousEvent(
            event_id=event_id,
            track_id=track_id,
            camera_id=cam_id,
            timestamp=timestamp,
            start_time=primary.timestamp,
            end_time=timestamp,
            event_type=event_type,
            risk_level=assessment.risk_level,
            score=assessment.score,
            evidence={
                "rule_name": primary.rule_name,
                "rule_score": primary.score,
                "rule_evidence": primary.evidence,
                "total_risk_score": assessment.score,
                "all_triggered_rules": [m.rule_name for m in assessment.all_rules],
            },
            snapshot_path=None,
            clip_path=None,
            status="new",
        )

        self._events[event_id] = event
        self._last_alert_time[key] = timestamp

        logger.warning(
            f"[SUSPICIOUS EVENT] ID={event_id[:8]} | Student #{track_id} | "
            f"Type='{primary.rule_name}' | Risk={assessment.risk_level} (Score={assessment.score})"
        )

        # Notify registered listeners
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Error in event listener: {e}")

        return event

    def get_event(self, event_id: str) -> Optional[SuspiciousEvent]:
        """Get event by ID."""
        return self._events.get(event_id)

    def list_events(
        self,
        risk_level: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[SuspiciousEvent]:
        """List events with optional filtering, sorted by newest first."""
        events = list(self._events.values())

        if risk_level:
            events = [e for e in events if e.risk_level.upper() == risk_level.upper()]
        if status:
            events = [e for e in events if e.status.lower() == status.lower()]

        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def update_event_status(
        self,
        event_id: str,
        new_status: str,
        reviewer_notes: Optional[str] = None,
    ) -> bool:
        """Update event review status by human invigilator."""
        event = self._events.get(event_id)
        if not event:
            return False

        valid_statuses = ("new", "reviewed", "confirmed", "confirmed_event", "dismissed")
        norm_status = new_status.lower().replace("-", "_")
        if norm_status not in valid_statuses:
            raise ValueError(f"Invalid status '{new_status}'. Allowed: {valid_statuses}")

        event.status = norm_status
        if reviewer_notes is not None:
            event.reviewer_notes = reviewer_notes

        logger.info(
            f"Event {event_id[:8]} status updated to '{event.status}' by invigilator. "
            f"(Notes: {reviewer_notes})"
        )
        return True
