"""
V4D Risk Aggregation & Evidence Scorer
=====================================
Maps observable event durations, independent cue concurrence, reliability weights,
and recurrence into auditable risk scores (0-100) and risk levels (LOW, MEDIUM, HIGH).

Strict Governance Rules:
- Risk is NOT probability of guilt or cheating.
- Single noisy frames or isolated glitches can NEVER escalate directly to HIGH.
- Correlated cues are discounted to prevent artificial risk inflation.
- Recurrence across time incrementally elevates risk.
"""

from typing import Dict, List, Optional, Any
from src.fusion.types import FusedEvent, RiskLevel, EventFamily


class RiskAggregator:
    """Aggregates observable multi-cue evidence into configured risk scores (0-100) and provisional risk levels."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        risk_cfg = self.config.get("risk_policy", {})
        thresh = risk_cfg.get("thresholds", {})
        self.low_to_medium = float(thresh.get("low_to_medium", 40.0))
        self.medium_to_high = float(thresh.get("medium_to_high", 75.0))
        self.max_single_frame_score = float(risk_cfg.get("max_single_frame_score", 25.0))

        # Track event recurrence: track_id -> count of historical events
        self._recurrence_counter: Dict[int, int] = {}

    def assess_event_risk(
        self,
        event: FusedEvent,
        active_cues_count: int = 1,
        independent_cues_count: int = 1,
        mean_reliability: float = 1.0,
    ) -> FusedEvent:
        """Calculate and attach risk_score and risk_level to a FusedEvent."""
        duration = max(0.0, event.duration)
        track_id = event.track_id
        ev_type = event.event_type

        # Base score by event family (Deterministic evidence priority policy)
        # Note: 0-100 score reflects auditable evidence strength, NOT probability of cheating.
        base_scores = {
            EventFamily.SUSTAINED_HEAD_REST.value: 25.0,
            EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION.value: 25.0,
            EventFamily.PHONE_ASSOCIATED.value: 65.0,  # Elevated so sustained clear phone resolves to HIGH (>= 75)
            EventFamily.DISCUSSION_CANDIDATE.value: 30.0,
            EventFamily.STANDING.value: 25.0,
            EventFamily.MULTI_CUE_ATTENTION_SHIFT.value: 35.0,
        }
        base = base_scores.get(ev_type, 25.0)

        # 1. Single-cue / short-duration safety
        if duration < 0.5:
            event.risk_score = min(base, self.max_single_frame_score)
            event.risk_level = RiskLevel.LOW.value
            return event

        # 2. Duration factor (logarithmic saturation)
        import math
        duration_factor = min(25.0, 10.0 * math.log1p(duration))

        # 3. Independent multi-cue concurrence bonus
        concurrence_bonus = 0.0
        if independent_cues_count >= 2:
            concurrence_bonus = 20.0 * (independent_cues_count - 1)

        # 4. Recurrence bonus
        past_events = self._recurrence_counter.get(track_id, 0)
        recurrence_bonus = min(15.0, past_events * 5.0)

        # 5. Reliability scaling
        raw_score = (base + duration_factor + concurrence_bonus + recurrence_bonus) * mean_reliability
        final_score = max(0.0, min(100.0, raw_score))

        # Assign Risk Level
        if final_score >= self.medium_to_high:
            level = RiskLevel.HIGH.value
        elif final_score >= self.low_to_medium:
            level = RiskLevel.MEDIUM.value
        else:
            level = RiskLevel.LOW.value

        event.risk_score = round(final_score, 2)
        event.risk_level = level
        event.evidence_summary["risk_assessment"] = {
            "base_score": base,
            "duration_factor": round(duration_factor, 2),
            "concurrence_bonus": concurrence_bonus,
            "recurrence_bonus": recurrence_bonus,
            "mean_reliability": round(mean_reliability, 3),
            "final_score": event.risk_score,
            "risk_level": event.risk_level,
        }

        # If event is closed, increment recurrence counter
        if event.status == "closed":
            self._recurrence_counter[track_id] = past_events + 1

        return event

    def reset(self):
        """Reset recurrence history."""
        self._recurrence_counter.clear()
