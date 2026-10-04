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

    EVENT_MIN_SEVERITY_ACTIVE = {
        EventFamily.SUSTAINED_HEAD_REST.value: RiskLevel.MEDIUM.value,
        EventFamily.STANDING.value: RiskLevel.MEDIUM.value,
        EventFamily.DISCUSSION_CANDIDATE.value: RiskLevel.MEDIUM.value,
        EventFamily.PHONE_ASSOCIATED.value: RiskLevel.MEDIUM.value,
        EventFamily.PHONE_VISIBLE_UNASSOCIATED.value: RiskLevel.MEDIUM.value,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            from pathlib import Path
            import yaml
            p = Path("configs/v4d_fusion.yaml")
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
        self.config = config or {}
        risk_cfg = self.config.get("risk_policy", {})
        thresh = risk_cfg.get("thresholds", {})
        self.low_to_medium = float(thresh.get("low_to_medium", 40.0))
        self.medium_to_high = float(thresh.get("medium_to_high", 75.0))
        self.max_single_frame_score = float(risk_cfg.get("max_single_frame_score", 25.0))
        self.phone_high_min_duration_sec = float(risk_cfg.get("phone_high_min_duration_sec", 2.5))
        turn_cfg = self.config.get("provisional_thresholds", {}).get("sustained_lateral_head_orientation", {})
        self.sustained_turn_duration_sec = float(turn_cfg.get("sustained_turn_min_duration_sec", 1.2))
        self.event_min_severity_active = dict(self.EVENT_MIN_SEVERITY_ACTIVE)
        if "event_min_severity_active" in risk_cfg:
            self.event_min_severity_active.update(risk_cfg["event_min_severity_active"])

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
            EventFamily.SUSTAINED_HEAD_REST.value: 35.0,
            EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION.value: 35.0,
            EventFamily.PHONE_ASSOCIATED.value: 50.0,
            EventFamily.PHONE_VISIBLE_UNASSOCIATED.value: 40.0,
            EventFamily.PHONE_VISUAL_CANDIDATE.value: 20.0,
            EventFamily.DISCUSSION_CANDIDATE.value: 35.0,
            EventFamily.STANDING.value: 35.0,
            EventFamily.MULTI_CUE_ATTENTION_SHIFT.value: 30.0,
        }
        base = base_scores.get(ev_type, 35.0)

        # 1. Single-cue / short-duration safety
        # Before temporal confirmation (e.g. transient spike < 0.35s), remains LOW
        if duration < 0.35:
            event.risk_score = min(base, self.max_single_frame_score)
            event.risk_level = RiskLevel.LOW.value
            event.review_status = "internal"
            if "risk" in getattr(event, "observation_snapshot", {}):
                event.observation_snapshot["risk"]["score"] = round(event.risk_score, 1)
                event.observation_snapshot["risk"]["level"] = event.risk_level
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

        # -------------------------------------------------------------
        # 6. Reviewability Policies per Event Family
        # -------------------------------------------------------------
        # A. PHONE_VISUAL_CANDIDATE: Internal candidate only (does NOT create review card)
        if ev_type == EventFamily.PHONE_VISUAL_CANDIDATE.value:
            final_score = min(final_score, self.low_to_medium - 10.0)
            event.risk_score = round(final_score, 2)
            event.risk_level = RiskLevel.LOW.value
            event.review_status = "internal"
            event.evidence_summary["is_reviewable"] = False
            event.evidence_summary["candidate_classification"] = "WEAK_VISUAL_CANDIDATE_INTERNAL"
            if "risk" in getattr(event, "observation_snapshot", {}):
                event.observation_snapshot["risk"]["score"] = round(event.risk_score, 1)
                event.observation_snapshot["risk"]["level"] = event.risk_level
            return event

        # B. MULTI_CUE_ATTENTION_SHIFT: Escalation metadata only (never duplicate review card)
        if ev_type == EventFamily.MULTI_CUE_ATTENTION_SHIFT.value:
            final_score = min(final_score, self.low_to_medium - 5.0)
            event.risk_score = round(final_score, 2)
            event.risk_level = RiskLevel.LOW.value
            event.review_status = "internal"
            event.evidence_summary["is_reviewable"] = False
            event.evidence_summary["review_policy"] = "INTERNAL_ESCALATION_METADATA_ONLY"
            if "risk" in getattr(event, "observation_snapshot", {}):
                event.observation_snapshot["risk"]["score"] = round(event.risk_score, 1)
                event.observation_snapshot["risk"]["level"] = event.risk_level
            return event

        # C. SUSTAINED_LATERAL_HEAD_ORIENTATION: Reviewability layer
        if ev_type == EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION.value:
            active_dur = float(event.evidence_summary.get("active_evidence_duration", duration))
            is_sustained = (active_dur >= self.sustained_turn_duration_sec)
            is_repeated = bool(
                event.evidence_summary.get("glance_burst", False) or
                event.evidence_summary.get("glance_count", 0) >= 3
            )
            is_multi_cue = (independent_cues_count >= 2)

            if is_sustained or is_repeated or is_multi_cue:
                # Reviewable event: clear sustained lateral orientation or repeated glances
                if final_score < self.low_to_medium:
                    final_score = self.low_to_medium
                event.review_status = "awaiting"
                event.evidence_summary["is_reviewable"] = True
                event.evidence_summary["glance_classification"] = (
                    "SUSTAINED_LATERAL_TURN" if is_sustained else ("REPEATED_GLANCES" if is_repeated else "MULTI_CUE_TURN")
                )
            else:
                # Brief isolated glance (< sustained_turn_duration_sec, no repetition): internal cue only!
                final_score = min(final_score, self.low_to_medium - 5.0)
                event.risk_score = round(final_score, 2)
                event.risk_level = RiskLevel.LOW.value
                event.review_status = "internal"
                event.evidence_summary["is_reviewable"] = False
                event.evidence_summary["glance_classification"] = "BRIEF_ISOLATED_GLANCE"
                if "risk" in getattr(event, "observation_snapshot", {}):
                    event.observation_snapshot["risk"]["score"] = round(event.risk_score, 1)
                    event.observation_snapshot["risk"]["level"] = event.risk_level
                return event

        # D. PHONE_ASSOCIATED: Authoritative Phone High Policy
        is_phone = (ev_type == EventFamily.PHONE_ASSOCIATED.value)
        escalation_reason = "NORMAL_EVALUATION"
        if is_phone:
            event.review_status = "awaiting"
            event.evidence_summary["is_reviewable"] = True
            phone_conf = event.evidence_summary.get("phone_confidence")
            is_cand = event.evidence_summary.get("is_candidate_only", False)
            latest_ev = event.evidence_summary.get("latest_evidence_score", phone_conf or 0.0)
            if is_cand and phone_conf is not None and float(phone_conf) < 0.35:
                # Weak candidate that got promoted: capped strictly at MEDIUM (< 65.0), never HIGH!
                final_score = min(final_score, 60.0)
                escalation_reason = "PHONE_WEAK_CANDIDATE_CONFIRMED"
            elif independent_cues_count >= 2 and final_score >= self.medium_to_high:
                escalation_reason = "MULTI_CUE_CONCURRENCE"
            elif duration >= self.phone_high_min_duration_sec and mean_reliability >= 0.70:
                # Sustained clear phone evidence confirmed: escalate to HIGH
                sustained_boost = max(0.0, self.medium_to_high - final_score + 5.0)
                final_score = min(100.0, final_score + sustained_boost)
                escalation_reason = "SUSTAINED_PHONE_TEMPORAL_CONFIRMED"
            else:
                # Under threshold: phone alone capped strictly below HIGH band (< 75.0)
                if final_score >= self.medium_to_high:
                    final_score = self.medium_to_high - 1.0
                escalation_reason = "PHONE_SINGLE_CUE_UNDER_THRESHOLD"

        # D2. PHONE_VISIBLE_UNASSOCIATED: Room-level unassociated phone cue policy
        if ev_type == EventFamily.PHONE_VISIBLE_UNASSOCIATED.value:
            phone_conf = event.evidence_summary.get("phone_confidence", 0.0)
            if float(phone_conf or 0.0) >= 0.55 and duration >= 1.5:
                event.review_status = "awaiting"
                event.evidence_summary["is_reviewable"] = True
            else:
                final_score = min(final_score, self.low_to_medium - 10.0)
                event.risk_score = round(final_score, 2)
                event.risk_level = RiskLevel.LOW.value
                event.review_status = "internal"
                event.evidence_summary["is_reviewable"] = False
                if "risk" in getattr(event, "observation_snapshot", {}):
                    event.observation_snapshot["risk"]["score"] = round(event.risk_score, 1)
                    event.observation_snapshot["risk"]["level"] = event.risk_level
                return event

        # 7. Event-Type Minimum Severity Floor once ACTIVE:
        # Sustained behaviors that passed temporal candidate confirmation must not display as LOW.
        min_sev = self.event_min_severity_active.get(ev_type)
        if min_sev == RiskLevel.MEDIUM.value:
            if final_score < self.low_to_medium:
                final_score = self.low_to_medium
        elif min_sev == RiskLevel.HIGH.value:
            if final_score < self.medium_to_high:
                final_score = self.medium_to_high

        # 8. Deterministic Single-Source-of-Truth Risk Level Assignment
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
            "escalation_reason": escalation_reason,
        }

        if "risk" in getattr(event, "observation_snapshot", {}):
            event.observation_snapshot["risk"]["score"] = round(event.risk_score, 1)
            event.observation_snapshot["risk"]["level"] = event.risk_level

        # If event is closed, increment recurrence counter
        if event.status == "closed":
            self._recurrence_counter[track_id] = past_events + 1

        return event

    def reset(self):
        """Reset recurrence history."""
        self._recurrence_counter.clear()
