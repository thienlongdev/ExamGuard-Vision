"""Configurable temporal rule engine for suspicious behavior evaluation."""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional
import yaml

from src.analysis.temporal_buffer import TemporalBuffer

logger = logging.getLogger(__name__)


@dataclass
class RuleMatch:
    """Represents a matched behavior rule for a specific student."""
    rule_id: str
    rule_name: str
    track_id: int
    severity: str          # 'low', 'medium', 'high'
    score: float           # Rule base score (0 - 100)
    evidence: Dict[str, Any]
    timestamp: float


@dataclass
class RuleDefinition:
    """Configured rule parameters."""
    rule_id: str
    name: str
    behavior: Optional[str] = None
    min_duration_seconds: float = 0.0
    min_occurrences: int = 0
    window_seconds: float = 30.0
    require_phone: bool = False
    severity: str = "medium"
    score: float = 50.0
    description: str = ""


class BehaviorRuleEngine:
    """Evaluates temporal evidence in the sliding window against configured rules."""

    def __init__(self, rules_config_path: Optional[str] = None):
        self.rules: Dict[str, RuleDefinition] = {}
        if rules_config_path:
            self.load_rules(rules_config_path)

    def load_rules(self, config_path: str) -> None:
        """Load rules definitions from YAML file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            raw_rules = data.get("rules", {})
            self.rules.clear()

            for r_id, r_cfg in raw_rules.items():
                self.rules[r_id] = RuleDefinition(
                    rule_id=r_id,
                    name=r_cfg.get("name", r_id),
                    behavior=r_cfg.get("behavior"),
                    min_duration_seconds=float(r_cfg.get("min_duration_seconds", 0.0)),
                    min_occurrences=int(r_cfg.get("min_occurrences", 0)),
                    window_seconds=float(r_cfg.get("window_seconds", 30.0)),
                    require_phone=bool(r_cfg.get("require_phone", False)),
                    severity=r_cfg.get("severity", "medium").lower(),
                    score=float(r_cfg.get("score", 50.0)),
                    description=r_cfg.get("description", ""),
                )
            logger.info(f"Loaded {len(self.rules)} rules from '{config_path}'")
        except Exception as e:
            logger.error(f"Failed to load rules from '{config_path}': {e}")
            raise

    def add_rule(self, rule: RuleDefinition) -> None:
        """Add or override a rule definition programmatically."""
        self.rules[rule.rule_id] = rule

    def evaluate_track(
        self,
        track_id: int,
        temporal_buffer: TemporalBuffer,
        timestamp: float,
    ) -> List[RuleMatch]:
        """Evaluate all loaded rules against a single student's temporal history.

        Args:
            track_id: Student track ID.
            temporal_buffer: Historical temporal buffer.
            timestamp: Current timestamp in seconds.

        Returns:
            List of RuleMatch objects for all triggered rules.
        """
        matches: List[RuleMatch] = []

        for rule in self.rules.values():
            triggered = False
            evidence: Dict[str, Any] = {}

            # Check phone requirement if specified
            has_recent_phone = temporal_buffer.recent_phone_presence(
                track_id=track_id,
                window_seconds=rule.window_seconds,
            )

            if rule.require_phone and not has_recent_phone:
                continue

            # 1. Duration-based rule evaluation
            if rule.min_duration_seconds > 0 and rule.behavior:
                duration = temporal_buffer.duration_behavior(
                    track_id=track_id,
                    behavior=rule.behavior,
                    window_seconds=rule.window_seconds,
                )
                evidence["measured_duration_seconds"] = round(duration, 2)
                evidence["threshold_duration_seconds"] = rule.min_duration_seconds
                evidence["behavior"] = rule.behavior

                if duration >= rule.min_duration_seconds:
                    triggered = True

            # 2. Frequency/occurrence-based rule evaluation
            if rule.min_occurrences > 0 and rule.behavior:
                count = temporal_buffer.count_behavior(
                    track_id=track_id,
                    behavior=rule.behavior,
                    window_seconds=rule.window_seconds,
                )
                evidence["measured_occurrences"] = count
                evidence["threshold_occurrences"] = rule.min_occurrences
                evidence["behavior"] = rule.behavior

                if count >= rule.min_occurrences:
                    triggered = True

            if triggered:
                if has_recent_phone:
                    evidence["phone_detected_nearby"] = True

                matches.append(
                    RuleMatch(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        track_id=track_id,
                        severity=rule.severity,
                        score=rule.score,
                        evidence=evidence,
                        timestamp=timestamp,
                    )
                )

        return matches
