"""Suspicion risk scoring module.

Maps rule matches and temporal evidence to provisional risk levels (LOW, MEDIUM, HIGH).
Crucial Rule: AI produces risk scores indicating need for human review, NEVER proof of cheating.
"""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple
import yaml

from src.behavior.rules import RuleMatch

logger = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    """Calculated risk score and level for a student track."""
    track_id: int
    score: float           # 0.0 to 100.0
    risk_level: str        # 'LOW', 'MEDIUM', 'HIGH'
    primary_rule: Optional[RuleMatch] = None
    all_rules: List[RuleMatch] = None

    def __post_init__(self):
        if self.all_rules is None:
            self.all_rules = []


class RiskScorer:
    """Calculates risk levels based on triggered rules, weights, and evidence combinations."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        low_threshold: float = 25.0,
        medium_threshold: float = 50.0,
        high_threshold: float = 75.0,
    ):
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> None:
        """Load scoring thresholds from YAML config."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            scoring_cfg = data.get("scoring", {})
            thresh = scoring_cfg.get("thresholds", {})
            self.low_threshold = float(thresh.get("low", self.low_threshold))
            self.medium_threshold = float(thresh.get("medium", self.medium_threshold))
            self.high_threshold = float(thresh.get("high", self.high_threshold))

            logger.info(
                f"RiskScorer loaded: LOW >= {self.low_threshold}, "
                f"MEDIUM >= {self.medium_threshold}, HIGH >= {self.high_threshold}"
            )
        except Exception as e:
            logger.warning(f"Could not load scoring thresholds from {config_path}: {e}")

    def score_matches(self, track_id: int, matches: List[RuleMatch]) -> RiskAssessment:
        """Aggregate rule matches into a total risk score and categorical risk level.

        Args:
            track_id: Student track ID.
            matches: Triggered rules for this student.

        Returns:
            RiskAssessment object.
        """
        if not matches:
            return RiskAssessment(
                track_id=track_id,
                score=0.0,
                risk_level="NORMAL",
                primary_rule=None,
                all_rules=[],
            )

        # Base score is the highest individual rule score
        # Extra simultaneous rule triggers add diminishing cumulative risk (+10% of subsequent rule scores)
        sorted_matches = sorted(matches, key=lambda m: m.score, reverse=True)
        primary = sorted_matches[0]

        total_score = primary.score
        for extra in sorted_matches[1:]:
            total_score += extra.score * 0.15

        total_score = min(100.0, max(0.0, total_score))

        # Categorize into provisional engineering risk levels
        if total_score >= self.high_threshold:
            risk_level = "HIGH"
        elif total_score >= self.medium_threshold:
            risk_level = "MEDIUM"
        elif total_score >= self.low_threshold:
            risk_level = "LOW"
        else:
            risk_level = "NORMAL"

        return RiskAssessment(
            track_id=track_id,
            score=round(total_score, 1),
            risk_level=risk_level,
            primary_rule=primary,
            all_rules=matches,
        )
