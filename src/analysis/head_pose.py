"""Optional head pose estimation interface and implementation.

Crucial Architectural Rule:
Head pose estimation is strictly OPTIONAL. Wall-mounted CCTV cameras often capture
small student faces (<50px) where gaze/face landmarks are unreliable. The pipeline
must function completely whether head pose is enabled, disabled, or unavailable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Optional
import numpy as np

from src.detection.types import BBox

logger = logging.getLogger(__name__)


@dataclass
class HeadPoseResult:
    """Estimated head orientation angles in degrees."""
    pitch: float       # Vertical tilt (negative: looking down, positive: looking up)
    yaw: float         # Horizontal turn (negative: left, positive: right)
    roll: float        # Lateral tilt
    orientation: str   # 'center', 'left', 'right', 'down', 'up'


class HeadPoseEstimator(ABC):
    """Abstract interface for head pose estimation."""

    @abstractmethod
    def estimate(self, frame: np.ndarray, person_bbox: BBox) -> Optional[HeadPoseResult]:
        """Estimate head pose for a person crop.

        Returns:
            HeadPoseResult or None if disabled, face too small, or landmarks missing.
        """
        pass


class OptionalHeadPoseEstimator(HeadPoseEstimator):
    """Optional adapter that gracefully disables head pose or applies lightweight estimation."""

    def __init__(
        self,
        enabled: bool = False,
        min_face_size_pixels: int = 60,
        yaw_threshold: float = 35.0,
        pitch_threshold_down: float = -20.0,
    ):
        self.enabled = enabled
        self.min_face_size = min_face_size_pixels
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold_down = pitch_threshold_down

        if not self.enabled:
            logger.info("HeadPoseEstimator is DISABLED by configuration (standard for CCTV).")
        else:
            logger.info("HeadPoseEstimator is ENABLED.")

    def estimate(self, frame: np.ndarray, person_bbox: BBox) -> Optional[HeadPoseResult]:
        """Estimate head pose if enabled and person box is sufficiently large."""
        if not self.enabled:
            return None

        # Check if person box is large enough to contain a resolvable face
        # Upper 1/3 of person bbox is approximately the head region
        head_h = person_bbox.height * 0.35
        head_w = person_bbox.width * 0.50

        if head_h < self.min_face_size or head_w < self.min_face_size:
            # Face too small for reliable landmarks (typical CCTV scenario)
            return None

        # If MediaPipe or OpenCV face landmark model is added in future phases,
        # it executes here. When absent, returns None safely.
        return None
