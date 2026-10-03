"""Multi-Object Tracking abstraction layer."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.detection.types import BBox, Detection


@dataclass
class Track:
    """Normalized multi-object track state with persistent track ID."""
    track_id: int
    bbox: BBox
    confidence: float
    timestamp: float
    class_id: int = 0
    class_name: str = "person"
    lost: bool = False


class BaseTracker(ABC):
    """Abstract interface for student tracking (ByteTrack, BoT-SORT, etc.)."""

    @abstractmethod
    def update(
        self,
        detections: List[Detection],
        timestamp: float,
        frame_shape: Tuple[int, int],
    ) -> List[Track]:
        """Update tracker state with new frame detections.

        Args:
            detections: Normalized detections for this frame (usually filtered to 'person').
            timestamp: Current frame timestamp in seconds.
            frame_shape: (height, width) of the frame.

        Returns:
            List of active Track objects with persistent track_ids.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal tracking state (e.g. on stream reconnect or session start)."""
        pass
