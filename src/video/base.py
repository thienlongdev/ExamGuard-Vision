"""Video Source Abstraction Layer.

Defines the common interface for all video sources (webcam, video files, RTSP streams).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Optional
import numpy as np


@dataclass
class VideoFrame:
    """Represents a single decoded video frame with comprehensive metadata."""
    frame: np.ndarray
    timestamp: float        # Monotonic or wall-clock epoch timestamp (seconds)
    frame_idx: int          # Monotonically increasing frame counter
    fps: float              # Nominal or measured FPS
    width: int              # Width in pixels
    height: int             # Height in pixels
    source_id: str          # Identifier of the originating camera/source


class VideoSource(ABC):
    """Abstract Base Class for all video sources.

    Guarantees that downstream processing (detection, tracking, temporal analysis)
    is completely decoupled from whether the source is a laptop webcam, RTSP stream,
    or local video file.
    """

    def __init__(self, source_id: str, source_type: str):
        self.source_id = source_id
        self.source_type = source_type
        self._frame_count = 0
        self._start_time: Optional[float] = None

    @abstractmethod
    def open(self) -> bool:
        """Connect to and initialize the video source.

        Returns:
            True if connection was successful, False otherwise.
        """
        pass

    @abstractmethod
    def read(self) -> Optional[VideoFrame]:
        """Read and decode the next frame from the video stream.

        Returns:
            VideoFrame if successful, or None if the stream ended or failed.
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Release underlying camera/stream resources."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Check whether the video stream is active and readable."""
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        """Nominal or configured frames per second."""
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        """Frame width in pixels."""
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        """Frame height in pixels."""
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
