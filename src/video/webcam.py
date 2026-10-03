"""Webcam video source implementation for local development and demos."""

import logging
import time
from typing import Optional, Union
import cv2
import numpy as np

from src.video.base import VideoFrame, VideoSource

logger = logging.getLogger(__name__)


class WebcamSource(VideoSource):
    """VideoSource implementation for USB/laptop webcams."""

    def __init__(
        self,
        source: Union[int, str] = 0,
        source_id: str = "webcam-0",
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
    ):
        super().__init__(source_id=source_id, source_type="webcam")
        self.device_index = int(source) if isinstance(source, str) and source.isdigit() else (source if isinstance(source, int) else 0)
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._actual_width = width
        self._actual_height = height
        self._actual_fps = fps

    def open(self) -> bool:
        """Open the webcam device with backend fallback."""
        if self._cap is not None and self._cap.isOpened():
            return True

        logger.info(f"Opening webcam device index={self.device_index} (id={self.source_id})")
        
        # On Windows, try CAP_DSHOW, CAP_MSMF, then default CAP_ANY
        candidate_backends = [
            ("CAP_DSHOW", cv2.CAP_DSHOW),
            ("CAP_MSMF", cv2.CAP_MSMF),
            ("CAP_ANY", cv2.CAP_ANY),
        ] if hasattr(cv2, "CAP_DSHOW") else [("CAP_ANY", cv2.CAP_ANY)]

        self._cap = None
        self._backend_name = "NONE"

        for b_name, b_flag in candidate_backends:
            try:
                cap = cv2.VideoCapture(self.device_index, b_flag)
                if cap.isOpened():
                    self._cap = cap
                    self._backend_name = b_name
                    break
                cap.release()
            except Exception as e:
                logger.debug(f"Failed backend {b_name} on index {self.device_index}: {e}")

        if self._cap is None or not self._cap.isOpened():
            logger.error(f"Failed to open webcam index {self.device_index} across all candidate backends")
            return False

        # Set requested resolution and FPS
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        if self.requested_fps > 0:
            self._cap.set(cv2.CAP_PROP_FPS, self.requested_fps)

        # Read actual properties
        self._actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.requested_width
        self._actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.requested_height
        detected_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._actual_fps = detected_fps if detected_fps and detected_fps > 0 else self.requested_fps

        self._frame_count = 0
        self._start_time = time.time()
        self._last_timestamp = -1.0
        logger.info(
            f"Webcam opened via {self._backend_name}: {self._actual_width}x{self._actual_height} @ {self._actual_fps:.1f} FPS"
        )
        return True

    def read(self) -> Optional[VideoFrame]:
        """Capture and return the next frame from webcam with strictly monotonic timestamps."""
        if self._cap is None or not self._cap.isOpened():
            if not self.open():
                return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning(f"Webcam {self.source_id}: frame read returned empty")
            return None

        current_time = time.time()
        if hasattr(self, "_last_timestamp") and current_time <= self._last_timestamp:
            current_time = self._last_timestamp + 0.0001
        self._last_timestamp = current_time

        self._frame_count += 1

        return VideoFrame(
            frame=frame,
            timestamp=current_time,
            frame_idx=self._frame_count,
            fps=self._actual_fps,
            width=self._actual_width,
            height=self._actual_height,
            source_id=self.source_id,
        )

    def release(self) -> None:
        """Release the webcam handle."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info(f"Webcam {self.source_id} released.")

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        return self._actual_fps

    @property
    def width(self) -> int:
        return self._actual_width

    @property
    def height(self) -> int:
        return self._actual_height
