"""RTSP / IP CCTV camera source implementation with robust auto-reconnection."""

import logging
import os
import time
from typing import Optional
import cv2
import numpy as np

from src.video.base import VideoFrame, VideoSource

logger = logging.getLogger(__name__)


class RTSPSource(VideoSource):
    """VideoSource for RTSP / IP cameras in school examination halls."""

    def __init__(
        self,
        rtsp_url: str,
        source_id: str = "cctv-rtsp",
        max_reconnect_attempts: int = 10,
        initial_reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 30.0,
        expected_fps: float = 25.0,
    ):
        super().__init__(source_id=source_id, source_type="rtsp")
        self.source_origin = "RTSP_STREAM"
        self.rtsp_url = rtsp_url
        self.max_reconnect_attempts = max_reconnect_attempts
        self.initial_reconnect_delay = initial_reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.expected_fps = expected_fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._fps = expected_fps
        self._width = 1920
        self._height = 1080
        self._reconnect_count = 0

    def open(self) -> bool:
        """Connect to RTSP stream."""
        if self._cap is not None and self._cap.isOpened():
            return True

        # Use TCP transport for RTSP stability if supported by environment
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        logger.info(f"Connecting to RTSP stream {self.source_id}...")
        self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        if not self._cap.isOpened():
            logger.error(f"Failed to connect to RTSP stream: {self.rtsp_url}")
            return False

        detected_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = detected_fps if detected_fps and detected_fps > 0 else self.expected_fps
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        self._reconnect_count = 0

        logger.info(
            f"RTSP stream connected ({self.source_id}): {self._width}x{self._height} @ {self._fps:.1f} FPS"
        )
        return True

    def _attempt_reconnect(self) -> bool:
        """Reconnect to RTSP stream using exponential backoff."""
        delay = self.initial_reconnect_delay
        for attempt in range(1, self.max_reconnect_attempts + 1):
            logger.warning(
                f"RTSP {self.source_id} connection lost. Reconnect attempt {attempt}/{self.max_reconnect_attempts} in {delay:.1f}s..."
            )
            time.sleep(delay)
            self.release()

            if self.open():
                logger.info(f"RTSP {self.source_id} successfully reconnected on attempt {attempt}")
                return True

            delay = min(delay * 2.0, self.max_reconnect_delay)

        logger.critical(f"RTSP {self.source_id} failed to reconnect after {self.max_reconnect_attempts} attempts.")
        return False

    def read(self) -> Optional[VideoFrame]:
        """Read the next frame from the RTSP stream, handling dropouts."""
        if self._cap is None or not self._cap.isOpened():
            if not self._attempt_reconnect():
                return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning(f"RTSP {self.source_id}: frame read failed. Triggering reconnect.")
            if not self._attempt_reconnect():
                return None
            ret, frame = self._cap.read()
            if not ret or frame is None:
                return None

        current_time = time.time()
        self._frame_count += 1

        return VideoFrame(
            frame=frame,
            timestamp=current_time,
            frame_idx=self._frame_count,
            fps=self._fps,
            width=self._width,
            height=self._height,
            source_id=self.source_id,
        )

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info(f"RTSP source {self.source_id} released.")

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height
