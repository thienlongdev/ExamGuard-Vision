"""Video file source implementation for offline development, evaluation, and CI."""

import logging
import os
import time
from typing import Optional
import cv2
import numpy as np

from src.video.base import VideoFrame, VideoSource

logger = logging.getLogger(__name__)


class VideoFileSource(VideoSource):
    """VideoSource for recorded video files (MP4, AVI, MKV, etc.)."""

    def __init__(
        self,
        file_path: str,
        source_id: Optional[str] = None,
        loop: bool = False,
        realtime_pace: bool = True,
    ):
        file_id = source_id or os.path.splitext(os.path.basename(file_path))[0]
        super().__init__(source_id=file_id, source_type="video_file")
        self.source_origin = "VIDEO_FILE"
        self.file_path = file_path
        self.loop = loop
        self.realtime_pace = realtime_pace
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps = 30.0
        self._width = 1280
        self._height = 720
        self._last_read_wall_time = 0.0
        self._simulated_timestamp = 0.0

    def open(self) -> bool:
        """Open the video file."""
        if not os.path.exists(self.file_path):
            normalized = self.file_path.replace("\\", "/")
            if normalized.startswith("samples/"):
                fixture_candidate = normalized.replace("samples/", "tests/fixtures/")
                if os.path.exists(fixture_candidate):
                    self.file_path = fixture_candidate
        if not os.path.exists(self.file_path):
            logger.error(f"Video file not found: {self.file_path}")
            return False

        if self._cap is not None and self._cap.isOpened():
            return True

        self._cap = cv2.VideoCapture(self.file_path)
        if not self._cap.isOpened():
            logger.error(f"Cannot open video file: {self.file_path}")
            return False

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 0 else 30.0
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._total_video_frames = total_frames if total_frames > 0 else 0
        self._source_duration = (self._total_video_frames / self._fps) if self._total_video_frames > 0 else 0.0
        self._frame_count = 0
        self._loop_count = 0
        self._loop_timestamp_offset = 0.0
        self._simulated_timestamp = time.time()
        self._last_read_wall_time = time.time()

        logger.info(
            f"Opened video file '{self.file_path}': {self._width}x{self._height} @ {self._fps:.1f} FPS "
            f"({self._total_video_frames} frames, duration={self._source_duration:.2f}s)"
        )
        return True

    def read(self) -> Optional[VideoFrame]:
        """Read the next frame from the file."""
        if self._cap is None or not self._cap.isOpened():
            if not self.open():
                return None

        # Real-time pacing simulation
        if self.realtime_pace and self._frame_count > 0:
            target_interval = 1.0 / self._fps
            elapsed = time.time() - self._last_read_wall_time
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

        ret, frame = self._cap.read()
        if not ret or frame is None:
            if self.loop:
                self._loop_count += 1
                if self._source_duration > 0.0:
                    self._loop_timestamp_offset = self._loop_count * self._source_duration
                logger.info(
                    f"Looping video file: {self.file_path} (loop_count={self._loop_count}, offset={self._loop_timestamp_offset:.3f}s)"
                )
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    return None
            else:
                logger.info(f"Reached end of video file: {self.file_path}")
                return None

        self._last_read_wall_time = time.time()
        self._frame_count += 1
        self._simulated_timestamp += (1.0 / self._fps)

        return VideoFrame(
            frame=frame,
            timestamp=self._simulated_timestamp,
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
            logger.info(f"Video file source {self.source_id} released.")

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

    @property
    def loop_count(self) -> int:
        return self._loop_count

    @property
    def source_duration(self) -> float:
        return self._source_duration

    @property
    def loop_timestamp_offset(self) -> float:
        return self._loop_timestamp_offset
