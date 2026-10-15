"""Webcam video source implementation for local development and demos."""

import logging
import time
import threading
from collections import deque
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
        preferred_backend: Optional[str] = None,
        fourcc: Optional[str] = None,
        async_capture: bool = False,
    ):
        super().__init__(source_id=source_id, source_type="webcam")
        self.source_origin = "PHYSICAL_LIVE_CAMERA"
        self.device_index = int(source) if isinstance(source, str) and source.isdigit() else (source if isinstance(source, int) else 0)
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self.preferred_backend = preferred_backend
        self.requested_fourcc = fourcc
        self.actual_fourcc = "UNKNOWN"
        self.async_capture = async_capture
        self.identity_check = None
        self.failure_reason = None
        self._capture_stop = threading.Event()
        self._condition = threading.Condition()
        self._capture_thread = None
        self._latest = None
        self._consumed_idx = 0
        self.skipped_frames = 0
        self.capture_times = deque(maxlen=120)
        self._cap: Optional[cv2.VideoCapture] = None
        self._actual_width = width
        self._actual_height = height
        self._actual_fps = fps
        self._backend_name = "NONE"

    def open(self) -> bool:
        """Open the webcam device with backend fallback."""
        if self._cap is not None and self._cap.isOpened():
            return True

        if self.identity_check is not None and not self.identity_check():
            self.failure_reason = "Saved physical camera identity changed. Explicit selection required."
            logger.error(self.failure_reason)
            return False

        logger.info(f"Opening webcam device index={self.device_index} (id={self.source_id})")
        
        # On Windows, hardware benchmarks confirmed CAP_MSMF achieves ~30 FPS at 1280x720,
        # whereas CAP_DSHOW locks to ~7.5 FPS YUY2. Priority: preferred -> CAP_MSMF -> CAP_DSHOW -> CAP_ANY.
        candidate_backends = []
        if self.preferred_backend:
            pb = self.preferred_backend.upper().strip()
            if pb == "CAP_MSMF" and hasattr(cv2, "CAP_MSMF"):
                candidate_backends.append(("CAP_MSMF", cv2.CAP_MSMF))
            elif pb == "CAP_DSHOW" and hasattr(cv2, "CAP_DSHOW"):
                candidate_backends.append(("CAP_DSHOW", cv2.CAP_DSHOW))

        if hasattr(cv2, "CAP_MSMF") and ("CAP_MSMF", cv2.CAP_MSMF) not in candidate_backends:
            candidate_backends.append(("CAP_MSMF", cv2.CAP_MSMF))
        if hasattr(cv2, "CAP_DSHOW") and ("CAP_DSHOW", cv2.CAP_DSHOW) not in candidate_backends:
            candidate_backends.append(("CAP_DSHOW", cv2.CAP_DSHOW))
        candidate_backends.append(("CAP_ANY", cv2.CAP_ANY))

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
        if self.requested_fourcc:
            # Verified DSHOW order on the owner's UVC camera. Check the accepted values below.
            code = cv2.VideoWriter_fourcc(*self.requested_fourcc)
            self._cap.set(cv2.CAP_PROP_FOURCC, code)
            if int(self._cap.get(cv2.CAP_PROP_FOURCC)) != code:
                # Some drivers need FOURCC before dimensions; retry on the same exclusive handle.
                self._cap.set(cv2.CAP_PROP_FOURCC, code)
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
                self._cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
            actual = int(self._cap.get(cv2.CAP_PROP_FOURCC))
            self.actual_fourcc = "".join(chr((actual >> (8 * i)) & 255) for i in range(4))
            ok, first = self._cap.read()
            if (not ok or first is None or first.shape[:2] != (self.requested_height, self.requested_width)
                    or self.actual_fourcc != self.requested_fourcc):
                logger.error("Requested camera mode rejected: %sx%s %s; accepted FOURCC=%s", self.requested_width, self.requested_height, self.requested_fourcc, self.actual_fourcc)
                self.release()
                return False

        # Read actual properties
        self._actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.requested_width
        self._actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.requested_height
        detected_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._actual_fps = detected_fps if detected_fps and detected_fps > 0 else self.requested_fps

        self._frame_count = 0
        self._start_time = time.time()
        self._last_timestamp = -1.0
        self.failure_reason = None
        self.capture_times.clear()
        self._latest = None
        self._consumed_idx = 0
        self.skipped_frames = 0
        logger.info(
            f"Webcam opened via {self._backend_name}: {self._actual_width}x{self._actual_height} @ {self._actual_fps:.1f} FPS"
        )
        if self.async_capture:
            self._capture_stop.clear()
            self._capture_thread = threading.Thread(target=self._capture_worker, name=f"capture-{self.source_id}", daemon=True)
            self._capture_thread.start()
        return True

    def read(self) -> Optional[VideoFrame]:
        """Capture and return the next frame from webcam with strictly monotonic timestamps."""
        if self.async_capture:
            with self._condition:
                self._condition.wait_for(lambda: self.failure_reason or self._capture_stop.is_set() or
                                         (self._latest is not None and self._latest.frame_idx > self._consumed_idx), timeout=2.0)
                if self.failure_reason or self._latest is None or self._latest.frame_idx <= self._consumed_idx:
                    return None
                frame = self._latest
                self.skipped_frames += max(0, frame.frame_idx - self._consumed_idx - 1)
                self._consumed_idx = frame.frame_idx
                return frame
        return self._read_capture()

    def _capture_worker(self):
        while not self._capture_stop.is_set():
            frame = self._read_capture()
            with self._condition:
                if frame is None:
                    self.failure_reason = "Camera stopped delivering frames; monitoring interrupted. Reconnect and explicitly restart the session."
                    self._condition.notify_all()
                    break
                self._latest = frame
                self._condition.notify_all()

    @property
    def observed_capture_fps(self):
        times = list(self.capture_times)
        return (len(times)-1)/(times[-1]-times[0]) if len(times) > 4 and times[-1] > times[0] else None

    def _read_capture(self) -> Optional[VideoFrame]:
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
        self.capture_times.append(time.perf_counter())

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
        self._capture_stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=3.0)
            if self._capture_thread.is_alive():
                raise RuntimeError("Camera driver read blocked; refusing to release a handle still in use.")
            self._capture_thread = None
        self._latest = None
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

    @property
    def backend_name(self) -> str:
        return getattr(self, "_backend_name", "NONE")
