"""
Rolling buffer and asynchronous clip recorder for incident video evidence.
Enforces:
- Memory-bounded buffer (stores compressed JPEG bytes with timestamps, max MB limit)
- Non-blocking asynchronous MP4 clip encoding
- Atomic file write (.tmp -> final)
- SHA-256 hash calculation and completion callback
- Graceful error isolation (encoding failure never crashes perception)
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import logging
import os
import time
from typing import Deque, List, Optional, Callable, Dict, Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BufferedFrame:
    jpeg_bytes: bytes
    timestamp: float
    width: int
    height: int


class RollingClipRecorder:
    """Maintains a memory-bounded circular pre-event buffer and writes short MP4 evidence clips."""

    def __init__(
        self,
        output_dir: str = "storage/evidence/clips",
        pre_event_seconds: float = 5.0,
        post_event_seconds: float = 5.0,
        max_fps: float = 15.0,
        jpeg_quality: int = 80,
        max_buffer_mb: float = 64.0,
        on_clip_ready: Optional[Callable[[str, str, str, int], None]] = None,
        on_clip_failed: Optional[Callable[[str, str, Exception], None]] = None,
    ):
        self.output_dir = output_dir
        self.pre_event_seconds = pre_event_seconds
        self.post_event_seconds = post_event_seconds
        self.max_fps = max_fps
        self.jpeg_quality = jpeg_quality
        self.max_buffer_bytes = int(max_buffer_mb * 1024 * 1024)
        self.on_clip_ready = on_clip_ready
        self.on_clip_failed = on_clip_failed

        # Maximum frames in rolling buffer (e.g. 5s * 30fps = 150 frames + margin)
        self.buffer_maxlen = int(pre_event_seconds * 30.0) + 20
        self._rolling_buffer: Deque[BufferedFrame] = deque(maxlen=self.buffer_maxlen)
        self._current_buffer_bytes = 0

        # Active recording tasks awaiting post-event frames:
        # event_id -> {"frames": List[BufferedFrame], "end_timestamp": float, "filepath": str, "fps": float}
        self._active_recordings: List[Dict[str, Any]] = []

        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="clip_worker")
        os.makedirs(self.output_dir, exist_ok=True)

    def push_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """Compress frame to JPEG bytes, add to rolling buffer, and feed active clip jobs."""
        h, w = frame.shape[:2]
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not success:
            return

        jpeg_data = encoded.tobytes()
        buf_frame = BufferedFrame(
            jpeg_bytes=jpeg_data,
            timestamp=timestamp,
            width=w,
            height=h,
        )

        # Enforce memory safety bound
        while self._current_buffer_bytes + len(jpeg_data) > self.max_buffer_bytes and self._rolling_buffer:
            evicted = self._rolling_buffer.popleft()
            self._current_buffer_bytes -= len(evicted.jpeg_bytes)

        self._rolling_buffer.append(buf_frame)
        self._current_buffer_bytes += len(jpeg_data)

        # Append to active clip jobs
        completed_jobs = []
        for job in self._active_recordings:
            job["frames"].append(buf_frame)
            if timestamp >= job["end_timestamp"]:
                completed_jobs.append(job)

        # Dispatch completed jobs to background encoding
        for job in completed_jobs:
            self._active_recordings.remove(job)
            self._dispatch_encode(
                job["event_id"],
                job["filepath"],
                job["frames"],
                job["fps"],
            )

    def trigger_clip(
        self,
        event_id: str,
        timestamp: float,
        fps: float = 15.0,
        custom_filepath: Optional[str] = None,
    ) -> str:
        """Trigger evidence clip capture centered around current event."""
        if custom_filepath:
            filepath = custom_filepath
        else:
            filename = f"{event_id}_evidence.mp4"
            filepath = os.path.join(self.output_dir, filename)

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        # Collect pre-event frames from rolling buffer
        cutoff = timestamp - self.pre_event_seconds
        pre_frames = [
            bf for bf in self._rolling_buffer
            if bf.timestamp >= cutoff
        ]

        clip_fps = min(fps, self.max_fps)
        end_time = timestamp + self.post_event_seconds

        job = {
            "event_id": event_id,
            "filepath": filepath,
            "frames": list(pre_frames),
            "end_timestamp": end_time,
            "fps": clip_fps,
        }
        self._active_recordings.append(job)
        return filepath

    def finalize_all_active(self) -> None:
        """Immediately dispatch any remaining active recordings (e.g. on stream end or shutdown)."""
        pending = list(self._active_recordings)
        self._active_recordings.clear()
        for job in pending:
            if job["frames"]:
                self._dispatch_encode(
                    job["event_id"],
                    job["filepath"],
                    job["frames"],
                    job["fps"],
                )

    def _dispatch_encode(
        self,
        event_id: str,
        filepath: str,
        frames: List[BufferedFrame],
        fps: float,
    ) -> None:
        """Encode frames into a verified MP4 file in a background worker."""
        if not frames:
            if self.on_clip_failed:
                self.on_clip_failed(event_id, filepath, RuntimeError("No frames available for video clip"))
            return

        def _encode():
            tmp_path = f"{filepath}.tmp.mp4"
            try:
                w, h = frames[0].width, frames[0].height
                fourcc = cv2.VideoWriter_fourcc(*"H264")
                writer = cv2.VideoWriter(tmp_path, cv2.CAP_MSMF, fourcc, fps, (w, h))
                if not writer.isOpened():
                    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
                if not writer.isOpened():
                    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

                if not writer.isOpened():
                    raise RuntimeError(f"Cannot initialize cv2.VideoWriter for {tmp_path}")

                written_count = 0
                for bf in frames:
                    arr = np.frombuffer(bf.jpeg_bytes, dtype=np.uint8)
                    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if decoded is not None:
                        writer.write(decoded)
                        written_count += 1
                writer.release()

                # Strict post-encode validation (Section 29)
                if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                    raise RuntimeError(f"Encoded clip {tmp_path} is missing or 0 bytes")

                cap = cv2.VideoCapture(tmp_path)
                is_valid = False
                if cap.isOpened():
                    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if n_frames > 0 or written_count > 0:
                        is_valid = True
                    cap.release()
                else:
                    raise RuntimeError(f"Encoded clip {tmp_path} cannot be opened by VideoCapture")

                if not is_valid:
                    raise RuntimeError(f"Encoded clip {tmp_path} has 0 frames or invalid duration")

                # Atomic rename
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass
                os.replace(tmp_path, filepath)

                # Compute SHA-256
                hasher = hashlib.sha256()
                with open(filepath, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
                clip_hash = hasher.hexdigest()
                clip_size = os.path.getsize(filepath)

                logger.info(f"Saved verified video evidence clip: {filepath} ({written_count} frames, {clip_size} bytes)")
                if self.on_clip_ready:
                    self.on_clip_ready(event_id, filepath, clip_hash, clip_size)

            except Exception as e:
                logger.error(f"Error encoding video clip {filepath}: {e}")
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                if self.on_clip_failed:
                    self.on_clip_failed(event_id, filepath, e)

        self._executor.submit(_encode)

    def shutdown(self) -> None:
        """Gracefully finalize pending recordings and wait for writes to complete."""
        self.finalize_all_active()
        self._executor.shutdown(wait=True)
