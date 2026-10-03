"""Rolling buffer and asynchronous clip recorder for incident video evidence."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import os
import time
from typing import Deque, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BufferedFrame:
    frame: np.ndarray
    timestamp: float


class RollingClipRecorder:
    """Maintains a circular pre-event buffer and writes short MP4 evidence clips asynchronously."""

    def __init__(
        self,
        output_dir: str = "storage/evidence/clips",
        pre_event_seconds: float = 4.0,
        post_event_seconds: float = 4.0,
        max_fps: float = 15.0,
    ):
        self.output_dir = output_dir
        self.pre_event_seconds = pre_event_seconds
        self.post_event_seconds = post_event_seconds
        self.max_fps = max_fps

        # Maximum frames in rolling buffer (e.g. 4s * 30fps = 120 frames)
        self.buffer_maxlen = int(pre_event_seconds * 30.0) + 10
        self._rolling_buffer: Deque[BufferedFrame] = deque(maxlen=self.buffer_maxlen)

        # Active recording tasks awaiting post-event frames:
        # event_id -> {"frames": List[np.ndarray], "end_timestamp": float, "filepath": str, "fps": float}
        self._active_recordings: List[dict] = []

        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="clip_worker")
        os.makedirs(self.output_dir, exist_ok=True)

    def push_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """Add current frame to the rolling history and append to any active recording jobs."""
        # Store in rolling buffer
        self._rolling_buffer.append(BufferedFrame(frame=frame.copy(), timestamp=timestamp))

        # Check if any active recording jobs are collecting post-event frames
        completed_jobs = []
        for job in self._active_recordings:
            job["frames"].append(frame.copy())
            if timestamp >= job["end_timestamp"]:
                completed_jobs.append(job)

        # Dispatch completed jobs to background encoding
        for job in completed_jobs:
            self._active_recordings.remove(job)
            self._dispatch_encode(job["filepath"], job["frames"], job["fps"])

    def trigger_clip(
        self,
        event_id: str,
        timestamp: float,
        fps: float = 15.0,
    ) -> str:
        """Trigger evidence clip capture centered around current event.

        Returns:
            Relative file path where the video clip will be stored.
        """
        filename = f"{event_id}_evidence.mp4"
        filepath = os.path.join(self.output_dir, filename)

        # Collect pre-event frames from rolling buffer
        cutoff = timestamp - self.pre_event_seconds
        pre_frames = [
            bf.frame.copy()
            for bf in self._rolling_buffer
            if bf.timestamp >= cutoff
        ]

        clip_fps = min(fps, self.max_fps)
        end_time = timestamp + self.post_event_seconds

        job = {
            "event_id": event_id,
            "filepath": filepath,
            "frames": pre_frames,
            "end_timestamp": end_time,
            "fps": clip_fps,
        }
        self._active_recordings.append(job)
        return filepath

    def _dispatch_encode(self, filepath: str, frames: List[np.ndarray], fps: float) -> None:
        """Encode frames into an MP4 file in a background worker."""
        if not frames:
            return

        def _encode():
            try:
                h, w = frames[0].shape[:2]
                # FourCC: mp4v is universally supported on Windows OpenCV
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(filepath, fourcc, fps, (w, h))

                if not writer.isOpened():
                    logger.error(f"Cannot initialize VideoWriter for {filepath}")
                    return

                for f in frames:
                    writer.write(f)
                writer.release()
                logger.debug(f"Saved video evidence clip: {filepath} ({len(frames)} frames)")
            except Exception as e:
                logger.error(f"Error encoding video clip {filepath}: {e}")

        self._executor.submit(_encode)

    def shutdown(self) -> None:
        """Finish pending encoding jobs."""
        self._executor.shutdown(wait=True)
