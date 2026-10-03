"""Asynchronous snapshot capture for suspicious event evidence."""

from concurrent.futures import ThreadPoolExecutor
import logging
import os
import time
from typing import Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SnapshotCapture:
    """Saves JPEG evidence snapshots in a background thread to prevent inference stutter."""

    def __init__(
        self,
        output_dir: str = "storage/evidence/snapshots",
        jpeg_quality: int = 90,
    ):
        self.output_dir = output_dir
        self.jpeg_quality = jpeg_quality
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="snapshot_worker")
        os.makedirs(self.output_dir, exist_ok=True)

    def capture_async(
        self,
        frame: np.ndarray,
        event_id: str,
        suffix: str = "annotated",
    ) -> str:
        """Schedule an asynchronous write of a frame snapshot.

        Returns:
            Relative file path where the snapshot will be saved.
        """
        filename = f"{event_id}_{suffix}.jpg"
        filepath = os.path.join(self.output_dir, filename)

        # Make a copy of the frame numpy array so the caller can continue modifying frame buffers
        frame_copy = frame.copy()

        def _write():
            try:
                cv2.imwrite(
                    filepath,
                    frame_copy,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                logger.debug(f"Saved snapshot evidence: {filepath}")
            except Exception as e:
                logger.error(f"Failed to write snapshot {filepath}: {e}")

        self._executor.submit(_write)
        return filepath

    def shutdown(self) -> None:
        """Gracefully wait for pending snapshot writes to complete."""
        self._executor.shutdown(wait=True)
