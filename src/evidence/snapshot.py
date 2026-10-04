"""
Asynchronous snapshot capture for suspicious event evidence.
Enforces:
- Atomic file write (.tmp -> final)
- SHA-256 calculation and completion callback
- Strict path creation and graceful failure isolation
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import logging
import os
import time
from typing import Optional, Callable
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SnapshotCapture:
    """Saves JPEG evidence snapshots asynchronously with atomic writes and SHA-256 checksums."""

    def __init__(
        self,
        output_dir: str = "storage/evidence/snapshots",
        jpeg_quality: int = 90,
        on_snapshot_ready: Optional[Callable[[str, str, str, int], None]] = None,
    ):
        self.output_dir = output_dir
        self.jpeg_quality = jpeg_quality
        self.on_snapshot_ready = on_snapshot_ready
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="snapshot_worker")
        os.makedirs(self.output_dir, exist_ok=True)

    def capture_async(
        self,
        frame: np.ndarray,
        event_id: str,
        suffix: str = "annotated",
        custom_filepath: Optional[str] = None,
    ) -> str:
        """Schedule an asynchronous atomic write of a frame snapshot.

        Returns:
            Relative or absolute file path where the snapshot will be saved.
        """
        if custom_filepath:
            filepath = custom_filepath
        else:
            filename = f"{event_id}_{suffix}.jpg"
            filepath = os.path.join(self.output_dir, filename)

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        frame_copy = frame.copy()

        def _write():
            tmp_path = f"{filepath}.tmp.jpg"
            try:
                cv2.imwrite(
                    tmp_path,
                    frame_copy,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass
                os.replace(tmp_path, filepath)

                # Calculate SHA-256
                hasher = hashlib.sha256()
                with open(filepath, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
                snap_hash = hasher.hexdigest()
                snap_size = os.path.getsize(filepath)

                logger.debug(f"Saved snapshot evidence: {filepath} ({snap_size} bytes)")
                if self.on_snapshot_ready:
                    self.on_snapshot_ready(event_id, filepath, snap_hash, snap_size)
            except Exception as e:
                logger.error(f"Failed to write snapshot {filepath}: {e}")
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

        self._executor.submit(_write)
        return filepath

    def shutdown(self) -> None:
        """Gracefully wait for pending snapshot writes to complete."""
        self._executor.shutdown(wait=True)
