"""
Stage 2 Integrated Evidence Manager
===================================
Coordinates bounded JPEG snapshots and MP4 rolling clip recordings for
FusedEvent lifecycle transitions (OPEN, UPDATE, CLOSE).
Enforces:
- Privacy: Anonymous track IDs and event IDs only (no facial recognition, no names)
- Rate-limiting: Strictly bounded snapshots per event (max 3)
- Disk safety: Bounded directories, asynchronous writes, graceful failure handling
"""

import logging
import os
import json
from typing import Dict, Optional, Any, List
import numpy as np

from src.fusion.types import FusedEvent
from src.evidence.snapshot import SnapshotCapture
from src.evidence.clip_recorder import RollingClipRecorder

logger = logging.getLogger(__name__)


class IntegratedEvidenceManager:
    """Manages evidence capture lifecycle tied to V4D FusedEvents."""

    def __init__(
        self,
        output_dir: str = "storage/evidence",
        max_snapshots_per_event: int = 3,
        pre_event_seconds: float = 3.0,
        post_event_seconds: float = 3.0,
        enabled: bool = True,
    ):
        self.output_dir = output_dir
        self.max_snapshots_per_event = max_snapshots_per_event
        self.enabled = enabled

        self.snapshots_dir = os.path.join(output_dir, "snapshots")
        self.clips_dir = os.path.join(output_dir, "clips")
        self.metadata_dir = os.path.join(output_dir, "metadata")

        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(self.clips_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        self.snapshot_capture = SnapshotCapture(output_dir=self.snapshots_dir) if enabled else None
        self.clip_recorder = RollingClipRecorder(
            output_dir=self.clips_dir,
            pre_event_seconds=pre_event_seconds,
            post_event_seconds=post_event_seconds,
        ) if enabled else None

        # Tracking snapshot counts per event_id
        self._event_snapshot_counts: Dict[str, int] = {}
        self._event_clip_paths: Dict[str, str] = {}

    def push_frame(self, frame: np.ndarray, timestamp_sec: float) -> None:
        """Feed every incoming video frame to rolling clip buffer."""
        if self.enabled and self.clip_recorder is not None:
            self.clip_recorder.push_frame(frame, timestamp_sec)

    def handle_event_lifecycle(
        self,
        event: FusedEvent,
        action: str,
        frame: np.ndarray,
        timestamp_sec: float,
        fps: float = 30.0,
    ) -> None:
        """Handle lifecycle transitions (OPEN, UPDATE, CLOSE)."""
        if not self.enabled:
            return

        ev_id = event.event_id

        # 1. OPEN Action
        if action == "OPEN":
            count = self._event_snapshot_counts.get(ev_id, 0)
            if count < self.max_snapshots_per_event and self.snapshot_capture is not None:
                try:
                    snap_path = self.snapshot_capture.capture_async(frame, ev_id, suffix="open")
                    event.evidence_summary["open_snapshot_path"] = snap_path
                    self._event_snapshot_counts[ev_id] = count + 1
                except Exception as e:
                    logger.warning(f"Snapshot capture failed for {ev_id}: {e}")

            if self.clip_recorder is not None:
                try:
                    clip_path = self.clip_recorder.trigger_clip(ev_id, timestamp_sec, fps=fps)
                    self._event_clip_paths[ev_id] = clip_path
                    event.evidence_summary["clip_path"] = clip_path
                except Exception as e:
                    logger.warning(f"Clip trigger failed for {ev_id}: {e}")

        # 2. UPDATE Action (capture escalation snapshot if newly HIGH risk)
        elif action == "UPDATE":
            count = self._event_snapshot_counts.get(ev_id, 0)
            if event.risk_level == "HIGH" and count < self.max_snapshots_per_event and self.snapshot_capture is not None:
                if "escalation_snapshot_path" not in event.evidence_summary:
                    try:
                        snap_path = self.snapshot_capture.capture_async(frame, ev_id, suffix="escalation")
                        event.evidence_summary["escalation_snapshot_path"] = snap_path
                        self._event_snapshot_counts[ev_id] = count + 1
                    except Exception as e:
                        logger.warning(f"Escalation snapshot capture failed for {ev_id}: {e}")

        # 3. CLOSE Action
        elif action == "CLOSE":
            # Save event audit metadata JSON
            try:
                meta_path = os.path.join(self.metadata_dir, f"{ev_id}_metadata.json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(event.to_dict(), f, indent=2)
                event.evidence_summary["metadata_path"] = meta_path
            except Exception as e:
                logger.warning(f"Failed to write event metadata json for {ev_id}: {e}")

            # Cleanup tracking counters to prevent dictionary leak
            self._event_snapshot_counts.pop(ev_id, None)
            self._event_clip_paths.pop(ev_id, None)

    def reset(self) -> None:
        """Reset internal tracking and in-memory clip buffer."""
        self._event_snapshot_counts.clear()
        self._event_clip_paths.clear()
        if self.clip_recorder is not None:
            self.clip_recorder._rolling_buffer.clear()

    def shutdown(self) -> None:
        """Flush workers and close handlers."""
        if self.snapshot_capture is not None:
            self.snapshot_capture.shutdown()
        if self.clip_recorder is not None:
            self.clip_recorder.shutdown()
