"""ByteTrack tracker implementation for persistent student tracking."""

import logging
from types import SimpleNamespace
from typing import List, Optional, Tuple
import numpy as np
import torch

from src.detection.types import BBox, Detection
from src.tracking.tracker import BaseTracker, Track

logger = logging.getLogger(__name__)


class ByteTrackTracker(BaseTracker):
    """ByteTrack adapter for persistent student identification across video frames."""

    def __init__(
        self,
        track_high_thresh: float = 0.45,
        track_low_thresh: float = 0.10,
        new_track_thresh: float = 0.50,
        track_buffer: int = 30,
        match_thresh: float = 0.80,
        frame_rate: int = 30,
    ):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate

        self._tracker = None
        self._init_tracker()

    def _init_tracker(self) -> None:
        """Initialize the underlying BYTETracker instance."""
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker

            args = SimpleNamespace(
                track_high_thresh=self.track_high_thresh,
                track_low_thresh=self.track_low_thresh,
                new_track_thresh=self.new_track_thresh,
                track_buffer=self.track_buffer,
                match_thresh=self.match_thresh,
                fuse_score=True,
                frame_rate=self.frame_rate,
            )
            self._tracker = BYTETracker(args)
            logger.info("ByteTrackTracker initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ByteTrack: {e}")
            raise

    def update(
        self,
        detections: List[Detection],
        timestamp: float,
        frame_shape: Tuple[int, int],
    ) -> List[Track]:
        """Update tracker with person detections and return active persistent tracks."""
        if self._tracker is None:
            return []

        h, w = frame_shape

        # Filter strictly to person detections by verified class name
        person_dets = [d for d in detections if d.class_name == "person"]

        if not person_dets:
            # Send empty tensor to maintain Kalman filter predictions for lost tracks
            try:
                from ultralytics.engine.results import Boxes
                empty_boxes = Boxes(torch.empty((0, 6)), orig_shape=(h, w))
                raw_tracks = self._tracker.update(empty_boxes)
            except Exception:
                raw_tracks = []
        else:
            # Construct tensor: [x1, y1, x2, y2, conf, cls]
            boxes_data = [
                [
                    d.bbox.x1,
                    d.bbox.y1,
                    d.bbox.x2,
                    d.bbox.y2,
                    d.confidence,
                    float(d.class_id),
                ]
                for d in person_dets
            ]
            tensor_data = torch.tensor(boxes_data, dtype=torch.float32)

            from ultralytics.engine.results import Boxes
            boxes_obj = Boxes(tensor_data, orig_shape=(h, w))
            raw_tracks = self._tracker.update(boxes_obj)

        tracks: List[Track] = []
        if len(raw_tracks) == 0:
            return tracks

        # raw_tracks is numpy array: [x1, y1, x2, y2, track_id, conf, cls, idx]
        for row in raw_tracks:
            x1, y1, x2, y2, track_id, conf = row[:6]
            cls_id = int(row[6]) if len(row) > 6 else 0

            bbox = BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
            tracks.append(
                Track(
                    track_id=int(track_id),
                    bbox=bbox,
                    confidence=float(conf),
                    timestamp=timestamp,
                    class_id=cls_id,
                    class_name="person",
                )
            )

        return tracks

    def reset(self) -> None:
        """Reset internal tracker state."""
        self._init_tracker()
