"""Observation fusion engine.

Merges multi-object tracking, behavior detections, object associations,
and optional head pose into a unified StudentObservation per student.
"""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional
import numpy as np

from src.detection.types import BBox, BehaviorDetection
from src.tracking.tracker import Track
from src.analysis.object_association import AssociatedObjects
from src.analysis.head_pose import HeadPoseEstimator, HeadPoseResult

logger = logging.getLogger(__name__)


@dataclass
class StudentObservation:
    """Unified observation of a tracked student at a specific point in time."""
    track_id: int
    timestamp: float
    bbox: BBox
    behavior: str
    behavior_confidence: float
    phone_present: bool = False
    phone_confidence: float = 0.0
    head_pose: Optional[HeadPoseResult] = None


class FusionEngine:
    """Fuses tracking, behavior detections, secondary object associations, and head pose."""

    def __init__(self, head_pose_estimator: Optional[HeadPoseEstimator] = None):
        self.head_pose_estimator = head_pose_estimator

    def fuse(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        behavior_detections: List[BehaviorDetection],
        associated_objects: Dict[int, AssociatedObjects],
        timestamp: float,
    ) -> List[StudentObservation]:
        """Fuse all per-frame signals into StudentObservation objects.

        Args:
            frame: Full video frame.
            tracks: Active student tracks.
            behavior_detections: Detected behaviors in the frame.
            associated_objects: Secondary object associations per track.
            timestamp: Frame timestamp in seconds.

        Returns:
            List of StudentObservation, one per active track.
        """
        observations: List[StudentObservation] = []

        for track in tracks:
            t_id = track.track_id
            t_box = track.bbox

            # 1. Match behavior detection to this student's bounding box
            matched_behavior = "normal"
            matched_conf = 0.50
            best_iou = -1.0

            for b_det in behavior_detections:
                iou = t_box.iou(b_det.bbox)
                if iou > 0.30 and iou > best_iou:
                    best_iou = iou
                    matched_behavior = b_det.behavior
                    matched_conf = b_det.confidence

            # 2. Extract object association (e.g. phone)
            assoc = associated_objects.get(t_id)
            phone_present = assoc.has_phone if assoc else False
            phone_conf = assoc.highest_phone_confidence if assoc else 0.0

            # If phone is strongly associated with student, reflect in behavior if not already flagged
            if phone_present and matched_behavior in ("normal", "lean") and phone_conf >= 0.50:
                matched_behavior = "use_phone"
                matched_conf = max(matched_conf, phone_conf)

            # 3. Optional head pose
            head_pose_res: Optional[HeadPoseResult] = None
            if self.head_pose_estimator is not None:
                head_pose_res = self.head_pose_estimator.estimate(frame, t_box)

            # If head pose orientation is confident, it can refine head turn/down
            if head_pose_res is not None:
                if head_pose_res.orientation in ("left", "right") and matched_behavior == "normal":
                    matched_behavior = "turn_head"
                elif head_pose_res.orientation == "down" and matched_behavior == "normal":
                    matched_behavior = "head_down"

            obs = StudentObservation(
                track_id=t_id,
                timestamp=timestamp,
                bbox=t_box,
                behavior=matched_behavior,
                behavior_confidence=matched_conf,
                phone_present=phone_present,
                phone_confidence=phone_conf,
                head_pose=head_pose_res,
            )
            observations.append(obs)

        return observations
