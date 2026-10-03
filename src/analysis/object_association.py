"""Spatial object association module.

Associates secondary detected objects (e.g. cell phones) with specific student tracks
based on spatial geometry (IoU, containment within expanded lap/desk region, and center distance).
"""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.detection.types import BBox, Detection
from src.tracking.tracker import Track

logger = logging.getLogger(__name__)


@dataclass
class AssociatedObjects:
    """Objects associated with a specific student track."""
    track_id: int
    has_phone: bool
    phone_detections: List[Detection]
    highest_phone_confidence: float = 0.0


class ObjectAssociator:
    """Associates secondary objects (cell phones, notes) to tracked students."""

    def __init__(
        self,
        expand_student_bbox_ratio: float = 0.15,
        max_center_distance_ratio: float = 0.60,
        min_iou_overlap: float = 0.05,
    ):
        self.expand_ratio = expand_student_bbox_ratio
        self.max_distance_ratio = max_center_distance_ratio
        self.min_iou = min_iou_overlap

    def associate(
        self,
        tracks: List[Track],
        detections: List[Detection],
    ) -> Dict[int, AssociatedObjects]:
        """Associate secondary objects (e.g. phones) with active student tracks.

        Args:
            tracks: Active student tracks.
            detections: All object detections in the frame.

        Returns:
            Dictionary mapping track_id -> AssociatedObjects.
        """
        # Separate secondary target objects (e.g. cell phone)
        phone_dets = [
            d for d in detections
            if d.class_name in ("cell phone", "phone", "mobile phone") or d.class_id == 67
        ]

        # Initialize result mapping for every active track
        result: Dict[int, AssociatedObjects] = {
            t.track_id: AssociatedObjects(
                track_id=t.track_id,
                has_phone=False,
                phone_detections=[],
                highest_phone_confidence=0.0,
            )
            for t in tracks
        }

        if not phone_dets or not tracks:
            return result

        # For each phone detection, find the best matching student track
        for phone in phone_dets:
            phone_center = phone.bbox.center
            best_track_id: Optional[int] = None
            best_score = -1.0

            for track in tracks:
                student_box = track.bbox
                # Expand student box slightly to account for lap, desk surface in front, hands
                expanded_box = student_box.expand(self.expand_ratio, self.expand_ratio)

                # Check 1: phone center inside expanded student box
                is_inside = expanded_box.contains_point(phone_center[0], phone_center[1])

                # Check 2: overlap / IoU
                iou = student_box.iou(phone.bbox)
                overlap = student_box.overlap_ratio(phone.bbox)

                # Check 3: normalized center distance
                s_center = student_box.center
                diag = np.sqrt(student_box.width ** 2 + student_box.height ** 2) + 1e-6
                dist = np.sqrt((phone_center[0] - s_center[0]) ** 2 + (phone_center[1] - s_center[1]) ** 2)
                normalized_dist = dist / diag

                if (is_inside or iou >= self.min_iou or overlap >= self.min_iou) and normalized_dist <= self.max_distance_ratio:
                    # Combined association score: higher IoU and closer distance gives higher score
                    score = (1.0 - normalized_dist) + (iou * 2.0)
                    if score > best_score:
                        best_score = score
                        best_track_id = track.track_id

            if best_track_id is not None:
                assoc = result[best_track_id]
                assoc.has_phone = True
                assoc.phone_detections.append(phone)
                if phone.confidence > assoc.highest_phone_confidence:
                    assoc.highest_phone_confidence = phone.confidence

        return result
