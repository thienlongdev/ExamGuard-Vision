"""
Stage 2 Phone Spatial Association & Ambiguity Resolver
======================================================
Strict separation of phone detection from posture.
Associates phone bounding boxes with student tracks using spatial geometry
(containment, IoU, center distance) and strict ambiguity gating:
- Multiple students + one phone with similar proximity -> AMBIGUOUS (not associated)
- Clear closest student beyond ambiguity margin -> ASSOCIATED
- No phone in range -> UNASSOCIATED / NO_PHONE
"""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.detection.types import BBox, Detection
from src.tracking.tracker import Track
from src.fusion.types import PhoneAssociationStatus, PhoneCue, ObservationStatus

logger = logging.getLogger(__name__)


@dataclass
class TrackPhoneAssociation:
    """Phone association result for a specific student track."""
    track_id: int
    status: str                         # "ASSOCIATED", "AMBIGUOUS", "UNASSOCIATED", "NOT_EVALUATED"
    association_status_enum: PhoneAssociationStatus
    detected: bool
    confidence: float
    spatial_relation: str               # "DIRECT_CONTACT", "DESK_PROXIMITY", "NONE"
    phone_bbox: Optional[Tuple[float, float, float, float]] = None
    competing_track_ids: List[int] = None

    def to_phone_cue(self) -> PhoneCue:
        """Convert into V4D canonical PhoneCue."""
        obs_status = ObservationStatus.AVAILABLE
        if self.status == "NOT_EVALUATED":
            obs_status = ObservationStatus.NOT_EVALUATED

        is_assoc = (self.status == "ASSOCIATED")
        rel_weight = 1.0 if is_assoc else (0.4 if self.status == "AMBIGUOUS" else 0.0)

        return PhoneCue(
            status=obs_status,
            detected=is_assoc,
            association_confidence=self.confidence if is_assoc else 0.0,
            association_status=self.association_status_enum,
            spatial_relation=self.spatial_relation,
            phone_bbox=self.phone_bbox,
            reliability_weight=rel_weight,
        )


class PhoneAssociator:
    """Associates cell phone detections with active student tracks with strict ambiguity checks."""

    def __init__(
        self,
        expand_ratio: float = 0.20,
        max_distance_ratio: float = 0.65,
        min_iou_overlap: float = 0.04,
        ambiguity_margin: float = 0.15,
    ):
        self.expand_ratio = expand_ratio
        self.max_distance_ratio = max_distance_ratio
        self.min_iou = min_iou_overlap
        self.ambiguity_margin = ambiguity_margin

    def associate(
        self,
        tracks: List[Track],
        detections: List[Detection],
    ) -> Dict[int, TrackPhoneAssociation]:
        """Associate phone detections with student tracks."""
        # Initialize default unassociated result for every active track
        results: Dict[int, TrackPhoneAssociation] = {
            t.track_id: TrackPhoneAssociation(
                track_id=t.track_id,
                status="UNASSOCIATED",
                association_status_enum=PhoneAssociationStatus.NO_PHONE,
                detected=False,
                confidence=0.0,
                spatial_relation="NONE",
                phone_bbox=None,
                competing_track_ids=[],
            )
            for t in tracks
        }

        # Filter detections for phones
        phone_dets = [
            d for d in detections
            if d.class_name in ("cell phone", "phone", "mobile phone") or d.class_id == 67
        ]

        if not phone_dets or not tracks:
            return results

        # For each phone detection, evaluate candidate associations across all tracks
        for phone in phone_dets:
            phone_center = phone.bbox.center
            candidate_scores: List[Tuple[int, float, str]] = [] # (track_id, score, spatial_relation)

            for track in tracks:
                student_box = track.bbox
                expanded_box = student_box.expand(self.expand_ratio, self.expand_ratio)

                is_inside = expanded_box.contains_point(phone_center[0], phone_center[1])
                iou = student_box.iou(phone.bbox)
                overlap = student_box.overlap_ratio(phone.bbox)

                s_center = student_box.center
                diag = np.sqrt(student_box.width ** 2 + student_box.height ** 2) + 1e-6
                dist = np.sqrt((phone_center[0] - s_center[0]) ** 2 + (phone_center[1] - s_center[1]) ** 2)
                normalized_dist = dist / diag

                if (is_inside or iou >= self.min_iou or overlap >= self.min_iou) and normalized_dist <= self.max_distance_ratio:
                    score = (1.0 - normalized_dist) + (iou * 2.0)
                    relation = "DIRECT_CONTACT" if (iou > 0.10 or overlap > 0.15) else "DESK_PROXIMITY"
                    candidate_scores.append((track.track_id, score, relation))

            if not candidate_scores:
                continue

            # Sort candidate tracks by association score descending
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            best_track_id, best_score, best_relation = candidate_scores[0]

            # Check for ambiguity: if second candidate is close to first
            is_ambiguous = False
            competing_ids = [best_track_id]
            if len(candidate_scores) > 1:
                second_track_id, second_score, _ = candidate_scores[1]
                if abs(best_score - second_score) <= self.ambiguity_margin:
                    is_ambiguous = True
                    competing_ids.append(second_track_id)

            if is_ambiguous:
                # Mark both competing tracks as AMBIGUOUS (not associated!)
                for c_id in competing_ids:
                    results[c_id] = TrackPhoneAssociation(
                        track_id=c_id,
                        status="AMBIGUOUS",
                        association_status_enum=PhoneAssociationStatus.AMBIGUOUS_ASSOCIATION,
                        detected=False,
                        confidence=phone.confidence,
                        spatial_relation="DESK_PROXIMITY",
                        phone_bbox=phone.bbox.as_tuple(),
                        competing_track_ids=competing_ids,
                    )
            else:
                # Clear association for best track
                results[best_track_id] = TrackPhoneAssociation(
                    track_id=best_track_id,
                    status="ASSOCIATED",
                    association_status_enum=PhoneAssociationStatus.CLEAR_ASSOCIATION,
                    detected=True,
                    confidence=phone.confidence,
                    spatial_relation=best_relation,
                    phone_bbox=phone.bbox.as_tuple(),
                    competing_track_ids=[best_track_id],
                )

        return results
