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
import os
import time
from typing import Dict, List, Optional, Tuple, Any
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
    is_candidate_only: bool = False

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
            is_candidate_only=self.is_candidate_only,
        )


class PhoneAssociator:
    """Associates cell phone detections with active student tracks with strict ambiguity checks."""

    def __init__(
        self,
        expand_ratio: float = 0.20,
        expand_ratio_down: float = 0.35,
        max_distance_ratio: float = 0.75,
        min_iou_overlap: float = 0.03,
        ambiguity_margin: float = 0.15,
        phone_strong_confidence: float = 0.35,
        phone_candidate_window_sec: float = 3.0,
        phone_candidate_gap_tolerance_sec: float = 1.0,
        weak_candidate_min_hits: int = 3,
        weak_candidate_min_duration_sec: float = 0.8,
    ):
        self.expand_ratio = expand_ratio
        self.expand_ratio_down = expand_ratio_down
        self.max_distance_ratio = max_distance_ratio
        self.min_iou = min_iou_overlap
        self.ambiguity_margin = ambiguity_margin
        self.phone_strong_confidence = phone_strong_confidence
        self.phone_candidate_window_sec = phone_candidate_window_sec
        self.phone_candidate_gap_tolerance_sec = phone_candidate_gap_tolerance_sec
        self.weak_candidate_min_hits = weak_candidate_min_hits
        self.weak_candidate_min_duration_sec = weak_candidate_min_duration_sec

        # Temporal accumulation store: track_id -> List[(timestamp, conf, bbox)]
        self._recent_track_hits: Dict[int, List[Tuple[float, float, Tuple[float, float, float, float]]]] = {}
        # Unassociated strong phone detections from latest frame
        self.unassociated_phones: List[Detection] = []

    def cleanup_expired_tracks(self, active_track_ids: List[int]) -> None:
        """Prune stale track accumulation data."""
        active_set = set(active_track_ids)
        stale = [tid for tid in self._recent_track_hits if tid not in active_set]
        for tid in stale:
            self._recent_track_hits.pop(tid, None)

    def get_unassociated_phones(self) -> List[Dict[str, Any]]:
        """Return unassociated strong phone detections from latest frame."""
        return [
            {
                "bbox": p.bbox.as_tuple() if hasattr(p.bbox, "as_tuple") else p.bbox,
                "confidence": float(p.confidence),
                "class_id": int(p.class_id),
                "class_name": str(p.class_name),
            }
            for p in self.unassociated_phones
        ]

    def associate(
        self,
        tracks: List[Track],
        detections: List[Detection],
        timestamp_sec: Optional[float] = None,
    ) -> Dict[int, TrackPhoneAssociation]:
        """Associate phone detections with student tracks and accumulate short-memory evidence."""
        self.unassociated_phones = []
        diag_mode = os.environ.get("EXAMGUARD_DIAGNOSTIC_CUES") == "1"

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

        # Normalize detections if passed as dicts
        norm_dets: List[Detection] = []
        for d in detections:
            if isinstance(d, dict):
                raw_bbox = d.get("bbox", [0, 0, 0, 0])
                if isinstance(raw_bbox, (list, tuple)):
                    bx = BBox(raw_bbox[0], raw_bbox[1], raw_bbox[2], raw_bbox[3])
                else:
                    bx = raw_bbox
                norm_dets.append(
                    Detection(
                        bbox=bx,
                        class_id=int(d.get("class_id", 67)),
                        class_name=str(d.get("class_name", "cell phone")),
                        confidence=float(d.get("confidence", 0.0)),
                    )
                )
            else:
                norm_dets.append(d)

        # Filter detections for phones with handheld dimensions sanity check
        phone_dets = []
        for d in norm_dets:
            if d.class_name in ("cell phone", "phone", "mobile phone") or d.class_id == 67:
                bw = float(d.bbox.width)
                bh = float(d.bbox.height)
                # Filter out giant boxes (desk surfaces, TV/screens, chair backs > 220x250 px or area > 35,000)
                if bw <= 220.0 and bh <= 250.0 and (bw * bh) <= 35000.0:
                    phone_dets.append(d)

        now_ts = timestamp_sec if timestamp_sec is not None else time.time()

        if not phone_dets:
            # Check temporal accumulation across recent gap tolerance
            if timestamp_sec is not None:
                for t in tracks:
                    tid = t.track_id
                    hits = self._recent_track_hits.get(tid, [])
                    # Prune old hits
                    valid_hits = [h for h in hits if (now_ts - h[0]) <= self.phone_candidate_window_sec]
                    self._recent_track_hits[tid] = valid_hits
                    if valid_hits:
                        last_hit_ts, last_conf, last_bbox = valid_hits[-1]
                        gap = now_ts - last_hit_ts
                        has_strong = any(h[1] >= self.phone_strong_confidence for h in valid_hits)
                        hit_span = (valid_hits[-1][0] - valid_hits[0][0]) if len(valid_hits) >= 2 else 0.0
                        is_temporal_supported = (
                            has_strong or
                            (len(valid_hits) >= self.weak_candidate_min_hits and hit_span >= self.weak_candidate_min_duration_sec)
                        )
                        if gap <= self.phone_candidate_gap_tolerance_sec and len(valid_hits) >= 2 and is_temporal_supported:
                            # Bridge intermittent gap using accumulated evidence, preserving candidate status
                            is_cand = not is_temporal_supported
                            results[tid] = TrackPhoneAssociation(
                                track_id=tid,
                                status="ASSOCIATED",
                                association_status_enum=PhoneAssociationStatus.CLEAR_ASSOCIATION,
                                detected=True,
                                confidence=round(last_conf * max(0.60, 1.0 - (gap / self.phone_candidate_window_sec)), 3),
                                spatial_relation="DESK_PROXIMITY",
                                phone_bbox=last_bbox,
                                competing_track_ids=[tid],
                                is_candidate_only=is_cand,
                            )
            return results

        if not tracks:
            # Strong phones with no tracks present in frame: retain as unassociated
            for phone in phone_dets:
                if phone.confidence >= self.phone_strong_confidence:
                    self.unassociated_phones.append(phone)
            return results

        # For each phone detection, evaluate candidate associations across all tracks
        for phone in phone_dets:
            phone_center = phone.bbox.center
            candidate_scores: List[Tuple[int, float, str]] = [] # (track_id, score, spatial_relation)

            for track in tracks:
                student_box = track.bbox
                # Asymmetric expansion: generous downward into lap/desk region
                exp_x1 = student_box.x1 - self.expand_ratio * student_box.width
                exp_x2 = student_box.x2 + self.expand_ratio * student_box.width
                exp_y1 = student_box.y1 - self.expand_ratio * student_box.height
                exp_y2 = student_box.y2 + self.expand_ratio_down * student_box.height

                is_inside = (exp_x1 <= phone_center[0] <= exp_x2 and exp_y1 <= phone_center[1] <= exp_y2)
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
                # Phone detected but no student track close enough: retain as unassociated observable
                if phone.confidence >= self.phone_strong_confidence:
                    bw = float(phone.bbox.width)
                    bh = float(phone.bbox.height)
                    if bw <= 180.0 and bh <= 200.0:
                        self.unassociated_phones.append(phone)
                        if diag_mode:
                            logger.info(
                                f"[DIAGNOSTIC_CUES] Unassociated phone preserved: conf={phone.confidence:.3f} bbox={phone.bbox.as_tuple()}"
                            )
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
                # Mark competing tracks as AMBIGUOUS and preserve phone detection
                self.unassociated_phones.append(phone)
                for c_id in competing_ids:
                    if results[c_id].status != "ASSOCIATED":
                        results[c_id] = TrackPhoneAssociation(
                            track_id=c_id,
                            status="AMBIGUOUS",
                            association_status_enum=PhoneAssociationStatus.AMBIGUOUS_ASSOCIATION,
                            detected=False,
                            confidence=phone.confidence,
                            spatial_relation="DESK_PROXIMITY",
                            phone_bbox=phone.bbox.as_tuple(),
                            competing_track_ids=competing_ids,
                            is_candidate_only=True,
                        )
            else:
                # Clear association for best track: takes precedence over UNASSOCIATED or AMBIGUOUS
                hits = self._recent_track_hits.setdefault(best_track_id, [])
                hits.append((now_ts, phone.confidence, phone.bbox.as_tuple()))
                valid_hits = [
                    h for h in hits if (now_ts - h[0]) <= self.phone_candidate_window_sec
                ]
                self._recent_track_hits[best_track_id] = valid_hits

                has_strong_hit = (phone.confidence >= self.phone_strong_confidence)
                hit_span = (valid_hits[-1][0] - valid_hits[0][0]) if len(valid_hits) >= 2 else 0.0
                is_temporal_supported = (
                    has_strong_hit or
                    (len(valid_hits) >= self.weak_candidate_min_hits and hit_span >= self.weak_candidate_min_duration_sec)
                )
                is_cand = not is_temporal_supported

                prev = results[best_track_id]
                if prev.status != "ASSOCIATED" or phone.confidence > prev.confidence or (prev.is_candidate_only and not is_cand):
                    results[best_track_id] = TrackPhoneAssociation(
                        track_id=best_track_id,
                        status="ASSOCIATED",
                        association_status_enum=PhoneAssociationStatus.CLEAR_ASSOCIATION,
                        detected=True,
                        confidence=phone.confidence,
                        spatial_relation=best_relation,
                        phone_bbox=phone.bbox.as_tuple(),
                        competing_track_ids=[best_track_id],
                        is_candidate_only=is_cand,
                    )

        return results
