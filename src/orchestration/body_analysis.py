"""Scheduled optional COCO pose and conservative, track-local 2D body cues.

No wrist/arm cue, proximity rule, face-absence rule or fabricated 3D angle.
Hidden shoulders/hips cannot generate torso evidence. Calibration requires a
visible frontal neutral posture. Shoulder ordering is only a rearward candidate,
because a monocular estimator can swap anatomical labels when occluded.
"""
from dataclasses import dataclass, field, replace
import logging
import math
import time
from typing import Dict
import numpy as np
from src.detection.types import BBox
from src.fusion.types import BodyCue, ObservationStatus

logger = logging.getLogger(__name__)


@dataclass
class BodyBaseline:
    last_seen: float = -1.0
    bbox: object = None
    samples: list = field(default_factory=list)
    baseline: object = None
    movement_starts: list = field(default_factory=list)
    last_movement: str = "NEUTRAL"
    upper_samples: list = field(default_factory=list)
    upper_baseline: object = None


class BodyGeometry:
    """Scale invariant measurements with bounded neutral calibration per track."""
    def __init__(self, keypoint_confidence=0.55):
        self.confidence = keypoint_confidence
        self.states: Dict[object, BodyBaseline] = {}

    def measure(self, key, points, bbox, timestamp, neutral=False):
        cue = BodyCue(status=ObservationStatus.UNAVAILABLE, observed_at=timestamp,
                      fresh=True, uncertainty="SHOULDERS_OR_HIPS_HIDDEN")
        p = np.asarray(points, dtype=float)
        if p.shape != (17, 3) or not np.isfinite(p).all():
            return cue
        # Reject hallucinated points outside their owner's box, not just low scores.
        def visible(i):
            x, y, confidence = p[i]
            return bool(confidence >= self.confidence and bbox.contains_point(x, y) and x > 0 and y > 0)

        face_ids = [i for i in (0, 1, 2) if visible(i)]
        cue.face_visible = len(face_ids) >= 2 and visible(0)
        head_ids = [i for i in range(5) if visible(i)]
        if len(head_ids) >= 2:
            pts = p[head_ids, :2]
            center = pts.mean(axis=0)
            size = max(float(np.ptp(pts, axis=0).max()) * 2.2, bbox.height * .13)
            cue.head_bbox = (max(bbox.x1, center[0] - size / 2), max(bbox.y1, center[1] - size * .6),
                             min(bbox.x2, center[0] + size / 2), min(bbox.y2, center[1] + size * .6))
        state = self.states.setdefault(key, BodyBaseline())
        if state.bbox is not None:
            distance = math.dist(bbox.center, state.bbox.center) / max(state.bbox.height, 1)
            if timestamp - state.last_seen > .8 or distance > .65:
                state = self.states[key] = BodyBaseline()
        state.last_seen, state.bbox = timestamp, bbox
        # A desk often hides hips. Visible head displacement PLUS shoulder roll
        # can support upper-body leaning, but cannot supply rotation/rearward cues.
        upper = None
        if visible(5) and visible(6) and len(head_ids) >= 2:
            shoulders = p[[5, 6], :2]
            shoulders = shoulders[np.argsort(shoulders[:, 0])]
            span = float(shoulders[1, 0] - shoulders[0, 0])
            if span >= .12 * bbox.height:
                slope = float((shoulders[1, 1] - shoulders[0, 1]) / span)
                head_delta = float((p[head_ids, 0].mean() - shoulders[:, 0].mean()) / span)
                upper = (slope, head_delta)
                if state.upper_baseline is None:
                    if neutral and cue.face_visible and abs(slope) < .25 and abs(head_delta) < .3:
                        state.upper_samples.append((timestamp, slope, head_delta))
                        state.upper_samples = state.upper_samples[-8:]
                        if len(state.upper_samples) >= 4 and timestamp - state.upper_samples[0][0] >= .75:
                            state.upper_baseline = np.median(np.asarray(state.upper_samples)[:, 1:], axis=0)
                    else:
                        state.upper_samples.clear()
        if not all(visible(i) for i in (5, 6, 11, 12)):
            # Partial neutral samples must not span an unobserved interval.
            state.samples.clear()
            if upper is not None and state.upper_baseline is not None:
                slope_delta, head_delta = np.asarray(upper) - state.upper_baseline
                cue.status = ObservationStatus.AVAILABLE
                cue.calibrated = True
                cue.reliability = float(min(p[[5, 6] + head_ids, 2])) * .8
                cue.rotation_observable = cue.rearward_observable = False
                cue.direction = "RIGHT" if head_delta > 0 else "LEFT" if head_delta < 0 else "UNKNOWN"
                cue.geometry = dict(shoulder_roll_change=float(slope_delta), head_to_shoulders_change=float(head_delta))
                # Require strong head displacement together with shoulder roll;
                # isolated head or arm movement is insufficient.
                if abs(slope_delta) >= .18 and abs(head_delta) >= .45 and slope_delta * head_delta > 0:
                    cue.lean_score = .65
                cue.uncertainty = "UPPER_BODY_ONLY_HIPS_HIDDEN"
            else:
                cue.lean_observable = cue.rotation_observable = cue.rearward_observable = False
            return cue
        shoulder = (p[5, :2] + p[6, :2]) / 2
        hip = (p[11, :2] + p[12, :2]) / 2
        height = float(hip[1] - shoulder[1])
        if height < .12 * bbox.height:
            cue.uncertainty = "TORSO_TOO_SMALL_OR_FOLDED"
            return cue
        shoulder_width = float((p[5, 0] - p[6, 0]) / height)
        hip_width = float((p[11, 0] - p[12, 0]) / height)
        offset = float((shoulder[0] - hip[0]) / height)
        head_offset = float((p[head_ids, 0].mean() - shoulder[0]) / height) if head_ids else None
        cue.status = ObservationStatus.AVAILABLE
        cue.reliability = float(min(p[[5, 6, 11, 12], 2]))
        cue.geometry = dict(shoulder_span=round(shoulder_width, 4), hip_span=round(hip_width, 4),
                            torso_lateral_offset=round(offset, 4), head_lateral_offset=head_offset)
        if state.baseline is None:
            if neutral and cue.face_visible and abs(offset) < .25 and abs(shoulder_width) > .35 and abs(hip_width) > .18:
                state.samples.append((timestamp, shoulder_width, hip_width, offset, head_offset or 0))
                state.samples = state.samples[-8:]
                if len(state.samples) >= 4 and timestamp - state.samples[0][0] >= .75:
                    state.baseline = np.median(np.asarray(state.samples)[:, 1:], axis=0)
            else:
                state.samples.clear()
            cue.uncertainty = "NEUTRAL_CALIBRATION_REQUIRED"
            return cue
        cue.calibrated = True
        sw, hw, base_offset, base_head = state.baseline
        displacement = offset - base_offset
        head_displacement = (head_offset - base_head) if head_offset is not None else 0
        cue.direction = "RIGHT" if displacement > 0 else "LEFT" if displacement < 0 else "UNKNOWN"
        # Both upper torso AND head must move laterally: arm movement alone does nothing.
        if abs(displacement) >= .35 and abs(displacement + head_displacement) >= .45:
            cue.lean_score = min(.95, .65 + (abs(displacement) - .35) * .5)
        shoulder_ratio, hip_ratio = abs(shoulder_width / sw), abs(hip_width / hw)
        shoulders_reversed = shoulder_width * sw < 0 and shoulder_ratio > .40
        hips_reversed = hip_width * hw < 0 and hip_ratio > .40
        if shoulders_reversed and hips_reversed:
            cue.rearward_score = .80
            cue.rotation_score = .80
        elif shoulder_ratio < .62 and hip_ratio < .80 and cue.lean_score == 0:
            if not neutral:
                cue.rotation_score = .75
        cue.geometry.update(shoulder_compression=round(shoulder_ratio, 4), hip_compression=round(hip_ratio, 4),
                            shoulders_reversed=bool(shoulders_reversed), hips_reversed=bool(hips_reversed))
        cue.uncertainty = "MONOCULAR_REARWARD_CANDIDATE" if cue.rearward_score else "2D_GEOMETRY_ONLY"
        movement = "REARWARD" if cue.rearward_score else "ROTATION" if cue.rotation_score else (cue.direction if cue.lean_score else "NEUTRAL")
        if movement != "NEUTRAL" and movement != state.last_movement:
            if not state.movement_starts or timestamp - state.movement_starts[-1] >= .5:
                state.movement_starts.append(timestamp)
        state.movement_starts = [t for t in state.movement_starts[-20:] if timestamp - t <= 10]
        state.last_movement = movement
        cue.repeated_movement_count = len(state.movement_starts)
        return cue


class BodyPoseScheduler:
    def __init__(self, registry, config=None):
        cfg = (config or {}).get('models', {}).get('body_pose', {})
        self.registry = registry
        self.enabled = bool(cfg.get('enabled', False))
        self.image_size = int(cfg.get('image_size', 640))
        self.interval = 1 / max(.5, float(cfg.get('normal_hz', 4)))
        self.attention_interval = 1 / max(.5, float(cfg.get('attention_hz', 6)))
        self.last_run = -math.inf
        self.cache = {}
        self.geometry = BodyGeometry(float(cfg.get('keypoint_confidence', .55)))

    def reset(self):
        self.last_run = -math.inf
        self.cache.clear()
        self.geometry.states.clear()

    def infer(self, frame, tracks, timestamp, camera_id, posture_cues, headpose_cues):
        start = time.perf_counter()
        keys = {(camera_id, t.track_id) for t in tracks if not t.lost}
        for key in list(self.geometry.states):
            if key not in keys:
                self.geometry.states.pop(key, None)
                self.cache.pop(key, None)
        for key in list(self.cache):
            if key not in keys:
                self.cache.pop(key, None)
        model = getattr(self.registry, 'body_pose_model', None)
        if not self.enabled or model is None or not tracks:
            return {}, 0.0
        attention = any(c.rotation_score or c.lean_score or c.rearward_score for c in self.cache.values())
        # Adaptive downshift for dense scenes avoids pose dominating GPU work.
        interval = max(self.attention_interval if attention else self.interval, .5 if len(tracks) > 8 else 0)
        if timestamp - self.last_run < interval:
            return {t.track_id: replace(self.cache[(camera_id, t.track_id)], fresh=False)
                    for t in tracks if (camera_id, t.track_id) in self.cache}, 0.0
        self.last_run = timestamp
        try:
            with self.registry.inference_lock:
                output = model.predict(frame, imgsz=self.image_size, device=str(self.registry.device),
                                       verbose=False, conf=.45, max_det=32)[0]
            boxes = output.boxes.xyxy.cpu().numpy()
            points = output.keypoints.data.cpu().numpy() if output.keypoints is not None else []
        except Exception:
            logger.warning('Optional body pose failed; cues unavailable', exc_info=True)
            self.cache.clear()
            return {}, (time.perf_counter() - start) * 1000
        cues = {}
        overlaps = {(t.track_id, i): t.bbox.iou(BBox(*box)) for t in tracks for i, box in enumerate(boxes)}
        for track in tracks:
            key = (camera_id, track.track_id)
            ranked = sorted(((overlaps[(track.track_id, i)], i) for i in range(len(boxes))), reverse=True)
            matched = ranked and ranked[0][0] >= .35 and len(points) > ranked[0][1]
            ambiguous = track.lost or any(track.bbox.iou(other.bbox) > .30 for other in tracks if other.track_id != track.track_id)
            if matched:
                score, index = ranked[0]
                ambiguous |= len(ranked) > 1 and score - ranked[1][0] < .15
                ambiguous |= any(score - overlaps[(other.track_id, index)] < .15 for other in tracks if other.track_id != track.track_id)
            if ambiguous:
                self.geometry.states.pop(key, None)
            if not matched or ambiguous:
                cue = BodyCue(status=ObservationStatus.UNAVAILABLE, observed_at=timestamp, fresh=True,
                              identity_ambiguous=bool(ambiguous), uncertainty='AMBIGUOUS_OWNER' if ambiguous else 'POSE_NOT_MATCHED')
            else:
                posture, headpose = posture_cues.get(track.track_id), headpose_cues.get(track.track_id)
                neutral = bool(posture and posture.probabilities.get('NORMAL_UPRIGHT', 0) >= .6 and
                               headpose and headpose.yaw_deg is not None and abs(headpose.yaw_deg) < 18)
                cue = self.geometry.measure(key, points[index], track.bbox, timestamp, neutral=neutral)
            cues[track.track_id] = cue
            self.cache[key] = cue
        return cues, (time.perf_counter() - start) * 1000
