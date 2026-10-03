"""Exam Suspicious Behavior Monitoring Pipeline.

Integrates video ingestion, object detection, ByteTrack tracking, secondary object association,
behavior detection, temporal buffer analysis, YAML rules, risk scoring, and evidence capture.
"""

from dataclasses import dataclass
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.video.base import VideoFrame, VideoSource
from src.detection.types import BBox, Detection, BehaviorDetection
from src.detection.object_detector import ObjectDetector, YOLOObjectDetector
from src.detection.behavior_detector import BehaviorDetector, YOLOBehaviorDetector
from src.tracking.tracker import BaseTracker, Track
from src.tracking.bytetrack import ByteTrackTracker
from src.analysis.object_association import ObjectAssociator, AssociatedObjects
from src.analysis.head_pose import HeadPoseEstimator, OptionalHeadPoseEstimator
from src.analysis.fusion import FusionEngine, StudentObservation
from src.analysis.temporal_buffer import TemporalBuffer
from src.behavior.rules import BehaviorRuleEngine, RuleMatch
from src.behavior.scorer import RiskScorer, RiskAssessment
from src.behavior.event_manager import EventManager, SuspiciousEvent
from src.evidence.snapshot import SnapshotCapture
from src.evidence.clip_recorder import RollingClipRecorder

logger = logging.getLogger(__name__)


@dataclass
class PipelineFrameResult:
    """Summary of processing result for a single video frame."""
    frame_idx: int
    timestamp: float
    annotated_frame: np.ndarray
    tracks: List[Track]
    observations: List[StudentObservation]
    assessments: List[RiskAssessment]
    new_events: List[SuspiciousEvent]
    fps_metric: float
    inference_fps_metric: float


class ExamMonitoringPipeline:
    """End-to-End Orchestrator for Exam Behavior Monitoring."""

    def __init__(
        self,
        video_source: VideoSource,
        object_detector: ObjectDetector,
        behavior_detector: BehaviorDetector,
        tracker: BaseTracker,
        rule_engine: BehaviorRuleEngine,
        risk_scorer: RiskScorer,
        event_manager: EventManager,
        object_associator: Optional[ObjectAssociator] = None,
        head_pose_estimator: Optional[HeadPoseEstimator] = None,
        snapshot_capture: Optional[SnapshotCapture] = None,
        clip_recorder: Optional[RollingClipRecorder] = None,
        target_inference_fps: float = 12.0,
    ):
        self.source = video_source
        self.object_detector = object_detector
        self.behavior_detector = behavior_detector
        self.tracker = tracker
        self.rule_engine = rule_engine
        self.risk_scorer = risk_scorer
        self.event_manager = event_manager

        self.object_associator = object_associator or ObjectAssociator()
        self.head_pose_estimator = head_pose_estimator or OptionalHeadPoseEstimator(enabled=False)
        self.fusion_engine = FusionEngine(head_pose_estimator=self.head_pose_estimator)
        self.temporal_buffer = TemporalBuffer(max_history_seconds=300.0, eviction_inactive_seconds=60.0)

        self.snapshot_capture = snapshot_capture
        self.clip_recorder = clip_recorder

        self.target_inference_fps = target_inference_fps
        self._min_inference_interval = 1.0 / max(1.0, target_inference_fps)
        self._last_inference_time = 0.0

        # Performance metrics
        self._frame_times: List[float] = []
        self._inference_times: List[float] = []
        self._last_perf_log = time.time()

        # Cache last detections to reuse during skipped frames if needed
        self._last_tracks: List[Track] = []
        self._last_assessments: List[RiskAssessment] = []
        self._last_observations: List[StudentObservation] = []

    def process_frame(self, video_frame: VideoFrame) -> PipelineFrameResult:
        """Process a single incoming frame through the full inspection pipeline."""
        start_time = time.time()
        frame = video_frame.frame
        ts = video_frame.timestamp
        h, w = frame.shape[:2]

        # Push to rolling clip recorder regardless of inference skipping
        if self.clip_recorder is not None:
            self.clip_recorder.push_frame(frame, ts)

        # Check if inference should run on this frame or if we should skip
        should_infer = (
            self._last_inference_time == 0.0
            or (ts - self._last_inference_time) >= (self._min_inference_interval - 1e-4)
        )

        new_events: List[SuspiciousEvent] = []

        if should_infer:
            inf_start = time.time()
            self._last_inference_time = ts

            # 1. Object detection (person, cell phone)
            detections = self.object_detector.detect(frame)

            # 2. Multi-object tracking (persistent student IDs)
            self._last_tracks = self.tracker.update(detections, ts, (h, w))

            # 3. Spatial secondary object association (phone to student)
            associated_objs = self.object_associator.associate(self._last_tracks, detections)

            # 4. Behavior detection (custom YOLO model or adapter)
            person_boxes = [t.bbox for t in self._last_tracks]
            behavior_dets = self.behavior_detector.detect(frame, person_boxes)

            # 5. Signal fusion into per-student observation
            self._last_observations = self.fusion_engine.fuse(
                frame=frame,
                tracks=self._last_tracks,
                behavior_detections=behavior_dets,
                associated_objects=associated_objs,
                timestamp=ts,
            )

            # 6. Temporal buffer update
            for obs in self._last_observations:
                self.temporal_buffer.push(obs)

            # Evict stale tracks every 30 seconds
            if video_frame.frame_idx % 90 == 0:
                self.temporal_buffer.evict_expired(ts)

            # 7. Behavior rules & risk scoring
            self._last_assessments = []
            for obs in self._last_observations:
                t_id = obs.track_id
                matches = self.rule_engine.evaluate_track(t_id, self.temporal_buffer, ts)
                assessment = self.risk_scorer.score_matches(t_id, matches)
                self._last_assessments.append(assessment)

                # 8. Event creation & debounced dispatch
                event = self.event_manager.process_assessment(assessment, ts, self.source.source_id)
                if event:
                    new_events.append(event)

            inf_duration = time.time() - inf_start
            self._inference_times.append(inf_duration)
            if len(self._inference_times) > 30:
                self._inference_times.pop(0)

        # 9. Handle evidence capture for newly triggered events
        if new_events:
            for ev in new_events:
                # Save snapshot
                if self.snapshot_capture is not None:
                    # Save clean frame and annotated frame
                    clean_path = self.snapshot_capture.capture_async(frame, ev.event_id, "clean")
                    ev.snapshot_path = clean_path

                # Trigger video clip recording
                if self.clip_recorder is not None:
                    clip_path = self.clip_recorder.trigger_clip(ev.event_id, ts, fps=self.source.fps)
                    ev.clip_path = clip_path

        # 10. Measure overall processing FPS
        total_dt = time.time() - start_time
        self._frame_times.append(total_dt)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)

        fps_metric = 1.0 / (sum(self._frame_times) / len(self._frame_times)) if self._frame_times else 0.0
        avg_inf_dt = sum(self._inference_times) / len(self._inference_times) if self._inference_times else 0.001
        inf_fps_metric = 1.0 / avg_inf_dt if avg_inf_dt > 0 else 0.0

        # 11. Render annotated frame for UI/live monitor
        annotated_frame = self.annotate_frame(
            frame=frame.copy(),
            tracks=self._last_tracks,
            observations=self._last_observations,
            assessments=self._last_assessments,
            fps=fps_metric,
            inf_fps=inf_fps_metric,
            source_id=self.source.source_id,
        )

        return PipelineFrameResult(
            frame_idx=video_frame.frame_idx,
            timestamp=ts,
            annotated_frame=annotated_frame,
            tracks=self._last_tracks,
            observations=self._last_observations,
            assessments=self._last_assessments,
            new_events=new_events,
            fps_metric=round(fps_metric, 1),
            inference_fps_metric=round(inf_fps_metric, 1),
        )

    def annotate_frame(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        observations: List[StudentObservation],
        assessments: List[RiskAssessment],
        fps: float,
        inf_fps: float,
        source_id: str,
    ) -> np.ndarray:
        """Render informative visual annotations on frame."""
        h, w = frame.shape[:2]
        obs_map = {obs.track_id: obs for obs in observations}
        risk_map = {ass.track_id: ass for ass in assessments}

        # Draw each tracked student
        for track in tracks:
            t_id = track.track_id
            x1, y1, x2, y2 = track.bbox.as_int_tuple()

            obs = obs_map.get(t_id)
            ass = risk_map.get(t_id)

            behavior_str = obs.behavior if obs else "normal"
            conf = obs.behavior_confidence if obs else 0.50
            risk_level = ass.risk_level if ass else "NORMAL"
            score = ass.score if ass else 0.0

            # Bounding box color based on configured risk level
            if risk_level == "HIGH":
                color = (0, 0, 235)       # Red
                thickness = 3
            elif risk_level == "MEDIUM":
                color = (0, 165, 255)     # Orange
                thickness = 2
            elif risk_level == "LOW":
                color = (0, 215, 255)     # Yellow
                thickness = 2
            else:
                color = (0, 200, 0)       # Green (Normal)
                thickness = 2

            # Student box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Label banner
            label = f"Student #{t_id} | {behavior_str} ({int(conf * 100)}%)"
            if risk_level != "NORMAL":
                label += f" | {risk_level} ({score:.0f})"
            if obs and obs.phone_present:
                label += " [PHONE]"

            # Background for label
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + lw + 8, max(0, y1)), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 4, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Top status dashboard banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (20, 24, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        status_text = (
            f"Source: {source_id} | Video: {fps:.1f} FPS | AI: {inf_fps:.1f} FPS | "
            f"Students: {len(tracks)} | Total Events: {len(self.event_manager._events)}"
        )
        cv2.putText(
            frame,
            status_text,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 220),
            1,
            cv2.LINE_AA,
        )

        return frame

    def close(self) -> None:
        """Clean up pipeline resources."""
        self.source.release()
        if self.snapshot_capture:
            self.snapshot_capture.shutdown()
        if self.clip_recorder:
            self.clip_recorder.shutdown()
        logger.info("ExamMonitoringPipeline shut down cleanly.")


# Stage 2 Orchestration Re-export
try:
    from src.orchestration.stage2_pipeline import Stage2Pipeline
except ImportError:
    pass
