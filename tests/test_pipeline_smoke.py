"""Integration smoke test for end-to-end ExamMonitoringPipeline."""

from typing import List, Optional
import numpy as np
import pytest

from src.video.base import VideoFrame, VideoSource
from src.detection.types import BBox, Detection, BehaviorDetection
from src.detection.object_detector import ObjectDetector
from src.detection.behavior_detector import BehaviorDetector
from src.tracking.bytetrack import ByteTrackTracker
from src.analysis.object_association import ObjectAssociator
from src.analysis.head_pose import OptionalHeadPoseEstimator
from src.behavior.rules import BehaviorRuleEngine, RuleDefinition
from src.behavior.scorer import RiskScorer
from src.behavior.event_manager import EventManager
from src.pipeline import ExamMonitoringPipeline


class SyntheticVideoSource(VideoSource):
    """Generates synthetic video frames for integration tests."""

    def __init__(self, frame_count: int = 20, fps: float = 10.0):
        super().__init__(source_id="synthetic-cam", source_type="synthetic")
        self.total_frames = frame_count
        self._fps = fps
        self._width = 640
        self._height = 480
        self._opened = True

    def open(self) -> bool:
        self._opened = True
        return True

    def read(self) -> Optional[VideoFrame]:
        if self._frame_count >= self.total_frames:
            return None

        self._frame_count += 1
        # Create blank 3-channel image
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        ts = 1000.0 + (self._frame_count / self._fps)

        return VideoFrame(
            frame=frame,
            timestamp=ts,
            frame_idx=self._frame_count,
            fps=self._fps,
            width=self._width,
            height=self._height,
            source_id=self.source_id,
        )

    def release(self) -> None:
        self._opened = False

    def is_opened(self) -> bool:
        return self._opened

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height


class MockObjectDetector(ObjectDetector):
    """Outputs synthetic person and phone detections."""
    def detect(self, frame: np.ndarray) -> List[Detection]:
        return [
            # Tracked person in center
            Detection(bbox=BBox(150, 100, 350, 400), class_id=0, class_name="person", confidence=0.92),
            # Associated phone
            Detection(bbox=BBox(220, 300, 260, 360), class_id=67, class_name="cell phone", confidence=0.88),
        ]


class MockBehaviorDetector(BehaviorDetector):
    """Outputs synthetic behavior detections."""
    def detect(self, frame: np.ndarray, person_boxes: Optional[List[BBox]] = None) -> List[BehaviorDetection]:
        return [
            BehaviorDetection(bbox=BBox(150, 100, 350, 400), behavior="use_phone", confidence=0.89)
        ]


def test_pipeline_end_to_end_smoke():
    source = SyntheticVideoSource(frame_count=15, fps=5.0)

    obj_detector = MockObjectDetector()
    beh_detector = MockBehaviorDetector()
    tracker = ByteTrackTracker(track_high_thresh=0.3, track_buffer=10, frame_rate=5)

    rule_engine = BehaviorRuleEngine()
    rule_engine.add_rule(
        RuleDefinition(
            rule_id="prolonged_phone_use",
            name="Prolonged Phone Use",
            behavior="use_phone",
            require_phone=True,
            min_duration_seconds=1.5,
            window_seconds=10.0,
            severity="high",
            score=85.0,
        )
    )

    risk_scorer = RiskScorer(high_threshold=75.0)
    event_manager = EventManager(camera_id="synthetic-cam", cooldown_seconds=5.0)

    # Collect events dispatched to listener
    dispatched_events = []
    event_manager.add_listener(lambda ev: dispatched_events.append(ev))

    pipeline = ExamMonitoringPipeline(
        video_source=source,
        object_detector=obj_detector,
        behavior_detector=beh_detector,
        tracker=tracker,
        rule_engine=rule_engine,
        risk_scorer=risk_scorer,
        event_manager=event_manager,
        target_inference_fps=10.0,
    )

    total_processed = 0
    while True:
        vframe = source.read()
        if vframe is None:
            break
        res = pipeline.process_frame(vframe)
        total_processed += 1
        assert res.annotated_frame is not None
        assert res.annotated_frame.shape == (480, 640, 3)

    assert total_processed == 15
    pipeline.close()

    # ByteTrack should have tracked Student #1 and flagged suspicious phone event
    assert len(dispatched_events) >= 1
    ev = dispatched_events[0]
    assert ev.track_id == 1
    assert ev.risk_level == "HIGH"
    assert ev.status == "new"
    assert ev.score >= 85.0
