"""Integration smoke test with trained stage1_best.pt checkpoint."""

import argparse
import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from fastapi.testclient import TestClient
import numpy as np

from src.video.video_file import VideoFileSource
from src.detection.object_detector import YOLOObjectDetector
from src.detection.behavior_detector import YOLOBehaviorDetector
from src.tracking.bytetrack import ByteTrackTracker
from src.analysis.object_association import ObjectAssociator
from src.analysis.head_pose import OptionalHeadPoseEstimator
from src.behavior.rules import BehaviorRuleEngine, RuleDefinition
from src.behavior.scorer import RiskScorer
from src.behavior.event_manager import EventManager
from src.pipeline import ExamMonitoringPipeline
from src.api.main import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("smoke_test")


def run_smoke_test(
    weights_path: str = "models/trained/stage1_best.pt",
    video_path: str = "samples/sample_exam.mp4",
    num_frames: int = 25,
) -> bool:
    weights = Path(weights_path)
    if not weights.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {weights}")

    logger.info(f"--- Running Non-Destructive Stage 1 Smoke Test with {weights} ---")

    # 1. Initialize Video Source
    v_path = Path(video_path)
    if not v_path.exists():
        raise FileNotFoundError(f"Sample video not found at {v_path}")

    source = VideoFileSource(file_path=str(v_path), source_id="smoke-cam", loop=False)
    if not source.open():
        raise RuntimeError(f"Could not open video file {v_path}")

    # 2. Wire Components
    logger.info("Initializing YOLO object and Stage 1 behavior detectors...")
    # Standard COCO detector for person and phone
    obj_detector = YOLOObjectDetector(model_path="yolo11n.pt", device="0", confidence=0.30)
    # Custom Stage 1 behavior detector pointing to best.pt
    beh_detector = YOLOBehaviorDetector(
        model_path=str(weights),
        confidence=0.30,
        image_size=768,
        device="0",
        allow_fallback=False,
    )

    tracker = ByteTrackTracker(track_high_thresh=0.4, track_buffer=15, frame_rate=source.fps)
    associator = ObjectAssociator()
    rule_engine = BehaviorRuleEngine(rules_config_path="configs/risk_rules.yaml")
    risk_scorer = RiskScorer()
    event_manager = EventManager(camera_id="smoke-cam")

    pipeline = ExamMonitoringPipeline(
        video_source=source,
        object_detector=obj_detector,
        behavior_detector=beh_detector,
        tracker=tracker,
        object_associator=associator,
        rule_engine=rule_engine,
        risk_scorer=risk_scorer,
        event_manager=event_manager,
    )

    logger.info(f"Processing {num_frames} frames through pipeline...")
    processed_count = 0
    total_tracks_seen = set()

    for _ in range(num_frames):
        vframe = source.read()
        if vframe is None:
            break
        result = pipeline.process_frame(vframe)
        processed_count += 1
        for trk in result.tracks:
            total_tracks_seen.add(trk.track_id)

    source.release()
    pipeline.close()

    logger.info(f"Pipeline processed {processed_count} frames successfully.")
    logger.info(f"Active student tracks observed: {len(total_tracks_seen)}")

    # 3. Test FastAPI Integration
    logger.info("Testing FastAPI test client with event manager...")
    app = create_app(event_manager=event_manager, camera_id="smoke-cam", camera_type="file")
    client = TestClient(app)

    health_resp = client.get("/health")
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
    health_data = health_resp.json()
    assert health_data["status"] == "ok"

    events_resp = client.get("/api/events")
    assert events_resp.status_code == 200, f"Events check failed: {events_resp.text}"

    logger.info("Smoke test passed: ObjectDetector -> Stage1BehaviorDetector -> ByteTrack -> Buffer -> Rules -> FastAPI OK!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/trained/stage1_best.pt")
    parser.add_argument("--video", default="samples/sample_exam.mp4")
    parser.add_argument("--frames", type=int, default=25)
    args = parser.parse_args()

    success = run_smoke_test(weights_path=args.weights, video_path=args.video, num_frames=args.frames)
    sys.exit(0 if success else 1)
