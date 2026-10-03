"""Unified CLI entrypoint for Exam Suspicious Behavior Detection.

Supports:
1. Laptop Webcam demo:
   python scripts/run_demo.py --source webcam [--input 0] [--show] [--serve]

2. Video file offline testing:
   python scripts/run_demo.py --source video --input samples/exam.mp4 [--show]

3. School CCTV / RTSP stream:
   python scripts/run_demo.py --source rtsp --input rtsp://admin:pass@192.168.1.100:554/stream [--serve]
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time
from typing import Optional
import cv2
import uvicorn
import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.video.factory import create_video_source
from src.detection.object_detector import YOLOObjectDetector
from src.detection.behavior_detector import YOLOBehaviorDetector
from src.tracking.bytetrack import ByteTrackTracker
from src.analysis.object_association import ObjectAssociator
from src.analysis.head_pose import OptionalHeadPoseEstimator
from src.behavior.rules import BehaviorRuleEngine
from src.behavior.scorer import RiskScorer
from src.behavior.event_manager import EventManager
from src.evidence.snapshot import SnapshotCapture
from src.evidence.clip_recorder import RollingClipRecorder
from src.pipeline import ExamMonitoringPipeline
from src.api.main import create_app
from src.api.websocket import ConnectionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ExamMonitor")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exam Suspicious Behavior Detection - Multi-Source Demo"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="webcam",
        choices=["webcam", "video", "rtsp"],
        help="Video source type: 'webcam' (demo), 'video' (offline file), or 'rtsp' (CCTV)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Source input: device index (e.g. 0), file path (e.g. sample.mp4), or RTSP URL",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs"),
        help="Path to directory containing YAML configs",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=True,
        help="Display OpenCV live visualization window (default: True)",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Run headless without OpenCV GUI window",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        default=False,
        help="Start the FastAPI dashboard server concurrently on port 8000",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port for FastAPI web server (default: 8050)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device for inference ('cpu', 'cuda', '0')",
    )
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    args = parse_args()

    # Load configurations
    camera_cfg = load_yaml(os.path.join(args.config_dir, "camera.yaml")).get("camera", {})
    inference_cfg = load_yaml(os.path.join(args.config_dir, "inference.yaml")).get("inference", {})
    rules_cfg_path = os.path.join(args.config_dir, "risk_rules.yaml")
    evidence_cfg = load_yaml(os.path.join(args.config_dir, "evidence.yaml")).get("evidence", {})
    tracking_cfg = load_yaml(os.path.join(args.config_dir, "tracking.yaml"))

    # Resolve video source parameters
    source_type = args.source
    if args.input is not None:
        source_val = args.input
    else:
        if source_type == "webcam":
            source_val = camera_cfg.get("source", 0)
        elif source_type == "video":
            source_val = camera_cfg.get("source", "samples/demo.mp4")
        else:
            source_val = camera_cfg.get("source", "rtsp://localhost:554/live")

    source_id = camera_cfg.get("id", f"{source_type}-cam")

    logger.info("=" * 60)
    logger.info(" EXAM SUSPICIOUS BEHAVIOR DETECTION SYSTEM")
    logger.info(f" Mode: {source_type.upper()} | Input: {source_val}")
    logger.info(" Target CCTV Production: Same downstream pipeline as demo webcam.")
    logger.info("=" * 60)

    # 1. Video Source
    video_source = create_video_source(
        source_type=source_type,
        source=source_val,
        source_id=source_id,
        width=int(camera_cfg.get("width", 1280)),
        height=int(camera_cfg.get("height", 720)),
        fps=float(camera_cfg.get("fps", 30.0)),
        extra_config={
            "loop": True,
            "reconnect_attempts": int(camera_cfg.get("reconnect_attempts", 5)),
            "reconnect_delay_seconds": float(camera_cfg.get("reconnect_delay_seconds", 2.0)),
        },
    )

    if not video_source.open():
        logger.critical(f"Could not open video source '{source_type}:{source_val}'. Exiting.")
        sys.exit(1)

    # 2. Object Detector (person, cell phone)
    obj_model_path = inference_cfg.get("object_model_path", "yolo11n.pt")
    object_detector = YOLOObjectDetector(
        model_path=obj_model_path,
        confidence=float(inference_cfg.get("object_confidence", 0.35)),
        iou_threshold=float(inference_cfg.get("object_iou_threshold", 0.45)),
        image_size=int(inference_cfg.get("image_size", 640)),
        device=args.device,
    )

    # 3. Behavior Detector (custom model or fallback heuristic adapter)
    beh_model_path = inference_cfg.get("behavior_model_path", "models/trained/behavior_best.pt")
    behavior_detector = YOLOBehaviorDetector(
        model_path=beh_model_path,
        confidence=float(inference_cfg.get("behavior_confidence", 0.40)),
        image_size=int(inference_cfg.get("image_size", 640)),
        device=args.device,
        allow_fallback=bool(inference_cfg.get("behavior_fallback_heuristic", True)),
    )

    # 4. ByteTrack Multi-Object Tracker
    t_cfg = tracking_cfg.get("tracking", {})
    tracker = ByteTrackTracker(
        track_high_thresh=float(t_cfg.get("track_thresh", 0.45)),
        match_thresh=float(t_cfg.get("match_thresh", 0.80)),
        track_buffer=int(t_cfg.get("track_buffer", 30)),
        frame_rate=int(video_source.fps or 30),
    )

    # 5. Spatial Object Associator
    a_cfg = tracking_cfg.get("association", {})
    object_associator = ObjectAssociator(
        expand_student_bbox_ratio=float(a_cfg.get("expand_student_bbox_ratio", 0.15)),
        max_center_distance_ratio=float(a_cfg.get("max_center_distance_ratio", 0.50)),
        min_iou_overlap=float(a_cfg.get("min_iou_overlap", 0.05)),
    )

    # 6. Optional Head Pose Estimator
    head_pose_cfg = inference_cfg.get("head_pose", {})
    head_pose_estimator = OptionalHeadPoseEstimator(
        enabled=bool(head_pose_cfg.get("enabled", False)),
        min_face_size_pixels=int(head_pose_cfg.get("min_face_size_pixels", 60)),
    )

    # 7. Behavior Rule Engine & Risk Scorer
    rule_engine = BehaviorRuleEngine(rules_config_path=rules_cfg_path)
    risk_scorer = RiskScorer(config_path=rules_cfg_path)

    # 8. Event Manager & Evidence Capture
    event_manager = EventManager(camera_id=source_id, config_path=rules_cfg_path)
    connection_manager = ConnectionManager()

    snapshot_capture = SnapshotCapture(
        output_dir=evidence_cfg.get("snapshot_dir", "storage/evidence/snapshots")
    )
    clip_recorder = RollingClipRecorder(
        output_dir=evidence_cfg.get("clip_dir", "storage/evidence/clips"),
        pre_event_seconds=float(evidence_cfg.get("pre_event_seconds", 4.0)),
        post_event_seconds=float(evidence_cfg.get("post_event_seconds", 4.0)),
    )

    # 9. Pipeline Assembly
    target_inf_fps = float(inference_cfg.get("target_inference_fps", 12.0))
    pipeline = ExamMonitoringPipeline(
        video_source=video_source,
        object_detector=object_detector,
        behavior_detector=behavior_detector,
        tracker=tracker,
        rule_engine=rule_engine,
        risk_scorer=risk_scorer,
        event_manager=event_manager,
        object_associator=object_associator,
        head_pose_estimator=head_pose_estimator,
        snapshot_capture=snapshot_capture,
        clip_recorder=clip_recorder,
        target_inference_fps=target_inf_fps,
    )

    # 10. Start FastAPI Server in Background if --serve is specified
    if args.serve:
        app = create_app(
            event_manager=event_manager,
            connection_manager=connection_manager,
            camera_id=source_id,
            camera_type=source_type,
        )

        def _run_server():
            logger.info(f"Starting FastAPI Dashboard on http://localhost:{args.port}...")
            uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")

        server_thread = threading.Thread(target=_run_server, daemon=True)
        server_thread.start()
        time.sleep(1.0)
        logger.info(f"Web Dashboard ready at: http://localhost:{args.port}")

    # Handle graceful exit
    stop_event = threading.Event()

    def _sig_handler(sig, frame):
        logger.info("Shutdown signal received. Stopping pipeline...")
        stop_event.set()

    signal.signal(signal.SIGINT, _sig_handler)

    logger.info("Pipeline running. Press 'q' in OpenCV window or Ctrl+C to exit.")

    window_name = f"Exam Suspicious Behavior Detection [{source_type.upper()}]"
    if args.show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 576)

    try:
        while not stop_event.is_set():
            vframe = video_source.read()
            if vframe is None:
                logger.info("Video stream ended or frame is None.")
                break

            result = pipeline.process_frame(vframe)

            if args.show:
                cv2.imshow(window_name, result.annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit key pressed.")
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        pipeline.close()
        if args.show:
            cv2.destroyAllWindows()
        logger.info("Application exited successfully.")


if __name__ == "__main__":
    main()
