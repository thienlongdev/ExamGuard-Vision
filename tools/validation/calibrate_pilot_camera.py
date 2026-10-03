"""
Calibrate Pilot Camera CLI
==========================
Implements Part B requirements:
Usage:
  python scripts/calibrate_pilot_camera.py --source samples/sample_exam.mp4 --camera-id cam_01
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pilot.profile import CameraProfile, ViewpointProfile, Resolution
from src.pilot.calibration import CameraCalibrator
from src.video.video_file import VideoFileSource

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Calibrate Pilot CCTV Camera")
    parser.add_argument("--source", default="samples/sample_exam.mp4", help="Video file path, webcam index, or RTSP URL")
    parser.add_argument("--camera-id", default="cam_hall_01", help="Camera identifier")
    parser.add_argument("--source-type", default="video_file", choices=["video_file", "webcam", "rtsp"])
    parser.add_argument("--viewpoint", default="FRONT_OBLIQUE", choices=["CEILING_HIGH", "FRONT_OBLIQUE", "SIDE_OBLIQUE", "REAR_OBLIQUE", "DESK_LEVEL", "UNKNOWN"])
    parser.add_argument("--frames", type=int, default=90, help="Number of frames to collect for calibration")
    parser.add_argument("--duration", type=float, default=60.0, help="Maximum collection duration in seconds")
    parser.add_argument("--output-dir", default=None, help="Custom output directory")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    vp = ViewpointProfile(args.viewpoint)
    profile = CameraProfile(
        camera_id=args.camera_id,
        name=f"Camera {args.camera_id}",
        source_type=args.source_type,
        source_uri=args.source,
        viewpoint_profile=vp,
        nominal_fps=25.0,
        expected_student_count=15,
    )

    out_dir = args.output_dir or os.path.join("runs", "pilot", "calibration", args.camera_id)
    calibrator = CameraCalibrator(profile=profile, output_dir=out_dir)

    cap_summary, telemetry = calibrator.calibrate(
        max_frames=args.frames,
        max_duration_sec=args.duration,
    )

    print("\n" + "=" * 60)
    print(f"CAMERA CALIBRATION COMPLETE: {args.camera_id}")
    print(f"Posture Capability:  {cap_summary.posture_capability.value}")
    print(f"Headpose Capability: {cap_summary.headpose_capability.value}")
    print(f"Phone Capability:    {cap_summary.phone_capability.value}")
    print(f"Tracking Capability: {cap_summary.tracking_capability.value}")
    print(f"Quantitative Coverage (>=120px): {cap_summary.pct_height_gte_120:.1f}%")
    print(f"Quantitative Coverage (Head >=25x25): {cap_summary.pct_head_crop_gte_25:.1f}%")
    print(f"Artifacts saved to: {out_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
