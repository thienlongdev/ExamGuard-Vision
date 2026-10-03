"""
ASUS TUF Gaming A17 — School Demo One-Command Launcher
======================================================
Launches the full ExamGuard-Vision perception pipeline on the physical ASUS A17 laptop:
- Physical Webcam: 1280x720 @ 30 FPS (CAP_DSHOW)
- YOLO26m Object Detector + ByteTrack
- Specialized Posture Classifier (MobileNetV3) @ 224x224
- HopeNet-Yaw Headpose Estimator @ 224x224
- Macro Behavior Detector (Stage 1.5) @ 768x768
- Phone Spatial Association
- V4D Multi-Cue Temporal Fusion
- Live Invigilator Dashboard at http://127.0.0.1:8000/
- Real-time WebSocket broadcasting at ws://127.0.0.1:8000/ws/events
- Real-time OpenCV Debug HUD
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_local_live_validation import LiveValidationOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("asus_a17_demo")

DEMO_CONFIG_PATH = "configs/runtime/asus_a17_demo.yaml"


def main():
    parser = argparse.ArgumentParser(description="ASUS TUF Gaming A17 One-Command School Demo Launcher")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--port", type=int, default=8000, help="Localhost port for dashboard (default 8000)")
    parser.add_argument("--headless", action="store_true", help="Run without cv2.imshow GUI window")
    parser.add_argument("--burn-in-web-hud", action="store_true", help="Burn debug HUD into web stream (defaults to False for clean product stream)")
    parser.add_argument("--duration", type=float, default=None, help="Optional duration in seconds (defaults to infinite until stopped)")
    args = parser.parse_args()

    print("\n" + "=" * 76)
    print("       EXAMGUARD-VISION — LIVE EXAM BEHAVIOR EVIDENCE SYSTEM")
    print("                 ASUS TUF Gaming A17 (FA707RC)")
    print("=" * 76)
    print(f" Profile:       ASUS_A17_BALANCED_DEMO ({DEMO_CONFIG_PATH})")
    print(f" Target Device: NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB VRAM)")
    print(f" Camera Source: Physical UVC Webcam Index {args.camera_index} (CAP_DSHOW)")
    print(f" Resolution:    1280x720 @ 30.0 FPS")
    print(f" Web Stream:    {'DEBUG HUD (Burned-in)' if args.burn_in_web_hud else 'CLEAN PRODUCT STREAM'}")
    print("=" * 76)
    print(" Initializing perception pipeline and starting localhost backend...")

    orch = LiveValidationOrchestrator(
        camera_index=args.camera_index,
        port=args.port,
        headless=args.headless,
        config_path=DEMO_CONFIG_PATH,
        burn_in_web_hud=args.burn_in_web_hud,
    )

    if not orch.setup_pipeline():
        logger.error("ERROR: Could not open physical webcam. Please verify webcam connection.")
        sys.exit(1)

    if not orch.start_backend():
        logger.error("ERROR: Could not bind localhost port 8000.")
        sys.exit(1)

    orch.current_protocol_step = "SCHOOL_DEMO_LIVE"

    dashboard_url = f"http://127.0.0.1:{args.port}/"
    dashboard_alias = f"http://127.0.0.1:{args.port}/dashboard"
    ws_url = f"ws://127.0.0.1:{args.port}/ws/events"

    print("\n" + "*" * 76)
    print(" >>> SYSTEM IS LIVE AND OPERATIONAL <<<")
    print(f" Open your browser to view the Live Invigilator Dashboard:")
    print(f"   --> {dashboard_url}")
    print(f"   --> {dashboard_alias}")
    print(f" Real-time WebSocket: {ws_url}")
    print(f" GPU VRAM Allocated:  {orch.sys_info['gpu_vram_total_mb']} MB capacity")
    print(" To exit safely: Focus OpenCV window and press 'q' or ESC (or Ctrl+C).")
    print("*" * 76 + "\n")

    try:
        summary = orch.run_live_loop(duration_sec=args.duration)
    except KeyboardInterrupt:
        print("\nStopping demo...")
    finally:
        orch.shutdown()

    print("\nDemo session ended cleanly. Hardware handles released.")


if __name__ == "__main__":
    main()
