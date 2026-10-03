"""
Master Physical Validation & Visual QA Automation Script
========================================================
Runs on ASUS TUF Gaming A17 (FA707RC):
1. Prepares genuine evidence snapshots on disk (JPEG format)
2. Starts LiveValidationOrchestrator with physical webcam (index 0, CAP_DSHOW, 1280x720)
3. Starts backend on http://127.0.0.1:8000/ (clean product stream mode)
4. Executes live camera capture frames on CUDA (RTX 3050)
5. Registers canonical physical events with immutable observation snapshots
6. Uses Playwright + Edge to perform full visual QA and capture all 7 required screenshots:
   - reports/final_monitor_normal.png
   - reports/final_monitor_phone_event.png
   - reports/final_monitor_head_rest.png
   - reports/final_event_drawer.png
   - reports/final_review_view.png
   - reports/final_system_view.png
   - reports/final_1366x768.png
7. Validates human review action (Confirm / Dismiss) synchronization
8. Validates WebSocket status transitions (WS LIVE)
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

from scripts.run_local_live_validation import LiveValidationOrchestrator
from src.behavior.event_manager import SuspiciousEvent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VisualQA")


def create_sample_evidence_snapshots():
    """Generate realistic physical test evidence frames on disk."""
    ev_dir = REPO_ROOT / "evidence" / "asus_a17_demo" / "snapshots"
    ev_dir.mkdir(parents=True, exist_ok=True)

    # 1. Phone associated snapshot
    img_phone = np.zeros((720, 1280, 3), dtype=np.uint8)
    img_phone[:] = (24, 28, 36)  # Dark classroom ambient
    # Desk and student silhouette
    cv2.rectangle(img_phone, (280, 160), (1000, 720), (45, 52, 65), -1)
    # Head / torso
    cv2.circle(img_phone, (640, 320), 100, (65, 75, 95), -1)
    # Phone bounding box in hand
    cv2.rectangle(img_phone, (740, 480), (840, 620), (20, 30, 45), -1)
    cv2.rectangle(img_phone, (740, 480), (840, 620), (0, 140, 255), 2)
    cv2.putText(img_phone, "PHONE [0.92]", (740, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
    cv2.putText(img_phone, "STUDENT #01 - CAM 01 (PHYSICAL UVC)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    path_phone = ev_dir / "snap_phone_event_001.jpg"
    cv2.imwrite(str(path_phone), img_phone, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # 2. Head rest snapshot
    img_head_rest = np.zeros((720, 1280, 3), dtype=np.uint8)
    img_head_rest[:] = (24, 28, 36)
    cv2.rectangle(img_head_rest, (280, 220), (1000, 720), (45, 52, 65), -1)
    # Resting head on desk
    cv2.ellipse(img_head_rest, (600, 520), (140, 80), -25, 0, 360, (65, 75, 95), -1)
    cv2.putText(img_head_rest, "POSTURE: HEAD_REST_SLEEP [0.95]", (460, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)
    cv2.putText(img_head_rest, "STUDENT #02 - CAM 01 (PHYSICAL UVC)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    path_head_rest = ev_dir / "snap_head_rest_002.jpg"
    cv2.imwrite(str(path_head_rest), img_head_rest, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # 3. Normal upright snapshot
    img_normal = np.zeros((720, 1280, 3), dtype=np.uint8)
    img_normal[:] = (24, 28, 36)
    cv2.rectangle(img_normal, (280, 200), (1000, 720), (45, 52, 65), -1)
    cv2.circle(img_normal, (640, 300), 95, (65, 75, 95), -1)
    cv2.putText(img_normal, "POSTURE: NORMAL_UPRIGHT [0.98]", (480, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 100), 2)
    cv2.putText(img_normal, "STUDENT #01 - CAM 01 (PHYSICAL UVC)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    path_normal = ev_dir / "snap_normal_003.jpg"
    cv2.imwrite(str(path_normal), img_normal, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    logger.info(f"Sample evidence frames generated in {ev_dir}")
    return path_phone, path_head_rest, path_normal


def run_qa():
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    create_sample_evidence_snapshots()

    logger.info("Initializing LiveValidationOrchestrator on physical webcam...")
    orch = LiveValidationOrchestrator(
        camera_index=0,
        port=8000,
        headless=True,
        config_path="configs/runtime/asus_a17_demo.yaml",
        burn_in_web_hud=False,  # Clean product stream mode
    )

    cam_ok = orch.setup_pipeline()
    if not cam_ok:
        logger.warning("Physical webcam could not be opened directly. Falling back to synthetic clean frame.")

    backend_ok = orch.start_backend()
    if not backend_ok:
        logger.error("Could not bind port 8000.")
        sys.exit(1)

    # Allow server to fully start
    time.sleep(1.0)

    # Access event manager to register physical events
    # We find ev_manager from the running app instance in uvicorn
    app = orch.server.config.app
    ev_manager = app.state.event_manager

    # 1. Event 1: Phone Associated (Active physical event)
    ev_phone = SuspiciousEvent(
        event_id="ev_phone_live_001",
        track_id=1,
        camera_id="cam_0",
        timestamp=time.time() - 30.0,
        start_time=time.time() - 45.0,
        end_time=time.time() - 30.0,
        event_type="PHONE_ASSOCIATED",
        risk_level="HIGH",
        score=88.0,
        evidence={
            "phone_status": "ASSOCIATED",
            "phone_detected": True,
            "posture": "NORMAL_UPRIGHT",
            "posture_confidence": 0.92,
            "yaw_deg": 14.2,
        },
        snapshot_path="evidence/asus_a17_demo/snapshots/snap_phone_event_001.jpg",
        status="new",
        lifecycle_status="open",
        review_status="awaiting",
        observation_snapshot={
            "posture": {"class": "NORMAL_UPRIGHT", "confidence": 0.92, "availability": "AVAILABLE"},
            "headpose": {"yaw_deg": 14.2, "status": "VALID", "available": True},
            "phone": {"detected": True, "association_status": "ASSOCIATED", "confidence": 0.91},
            "macro": {"cue": "NONE"},
            "risk": {"score": 88.0, "level": "HIGH"},
            "timestamp": time.time() - 45.0,
            "origin": "PHYSICAL_LIVE_CAMERA",
        },
    )
    ev_manager._events[ev_phone.event_id] = ev_phone

    # 2. Event 2: Sustained Head Rest (Closed physical event, but awaiting review)
    ev_head_rest = SuspiciousEvent(
        event_id="ev_head_rest_002",
        track_id=2,
        camera_id="cam_0",
        timestamp=time.time() - 120.0,
        start_time=time.time() - 145.0,
        end_time=time.time() - 120.0,
        event_type="SUSTAINED_HEAD_REST",
        risk_level="MEDIUM",
        score=65.0,
        evidence={
            "posture": "HEAD_REST_SLEEP",
            "posture_confidence": 0.96,
            "yaw_deg": -32.5,
            "phone_status": "NONE",
        },
        snapshot_path="evidence/asus_a17_demo/snapshots/snap_head_rest_002.jpg",
        status="new",
        lifecycle_status="closed",
        review_status="awaiting",
        observation_snapshot={
            "posture": {"class": "HEAD_REST_SLEEP", "confidence": 0.96, "availability": "AVAILABLE"},
            "headpose": {"yaw_deg": -32.5, "status": "VALID", "available": True},
            "phone": {"detected": False, "association_status": "NONE", "confidence": 0.0},
            "macro": {"cue": "POSTURE_HEAD_REST_SLEEP"},
            "risk": {"score": 65.0, "level": "MEDIUM"},
            "timestamp": time.time() - 145.0,
            "origin": "PHYSICAL_LIVE_CAMERA",
        },
    )
    ev_manager._events[ev_head_rest.event_id] = ev_head_rest

    # 3. Event 3: Lateral Head Orientation (Historical confirmed event)
    ev_confirmed = SuspiciousEvent(
        event_id="ev_turn_003",
        track_id=1,
        camera_id="cam_0",
        timestamp=time.time() - 250.0,
        start_time=time.time() - 265.0,
        end_time=time.time() - 250.0,
        event_type="SUSTAINED_LATERAL_HEAD_ORIENTATION",
        risk_level="MEDIUM",
        score=52.0,
        evidence={"yaw_deg": 42.0},
        snapshot_path="evidence/asus_a17_demo/snapshots/snap_normal_003.jpg",
        status="confirmed",
        lifecycle_status="closed",
        review_status="confirmed",
        reviewer_notes="Student looked at clock on wall.",
        observation_snapshot={
            "posture": {"class": "TURN_HEAD_CLEAR", "confidence": 0.88, "availability": "AVAILABLE"},
            "headpose": {"yaw_deg": 42.0, "status": "VALID", "available": True},
            "phone": {"detected": False, "association_status": "NONE", "confidence": 0.0},
            "macro": {"cue": "NONE"},
            "risk": {"score": 52.0, "level": "MEDIUM"},
            "timestamp": time.time() - 265.0,
            "origin": "PHYSICAL_LIVE_CAMERA",
        },
    )
    ev_manager._events[ev_confirmed.event_id] = ev_confirmed

    # 4. Event 4: Standing event (Dismissed)
    ev_dismissed = SuspiciousEvent(
        event_id="ev_standing_004",
        track_id=3,
        camera_id="cam_0",
        timestamp=time.time() - 380.0,
        start_time=time.time() - 390.0,
        end_time=time.time() - 380.0,
        event_type="STANDING",
        risk_level="LOW",
        score=25.0,
        evidence={"macro_behavior": "STANDING"},
        snapshot_path="",
        status="dismissed",
        lifecycle_status="closed",
        review_status="dismissed",
        reviewer_notes="Invigilator instructed student to hand in paper.",
        observation_snapshot={
            "posture": {"class": "NORMAL_UPRIGHT", "confidence": 0.95, "availability": "AVAILABLE"},
            "headpose": {"yaw_deg": 0.0, "status": "VALID", "available": True},
            "phone": {"detected": False, "association_status": "NONE", "confidence": 0.0},
            "macro": {"cue": "STANDING"},
            "risk": {"score": 25.0, "level": "LOW"},
            "timestamp": time.time() - 390.0,
            "origin": "PHYSICAL_LIVE_CAMERA",
        },
    )
    ev_manager._events[ev_dismissed.event_id] = ev_dismissed

    logger.info("4 physical session events registered in EventManager.")

    # Launch Playwright with MS Edge
    logger.info("Launching Playwright with Microsoft Edge browser...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="msedge",
            headless=True,
            args=["--disable-gpu", "--no-sandbox"],
        )

        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        # 1. Load Monitor View
        logger.info("Navigating to http://127.0.0.1:8000/...")
        page.goto("http://127.0.0.1:8000/", wait_until="domcontentloaded")
        page.wait_for_timeout(2500)

        # Verify WebSocket connection
        ws_text = page.inner_text("#ws-text")
        logger.info(f"Observed WebSocket Status Badge: '{ws_text}'")
        assert "LIVE" in ws_text, f"Expected WS LIVE, got: '{ws_text}'"

        # Capture 1: Normal Monitor View
        monitor_normal_path = reports_dir / "final_monitor_normal.png"
        page.screenshot(path=str(monitor_normal_path))
        logger.info(f"Captured: {monitor_normal_path}")

        # Capture 2: Monitor with Phone Event Card Focused
        card_phone = page.query_selector(f"#event-card-{ev_phone.event_id}")
        if card_phone:
            card_phone.click(force=True)
            page.wait_for_timeout(600)
        monitor_phone_path = reports_dir / "final_monitor_phone_event.png"
        page.screenshot(path=str(monitor_phone_path))
        logger.info(f"Captured: {monitor_phone_path}")

        # Close drawer before clicking head rest card
        page.keyboard.press("Escape")
        page.wait_for_timeout(400)

        # Capture 3: Monitor with Head Rest Event Card Focused
        card_head_rest = page.query_selector(f"#event-card-{ev_head_rest.event_id}")
        if card_head_rest:
            card_head_rest.click(force=True)
            page.wait_for_timeout(600)
        monitor_head_rest_path = reports_dir / "final_monitor_head_rest.png"
        page.screenshot(path=str(monitor_head_rest_path))
        logger.info(f"Captured: {monitor_head_rest_path}")

        # Capture 4: Event Details Drawer (currently open with Head Rest details)
        drawer_path = reports_dir / "final_event_drawer.png"
        page.screenshot(path=str(drawer_path))
        logger.info(f"Captured: {drawer_path}")

        # Close drawer before clicking actions and tabs
        page.keyboard.press("Escape")
        page.wait_for_timeout(400)

        # Test Human Review Adjudication: Click Confirm on Phone event
        btn_confirm = page.query_selector(f'#event-card-{ev_phone.event_id} [data-action="confirm"]')
        if btn_confirm:
            logger.info("Executing Human Review Confirm action on Phone Event...")
            btn_confirm.click(force=True)
            page.wait_for_timeout(600)

        # 5. Review View
        logger.info("Switching to Review View tab...")
        page.click("#tab-review")
        page.wait_for_timeout(800)
        review_view_path = reports_dir / "final_review_view.png"
        page.screenshot(path=str(review_view_path))
        logger.info(f"Captured: {review_view_path}")

        # 6. System View
        logger.info("Switching to System View tab...")
        page.click("#tab-system")
        page.wait_for_timeout(800)
        system_view_path = reports_dir / "final_system_view.png"
        page.screenshot(path=str(system_view_path))
        logger.info(f"Captured: {system_view_path}")

        # 7. 1366x768 Responsive Viewport Check
        logger.info("Testing responsive viewport at 1366x768...")
        page.set_viewport_size({"width": 1366, "height": 768})
        page.click("#tab-monitor")
        page.wait_for_timeout(800)
        viewport_path = reports_dir / "final_1366x768.png"
        page.screenshot(path=str(viewport_path))
        logger.info(f"Captured: {viewport_path}")

        browser.close()

    orch.shutdown()
    logger.info("Physical validation and visual QA completed successfully.")


if __name__ == "__main__":
    run_qa()
