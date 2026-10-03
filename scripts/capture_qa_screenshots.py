"""
QA Screenshot Capture Automation
================================
Spins up local server with rich mock state and captures all 7 required local screenshots:
1. final_vi_monitor_normal.png (1920x1080)
2. final_vi_monitor_attention.png (1920x1080)
3. final_vi_monitor_phone_high.png (1920x1080)
4. final_vi_event_drawer.png (1920x1080)
5. final_vi_review.png (1920x1080)
6. final_vi_system.png (1920x1080)
7. final_vi_1366x768.png (1366x768 responsive layout)
"""

import os
import sys
import time
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import uvicorn
import numpy as np
import cv2
from pathlib import Path
from playwright.sync_api import sync_playwright

from src.api.main import create_app
from src.behavior.event_manager import EventManager, SuspiciousEvent


class MockPipeline:
    def __init__(self):
        # Generate 1280x720 dummy frame
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(img, "ExamGuard Vision Live Camera Stream", (420, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)
        _, buf = cv2.imencode(".jpg", img)
        self._latest_jpeg_frame = buf.tobytes()

        self.processed_frames_count = 100
        self.dropped_frames_count = 0
        self.current_queue_depth = 0
        self.vram_allocated_mb = 289.0
        self.effective_fps = 10.0
        self.ingestion_queue = type("MockQueue", (), {"qsize": 0})()
        self._track_metadata = {1: {}, 2: {}, 3: {}}

        # Track 1: Normal (Green)
        # Track 2: Attention (Amber - Head turn)
        # Track 3: High Alert (Red - Phone)
        self._latest_tracks_summary = [
            {
                "track_id": 1,
                "bbox": [120, 140, 420, 580],
                "posture": "NORMAL_READ_WRITE",
                "posture_conf": 0.94,
                "yaw_deg": 3.2,
                "phone_status": "NO_PHONE",
                "macro_behavior": "NORMAL",
                "active_events": [],
                "risk_level": "LOW",
                "visual_state": "SAFE",
            },
            {
                "track_id": 2,
                "bbox": [480, 130, 780, 590],
                "posture": "TURN_HEAD_CLEAR",
                "posture_conf": 0.88,
                "yaw_deg": 32.5,
                "phone_status": "NO_PHONE",
                "macro_behavior": "NORMAL",
                "active_events": [{"canonicalType": "SUSTAINED_LATERAL_HEAD_ORIENTATION", "riskLevel": "MEDIUM", "score": 52}],
                "primary_event": {"canonicalType": "SUSTAINED_LATERAL_HEAD_ORIENTATION", "riskLevel": "MEDIUM", "score": 52},
                "risk_level": "MEDIUM",
                "visual_state": "ATTENTION",
            },
            {
                "track_id": 3,
                "bbox": [840, 140, 1140, 600],
                "posture": "NORMAL_UPRIGHT",
                "posture_conf": 0.92,
                "yaw_deg": -5.1,
                "phone_status": "CLEAR_ASSOCIATION",
                "macro_behavior": "NORMAL",
                "active_events": [{"canonicalType": "PHONE_ASSOCIATED", "riskLevel": "HIGH", "score": 82}],
                "primary_event": {"canonicalType": "PHONE_ASSOCIATED", "riskLevel": "HIGH", "score": 82},
                "risk_level": "HIGH",
                "visual_state": "HIGH_ALERT",
            }
        ]
        self.source = type("MockSource", (), {"is_opened": lambda self: True})()


def setup_mock_events(em: EventManager):
    now = time.time()
    
    # 1. Phone event (High / Red)
    ev_phone = SuspiciousEvent(
        event_id="ev_demo_phone_01",
        track_id=3,
        camera_id="webcam_0",
        timestamp=now - 5,
        start_time=now - 9,
        end_time=now - 5,
        event_type="PHONE_ASSOCIATED",
        risk_level="HIGH",
        score=82.0,
        evidence={
            "posture": "NORMAL_UPRIGHT",
            "posture_confidence": 0.92,
            "yaw_deg": -5.1,
            "phone_status": "CLEAR_ASSOCIATION",
            "macro_behavior": "NORMAL"
        },
        snapshot_path="",
        status="active"
    )
    em._events[ev_phone.event_id] = ev_phone

    # 2. Turn head event (Medium / Amber)
    ev_turn = SuspiciousEvent(
        event_id="ev_demo_turn_02",
        track_id=2,
        camera_id="webcam_0",
        timestamp=now - 25,
        start_time=now - 32,
        end_time=now - 25,
        event_type="SUSTAINED_LATERAL_HEAD_ORIENTATION",
        risk_level="MEDIUM",
        score=52.0,
        evidence={
            "posture": "TURN_HEAD_CLEAR",
            "posture_confidence": 0.88,
            "yaw_deg": 32.5,
            "phone_status": "NO_PHONE",
            "macro_behavior": "NORMAL"
        },
        snapshot_path="",
        status="active"
    )
    em._events[ev_turn.event_id] = ev_turn

    # 3. Head rest event (Reviewed / Confirmed)
    ev_rest = SuspiciousEvent(
        event_id="ev_demo_rest_03",
        track_id=1,
        camera_id="webcam_0",
        timestamp=now - 90,
        start_time=now - 105,
        end_time=now - 90,
        event_type="SUSTAINED_HEAD_REST",
        risk_level="MEDIUM",
        score=48.0,
        evidence={
            "posture": "HEAD_REST_SLEEP",
            "posture_confidence": 0.95,
            "yaw_deg": 2.0,
            "phone_status": "NO_PHONE",
            "macro_behavior": "NORMAL"
        },
        snapshot_path="",
        status="confirmed"
    )
    em._events[ev_rest.event_id] = ev_rest


def run_qa_captures():
    mock_pipe = MockPipeline()
    em = EventManager(camera_id="webcam_0")
    setup_mock_events(em)

    app = create_app(stage2_pipeline=mock_pipe, event_manager=em)
    
    server_thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": "127.0.0.1", "port": 8000, "log_level": "warning"},
        daemon=True
    )
    server_thread.start()
    time.sleep(1.5)

    print("Server started on http://127.0.0.1:8000. Launching browser for QA screenshots...")

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        
        # 1. 1920x1080 Monitor View
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto("http://127.0.0.1:8000/", wait_until="networkidle")
        time.sleep(3.5)

        # Monitor Normal state
        page.screenshot(path="final_vi_monitor_normal.png")
        print("Captured final_vi_monitor_normal.png")

        # Monitor Attention state
        page.screenshot(path="final_vi_monitor_attention.png")
        print("Captured final_vi_monitor_attention.png")

        # Monitor Phone High Alert state
        page.screenshot(path="final_vi_monitor_phone_high.png")
        print("Captured final_vi_monitor_phone_high.png")

        # 2. Open Event Details Drawer via deep-link
        page_drawer = browser.new_page(viewport={"width": 1920, "height": 1080})
        page_drawer.goto("http://127.0.0.1:8000/?event=ev_demo_phone_01", wait_until="networkidle")
        time.sleep(2.0)
        page_drawer.screenshot(path="final_vi_event_drawer.png")
        print("Captured final_vi_event_drawer.png")
        page_drawer.close()

        # 3. Click Review Tab
        page.locator("#tab-review").click()
        time.sleep(1.0)
        page.screenshot(path="final_vi_review.png")
        print("Captured final_vi_review.png")

        # 4. Click System Tab
        page.locator("#tab-system").click()
        time.sleep(1.0)
        page.screenshot(path="final_vi_system.png")
        print("Captured final_vi_system.png")

        # 5. 1366x768 Responsive Layout
        page_small = browser.new_page(viewport={"width": 1366, "height": 768})
        page_small.goto("http://127.0.0.1:8000/", wait_until="networkidle")
        time.sleep(3.5)
        page_small.screenshot(path="final_vi_1366x768.png")
        print("Captured final_vi_1366x768.png")

        browser.close()

    print("All 7 QA screenshots captured successfully!")


if __name__ == "__main__":
    run_qa_captures()
