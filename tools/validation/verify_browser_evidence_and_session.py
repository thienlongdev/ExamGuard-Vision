"""
ExamGuard Vision — Physical Browser Acceptance Validation Script (Workstream 50 & 51)
Launches the real Google Chrome browser on Windows to physically verify:
1. Explicit Monitoring Session start & end
2. Snapshot image visibility (naturalWidth > 0)
3. Chromium HTML5 Video playback (duration > 0, seeking works, no 0:00 freeze)
4. Truthful integrity badge display
5. Multi-session isolation (zero event leakage)
"""

import os
import sys
import time
import tempfile
import threading
from pathlib import Path
import uvicorn
import numpy as np
import cv2

# Project root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from playwright.sync_api import sync_playwright
from src.api.main import create_app
from src.persistence.service import PersistenceService
from src.persistence.models import PersistedEvent
from src.evidence.clip_recorder import probe_video_codec
from src.security.key_provider import InMemoryKeyProvider, set_key_provider


CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
TEST_PORT = 8012


def create_synthetic_h264_clip(file_path: str, duration_sec: float = 3.0, fps: float = 30.0):
    """Write a synthetic browser-compatible H.264 video clip."""
    n_frames = int(duration_sec * fps)
    w, h = 1280, 720
    codec_info = probe_video_codec()
    fourcc = cv2.VideoWriter_fourcc(*codec_info.fourcc_str)
    writer = None
    if codec_info.backend_api != 0:
        writer = cv2.VideoWriter(file_path, codec_info.backend_api, fourcc, fps, (w, h))
    if not writer or not writer.isOpened():
        if os.name == "nt":
            writer = cv2.VideoWriter(file_path, cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"H264"), fps, (w, h))
    if not writer or not writer.isOpened():
        writer = cv2.VideoWriter(file_path, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    if not writer or not writer.isOpened():
        writer = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for i in range(n_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (30, 35, 45)
        x_pos = int((i / n_frames) * 1000) + 100
        cv2.rectangle(img, (x_pos, 250), (x_pos + 120, 450), (0, 165, 255), -1)
        cv2.putText(img, f"EXAMGUARD PLAYBACK TEST - FRAME {i}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(img, f"TIME: {i/fps:.2f}s", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120), 2)
        writer.write(img)
    writer.release()

    cap = cv2.VideoCapture(file_path)
    ok = cap.isOpened()
    f_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else 0
    f_fps = cap.get(cv2.CAP_PROP_FPS) if ok else 0.0
    dur = f_count / f_fps if (f_fps and f_fps > 0) else 0.0
    cap.release()
    return ok and dur > 0, dur, f_count


def main():
    print("\n" + "=" * 76)
    print("   EXAMGUARD VISION — PHYSICAL CHROMIUM VALIDATION SUITE (P0)")
    print("=" * 76)

    # 1. Setup isolated persistence and server
    temp_dir = tempfile.mkdtemp(prefix="eg_browser_test_")
    db_file = os.path.join(temp_dir, "test_browser.sqlite3")
    kp = InMemoryKeyProvider()
    set_key_provider(kp)

    ps = PersistenceService(db_path=db_file)
    app = create_app(persistence_service=ps, enforce_auth=False)

    config = uvicorn.Config(app, host="127.0.0.1", port=TEST_PORT, log_level="warning")
    server = uvicorn.Server(config)

    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Wait for server ready
    time.sleep(1.5)
    print(f"[*] Test API Server listening at http://127.0.0.1:{TEST_PORT}")

    results = {}

    with sync_playwright() as p:
        print(f"[*] Launching Google Chrome: {CHROME_PATH}")
        browser = p.chromium.launch(
            executable_path=CHROME_PATH,
            headless=True,
            args=["--disable-web-security", "--autoplay-policy=no-user-gesture-required"]
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        # STEP 1: Navigate to Dashboard
        print("\n--- STEP 1: INITIAL NAVIGATION (NO ACTIVE SESSION) ---")
        page.goto(f"http://127.0.0.1:{TEST_PORT}/")
        page.wait_for_selector("#header-session-pill", timeout=5000)

        # Verify no active session overlay is visible
        overlay_visible = page.is_visible("#camera-no-session-overlay")
        session_text = page.inner_text("#header-session-text")
        print(f"  • Header Session Text: '{session_text}'")
        print(f"  • Start Session Overlay Visible: {overlay_visible}")
        results["NO_SILENT_SESSION_ON_START"] = (session_text == "Chưa có phiên" and overlay_visible)

        # STEP 2: Explicitly Start Monitoring Session
        print("\n--- STEP 2: EXPLICIT START MONITORING SESSION ---")
        page.fill("#cam-input-session-name", "Thi Cuối Kỳ - Môn CSDL")
        page.fill("#cam-input-room", "Phòng A203")
        page.click("#btn-cam-submit-session")

        # Wait for overlay to hide and header to update
        page.wait_for_selector("#camera-no-session-overlay", state="hidden", timeout=5000)
        time.sleep(0.5)
        new_session_text = page.inner_text("#header-session-text")
        print(f"  • Updated Session Text: '{new_session_text}'")
        results["EXPLICIT_START_SESSION"] = ("Thi Cuối Kỳ" in new_session_text and "Phòng A203" in new_session_text)

        current_sess = ps.get_current_monitoring_session()
        sid = current_sess.session_id
        print(f"  • Created Session ID: {sid} (Status: {current_sess.status})")

        # STEP 3: Inject Real Encrypted Evidence Events
        print("\n--- STEP 3: GENERATE EVIDENCE EVENTS (PHONE, HEAD TURN) ---")
        storage_dir = Path(temp_dir) / "storage" / "sessions" / sid / "evidence"
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Event 1: Phone suspected (HIGH)
        ev1_id = "ev_phone_chrome_01"
        ev1_dir = storage_dir / ev1_id
        ev1_dir.mkdir(parents=True, exist_ok=True)

        snap1_file = str(ev1_dir / "snapshot.jpg")
        img1 = np.zeros((720, 1280, 3), dtype=np.uint8)
        img1[:] = (20, 25, 35)
        cv2.rectangle(img1, (600, 300), (700, 450), (0, 165, 255), 2)
        cv2.putText(img1, "PHONE SUSPECTED [0.94]", (580, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.imwrite(snap1_file, img1)

        clip1_file = str(ev1_dir / "clip.mp4")
        clip_ok, clip_dur, clip_frames = create_synthetic_h264_clip(clip1_file, duration_sec=3.0)
        assert clip_ok, "Failed to produce synthetic H.264 clip"

        # Register event and encrypted evidence
        ev1 = PersistedEvent(
            event_id=ev1_id,
            session_id=sid,
            camera_id="cam01",
            track_id=1,
            event_type="PHONE_SUSPECTED",
            opened_at="2026-10-04T12:00:00",
            duration_sec=3.0,
            severity="HIGH",
            score=94.0,
            lifecycle_status="closed",
            review_status="awaiting",
            source_origin="ai_detector",
        )
        ps.events.upsert_event(ev1)
        ev_snap1 = ps.record_evidence_file(ev1_id, "SNAPSHOT", snap1_file, mime_type="image/jpeg", encrypt=True)
        ev_clip1 = ps.record_evidence_file(ev1_id, "VIDEO_CLIP", clip1_file, mime_type="video/mp4", encrypt=True)

        # Event 2: Lateral head turn (MEDIUM)
        ev2_id = "ev_turn_chrome_02"
        ev2_dir = storage_dir / ev2_id
        ev2_dir.mkdir(parents=True, exist_ok=True)
        snap2_file = str(ev2_dir / "snapshot.jpg")
        img2 = np.zeros((720, 1280, 3), dtype=np.uint8)
        img2[:] = (25, 30, 40)
        cv2.putText(img2, "HEAD TURN LATERAL [38 deg]", (500, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imwrite(snap2_file, img2)
        ev2 = PersistedEvent(
            event_id=ev2_id,
            session_id=sid,
            camera_id="cam01",
            track_id=2,
            event_type="SUSTAINED_LATERAL_HEAD_ORIENTATION",
            opened_at="2026-10-04T12:01:00",
            duration_sec=2.5,
            severity="MEDIUM",
            score=68.0,
            lifecycle_status="closed",
            review_status="awaiting",
            source_origin="ai_detector",
        )
        ps.events.upsert_event(ev2)
        ps.record_evidence_file(ev2_id, "SNAPSHOT", snap2_file, mime_type="image/jpeg", encrypt=True)

        # Refresh page so WebSocket & state pick up both events
        page.reload()
        page.wait_for_selector(f"#event-card-{ev1_id}", timeout=5000)
        print("  • Events rendered in Live Review Queue!")

        # STEP 4: Inspect Live Queue & Thumbnail States
        print("\n--- STEP 4: LIVE REVIEW QUEUE & THUMBNAIL VERIFICATION ---")
        card1 = page.locator(f"#event-card-{ev1_id}")
        card1_img = card1.locator(".card-thumbnail-img")
        card1_img.wait_for(state="visible", timeout=5000)

        # Verify image actually decoded in Chromium (naturalWidth > 0)
        img_loaded = page.evaluate("""(eid) => {
            const img = document.querySelector(`#event-card-${eid} .card-thumbnail-img`);
            return img && img.naturalWidth > 0 && img.naturalHeight > 0;
        }""", ev1_id)
        print(f"  • Queue Thumbnail Loaded in Chrome: {img_loaded}")
        results["QUEUE_THUMBNAIL_LOADED"] = img_loaded

        # STEP 5: Event Drawer Evidence Forensic Validation
        print("\n--- STEP 5: EVENT DRAWER FORENSIC AUDIT (IMAGE & VIDEO PLAYBACK) ---")
        # Click on phone event card to open drawer
        card1.click()
        page.wait_for_selector("#event-drawer.open", timeout=5000)

        # 5.1 Snapshot in drawer
        drawer_snap_loaded = page.evaluate("""() => {
            const img = document.getElementById('drawer-snapshot-img');
            return img && img.style.display !== 'none' && img.naturalWidth > 0 && img.naturalHeight > 0;
        }""")
        print(f"  • Drawer Snapshot Decoded & Visible: {drawer_snap_loaded}")
        results["DRAWER_SNAPSHOT_VISIBLE"] = drawer_snap_loaded

        # 5.2 Video playback in drawer
        print("  • Testing HTML5 Video Player in Chrome...")
        video_el = page.locator("#drawer-evidence-video")
        video_el.wait_for(state="visible", timeout=8000)

        # Wait for video metadata to load
        page.wait_for_function("""() => {
            const vid = document.getElementById('drawer-evidence-video');
            return vid && vid.readyState >= 1 && vid.duration > 0;
        }""", timeout=8000)

        video_meta = page.evaluate("""() => {
            const vid = document.getElementById('drawer-evidence-video');
            return {
                duration: vid.duration,
                readyState: vid.readyState,
                videoWidth: vid.videoWidth,
                videoHeight: vid.videoHeight,
                paused: vid.paused,
                currentSrc: vid.currentSrc,
            };
        }""")
        print(f"  • Video Metadata in Chrome: {video_meta}")
        results["VIDEO_DURATION_GREATER_THAN_ZERO"] = (video_meta["duration"] > 0)
        results["VIDEO_ZERO_SECOND_BUG_FIXED"] = (video_meta["duration"] >= 2.0)

        # 5.3 Video Play Test
        page.evaluate("""() => {
            const vid = document.getElementById('drawer-evidence-video');
            return vid.play();
        }""")
        time.sleep(1.0)

        play_state = page.evaluate("""() => {
            const vid = document.getElementById('drawer-evidence-video');
            return {
                currentTime: vid.currentTime,
                paused: vid.paused,
                ended: vid.ended,
            };
        }""")
        print(f"  • Video Play Progress: {play_state}")
        results["VIDEO_PLAYS_AND_ADVANCES"] = (play_state["currentTime"] > 0.5)

        # 5.4 Video Seek Test
        page.evaluate("""() => {
            const vid = document.getElementById('drawer-evidence-video');
            vid.currentTime = 2.0;
        }""")
        time.sleep(0.5)
        seek_time = page.evaluate("() => document.getElementById('drawer-evidence-video').currentTime")
        print(f"  • Seek to 2.0s -> Current Time: {seek_time:.2f}s")
        results["VIDEO_SEEKING_WORKS"] = (seek_time >= 1.8)

        # 5.5 Integrity Badge
        time.sleep(0.5)
        badge_text = page.inner_text("#drawer-integrity-badge")
        print(f"  • Evidence Integrity Badge: '{badge_text}'")
        results["INTEGRITY_BADGE_TRUTHFUL"] = ("Hợp lệ" in badge_text or "ĐẦY ĐỦ" in badge_text or "Khớp" in badge_text)

        # STEP 6: Review Adjudication
        print("\n--- STEP 6: HUMAN REVIEW ACTION & REMOVAL FROM QUEUE ---")
        page.click("#btn-drawer-confirm")
        time.sleep(0.5)

        # Verify event removed from pending queue
        card1_exists = page.locator(f"#event-card-{ev1_id}").count()
        print(f"  • Confirmed Event Removed from Pending Queue: {card1_exists == 0}")
        results["CONFIRMED_REMOVED_FROM_PENDING"] = (card1_exists == 0)

        # STEP 7: End Monitoring Session
        print("\n--- STEP 7: EXPLICIT END SESSION & HISTORY VERIFICATION ---")
        page.click("#header-session-pill")
        page.wait_for_selector("#session-dropdown-menu", state="visible", timeout=3000)
        page.click("#menu-end-session")

        page.wait_for_selector("#modal-end-confirm", timeout=3000)
        page.click("#modal-end-confirm")
        time.sleep(1.0)

        # Verify header indicates no session
        end_session_text = page.inner_text("#header-session-text")
        print(f"  • Post-End Session Header: '{end_session_text}'")
        results["EXPLICIT_END_SESSION"] = (end_session_text == "Chưa có phiên")

        # Navigate to History Tab
        page.click('button[data-view="history"]')
        page.wait_for_selector(".history-view-container", timeout=5000)
        time.sleep(0.8)

        history_has_session = page.evaluate("""(sid) => {
            const row = document.querySelector(`tr[data-session-id="${sid}"]`);
            return row !== null;
        }""", sid)
        print(f"  • Ended Session Preserved in History: {history_has_session}")
        results["OLD_SESSION_PRESERVED_IN_HISTORY"] = history_has_session

        # STEP 8: Start New Session & Clean State Isolation
        print("\n--- STEP 8: START NEW SESSION & ZERO EVENT CONTAMINATION ---")
        page.click('button[data-view="monitor"]')
        page.wait_for_selector("#camera-no-session-overlay", timeout=5000)

        # Start Room B session
        page.fill("#cam-input-session-name", "Thi Môn Mạng Máy Tính")
        page.fill("#cam-input-room", "Phòng B102")
        page.click("#btn-cam-submit-session")
        page.wait_for_selector("#camera-no-session-overlay", state="hidden", timeout=5000)
        time.sleep(0.5)

        new_sess_b = ps.get_current_monitoring_session()
        print(f"  • New Session ID: {new_sess_b.session_id} (Name: {new_sess_b.name})")

        # Verify live queue for new session has 0 pending items
        pending_count = page.inner_text("#queue-pending-count")
        print(f"  • New Session Live Queue Pending Count: {pending_count}")
        results["NEW_SESSION_COUNTERS_CLEAN"] = (pending_count == "0")
        results["ZERO_CROSS_SESSION_EVENT_CONTAMINATION"] = (pending_count == "0" and new_sess_b.session_id != sid)

        browser.close()

    print("\n" + "=" * 76)
    print("      PHYSICAL CHROMIUM ACCEPTANCE VALIDATION RESULTS")
    print("=" * 76)
    all_passed = True
    for test_name, passed in results.items():
        status_str = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_passed = False
        print(f"  {status_str} : {test_name}")
    print("=" * 76)

    if all_passed:
        print("\n>>> ALL PHYSICAL ACCEPTANCE CRITERIA ON CHROME ARE 100% SATISFIED! <<<\n")
    else:
        print("\n>>> ONE OR MORE PHYSICAL ACCEPTANCE TESTS FAILED! <<<\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
