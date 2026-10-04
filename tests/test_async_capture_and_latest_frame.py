"""
Unit and Integration Tests for Decoupled Capture, Latest-Frame Buffer,
and Independent Evidence Ingestion.
"""

import time
import queue
import threading
import numpy as np
import pytest

from src.video.base import VideoFrame
from src.video.webcam import WebcamSource
from src.orchestration.stage2_pipeline import BoundedFrameQueue, Stage2Pipeline, Stage2FrameMetrics


def test_latest_frame_buffer_freshness():
    """Verify that LatestFrameBuffer always provides the newest frame when AI lags."""
    q = BoundedFrameQueue(maxsize=2, drop_policy="DROP_STALE_ON_BACKPRESSURE")

    # Push 5 frames in succession
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(1, 6):
        vf = VideoFrame(
            frame=dummy_img,
            timestamp=float(i),
            frame_idx=i,
            fps=30.0,
            width=100,
            height=100,
            source_id="test_cam",
        )
        q.push(vf)

    # We pushed 5 frames into capacity 2:
    # 3 frames should have been marked stale_skipped
    assert q.stale_skipped_count >= 3
    assert q.captured_count == 5

    # Pop should return the latest frame (frame_idx = 5)
    popped = q.pop_latest(timeout=0.1)
    assert popped is not None
    assert popped.frame_idx == 5
    assert q.ai_selected_count == 1


def test_latest_frame_buffer_intermediate_drain():
    """Verify that pop_latest drains older buffered frames to guarantee fresh observation."""
    q = BoundedFrameQueue(maxsize=2, drop_policy="DROP_STALE_ON_BACKPRESSURE")
    dummy_img = np.zeros((50, 50, 3), dtype=np.uint8)

    # Frame 1 and Frame 2 arrive
    f1 = VideoFrame(frame=dummy_img, timestamp=1.0, frame_idx=1, fps=30.0, width=50, height=50, source_id="c")
    f2 = VideoFrame(frame=dummy_img, timestamp=1.033, frame_idx=2, fps=30.0, width=50, height=50, source_id="c")
    q.push(f1)
    q.push(f2)

    # With drain_stale=True, pop should return f2 (the freshest) and f1 should be recorded as stale_skipped
    popped = q.pop(timeout=0.1, drain_stale=True)
    assert popped.frame_idx == 2
    assert q.stale_skipped_count == 1


def test_metrics_ai_frame_age_computation():
    """Verify that Stage2FrameMetrics properly computes ai_frame_age_ms."""
    m = Stage2FrameMetrics(
        frame_idx=10,
        timestamp_sec=100.0,
        ai_frame_age_ms=45.2,
        stale_skipped_count=3,
    )
    d = m.to_dict()
    assert d["ai_frame_age_ms"] == 45.2
    assert d["stale_skipped_count"] == 3


def test_webcam_preferred_backend_selection():
    """Verify that WebcamSource supports preferred_backend and default ordering."""
    ws = WebcamSource(source=0, preferred_backend="CAP_DSHOW")
    assert ws.preferred_backend == "CAP_DSHOW"
    assert ws.backend_name in ["NONE", "CAP_DSHOW", "CAP_MSMF", "CAP_ANY"]

    ws_auto = WebcamSource(source=0)
    assert ws_auto.preferred_backend is None


def test_evidence_buffer_decoupled_ingest():
    """Verify that evidence buffer can receive frames directly from capture without waiting for AI."""
    from src.orchestration.evidence_manager import IntegratedEvidenceManager
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        ev_mgr = IntegratedEvidenceManager(
            output_dir=tmpdir,
            pre_event_seconds=2.0,
            post_event_seconds=2.0,
            enabled=True,
        )

        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Ingest 30 frames at 30 Hz directly into evidence ring buffer
        t0 = time.time()
        for i in range(30):
            ev_mgr.push_frame(dummy_img, t0 + i * 0.033)

        # Ring buffer must have accumulated frames
        assert len(ev_mgr.clip_recorder._rolling_buffer) == 30
        ev_mgr.reset()
