"""
Tests for Stage 2 Full Pipeline Orchestration
"""

import pytest
import numpy as np
import time

from src.orchestration.stage2_pipeline import Stage2Pipeline
from src.video.base import VideoFrame


@pytest.fixture(scope="module")
def pipeline():
    return Stage2Pipeline(config_path="configs/stage2_pipeline.yaml", enable_debug_overlay=True)


def test_pipeline_single_frame_orchestration(pipeline):
    # Create test frame
    frame = np.full((720, 1280, 3), (200, 200, 200), dtype=np.uint8)
    vframe = VideoFrame(
        frame=frame,
        timestamp=1.0,
        frame_idx=1,
        fps=30.0,
        width=1280,
        height=720,
        source_id="test_cam",
    )

    res = pipeline.process_frame(vframe)
    assert res.frame_idx == 1
    assert res.timestamp_sec == 1.0
    assert res.metrics.total_pipeline_ms >= 0.0
    assert res.metrics.detector_ms >= 0.0
    assert res.annotated_frame is not None
    assert res.annotated_frame.shape == (720, 1280, 3)


def test_pipeline_observable_language_guarantee(pipeline):
    # Verify debug overlay and string representations contain NO cheating terminology
    forbidden_terms = ["CHEATING", "CHEATER", "GUILTY", "FRAUD"]

    frame = np.full((720, 1280, 3), (200, 200, 200), dtype=np.uint8)
    vframe = VideoFrame(
        frame=frame,
        timestamp=2.0,
        frame_idx=2,
        fps=30.0,
        width=1280,
        height=720,
        source_id="test_cam",
    )

    res = pipeline.process_frame(vframe)
    for ev in res.active_events:
        for term in forbidden_terms:
            assert term not in ev.event_type.upper()
            assert term not in ev.risk_level.upper()
            assert term not in str(ev.evidence_summary).upper()


def test_pipeline_stream_run_bounded(pipeline):
    # Run a short 10-frame loop using the configured source
    metrics = pipeline.run_stream(max_frames=10)
    assert len(metrics) <= 10
    if len(metrics) > 0:
        assert metrics[0].total_pipeline_ms > 0.0
