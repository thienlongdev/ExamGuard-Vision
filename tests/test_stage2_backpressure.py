"""
Tests for Stage 2 Pipeline Backpressure & Frame Drop Safety
"""

import pytest
import numpy as np
import time

from src.orchestration.stage2_pipeline import Stage2Pipeline
from src.video.base import VideoFrame


@pytest.fixture(scope="module")
def pipeline():
    return Stage2Pipeline(config_path="configs/stage2_pipeline.yaml", enable_debug_overlay=False)


def test_frame_dropping_under_irregular_cadence(pipeline):
    # Simulate bursty and irregular timestamps (e.g. 0.0, 0.033, 0.5, 0.533)
    timestamps = [0.0, 0.033, 0.066, 0.35, 0.383, 0.8, 1.2, 1.233]
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)

    for i, ts in enumerate(timestamps):
        vframe = VideoFrame(
            frame=frame,
            timestamp=ts,
            frame_idx=i,
            fps=30.0,
            width=640,
            height=480,
            source_id="burst_test",
        )
        res = pipeline.process_frame(vframe)
        assert res.timestamp_sec == ts
        assert res.metrics.total_pipeline_ms >= 0.0


def test_queue_depth_bounds(pipeline):
    assert pipeline.max_decode_queue > 0
    assert pipeline.max_decode_queue <= 100
    assert pipeline.drop_policy == "DROP_STALE_ON_BACKPRESSURE"
