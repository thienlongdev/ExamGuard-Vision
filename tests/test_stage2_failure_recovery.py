"""
Tests for Stage 2 Failure Recovery & Error Isolation
"""

import pytest
import numpy as np

from src.orchestration.stage2_pipeline import Stage2Pipeline
from src.video.base import VideoFrame
from src.tracking.tracker import Track
from src.detection.types import BBox


@pytest.fixture(scope="module")
def pipeline():
    return Stage2Pipeline(config_path="configs/stage2_pipeline.yaml", enable_debug_overlay=False)


def test_corrupted_zero_frame_handling(pipeline):
    # Completely black / empty frame
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    vframe = VideoFrame(
        frame=empty_frame,
        timestamp=100.0,
        frame_idx=100,
        fps=30.0,
        width=640,
        height=480,
        source_id="zero_test",
    )

    # Must process without throwing unhandled exception
    res = pipeline.process_frame(vframe)
    assert res is not None
    assert res.frame_idx == 100
    assert res.metrics.total_pipeline_ms >= 0.0


def test_crop_scheduler_sub_resolution_degradation(pipeline):
    # Track with 5x5 bounding box (sub-resolution)
    tiny_track = Track(track_id=999, bbox=BBox(10, 10, 15, 15), confidence=0.7, timestamp=200.0)
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    pos, hp, timings = pipeline.crop_scheduler.schedule_and_infer(frame, [tiny_track], timestamp_sec=200.0)

    # Must gracefully degrade to UNAVAILABLE without crashing CUDA
    assert 999 in pos
    assert 999 in hp
    assert pos[999].status.value == "UNAVAILABLE"
    assert hp[999].status.value == "UNAVAILABLE"
