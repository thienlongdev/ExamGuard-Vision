"""
Tests for Stage 2 Memory Bounds & Resource Leak Gate
"""

import pytest
import numpy as np
import psutil
import os
import torch

from src.orchestration.stage2_pipeline import Stage2Pipeline
from src.video.base import VideoFrame


@pytest.fixture(scope="module")
def pipeline():
    return Stage2Pipeline(config_path="configs/stage2_pipeline.yaml", enable_debug_overlay=False)


def test_memory_boundedness_on_sequential_run(pipeline):
    process = psutil.Process(os.getpid())
    initial_rss = process.memory_info().rss / (1024 * 1024)

    frame = np.full((360, 640, 3), 120, dtype=np.uint8)

    # Process 60 frames
    for i in range(60):
        vframe = VideoFrame(
            frame=frame,
            timestamp=float(i) * 0.033,
            frame_idx=i,
            fps=30.0,
            width=640,
            height=360,
            source_id="mem_test",
        )
        res = pipeline.process_frame(vframe)
        assert res is not None

    final_rss = process.memory_info().rss / (1024 * 1024)
    rss_growth = final_rss - initial_rss

    # Memory growth over 60 frames must be tightly bounded (< 150 MB)
    assert rss_growth < 150.0

    # Ensure buffer sample limits are enforced
    for t_id, buf in pipeline.temporal_buffer._tracks.items():
        assert buf.sample_count <= pipeline.temporal_buffer.max_samples_per_track


def test_expired_track_eviction(pipeline):
    # Ingest a track at t = 1.0
    u1 = pipeline.temporal_buffer.push
    from src.fusion.types import UnifiedTrackUpdate
    pipeline.temporal_buffer.push(UnifiedTrackUpdate(track_id=888, timestamp_sec=1.0))
    assert 888 in pipeline.temporal_buffer.active_track_ids()

    # Advance time past eviction threshold (t = 25.0, where threshold is 15.0s)
    evicted = pipeline.temporal_buffer.evict_inactive_tracks(25.0)
    assert 888 in evicted
    assert 888 not in pipeline.temporal_buffer.active_track_ids()
