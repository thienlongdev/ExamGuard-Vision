"""
Tests for Stage 2 Batch Track Mapping & Scheduler Integrity
"""

import pytest
import numpy as np
import torch

from src.tracking.tracker import Track
from src.detection.types import BBox
from src.orchestration.crop_scheduler import CropScheduler
from src.orchestration.model_registry import ModelRegistry
from src.fusion.types import ObservationStatus, HeadPoseSupportStatus


@pytest.fixture(scope="module")
def registry():
    reg = ModelRegistry.get_instance()
    reg.initialize_models()
    return reg


def test_batch_track_mapping_preservation(registry):
    scheduler = CropScheduler(model_registry=registry)

    # Create synthetic frame with 3 distinct students
    frame = np.full((720, 1280, 3), 150, dtype=np.uint8)

    # Student 1: left desk
    tracks = [
        Track(track_id=101, bbox=BBox(100, 100, 250, 400), confidence=0.9, timestamp=1.0),
        Track(track_id=102, bbox=BBox(400, 100, 550, 400), confidence=0.85, timestamp=1.0),
        Track(track_id=103, bbox=BBox(700, 100, 850, 400), confidence=0.88, timestamp=1.0),
    ]

    posture_cues, headpose_cues, timings = scheduler.schedule_and_infer(frame, tracks, timestamp_sec=1.0)

    # Assert exactly each track ID has corresponding cues
    assert set(posture_cues.keys()) == {101, 102, 103}
    assert set(headpose_cues.keys()) == {101, 102, 103}

    for t_id in [101, 102, 103]:
        p = posture_cues[t_id]
        assert p.status in (ObservationStatus.AVAILABLE, ObservationStatus.UNAVAILABLE)
        assert sum(p.probabilities.values()) >= 0.0

        hp = headpose_cues[t_id]
        assert hp.status in (ObservationStatus.AVAILABLE, ObservationStatus.UNAVAILABLE)


def test_capability_gating_small_head(registry):
    scheduler = CropScheduler(model_registry=registry)
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)

    # Very small person crop (< min_head_dim)
    tiny_track = Track(track_id=201, bbox=BBox(50, 50, 65, 75), confidence=0.8, timestamp=1.0)
    posture_cues, headpose_cues, _ = scheduler.schedule_and_infer(frame, [tiny_track], timestamp_sec=1.0)

    hp = headpose_cues[201]
    # Small face must NOT fabricate yaw = 0, must be UNAVAILABLE
    assert hp.status == ObservationStatus.UNAVAILABLE
    assert hp.yaw_deg is None
    assert hp.support_status == HeadPoseSupportStatus.FACE_UNRESOLVABLE


def test_cadence_scheduling_preservation(registry):
    scheduler = CropScheduler(model_registry=registry)
    frame = np.full((720, 1280, 3), 150, dtype=np.uint8)
    track = Track(track_id=301, bbox=BBox(200, 200, 400, 600), confidence=0.9, timestamp=1.0)

    # Frame 1: timestamp = 1.0 (runs posture & headpose)
    p1, h1, _ = scheduler.schedule_and_infer(frame, [track], timestamp_sec=1.0)
    assert p1[301].status == ObservationStatus.AVAILABLE

    # Frame 2: timestamp = 1.02 (20ms later, posture 10 Hz interval is 100ms -> should not run inference, but return cached)
    p2, h2, timings = scheduler.schedule_and_infer(frame, [track], timestamp_sec=1.02)
    assert p2[301].status == ObservationStatus.AVAILABLE
    # Timing for posture inference should be 0 because it was skipped by cadence!
    assert timings["posture_ms"] == 0.0
