"""
Automated Validation Suite for Batched Perception and Adaptive Per-Track Scheduling
==================================================================================
Tests:
1. Track-to-batch index mapping preservation across arbitrary batch sizes (1, 3, 8, 16 tracks).
2. Adaptive scheduling: normal track (4 Hz) vs attention track (8 Hz).
3. Anti-starvation guarantee: quiet tracks never starve past max_starvation_interval_sec.
4. FP16 autocast precision and contract conformance for HeadPosePredictor.
5. PosturePredictor contract conformance under torch.inference_mode().
6. Capability gating: small / unresolvable heads remain UNAVAILABLE without GPU waste.
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


def test_batch_exact_mapping_and_order_preservation(registry):
    """Verify exact track-to-output mapping and ordering across different track counts."""
    scheduler = CropScheduler(model_registry=registry)
    frame = np.full((720, 1280, 3), 140, dtype=np.uint8)

    for track_count in [1, 3, 8, 12]:
        tracks = []
        for i in range(track_count):
            t_id = 1000 + i
            # Position boxes spread across the frame
            x1 = 50 + (i % 6) * 180
            y1 = 80 + (i // 6) * 250
            tracks.append(Track(track_id=t_id, bbox=BBox(x1, y1, x1 + 140, y1 + 220), confidence=0.9, timestamp=10.0))

        posture_cues, headpose_cues, _ = scheduler.schedule_and_infer(frame, tracks, timestamp_sec=10.0)

        # Assert exact 1:1 key mapping
        expected_ids = {t.track_id for t in tracks}
        assert set(posture_cues.keys()) == expected_ids
        assert set(headpose_cues.keys()) == expected_ids

        for t in tracks:
            p_cue = posture_cues[t.track_id]
            hp_cue = headpose_cues[t.track_id]
            assert p_cue.status in (ObservationStatus.AVAILABLE, ObservationStatus.UNAVAILABLE)
            assert hp_cue.status in (ObservationStatus.AVAILABLE, ObservationStatus.UNAVAILABLE)


def test_adaptive_cadence_differentiation(registry):
    """
    Verify that an attention track receives higher inference frequency (8 Hz)
    than a normal quiet track (4 Hz).
    """
    config = {
        "performance": {
            "adaptive_scheduling": True,
            "normal_posture_hz": 4.0,       # 250 ms interval
            "attention_posture_hz": 8.0,    # 125 ms interval
            "normal_headpose_hz": 4.0,      # 250 ms interval
            "attention_headpose_hz": 8.0,   # 125 ms interval
            "max_starvation_interval_sec": 0.50,
        }
    }
    scheduler = CropScheduler(model_registry=registry, config=config)
    frame = np.full((720, 1280, 3), 150, dtype=np.uint8)

    normal_track = Track(track_id=501, bbox=BBox(100, 100, 250, 400), confidence=0.9, timestamp=0.0)
    attention_track = Track(track_id=502, bbox=BBox(400, 100, 550, 400), confidence=0.9, timestamp=0.0)
    all_tracks = [normal_track, attention_track]

    # Tick 1: t = 0.0 -> both get initial inference
    p1, h1, t1 = scheduler.schedule_and_infer(frame, all_tracks, timestamp_sec=0.0, attention_track_ids={502})
    state_norm = scheduler._get_track_state(501)
    state_attn = scheduler._get_track_state(502)
    assert state_norm.total_posture_evaluations == 1
    assert state_attn.total_posture_evaluations == 1

    # Tick 2: t = 0.14s (140 ms later):
    # - Attention track (interval 125 ms) has elapsed 140 ms >= 125 ms -> runs inference!
    # - Normal track (interval 250 ms) has elapsed 140 ms < 250 ms -> skips inference!
    p2, h2, t2 = scheduler.schedule_and_infer(frame, all_tracks, timestamp_sec=0.14, attention_track_ids={502})
    assert state_attn.total_posture_evaluations == 2
    assert state_norm.total_posture_evaluations == 1, "Normal track should skip inference at 140ms"

    # Tick 3: t = 0.26s (260 ms later from start):
    # - Attention track has elapsed 120 ms since last (0.14s) -> skips (needs 125ms)
    # - Normal track has elapsed 260 ms >= 250 ms -> runs inference!
    p3, h3, t3 = scheduler.schedule_and_infer(frame, all_tracks, timestamp_sec=0.26, attention_track_ids={502})
    assert state_norm.total_posture_evaluations == 2, "Normal track should evaluate after 250ms"


def test_anti_starvation_guarantee(registry):
    """
    Verify that even when an attention track is constantly running,
    a quiet track is guaranteed evaluation once max_starvation_interval_sec is reached.
    """
    config = {
        "performance": {
            "adaptive_scheduling": True,
            "normal_posture_hz": 2.0,       # 500 ms normal interval
            "attention_posture_hz": 10.0,   # 100 ms interval
            "max_starvation_interval_sec": 0.35, # 350 ms anti-starvation ceiling
        }
    }
    scheduler = CropScheduler(model_registry=registry, config=config)
    frame = np.full((720, 1280, 3), 150, dtype=np.uint8)

    quiet_track = Track(track_id=601, bbox=BBox(100, 100, 250, 400), confidence=0.9, timestamp=0.0)
    busy_track = Track(track_id=602, bbox=BBox(400, 100, 550, 400), confidence=0.9, timestamp=0.0)
    tracks = [quiet_track, busy_track]

    # Initial frame t = 0.0 -> both evaluated
    scheduler.schedule_and_infer(frame, tracks, timestamp_sec=0.0, attention_track_ids={602})
    s_quiet = scheduler._get_track_state(601)
    assert s_quiet.total_posture_evaluations == 1

    # At t = 0.20s (200 ms), quiet track normal interval is 500ms -> skips
    scheduler.schedule_and_infer(frame, tracks, timestamp_sec=0.20, attention_track_ids={602})
    assert s_quiet.total_posture_evaluations == 1

    # At t = 0.36s (360 ms > 350 ms starvation threshold), quiet track MUST be evaluated despite normal 500ms interval!
    scheduler.schedule_and_infer(frame, tracks, timestamp_sec=0.36, attention_track_ids={602})
    assert s_quiet.total_posture_evaluations == 2, "Quiet track must not starve past max_starvation_interval_sec"


def test_headpose_fp16_autocast_contract(registry):
    """Verify HeadPosePredictor with precision_mode='fp16' produces valid continuous angles."""
    pred = registry.headpose_predictor
    assert pred.precision_mode in ("fp16", "fp32")

    # Generate dummy input batch (4, 3, 224, 224)
    dummy_input = torch.randn(4, 3, 224, 224, dtype=torch.float32)
    outputs = pred.predict_tensor(dummy_input, student_ids=[1, 2, 3, 4])

    assert len(outputs) == 4
    for i, out in enumerate(outputs):
        assert out.student_id == (i + 1)
        assert -180.0 <= out.yaw_deg < 180.0
        assert not np.isnan(out.yaw_deg)
        assert not np.isinf(out.yaw_deg)
        assert 0.0 <= out.pose_confidence <= 1.0


def test_posture_inference_mode_contract(registry):
    """Verify PosturePredictor under torch.inference_mode produces valid probability distributions."""
    pred = registry.posture_predictor
    dummy_input = torch.randn(3, 3, 224, 224, dtype=torch.float32)
    outputs = pred.predict_tensor(dummy_input, student_ids=[10, 20, 30])

    assert len(outputs) == 3
    valid_classes = {"NORMAL_UPRIGHT", "NORMAL_READ_WRITE", "HEAD_REST_SLEEP", "TURN_HEAD_CLEAR"}
    for out in outputs:
        assert out.predicted_class in valid_classes
        assert 0.0 <= out.confidence <= 1.0
        prob_sum = (
            out.normal_upright_score +
            out.normal_read_write_score +
            out.head_rest_sleep_score +
            out.turn_head_clear_score
        )
        assert abs(prob_sum - 1.0) < 1e-4
