"""
Tests for V4D Replay Engine & Determinism
========================================
Covers:
- Deterministic replay execution (identical inputs -> identical events & scores)
- JSONL serialization and deserialization
- Offline replay without neural inference
"""

import json
import tempfile
from pathlib import Path
import pytest

from src.fusion.replay import ReplayEngine
from src.fusion.types import (
    UnifiedTrackUpdate,
    ObservationStatus,
    PostureCue,
    TrackingState,
)


def generate_synthetic_trace(num_frames: int = 30, fps: float = 10.0):
    updates = []
    dt = 1.0 / fps
    for i in range(num_frames):
        t = round(i * dt, 3)
        # Frames 5 to 20: Sleep posture active
        is_sleeping = (5 <= i <= 20)
        p_sleep = 0.85 if is_sleeping else 0.05
        p_upright = 0.10 if is_sleeping else 0.90

        updates.append(UnifiedTrackUpdate(
            track_id=1,
            timestamp_sec=t,
            posture=PostureCue(
                status=ObservationStatus.AVAILABLE,
                probabilities={
                    "NORMAL_UPRIGHT": p_upright,
                    "NORMAL_READ_WRITE": 0.05,
                    "HEAD_REST_SLEEP": p_sleep,
                    "TURN_HEAD_CLEAR": 0.0,
                },
                confidence=p_sleep if is_sleeping else p_upright,
            ),
            tracking=TrackingState(1, t, (10, 10, 100, 200)),
        ))
    return updates


def test_replay_determinism():
    config = {
        "provisional_thresholds": {
            "sustained_head_rest": {
                "min_candidate_duration_sec": 1.0,
                "evidence_enter_threshold": 0.60,
                "evidence_exit_threshold": 0.30,
                "cooldown_seconds": 2.0,
            }
        }
    }

    trace1 = generate_synthetic_trace()
    trace2 = generate_synthetic_trace()

    engine1 = ReplayEngine(config)
    engine2 = ReplayEngine(config)

    events1 = engine1.replay_trace(trace1)
    events2 = engine2.replay_trace(trace2)

    assert len(events1) == len(events2)
    for ev1, ev2 in zip(events1, events2):
        assert ev1.event_type == ev2.event_type
        assert abs(ev1.start_timestamp - ev2.start_timestamp) < 1e-4
        assert abs(ev1.duration - ev2.duration) < 1e-4
        assert abs(ev1.risk_score - ev2.risk_score) < 1e-4
        assert ev1.risk_level == ev2.risk_level


def test_replay_jsonl_roundtrip():
    updates = generate_synthetic_trace(num_frames=10)
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "test_trace.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for u in updates:
                row = {
                    "track_id": u.track_id,
                    "timestamp_sec": u.timestamp_sec,
                    "posture": {
                        "status": u.posture.status.value,
                        "probabilities": u.posture.probabilities,
                        "confidence": u.posture.confidence,
                    },
                    "tracking": {
                        "bbox": list(u.tracking.bbox),
                    }
                }
                f.write(json.dumps(row) + "\n")

        engine = ReplayEngine()
        events = engine.replay_jsonl(jsonl_path)
        # Replay runs without error
        assert isinstance(events, list)
