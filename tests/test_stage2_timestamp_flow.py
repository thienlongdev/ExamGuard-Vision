"""
Tests for Stage 2 Timestamp Flow & Temporal Continuity
"""

import pytest
import math
import numpy as np

from src.fusion.types import UnifiedTrackUpdate, TrackingState
from src.fusion.temporal_buffer import TemporalBuffer


def test_nan_inf_timestamp_rejection():
    buffer = TemporalBuffer()

    update_nan = UnifiedTrackUpdate(track_id=1, timestamp_sec=float("nan"))
    assert update_nan.is_valid() is False
    assert buffer.push(update_nan) is False

    update_inf = UnifiedTrackUpdate(track_id=1, timestamp_sec=float("inf"))
    assert update_inf.is_valid() is False
    assert buffer.push(update_inf) is False


def test_track_continuity_and_gaps():
    buffer = TemporalBuffer(track_continuity_tolerance_sec=2.0)

    # Frame at t = 1.0
    u1 = UnifiedTrackUpdate(track_id=5, timestamp_sec=1.0)
    assert buffer.push(u1) is True

    # Frame at t = 2.5 (gap = 1.5s <= 2.0s -> continuous)
    continuous, gap = buffer.check_identity_continuity(5, 2.5)
    assert continuous is True
    assert abs(gap - 1.5) < 1e-4

    # Frame at t = 5.5 (gap = 4.5s > 2.0s -> continuity break!)
    continuous_break, gap_break = buffer.check_identity_continuity(5, 5.5)
    assert continuous_break is False
    assert gap_break > 2.0


def test_duplicate_timestamp_tolerance():
    buffer = TemporalBuffer()

    u1 = UnifiedTrackUpdate(track_id=7, timestamp_sec=10.000)
    assert buffer.push(u1) is True

    # Duplicate timestamp within 1ms tolerance
    u2 = UnifiedTrackUpdate(track_id=7, timestamp_sec=10.0005)
    assert buffer.push(u2) is True
    assert buffer._tracks[7].sample_count == 1  # updated in place, no duplicate entry
