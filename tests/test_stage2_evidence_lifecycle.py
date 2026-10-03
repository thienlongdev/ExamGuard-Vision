"""
Tests for Stage 2 Evidence Lifecycle & Storage Safety
"""

import pytest
import os
import shutil
import numpy as np

from src.orchestration.evidence_manager import IntegratedEvidenceManager
from src.fusion.types import FusedEvent, EventFamily, RiskLevel


@pytest.fixture
def evidence_mgr(tmp_path):
    output_dir = str(tmp_path / "test_evidence")
    mgr = IntegratedEvidenceManager(
        output_dir=output_dir,
        max_snapshots_per_event=2,
        enabled=True,
    )
    yield mgr
    mgr.shutdown()


def test_evidence_lifecycle_open_and_close(evidence_mgr, tmp_path):
    frame = np.full((480, 640, 3), 150, dtype=np.uint8)
    event = FusedEvent(
        event_id="test_ev_001",
        track_id=10,
        camera_id="cam_01",
        event_type=EventFamily.SUSTAINED_HEAD_REST.value,
        start_timestamp=10.0,
        last_update_timestamp=12.0,
        duration=2.0,
        risk_level=RiskLevel.LOW.value,
    )

    # 1. OPEN transition
    evidence_mgr.handle_event_lifecycle(event, "OPEN", frame, timestamp_sec=10.0)
    assert "open_snapshot_path" in event.evidence_summary
    assert "clip_path" in event.evidence_summary

    # 2. Maximum snapshot limit check (prevent spam)
    # Simulate multiple UPDATE calls
    for _ in range(5):
        evidence_mgr.handle_event_lifecycle(event, "UPDATE", frame, timestamp_sec=11.0)
    # Should not exceed max_snapshots_per_event
    assert evidence_mgr._event_snapshot_counts.get("test_ev_001", 0) <= 2

    # 3. CLOSE transition
    evidence_mgr.handle_event_lifecycle(event, "CLOSE", frame, timestamp_sec=12.0)
    assert "metadata_path" in event.evidence_summary

    # Ensure memory cleanup: tracking dicts for this event must be popped
    assert "test_ev_001" not in evidence_mgr._event_snapshot_counts
    assert "test_ev_001" not in evidence_mgr._event_clip_paths
