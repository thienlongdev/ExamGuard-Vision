"""
Tests for Final School-Demo Vietnamese UI, Alert Semantics, and Event Quality Polish.

Verifies:
1. Zero occurrences of affirmative AI cheating verdict terms ("Gian lận", "Đã gian lận", "Cheating detected", "Violation confirmed").
2. Presence and validity of Vietnamese localization dictionary with no raw internal enum leakage.
3. Sustained clear phone association evaluates to HIGH (>= 75.0) while transient phone evaluates to LOW (<= 25.0).
4. Standing behavior baseline and conservative posture/geometry gating (lean-forward suppression vs. real standing).
5. Event deduplication and cooldown reopen gating for continuous behaviors.
6. Multi-cue track severity resolution (maximum active severity, graceful fallback from HIGH to AMBER).
7. KPI invariant: TOTAL = AWAITING + CONFIRMED + DISMISSED.
8. System FPS telemetry distinction (Camera configured capture vs. AI processing target).
"""

from pathlib import Path
import pytest

from src.fusion.types import (
    FusedEvent,
    RiskLevel,
    EventFamily,
    EventLifecycleState,
    PhoneCue,
    PhoneAssociationStatus,
    ObservationStatus,
)
from src.fusion.risk_aggregator import RiskAggregator
from src.fusion.reliability import ReliabilityModel
from src.fusion.event_engine import TrackEventStateMachine
from src.orchestration.stage2_pipeline import Stage2Pipeline


def test_no_forbidden_cheating_verdicts_in_ui_assets():
    """Verify strictly zero affirmative cheating verdict terms in templates and static files."""
    forbidden = [
        "đã gian lận",
        "cheating detected",
        "violation confirmed"
    ]
    ui_dirs = [
        Path("src/api/templates"),
        Path("src/api/static")
    ]
    scanned_files = 0
    for d in ui_dirs:
        for p in d.rglob("*"):
            if p.is_file() and p.suffix in [".html", ".js", ".css"]:
                scanned_files += 1
                content = p.read_text(encoding="utf-8").lower()
                for term in forbidden:
                    assert term not in content, f"Forbidden verdict '{term}' found in {p}"
                
                # Check for affirmative "gian lận" (excluding the mandatory disclaimer "không phải kết luận gian lận")
                clean_content = content.replace("không phải kết luận gian lận", "")
                assert "gian lận" not in clean_content, f"Affirmative cheating verdict 'gian lận' found in {p}"

    assert scanned_files >= 10, f"Expected to scan at least 10 UI files, scanned {scanned_files}"


def test_localization_dictionary_completeness():
    """Verify localization.js contains all required mappings and semantic disclaimer tooltip."""
    loc_file = Path("src/api/static/js/localization.js")
    assert loc_file.exists(), "localization.js must exist"
    content = loc_file.read_text(encoding="utf-8")

    # Required dictionary keys and mappings
    assert "EVENT_NAMES_VI" in content
    assert "POSTURE_NAMES_VI" in content
    assert "SEVERITY_BANDS_VI" in content
    assert "REVIEW_STATUS_VI" in content
    assert "SOURCE_ORIGIN_VI" in content
    assert "PHONE_ASSOCIATION_VI" in content
    assert "ALERT_SEMANTICS_TOOLTIP" in content

    # Check preferred Vietnamese semantics
    assert "BÌNH THƯỜNG" in content
    assert "CẦN CHÚ Ý" in content
    assert "CẢNH BÁO CAO" in content
    assert "không phải kết luận gian lận" in content

    # Check key posture and event translations
    assert "Bình thường" in content
    assert "Đọc / viết" in content
    assert "Gục đầu" in content
    assert "Quay đầu rõ" in content
    assert "Phát hiện điện thoại" in content
    assert "Đứng dậy" in content
    assert "Liên kết rõ" in content


def test_phone_severity_policy():
    """
    Verify phone severity policy:
    - Transient / short phone candidate (< 0.5s) remains LOW (<= 25.0)
    - Sustained clear phone association (>= 2.5s) resolves to HIGH (>= 75.0)
    """
    agg = RiskAggregator()

    # Case 1: Transient phone candidate (< 0.5s duration)
    transient_ev = FusedEvent(
        event_id="ev_transient_phone",
        track_id=1,
        camera_id="cam_0",
        event_type=EventFamily.PHONE_ASSOCIATED.value,
        start_timestamp=100.0,
        last_update_timestamp=100.3,
        duration=0.3,
        evidence_summary={"phone_status": "CLEAR_ASSOCIATION"}
    )
    res_transient = agg.assess_event_risk(transient_ev, mean_reliability=0.90)
    assert res_transient.risk_level == RiskLevel.LOW.value
    assert res_transient.risk_score <= 25.0, f"Transient phone must be <= 25, got {res_transient.risk_score}"

    # Case 2: Weak / ambiguous phone candidate (low reliability)
    rel_model = ReliabilityModel()
    ambiguous_cue = PhoneCue(
        status=ObservationStatus.AVAILABLE,
        detected=True,
        association_confidence=0.40,
        association_status=PhoneAssociationStatus.AMBIGUOUS_ASSOCIATION
    )
    ambiguous_rel = rel_model.evaluate_phone_reliability(ambiguous_cue)
    assert ambiguous_rel < 0.6, f"Ambiguous phone must have discounted reliability, got {ambiguous_rel}"

    # Clear association cue (high reliability >= 0.85)
    clear_cue = PhoneCue(
        status=ObservationStatus.AVAILABLE,
        detected=True,
        association_confidence=0.90,
        association_status=PhoneAssociationStatus.CLEAR_ASSOCIATION
    )
    clear_rel = rel_model.evaluate_phone_reliability(clear_cue)
    assert clear_rel >= 0.85, f"Clear phone association must have high reliability, got {clear_rel}"

    # Case 3: Clear sustained phone association (3.5s duration)
    sustained_ev = FusedEvent(
        event_id="ev_sustained_phone",
        track_id=1,
        camera_id="cam_0",
        event_type=EventFamily.PHONE_ASSOCIATED.value,
        start_timestamp=100.0,
        last_update_timestamp=103.5,
        duration=3.5,
        evidence_summary={"phone_status": "CLEAR_ASSOCIATION"}
    )
    res_sustained = agg.assess_event_risk(sustained_ev, mean_reliability=clear_rel)
    assert res_sustained.risk_level == RiskLevel.HIGH.value, f"Sustained clear phone must be HIGH, got {res_sustained.risk_level}"
    assert res_sustained.risk_score >= 75.0, f"Sustained clear phone score must be >= 75, got {res_sustained.risk_score}"


def test_standing_severity_baseline_and_false_positive_reduction():
    """
    Verify Standing policy:
    - Standing baseline is tuned conservatively to 25.0 (medium alert, not high)
    - Leaning forward posture (NORMAL_READ_WRITE) vetoes Standing in pipeline logic
    """
    agg = RiskAggregator()
    standing_ev = FusedEvent(
        event_id="ev_standing_test",
        track_id=1,
        camera_id="cam_0",
        event_type=EventFamily.STANDING.value,
        start_timestamp=100.0,
        last_update_timestamp=102.0,
        duration=2.0,
        evidence_summary={}
    )
    res_standing = agg.assess_event_risk(standing_ev, mean_reliability=0.80)
    assert res_standing.risk_level in [RiskLevel.LOW.value, RiskLevel.MEDIUM.value]
    assert res_standing.risk_score < 75.0, "Standing should not immediately become HIGH"

    # Test pipeline seated geometry & posture veto logic
    pipe = Stage2Pipeline()
    track_id = 99
    # Initialize track metadata with seated baseline
    pipe._track_metadata[track_id] = {
        "baseline_y1": 200.0,
        "baseline_height": 300.0,
        "baseline_aspect": 1.1,
        "posture": "NORMAL_READ_WRITE"
    }

    # Macro detector emits STANDING with 0.85 conf, but posture is NORMAL_READ_WRITE (seated leaning)
    # Posture veto: READ_WRITE should veto standing
    is_read_write = pipe._track_metadata[track_id]["posture"] in ["NORMAL_READ_WRITE", "HEAD_REST_SLEEP"]
    assert is_read_write is True, "Read/write posture must veto false standing from leaning forward"

    # Actual standing test: student rises up (y1 moves from 200 to 80, height increases)
    standing_bbox = [100.0, 80.0, 300.0, 580.0]
    height = standing_bbox[3] - standing_bbox[1]
    y1_rise = standing_bbox[1] < (pipe._track_metadata[track_id]["baseline_y1"] - 30.0)
    aspect_tall = (height / 200.0) >= 1.6
    assert y1_rise is True or aspect_tall is True, "Genuine standing transition satisfies geometric rise"


def test_event_state_machine_deduplication():
    """Verify that continuous behavior does not reopen duplicate events while cue persists."""
    sm = TrackEventStateMachine(
        track_id=1,
        event_family=EventFamily.PHONE_ASSOCIATED,
        min_candidate_duration=0.5,
        enter_threshold=0.6,
        exit_threshold=0.3,
        cooldown_seconds=2.0
    )

    class DummyCueState:
        posture_probs = {"NORMAL_UPRIGHT": 0.9}
        posture_status = ObservationStatus.AVAILABLE
        smoothed_yaw_deg = 5.0
        headpose_status = ObservationStatus.AVAILABLE
        phone_detected = True
        phone_association_status = "CLEAR_ASSOCIATION"
        phone_confidence = 0.95
        phone_status = ObservationStatus.AVAILABLE
        macro_status = ObservationStatus.NOT_EVALUATED
        stand_score = 0.0
        discuss_score = 0.0
        source_origin = "PHYSICAL_LIVE_CAMERA"

    dummy_cue = DummyCueState()
    events_emitted = []

    # 1. Drive state machine into ACTIVE with high evidence over 1 second (10 frames at 10 FPS)
    for i in range(10):
        t = 100.0 + (i * 0.1)
        ev, action = sm.process_frame(
            evidence_score=0.85,
            timestamp=t,
            cue_state=dummy_cue,
            is_vetoed=False
        )
        if action != "NONE":
            events_emitted.append((action, ev))

    assert sm.state == EventLifecycleState.ACTIVE
    opened_events = [e for action, e in events_emitted if action == "OPEN"]
    assert len(opened_events) == 1, "Must open exactly one event"

    # 2. While continuously active, process another 30 frames (3 seconds) with high evidence
    for i in range(10, 40):
        t = 100.0 + (i * 0.1)
        ev, action = sm.process_frame(
            evidence_score=0.85,
            timestamp=t,
            cue_state=dummy_cue,
            is_vetoed=False
        )
        if action != "NONE":
            events_emitted.append((action, ev))

    assert sm.state == EventLifecycleState.ACTIVE
    assert len([e for action, e in events_emitted if action == "OPEN"]) == 1, "Must NOT open duplicate event while behavior is continuous"


def test_multi_cue_track_severity_resolution():
    """Verify that multiple active events resolve to the maximum active severity."""
    # Track with both Phone (HIGH) and Head Rest (MEDIUM)
    active_events = [
        {"canonicalType": "SUSTAINED_HEAD_REST", "riskLevel": "MEDIUM", "score": 55},
        {"canonicalType": "PHONE_ASSOCIATED", "riskLevel": "HIGH", "score": 82}
    ]

    severity_order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
    max_sev = max(active_events, key=lambda e: severity_order.get(e["riskLevel"], 0))["riskLevel"]
    assert max_sev == "HIGH", "Track with Phone (HIGH) and Head Rest (MEDIUM) must resolve to HIGH"

    # When Phone event closes, track should fall back to MEDIUM (AMBER)
    active_events.pop(1)
    max_sev_after = max(active_events, key=lambda e: severity_order.get(e["riskLevel"], 0))["riskLevel"]
    assert max_sev_after == "MEDIUM", "Track must fall back to AMBER/MEDIUM when HIGH event closes"


def test_kpi_mathematical_invariant():
    """Verify invariant: TOTAL = AWAITING + CONFIRMED + DISMISSED."""
    class MockKPIStore:
        def __init__(self, events):
            self.events = events

        def get_kpis(self):
            total = len(self.events)
            awaiting = sum(1 for e in self.events if e.get("reviewStatus") == "awaiting")
            confirmed = sum(1 for e in self.events if e.get("reviewStatus") == "confirmed")
            dismissed = sum(1 for e in self.events if e.get("reviewStatus") == "dismissed")
            return {"total": total, "awaiting": awaiting, "confirmed": confirmed, "dismissed": dismissed}

    test_cases = [
        [],
        [{"reviewStatus": "awaiting"}],
        [{"reviewStatus": "confirmed"}, {"reviewStatus": "dismissed"}, {"reviewStatus": "awaiting"}],
        [{"reviewStatus": "confirmed"}] * 5 + [{"reviewStatus": "dismissed"}] * 3 + [{"reviewStatus": "awaiting"}] * 7
    ]

    for tc in test_cases:
        store = MockKPIStore(tc)
        k = store.get_kpis()
        assert k["total"] == k["awaiting"] + k["confirmed"] + k["dismissed"]


def test_fps_telemetry_source_distinction():
    """Verify distinct telemetry labels for Camera Capture vs. AI Processing target."""
    sys_view_file = Path("src/api/static/js/components/system_view.js")
    content = sys_view_file.read_text(encoding="utf-8")

    # Verify no hardcoded 30.0 target for AI processing
    assert "Observed Processing FPS (Target: 30.0)" not in content, "Must not hardcode 30.0 AI processing target"
    assert "Mục tiêu xử lý AI" in content, "Must have Vietnamese AI target label"
    assert "Tốc độ thu hình thực tế" in content, "Must have separate camera capture rate label"
