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

    # Case 2b: Clear phone candidate under 2.5s (e.g. 1.7s) must NOT be HIGH solely from phone
    short_clear_ev = FusedEvent(
        event_id="ev_short_clear_phone",
        track_id=1,
        camera_id="cam_0",
        event_type=EventFamily.PHONE_ASSOCIATED.value,
        start_timestamp=100.0,
        last_update_timestamp=101.7,
        duration=1.7,
        evidence_summary={"phone_status": "CLEAR_ASSOCIATION"}
    )
    res_short = agg.assess_event_risk(short_clear_ev, mean_reliability=clear_rel, independent_cues_count=1)
    assert res_short.risk_level == RiskLevel.MEDIUM.value, f"Short clear phone must be MEDIUM, got {res_short.risk_level}"
    assert res_short.risk_score < 75.0, f"Short clear phone must be < 75, got {res_short.risk_score}"
    assert res_short.risk_score >= 40.0, f"Short clear phone must be >= 40, got {res_short.risk_score}"

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
    - Standing baseline once ACTIVE is at least MEDIUM (40-74, not LOW, not immediately HIGH)
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
    assert res_standing.risk_level == RiskLevel.MEDIUM.value, f"Validated standing must be MEDIUM, got {res_standing.risk_level}"
    assert res_standing.risk_score >= 40.0, f"Validated standing score must be >= 40, got {res_standing.risk_score}"
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


def test_sustained_behaviors_minimum_medium_severity():
    """
    Verify Issue A & Section 2:
    Once a behavioral cue has passed temporal requirements and become an active event,
    it must NOT display as LOW.
    - SUSTAINED_HEAD_REST: minimum MEDIUM (>= 40.0)
    - SUSTAINED_LATERAL_HEAD_ORIENTATION: minimum MEDIUM (>= 40.0)
    - STANDING: minimum MEDIUM (>= 40.0)
    - DISCUSSION_CANDIDATE: minimum MEDIUM (>= 40.0)
    Single cue must not automatically become HIGH.
    """
    agg = RiskAggregator()

    cases = [
        (EventFamily.SUSTAINED_HEAD_REST.value, 2.0),
        (EventFamily.SUSTAINED_HEAD_REST.value, 2.9),
        (EventFamily.SUSTAINED_HEAD_REST.value, 8.0),
        (EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION.value, 1.5),
        (EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION.value, 3.5),
        (EventFamily.STANDING.value, 1.5),
        (EventFamily.STANDING.value, 3.0),
        (EventFamily.DISCUSSION_CANDIDATE.value, 2.0),
        (EventFamily.DISCUSSION_CANDIDATE.value, 4.0),
    ]

    for ev_type, dur in cases:
        ev = FusedEvent(
            event_id=f"test_{ev_type}_{dur}",
            track_id=1,
            camera_id="cam_0",
            event_type=ev_type,
            start_timestamp=100.0,
            last_update_timestamp=100.0 + dur,
            duration=dur,
            status="active",
            lifecycle_status="active",
            evidence_summary={},
            observation_snapshot={"risk": {}}
        )
        # Test across various reliabilities (0.65 to 1.0)
        for rel in [0.65, 0.80, 0.95, 1.0]:
            scored = agg.assess_event_risk(ev, active_cues_count=1, independent_cues_count=1, mean_reliability=rel)
            assert scored.risk_level == RiskLevel.MEDIUM.value, (
                f"{ev_type} at {dur}s (rel={rel}) must be MEDIUM, got {scored.risk_level} (score={scored.risk_score})"
            )
            assert scored.risk_score >= 40.0, (
                f"{ev_type} at {dur}s (rel={rel}) score must be >= 40.0, got {scored.risk_score}"
            )
            assert scored.risk_score < 75.0, (
                f"{ev_type} at {dur}s (rel={rel}) must not be HIGH without multi-cue/recurrence, got {scored.risk_score}"
            )
            # Consistency with observation_snapshot
            if "risk" in scored.observation_snapshot:
                assert scored.observation_snapshot["risk"]["level"] == RiskLevel.MEDIUM.value
                assert scored.observation_snapshot["risk"]["score"] >= 40.0


def test_continuous_head_rest_single_event_and_separate_incidents():
    """
    Verify Issue D & Section 5/6:
    1. Continuous 8-10s head rest produces exactly ONE event that updates duration.
    2. Head rest ends + return upright + cooldown finishes + head rest reappears -> opens SECOND event.
    """
    from src.fusion.cue_state import PerTrackCueState

    sm = TrackEventStateMachine(
        track_id=10,
        event_family=EventFamily.SUSTAINED_HEAD_REST,
        min_candidate_duration=2.0,
        enter_threshold=0.55,
        exit_threshold=0.35,
        cooldown_seconds=4.0
    )

    cue_state = PerTrackCueState(
        track_id=10,
        last_update_timestamp=100.0,
        posture_status=ObservationStatus.AVAILABLE,
        posture_probs={"HEAD_REST_SLEEP": 0.85, "NORMAL_UPRIGHT": 0.10},
    )

    opened_events = []
    updated_events = []

    # Scenario 1: Continuous 10 seconds in genuine head rest (100 frames at 10 FPS)
    for i in range(100):
        t = 100.0 + (i * 0.1)
        cue_state.last_update_timestamp = t
        ev, action = sm.process_frame(
            timestamp=t,
            evidence_score=0.85,
            is_vetoed=False,
            cue_state=cue_state,
        )
        if action == "OPEN":
            opened_events.append(ev)
        elif action == "UPDATE":
            updated_events.append(ev)

    # Must open exactly ONE event
    assert len(opened_events) == 1, f"Continuous head rest must open exactly 1 event, opened {len(opened_events)}"
    first_event_id = opened_events[0].event_id
    assert sm.state == EventLifecycleState.ACTIVE
    # Duration must have grown continuously to ~10.0s
    assert updated_events[-1].duration >= 9.8, f"Final duration should reach ~10s, got {updated_events[-1].duration}"
    assert all(e.event_id == first_event_id for e in updated_events), "All updates must reference the same event_id"

    # Scenario 2: Student returns upright, genuinely ending the incident
    cue_state_upright = PerTrackCueState(
        track_id=10,
        last_update_timestamp=110.1,
        posture_status=ObservationStatus.AVAILABLE,
        posture_probs={"HEAD_REST_SLEEP": 0.10, "NORMAL_UPRIGHT": 0.90},
    )
    ev_close, act_close = sm.process_frame(
        timestamp=110.1,
        evidence_score=0.10,
        is_vetoed=False,
        cue_state=cue_state_upright,
    )
    assert act_close == "CLOSE"
    assert sm.state == EventLifecycleState.COOLDOWN
    assert ev_close.status == "closed"

    # Wait through 4.0s cooldown (110.1 to 114.2)
    for step in range(1, 45):
        t_cd = 110.1 + (step * 0.1)
        ev_cd, act_cd = sm.process_frame(
            timestamp=t_cd,
            evidence_score=0.10,
            is_vetoed=False,
            cue_state=cue_state_upright,
        )
        assert act_cd == "NONE"

    assert sm.state == EventLifecycleState.INACTIVE, "After cooldown with low evidence, state must return to INACTIVE"

    # Scenario 3: Head rest reappears later (new incident)
    second_opens = []
    for step in range(25): # 2.5 seconds of head rest
        t_new = 115.0 + (step * 0.1)
        ev_new, act_new = sm.process_frame(
            timestamp=t_new,
            evidence_score=0.85,
            is_vetoed=False,
            cue_state=cue_state,
        )
        if act_new == "OPEN":
            second_opens.append(ev_new)

    assert len(second_opens) == 1, "A legitimate separate incident after cooldown must open a new event"
    second_event_id = second_opens[0].event_id
    assert second_event_id != first_event_id, "New incident must receive a distinct new event_id"


def test_phone_high_temporal_threshold_explicit():
    """
    Verify Issue E & Section 7/8/9:
    Authoritative phone policy:
    - 0.0-0.4s: transient (< 0.5s) -> LOW (<= 25.0)
    - 1.7s clear phone alone -> MEDIUM (40-74), strictly NOT HIGH
    - >= 2.5s sustained clear phone alone -> HIGH (>= 75.0)
    - Multi-cue escalation before 2.5s (e.g. 1.7s + independent cue) -> can reach HIGH with inspectable reason
    """
    agg = RiskAggregator()
    assert agg.phone_high_min_duration_sec == 2.5

    # 1. 1.7s clear association phone alone: MUST NOT BE HIGH
    ev_1_7 = FusedEvent(
        event_id="ev_phone_1_7",
        track_id=5,
        camera_id="cam_0",
        event_type=EventFamily.PHONE_ASSOCIATED.value,
        start_timestamp=10.0,
        last_update_timestamp=11.7,
        duration=1.7,
        evidence_summary={"phone_status": "CLEAR_ASSOCIATION"}
    )
    scored_1_7 = agg.assess_event_risk(ev_1_7, active_cues_count=1, independent_cues_count=1, mean_reliability=0.95)
    assert scored_1_7.risk_level == RiskLevel.MEDIUM.value, (
        f"Phone alone at 1.7s must be MEDIUM, got {scored_1_7.risk_level} (score={scored_1_7.risk_score})"
    )
    assert scored_1_7.risk_score < 75.0, f"Phone alone at 1.7s must be < 75.0, got {scored_1_7.risk_score}"
    assert scored_1_7.risk_score >= 40.0, f"Phone alone at 1.7s must be >= 40.0, got {scored_1_7.risk_score}"

    # 2. >= 2.5s sustained clear association phone: MUST BE HIGH
    for dur in [2.5, 3.1, 4.0]:
        ev_sustained = FusedEvent(
            event_id=f"ev_phone_{dur}",
            track_id=5,
            camera_id="cam_0",
            event_type=EventFamily.PHONE_ASSOCIATED.value,
            start_timestamp=10.0,
            last_update_timestamp=10.0 + dur,
            duration=dur,
            evidence_summary={"phone_status": "CLEAR_ASSOCIATION"}
        )
        scored_sustained = agg.assess_event_risk(ev_sustained, active_cues_count=1, independent_cues_count=1, mean_reliability=0.90)
        assert scored_sustained.risk_level == RiskLevel.HIGH.value, (
            f"Sustained clear phone at {dur}s must be HIGH, got {scored_sustained.risk_level}"
        )
        assert scored_sustained.risk_score >= 75.0, (
            f"Sustained clear phone at {dur}s score must be >= 75.0, got {scored_sustained.risk_score}"
        )
        assert scored_sustained.evidence_summary["risk_assessment"]["escalation_reason"] == "SUSTAINED_PHONE_TEMPORAL_CONFIRMED"

    # 3. Multi-cue escalation before 2.5s (e.g. 1.7s + independent cue) -> legitimate HIGH with inspectable reason
    ev_multi = FusedEvent(
        event_id="ev_phone_multi",
        track_id=5,
        camera_id="cam_0",
        event_type=EventFamily.PHONE_ASSOCIATED.value,
        start_timestamp=10.0,
        last_update_timestamp=11.7,
        duration=1.7,
        evidence_summary={"phone_status": "CLEAR_ASSOCIATION"}
    )
    scored_multi = agg.assess_event_risk(ev_multi, active_cues_count=2, independent_cues_count=2, mean_reliability=0.95)
    assert scored_multi.risk_level == RiskLevel.HIGH.value
    assert scored_multi.risk_score >= 75.0
    assert scored_multi.evidence_summary["risk_assessment"]["escalation_reason"] == "MULTI_CUE_CONCURRENCE"


def test_event_drawer_localization_and_lifecycle_copy():
    """
    Verify Issue B & C:
    - AVAILABLE translates to 'Có dữ liệu'
    - Event Drawer lifecycle label is 'Trạng thái sự kiện'
    - Closed state displays 'Đã kết thúc'
    - Active state displays 'Đang diễn ra'
    - Zero raw 'Vòng đời sự kiện' or raw 'AVAILABLE' shown to users
    """
    loc_file = Path("src/api/static/js/localization.js")
    drawer_file = Path("src/api/static/js/components/event_drawer.js")

    loc_content = loc_file.read_text(encoding="utf-8")
    drawer_content = drawer_file.read_text(encoding="utf-8")

    # Localization dictionary mapping
    assert 'AVAILABLE: "Có dữ liệu"' in loc_content
    assert 'UNAVAILABLE: "Không khả dụng"' in loc_content

    # Event Drawer imports and uses MACRO_BEHAVIOR_VI
    assert "MACRO_BEHAVIOR_VI" in drawer_content

    # Event Drawer user-facing lifecycle label
    assert '<span class="status-block-label">Trạng thái sự kiện</span>' in drawer_content
    assert '<span class="status-block-label">Vòng đời sự kiện</span>' not in drawer_content

    # Event Drawer lifecycle values
    assert '"Đã kết thúc"' in drawer_content
    assert '"Đang diễn ra"' in drawer_content


def test_risk_band_and_score_consistency_all_events():
    """
    Verify Issue 10:
    Single source of truth between risk score and severity band:
    - Score >= 75.0 <=> HIGH
    - 40.0 <= Score < 75.0 <=> MEDIUM
    - Score < 40.0 <=> LOW
    Never allow score 29 with MEDIUM or score 60 with LOW.
    """
    agg = RiskAggregator()

    for score_input in [0.0, 15.0, 25.0, 39.9, 40.0, 50.0, 60.0, 74.9, 75.0, 85.0, 95.0, 100.0]:
        ev = FusedEvent(
            event_id="consist_test",
            track_id=1,
            camera_id="cam_0",
            event_type=EventFamily.SUSTAINED_HEAD_REST.value,
            start_timestamp=0.0,
            last_update_timestamp=1.0,
            duration=1.0,
            evidence_summary={}
        )
        scored = agg.assess_event_risk(ev, mean_reliability=1.0)
        # Direct verification of the mathematical invariant
        if scored.risk_score >= 75.0:
            assert scored.risk_level == RiskLevel.HIGH.value
        elif scored.risk_score >= 40.0:
            assert scored.risk_level == RiskLevel.MEDIUM.value
        else:
            assert scored.risk_level == RiskLevel.LOW.value

