"""Deterministic geometry/lifecycle contracts, NOT physical accuracy evidence."""
from dataclasses import replace
import json
import threading
from types import SimpleNamespace
import numpy as np
import pytest
from src.detection.types import BBox
from src.tracking.tracker import Track
from src.orchestration.body_analysis import BodyGeometry, BodyPoseScheduler
from src.fusion.types import (BodyCue, ObservationStatus as S, EventFamily as F, HeadPoseCue,
                              PostureCue, TrackingState, UnifiedTrackUpdate)
from src.fusion.cue_state import PerTrackCueState
from src.fusion.fusion_engine import MultiCueFusionEngine
from src.fusion.event_engine import EventEngine, TrackEventStateMachine
from src.fusion.risk_aggregator import RiskAggregator


BOX = BBox(100, 50, 600, 650)


def points():
    p = np.zeros((17, 3))
    p[:, 2] = .95
    p[:5, :2] = [[300, 150], [315, 140], [285, 140], [330, 150], [270, 150]]
    p[5, :2], p[6, :2] = [360, 250], [240, 250]
    p[11, :2], p[12, :2] = [340, 450], [260, 450]
    return p


def calibrated(scale=1):
    g = BodyGeometry()
    b = BBox(*(x * scale for x in BOX.as_tuple()))
    p = points()
    p[:, :2] *= scale
    for t in (0, .25, .5, .75, 1):
        g.measure(('cam', 1), p, b, t, neutral=True)
    return g, b


def action_points(action):
    p = points()
    if action == 'lean':
        p[[5, 6], 0] += 100
        p[:5, 0] += 120
    elif action == 'rotation':
        p[[5, 6], 0] = [330, 270]
        p[[11, 12], 0] = [328, 272]
    elif action == 'rear':
        p[[5, 6]] = p[[6, 5]]
        p[[11, 12]] = p[[12, 11]]
        p[:3, 2] = .1
    return p


@pytest.mark.parametrize('action,field', [('lean', 'lean_score'), ('rotation', 'rotation_score'), ('rear', 'rearward_score')])
def test_geometry_is_scale_invariant(action, field):
    scores = []
    for scale in (1, 3, 6):
        g, b = calibrated(scale)
        p = action_points(action)
        p[:, :2] *= scale
        cue = g.measure(('cam', 1), p, b, 1.25)
        assert cue.calibrated
        scores.append(getattr(cue, field))
        json.dumps(replace(cue).__dict__)  # API payload cannot contain numpy bools.
    assert min(scores) >= .65
    assert scores == pytest.approx([scores[0]] * 3)


@pytest.mark.parametrize('action', ['write', 'read', 'page', 'pen', 'stationery', 'glasses', 'scratch', 'stretch', 'adjust', 'forward'])
def test_synthetic_normal_action_geometry_does_not_create_body_event(action):
    g, b = calibrated()
    p = points()
    # Head down / arm-only variations, not a claim that real movements were filmed.
    if action in ('write', 'read', 'forward'):
        p[:5, 1] += 65
    else:
        p[[7, 8, 9, 10], :2] = [[400, 200], [180, 300], [420, 100], [190, 450]]
    cue = g.measure(('cam', 1), p, b, 1.25)
    assert cue.lean_score == cue.rotation_score == cue.rearward_score == 0


def test_face_absence_alone_never_rearward_and_hidden_hips_unavailable():
    g, b = calibrated()
    p = points()
    p[:3, 2] = .1
    cue = g.measure(('cam', 1), p, b, 1.25)
    assert cue.face_visible is False
    assert cue.rearward_score == 0
    p[11, 2] = .1
    cue = g.measure(('cam', 1), p, b, 1.5)
    assert not cue.rotation_observable and not cue.rearward_observable
    assert cue.rearward_score == cue.rotation_score == cue.lean_score == 0


def test_hidden_hips_can_support_only_pronounced_upper_lean():
    g, b = calibrated()
    p = points()
    p[[11, 12], 2] = .1
    p[5, 1], p[6, 1] = 275, 225
    p[:5, 0] += 65
    cue = g.measure(('cam', 1), p, b, 1.25)
    assert cue.status == S.AVAILABLE and cue.lean_score >= .65
    assert not cue.rotation_observable and not cue.rearward_observable
    assert cue.rotation_score == cue.rearward_score == 0


def test_physical_upper_lean_geometry_and_normal_head_only_negative():
    g, b = calibrated()
    p = points()
    p[[11, 12], 2] = .1
    # Measured physical acceptance geometry: shoulder roll -.1906, head -.5197.
    p[5, 1], p[6, 1] = 238.6, 261.4
    p[:5, 0] -= 62.4
    cue = g.measure(('cam', 1), p, b, 1.25)
    assert cue.lean_score >= .65
    p[5, 1] = p[6, 1] = 250  # Head movement alone cannot generate a body cue.
    assert g.measure(('cam', 1), p, b, 1.5).lean_score == 0


def test_hysteresis_band_recovers_cooldown_and_next_clear_turn_opens():
    m = TrackEventStateMachine(1, F.SUSTAINED_LATERAL_HEAD_ORIENTATION, .6, .5, .3, 1, exit_grace_seconds=0, normal_reset_seconds=1)
    for i in range(11):
        m.process_frame(i * .1, .8, False, state(i * .1))
    m.process_frame(1.1, 0, False, state(1.1))
    for i in range(12, 26):
        m.process_frame(i * .1, .4, False, state(i * .1))
    opened = []
    for i in range(26, 38):
        ev, action = m.process_frame(i * .1, .9, False, state(i * .1))
        if action == 'OPEN':
            opened.append(ev)
    assert len(opened) == 1


def test_normal_action_158_second_head_episode_stays_internal():
    from src.fusion.types import FusedEvent
    ev = FusedEvent('normal-physical-probe', 1, 'cam', F.SUSTAINED_LATERAL_HEAD_ORIENTATION.value, 0, 2.88,
                    duration=2.88, evidence_summary={'active_evidence_duration': 1.58})
    scored = RiskAggregator().assess_event_risk(ev, independent_cues_count=2)
    assert scored.review_status == 'internal' and scored.risk_level == 'LOW'


def test_uncalibrated_rearward_has_no_fabricated_baseline():
    g = BodyGeometry()
    for t in np.arange(0, 4, .25):
        cue = g.measure(('cam', 1), action_points('rear'), BOX, float(t))
    assert not cue.calibrated and cue.rearward_score == 0


def test_identity_gap_jump_and_camera_scope_invalidate_calibration():
    for key, timestamp, box in [(('cam', 2), 1.25, BOX), (('other', 1), 1.25, BOX),
                                (('cam', 1), 3, BOX), (('cam', 1), 1.25, BBox(900, 50, 1400, 650))]:
        g, _ = calibrated()
        cue = g.measure(key, action_points('rear'), box, timestamp)
        assert not cue.calibrated and cue.rearward_score == 0


def state(t, body=None):
    return PerTrackCueState(track_id=1, last_update_timestamp=t, body=body or BodyCue())


def machine(family=F.SUSTAINED_TORSO_ROTATION):
    return TrackEventStateMachine(1, family, 2, .65, .4, 2, exit_grace_seconds=.5)


def test_candidate_pauses_missing_time_then_opens_once_and_closes():
    m = machine()
    opened, closed = [], []
    for i in range(40):
        t = i * .1
        observed = i not in (5, 6, 7)
        ev, action = m.process_frame(t, .8 if i < 32 else 0, False, state(t), observed=observed)
        if action == 'OPEN':
            opened.append((ev.event_id, t, ev.evidence_summary['active_evidence_duration']))
        if action == 'CLOSE':
            closed.append(ev.event_id)
    assert len(opened) == 1 and closed == [opened[0][0]]
    assert opened[0][1] >= 2.3  # The unavailable interval cannot satisfy duration.
    assert opened[0][2] >= 2


def test_long_gap_or_long_missing_cannot_open_or_keep_active():
    m = machine()
    m.process_frame(0, .8, False, state(0))
    assert m.process_frame(10, .8, False, state(10))[0] is None
    assert m.active_event is None
    for i in range(1, 24):
        t = 10 + i * .1
        m.process_frame(t, .8, False, state(t))
    assert m.active_event
    for i in range(24, 35):
        t = 10 + i * .1
        ev, action = m.process_frame(t, .8, False, state(t), observed=False)
        if action == 'CLOSE':
            assert ev.evidence_summary['closure_reason'] == 'OBSERVATION_EXPIRED'
    assert m.active_event is None


def test_opposite_directions_do_not_accumulate_one_sustained_head_candidate():
    m = machine(F.SUSTAINED_LATERAL_HEAD_ORIENTATION)
    for i in range(30):
        t = i * .1
        _, action = m.process_frame(t, .8, False, state(t), supporting_details={'direction': 'LEFT' if (i // 5) % 2 else 'RIGHT'})
        assert action != 'OPEN'


@pytest.mark.parametrize('family,field,duration', [(F.SUSTAINED_TORSO_ROTATION, 'rotation_score', 2),
    (F.SUSTAINED_SIDEWAYS_LEAN, 'lean_score', 2), (F.REARWARD_TURN_CANDIDATE, 'rearward_score', 2.5)])
def test_body_event_policy_dedup_medium_and_reviewable(family, field, duration):
    engine, risk = EventEngine(), RiskAggregator()
    opens = []
    for i in range(61):
        t = i * .1
        body = BodyCue(status=S.AVAILABLE, observed_at=t, fresh=True, reliability=.9, calibrated=True, **{field: .8})
        for ev, action in engine.process_cue_state(state(t, body), 'cam'):
            ev = risk.assess_event_risk(ev, independent_cues_count=10, mean_reliability=.9)
            if ev.event_type == family.value:
                assert ev.review_status == 'awaiting'
                assert ev.risk_level == 'MEDIUM'  # Even recurrence/concurrence cannot inflate pose to HIGH.
                assert ev.observation_snapshot['body']['calibrated']
                if action == 'OPEN':
                    opens.append((t, ev.event_id))
    assert len(opens) == 1 and opens[0][0] >= duration


def test_short_glance_writing_and_track_loss_do_not_open_review_event():
    fusion, engine = MultiCueFusionEngine(), EventEngine()
    emitted = []
    for i in range(40):
        t = i * .1
        yaw = 40 if i < 4 else 0
        update = UnifiedTrackUpdate(1, t, headpose=HeadPoseCue(status=S.AVAILABLE, yaw_deg=yaw),
                                    posture=PostureCue(status=S.AVAILABLE, probabilities={'NORMAL_READ_WRITE': .95}))
        cue = fusion.update_track(update)
        emitted.extend(engine.process_cue_state(cue))
    assert not [ev for ev, act in emitted if act == 'OPEN']


def test_fusion_camera_history_does_not_leak_and_track_jump_clears_history():
    f = MultiCueFusionEngine()
    def u(t, camera, yaw, box=BOX):
        return UnifiedTrackUpdate(1, t, camera_id=camera, headpose=HeadPoseCue(status=S.AVAILABLE, yaw_deg=yaw),
                                  tracking=TrackingState(1, t, box.as_tuple()))
    f.update_track(u(0, 'a', 50))
    other = f.update_track(u(.1, 'b', -40))
    assert other.smoothed_yaw_deg == -40 and other.continuity_valid
    jump = f.update_track(u(.2, 'a', 0, BBox(900, 50, 1400, 650)))
    assert not jump.continuity_valid and jump.smoothed_yaw_deg == 0


class Tensor:
    def __init__(self, data):
        self.data = np.array(data)
    def cpu(self):
        return self
    def numpy(self):
        return self.data


def test_pose_scheduler_overlap_suspends_owner_and_missing_model_falls_back():
    output = SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor([BOX.as_tuple()])), keypoints=SimpleNamespace(data=Tensor([points()])))
    registry = SimpleNamespace(body_pose_model=SimpleNamespace(predict=lambda *a, **k: [output]), device='cpu', inference_lock=threading.RLock())
    scheduler = BodyPoseScheduler(registry, {'models': {'body_pose': {'enabled': True}}})
    tracks = [Track(i, BOX, .95, 0) for i in (1, 2)]
    cues, _ = scheduler.infer(np.zeros((700, 700, 3)), tracks, 0, 'cam', {}, {})
    assert all(c.identity_ambiguous and c.status == S.UNAVAILABLE for c in cues.values())
    registry.body_pose_model = None
    assert scheduler.infer(None, tracks, 1, 'cam', {}, {}) == ({}, 0.0)


def test_repeated_leaning_is_metadata_not_automatic_high_priority():
    g, b = calibrated()
    for i, t in enumerate(np.arange(1.25, 6, .25)):
        cue = g.measure(('cam', 1), action_points('lean') if (i // 3) % 2 == 0 else points(), b, float(t))
    assert cue.repeated_movement_count >= 3
    assert cue.rearward_score == 0


def test_out_of_support_and_invisible_face_do_not_supply_yaw():
    from src.fusion.types import HeadPoseSupportStatus
    for hp, body in [(HeadPoseCue(status=S.AVAILABLE, yaw_deg=150), BodyCue()),
                     (HeadPoseCue(status=S.AVAILABLE, yaw_deg=50, support_status=HeadPoseSupportStatus.OUT_OF_SUPPORT), BodyCue()),
                     (HeadPoseCue(status=S.AVAILABLE, yaw_deg=50), BodyCue(face_visible=False))]:
        cue = MultiCueFusionEngine().update_track(UnifiedTrackUpdate(1, 1, headpose=hp, body=body))
        assert cue.smoothed_yaw_deg is None and cue.turn_fused_evidence == 0


def test_risk_recurrence_cannot_transfer_to_another_camera():
    from src.fusion.types import FusedEvent
    risk = RiskAggregator()
    closed = FusedEvent('old', 1, 'a', F.SUSTAINED_LATERAL_HEAD_ORIENTATION.value, 0, 2, duration=2, status='closed')
    risk.assess_event_risk(closed)
    other = FusedEvent('other', 1, 'b', F.SUSTAINED_LATERAL_HEAD_ORIENTATION.value, 0, 2, duration=2)
    new = RiskAggregator().assess_event_risk(replace(other))
    assert risk.assess_event_risk(other).risk_score == new.risk_score


def test_early_publication_precedes_slow_evidence(monkeypatch):
    from src.orchestration.stage2_pipeline import Stage2Pipeline
    from src.fusion.types import FusedEvent
    from src.video.base import VideoFrame
    import time
    pipe = Stage2Pipeline(enable_debug_overlay=False)
    order = []
    ev = FusedEvent('fixture-body-order', 1, 'cam', F.SUSTAINED_SIDEWAYS_LEAN.value, 100, 103, duration=3)
    monkeypatch.setattr(pipe.evidence_manager, 'push_frame', lambda *a: None)
    def evidence(*args, **kwargs):
        time.sleep(.025)
        order.append('evidence')
    monkeypatch.setattr(pipe.evidence_manager, 'handle_event_lifecycle', evidence)
    monkeypatch.setattr(pipe.event_engine, 'process_cue_state', lambda *a, **k: [(ev, 'OPEN')])
    pipe.add_event_listener(lambda *args: order.append('notify'))
    frame = np.zeros((700, 700, 3), dtype=np.uint8)
    pipe.process_frame(VideoFrame(frame=frame, timestamp=103, frame_idx=1, source_id='cam', fps=30, width=700, height=700), injected_tracks=[Track(1, BOX, .9, 103)])
    assert order == ['notify', 'evidence']


def test_api_ws_body_event_delivery_and_review_preserved(tmp_path):
    from fastapi.testclient import TestClient
    from src.api.main import create_app
    from src.api.websocket import ConnectionManager
    from src.behavior.event_manager import EventManager
    from src.persistence.service import PersistenceService
    from src.security.key_provider import InMemoryKeyProvider, set_key_provider
    from src.fusion.types import FusedEvent
    set_key_provider(InMemoryKeyProvider())
    ps = PersistenceService(db_path=str(tmp_path / 'body.db'))
    ps.start_monitoring_session(name='Synthetic body QA', room='fixture', camera_ids=['cam'])
    listeners = []
    pipeline = SimpleNamespace(add_event_listener=listeners.append)
    app = create_app(stage2_pipeline=pipeline, connection_manager=ConnectionManager(), event_manager=EventManager(camera_id='cam'), persistence_service=ps, enforce_auth=False)
    with TestClient(app) as client:
        with client.websocket_connect('/ws/events') as ws:
            ev = FusedEvent('body-test', 1, 'cam', F.REARWARD_TURN_CANDIDATE.value, 100, 103,
                            duration=3, risk_level='MEDIUM', risk_score=50,
                            evidence_summary={'clip_status': 'PENDING', 'is_reviewable': True})
            listeners[0](ev, 'OPEN')
            data = ws.receive_json()
            assert data['type'] == 'EVENT_OPEN'
            assert data['event']['track_id'] == 1 and data['event']['camera_id'] == 'cam'
            assert data['event']['evidence_summary']['clip_status'] == 'PENDING'
            assert client.get('/api/events/body-test').json()['event_type'] == F.REARWARD_TURN_CANDIDATE.value
            assert client.patch('/api/events/body-test', json={'status': 'dismissed', 'reviewer_notes': 'Synthetic review'}).status_code == 200
