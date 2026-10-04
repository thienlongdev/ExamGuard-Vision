"""
Session workflow (gate -> start/resume -> end), per-session History (pagination, filters,
isolation) and real playable video evidence (truthful READY/PENDING/FAILED states, Range).
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.evidence.clip_recorder import MIN_CLIP_DURATION_SEC, MIN_CLIP_FRAMES
from src.fusion.types import EventFamily, FusedEvent, RiskLevel
from src.orchestration.evidence_manager import IntegratedEvidenceManager
from src.persistence.models import EventEvidence, ExamSession, PersistedEvent
from src.persistence.service import PersistenceService
from src.security.key_provider import InMemoryKeyProvider, set_key_provider

STATIC = Path("src/api/static/js")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated DB, key provider and working dir (evidence lands under tmp_path/storage)."""
    set_key_provider(InMemoryKeyProvider())
    ps = PersistenceService(db_path=str(tmp_path / "eg.sqlite3"))
    app = create_app(persistence_service=ps, enforce_auth=False)
    client = TestClient(app)
    monkeypatch.chdir(tmp_path)
    yield client, ps, tmp_path
    ps.db.close()


def _start(client, name="DEMO VERIFY", room="Phòng A203"):
    res = client.post("/api/sessions/start", json={"name": name, "room": room, "invigilator_name": "GT A"})
    assert res.status_code == 200
    return res.json()


def _add_event(ps, session_id, event_id, review_status="awaiting"):
    ps.events.upsert_event(
        PersistedEvent(
            event_id=event_id,
            session_id=session_id,
            camera_id="cam_0",
            track_id=1,
            event_type="PHONE_ASSOCIATED",
            opened_at="2026-10-05T10:00:00",
            severity="HIGH",
            score=80.0,
            source_origin="ai_detector",
            review_status=review_status,
        )
    )


def _fused_event(event_id="ev_video_01", ts=100.0):
    return FusedEvent(
        event_id=event_id,
        track_id=3,
        camera_id="cam_0",
        event_type=EventFamily.PHONE_ASSOCIATED.value,
        start_timestamp=ts,
        last_update_timestamp=ts,
        duration=0.0,
        risk_level=RiskLevel.LOW.value,
    )


def _frame(i):
    f = np.full((240, 320, 3), (i * 5) % 255, dtype=np.uint8)
    cv2.putText(f, str(i), (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)
    return f


def _record_clip(ps, event_id="ev_video_01", pre=2.0, post=3.0, updates=3):
    """Drive the real evidence pipeline: ring buffer -> OPEN -> UPDATEs -> post-roll -> encode -> encrypt."""
    mgr = IntegratedEvidenceManager(
        output_dir="storage/evidence",
        pre_event_seconds=pre,
        post_event_seconds=post,
        persistence_service=ps,
    )
    t0, fps = time.time() - 10.0, 30.0
    event = _fused_event(event_id, ts=t0 + 3.0)
    for i in range(int(8 * fps)):
        ts = t0 + i / fps
        frame = _frame(i)
        mgr.push_frame(frame, ts)
        if i == int(3 * fps):
            mgr.handle_event_lifecycle(event, "OPEN", frame, ts, fps=fps)
        elif i in (int(3.5 * fps), int(4 * fps), int(4.5 * fps))[:updates]:
            mgr.handle_event_lifecycle(event, "UPDATE", frame, ts, fps=fps)
    mgr.shutdown()  # waits for background encode; never runs on the inference thread
    return event


def _clip_rows(ps, event_id):
    return [r for r in ps.evidence.list_evidence_for_event(event_id) if r.evidence_type == "VIDEO_CLIP"]


# ---------------------------------------------------------------------------
# SESSION FLOW
# ---------------------------------------------------------------------------


def test_login_without_active_session_shows_gate(env):
    client, ps, _ = env
    assert client.get("/api/sessions/current").status_code == 404
    html = client.get("/").text
    assert 'id="session-gate"' in html
    assert 'id="monitor-session-bar"' in html
    # Loading the dashboard must not create a session silently
    assert ps.sessions.list_sessions() == []


def test_start_refresh_and_relogin_keep_same_session(env):
    client, ps, _ = env
    started = _start(client)
    sid = started["session_id"]
    assert started["status"] == "ACTIVE"

    # Refresh: dashboard + current session again
    client.get("/")
    assert client.get("/api/sessions/current").json()["session_id"] == sid

    # Logout/login: a fresh client (new cookies) resumes the same session
    relogin = TestClient(client.app)
    assert relogin.get("/api/sessions/current").json()["session_id"] == sid
    assert len(ps.sessions.list_sessions()) == 1


def test_end_session_closes_normally_and_new_session_is_clean(env):
    client, ps, _ = env
    a = _start(client, name="Phiên A")
    _add_event(ps, a["session_id"], "ev_a1")
    _add_event(ps, a["session_id"], "ev_a2", review_status="confirmed")

    res = client.post(f"/api/sessions/{a['session_id']}/end", json={"reason": "COMPLETED"})
    assert res.status_code == 200
    closed = res.json()
    assert closed["status"] == "CLOSED"
    assert closed["status"] != "INTERRUPTED"
    assert closed["ended_at"]
    stored = ps.sessions.get_session(a["session_id"])
    assert stored.status == "CLOSED" and stored.ended_at and stored.close_reason == "COMPLETED"
    # Reviews / events preserved after close
    assert closed["summary"]["total_events"] == 2
    assert closed["summary"]["awaiting_count"] == 1
    assert closed["summary"]["confirmed_count"] == 1
    assert client.get("/api/sessions/current").status_code == 404

    b = _start(client, name="Phiên B")
    assert b["session_id"] != a["session_id"]
    s = b["summary"]
    assert (s["total_events"], s["awaiting_count"], s["confirmed_count"], s["dismissed_count"]) == (0, 0, 0, 0)
    assert client.get("/api/events").json() == []

    _add_event(ps, b["session_id"], "ev_b1")
    a_ids = {e["event_id"] for e in client.get(f"/api/sessions/{a['session_id']}/events").json()}
    b_ids = {e["event_id"] for e in client.get(f"/api/sessions/{b['session_id']}/events").json()}
    assert a_ids == {"ev_a1", "ev_a2"}
    assert b_ids == {"ev_b1"}


# ---------------------------------------------------------------------------
# HISTORY
# ---------------------------------------------------------------------------


def _seed_sessions(ps, names, status="CLOSED"):
    for i, name in enumerate(names):
        iso = f"2026-09-01T{8 + i // 60:02d}:{i % 60:02d}:00"
        ps.sessions.create_session(
            ExamSession(
                session_id=f"sess_seed_{i:03d}",
                name=name,
                room=f"Phòng {100 + i}",
                invigilator_name="GT",
                started_at=iso,
                ended_at=iso,
                status=status,
                camera_count=1,
                created_at=iso,
                updated_at=iso,
                last_heartbeat_at=iso,
                close_reason="COMPLETED",
            )
        )


def test_history_one_row_per_session_with_pagination(env):
    client, ps, _ = env
    _seed_sessions(ps, [f"Thi học kỳ {i}" for i in range(23)])
    for i in range(5):
        _add_event(ps, "sess_seed_000", f"ev_many_{i}")

    first = client.get("/api/sessions/page", params={"page": 1}).json()
    assert first["page_size"] == 10
    assert first["total"] == 23
    assert first["total_pages"] == 3
    assert len(first["items"]) == 10

    last = client.get("/api/sessions/page", params={"page": 3}).json()
    assert len(last["items"]) == 3

    all_ids = []
    for p in (1, 2, 3):
        all_ids += [s["session_id"] for s in client.get("/api/sessions/page", params={"page": p}).json()["items"]]
    assert len(all_ids) == len(set(all_ids)) == 23  # one row per session, events never become rows

    row = next(s for s in last["items"] + first["items"] if s["session_id"] == "sess_seed_000")
    assert row["summary"]["total_events"] == 5


def test_history_search_status_and_test_filters(env):
    client, ps, _ = env
    _seed_sessions(ps, ["Thi Toán", "Thi Văn", "FINAL 60M SOAK", "TEST_SESSION", "CARDINALITY_VERIFY"])
    _start(client, name="DEMO VERIFY", room="Phòng Z9")

    default = client.get("/api/sessions/page").json()
    names = {s["name"] for s in default["items"]}
    assert names == {"Thi Toán", "Thi Văn", "DEMO VERIFY"}
    assert default["hidden_test_count"] == 3

    with_test = client.get("/api/sessions/page", params={"include_test": "true"}).json()
    assert with_test["total"] == 6
    kinds = {s["name"]: s["session_kind"] for s in with_test["items"]}
    assert kinds["TEST_SESSION"] == "TEST"
    assert kinds["FINAL 60M SOAK"] == "VALIDATION"
    assert kinds["DEMO VERIFY"] == "OPERATIONAL"

    by_name = client.get("/api/sessions/page", params={"search": "Toán"}).json()
    assert [s["name"] for s in by_name["items"]] == ["Thi Toán"]
    by_room = client.get("/api/sessions/page", params={"search": "Z9"}).json()
    assert [s["name"] for s in by_room["items"]] == ["DEMO VERIFY"]

    active = client.get("/api/sessions/page", params={"status": "ACTIVE"}).json()
    assert [s["name"] for s in active["items"]] == ["DEMO VERIFY"]
    closed = client.get("/api/sessions/page", params={"status": "CLOSED"}).json()
    assert {s["status"] for s in closed["items"]} == {"CLOSED"}

    today = client.get("/api/sessions/page", params={"date_range": "today"}).json()
    assert [s["name"] for s in today["items"]] == ["DEMO VERIFY"]


# ---------------------------------------------------------------------------
# VIDEO EVIDENCE
# ---------------------------------------------------------------------------


def test_one_event_produces_one_snapshot_and_one_playable_video(env):
    client, ps, tmp_path = env
    sess = _start(client)
    _record_clip(ps, updates=3)

    clips = _clip_rows(ps, "ev_video_01")
    snaps = [r for r in ps.evidence.list_evidence_for_event("ev_video_01") if r.evidence_type == "SNAPSHOT"]
    assert len(clips) == 1, "UPDATE transitions must not create additional clips"
    assert len(snaps) == 1

    clip = clips[0]
    assert clip.artifact_state == "READY"
    assert clip.size_bytes > 0
    assert clip.encryption_state == "ENCRYPTED_V1"
    assert clip.frame_count >= MIN_CLIP_FRAMES
    assert MIN_CLIP_DURATION_SEC <= clip.duration_sec <= 8.0

    # Decrypted bytes are a finalized, decodable MP4 with real duration
    data, mime = ps.load_and_decrypt_evidence(clip.evidence_id)
    assert mime == "video/mp4"
    out = tmp_path / "decoded.mp4"
    out.write_bytes(data)
    cap = cv2.VideoCapture(str(out))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert n >= MIN_CLIP_FRAMES and fps > 0
    assert n / fps >= MIN_CLIP_DURATION_SEC
    assert b"moov" in data  # container finalized before encryption

    ev = client.get(f"/api/sessions/{sess['session_id']}/events").json()[0]
    assert ev["clip_status"] == "READY"
    assert ev["snapshot_status"] == "READY"


def test_unplayable_clip_is_never_ready(env):
    client, ps, _ = env
    _start(client)
    # The old demo config (0s pre/post roll) produced 2-frame clips shown as a 0:00 player
    _record_clip(ps, event_id="ev_short", pre=0.0, post=0.0, updates=0)
    assert all(r.artifact_state != "READY" for r in _clip_rows(ps, "ev_short"))
    ev = client.get("/api/events/ev_short").json()
    assert ev["clip_status"] == "FAILED"


def test_evidence_route_full_get_and_range_requests(env):
    client, ps, _ = env
    sess = _start(client)
    _record_clip(ps, updates=0)
    ev = client.get(f"/api/sessions/{sess['session_id']}/events").json()[0]
    url = ev["clip_path"]
    assert url.startswith("/api/evidence/") and ":" not in url  # no raw filesystem path

    full = client.get(url)
    assert full.status_code == 200
    assert full.headers["content-type"] == "video/mp4"
    total = int(full.headers["content-length"])
    assert total == len(full.content) > 0
    assert full.headers["accept-ranges"] == "bytes"

    part = client.get(url, headers={"Range": "bytes=0-99"})
    assert part.status_code == 206
    assert part.headers["content-range"] == f"bytes 0-99/{total}"
    assert len(part.content) == 100 and part.content == full.content[:100]

    open_ended = client.get(url, headers={"Range": f"bytes=10-{total + 500}"})
    assert open_ended.status_code == 206
    assert open_ended.headers["content-range"] == f"bytes 10-{total - 1}/{total}"

    suffix = client.get(url, headers={"Range": "bytes=-64"})
    assert suffix.status_code == 206 and suffix.content == full.content[-64:]

    assert client.get(url, headers={"Range": f"bytes={total}-"}).status_code == 416


def test_video_stays_playable_after_close_and_from_history(env):
    client, ps, _ = env
    sess = _start(client)
    sid = sess["session_id"]
    _record_clip(ps, updates=0)
    client.post(f"/api/sessions/{sid}/end", json={"reason": "COMPLETED"})
    _start(client, name="Phiên sau")

    # History -> Xem lại -> event -> video, after the session is CLOSED and another one started
    ev = client.get(f"/api/sessions/{sid}/events").json()[0]
    assert ev["clip_status"] == "READY"
    assert client.get(ev["clip_path"]).status_code == 200
    assert client.get(ev["snapshot_path"]).status_code == 200

    # Backend restart: a new app over the same persistent storage still serves it
    restarted = TestClient(create_app(persistence_service=ps, enforce_auth=False))
    again = restarted.get(f"/api/sessions/{sid}/events").json()[0]
    assert again["clip_status"] == "READY"
    assert restarted.get(again["clip_path"], headers={"Range": "bytes=0-15"}).status_code == 206


def test_pending_failed_and_legacy_states_are_truthful(env):
    client, ps, _ = env
    sess = _start(client)
    sid = sess["session_id"]

    _add_event(ps, sid, "ev_pending")
    assert client.get("/api/events/ev_pending").json()["clip_status"] == "PENDING"

    _add_event(ps, sid, "ev_failed")
    ps.events.update_evidence_summary("ev_failed", {"clip_state": "FAILED"})
    assert client.get("/api/events/ev_failed").json()["clip_status"] == "FAILED"

    # Legacy 2-frame / 0.13s clip that an older build marked READY
    _add_event(ps, sid, "ev_legacy")
    ps.evidence.add_evidence(
        EventEvidence(
            evidence_id="evd_legacy",
            event_id="ev_legacy",
            evidence_type="VIDEO_CLIP",
            relative_path="storage/sessions/x/evidence/ev_legacy/clip.mp4.enc",
            mime_type="video/mp4",
            sha256="0" * 64,
            size_bytes=336109,
            captured_at="2026-10-05T05:16:15",
            artifact_state="READY",
            duration_sec=0.13,
            frame_count=2,
            created_at="2026-10-05T05:16:15",
        )
    )
    assert client.get("/api/events/ev_legacy").json()["clip_status"] == "LEGACY_INVALID"

    # Events of a closed session that never got a clip are not shown as pending forever
    client.post(f"/api/sessions/{sid}/end", json={"reason": "COMPLETED"})
    hist = {e["event_id"]: e["clip_status"] for e in client.get(f"/api/sessions/{sid}/events").json()}
    assert hist["ev_pending"] == "LEGACY_INVALID"
    assert hist["ev_failed"] == "FAILED"
    assert hist["ev_legacy"] == "LEGACY_INVALID"


def test_evidence_requires_authentication(tmp_path):
    set_key_provider(InMemoryKeyProvider())
    ps = PersistenceService(db_path=str(tmp_path / "auth.sqlite3"))
    client = TestClient(create_app(persistence_service=ps, enforce_auth=True))
    assert client.get("/api/evidence/sessions/x/evidence/e/clip.mp4").status_code in (401, 403)
    assert client.get("/api/sessions/page").status_code in (401, 403)
    ps.db.close()


# ---------------------------------------------------------------------------
# FRONTEND CONTRACT
# ---------------------------------------------------------------------------


def test_frontend_session_gate_and_end_controls():
    gate = (STATIC / "components/session_gate.js").read_text(encoding="utf-8")
    for text in [
        "Khởi tạo phiên giám sát",
        "Bắt đầu phiên giám sát",
        "Bạn đang có một phiên giám sát đang hoạt động",
        "Tiếp tục phiên",
        "Kết thúc phiên hiện tại",
        "Xem chi tiết",
        "Kết thúc phiên giám sát?",
        "Các sự kiện và bằng chứng của phiên sẽ được lưu trong Lịch sử.",
        "Phiên đã kết thúc",
        "Bắt đầu phiên mới",
        "Xem lịch sử phiên",
        'id="btn-end-session"',
    ]:
        assert text in gate, text


def test_frontend_video_states_and_history_paging():
    drawer = (STATIC / "components/event_drawer.js").read_text(encoding="utf-8")
    assert "Đang chuẩn bị video bằng chứng..." in drawer
    assert "Không thể tạo video bằng chứng" in drawer
    assert "Video cũ không khả dụng" in drawer
    assert 'status === "READY" && ev.clipUrl' in drawer  # player only for READY

    history = (STATIC / "components/history_view.js").read_text(encoding="utf-8")
    for text in ["this.pageSize = 10", "Trước", "Sau", "Hiển thị phiên kiểm thử", "Tìm theo tên phiên hoặc phòng...", "inspectEvent"]:
        assert text in history, text
    assert "appState.upsertEvent" not in history  # historical events never enter live state


def test_session_review_survives_epoch_edge_timestamps(env):
    client, ps, _ = env
    sess = _start(client)
    _add_event(ps, sess["session_id"], "ev_ok")
    ps.events.upsert_event(
        PersistedEvent(
            event_id="ev_epoch_edge",
            session_id=sess["session_id"],
            camera_id="cam_0",
            track_id=2,
            event_type="PHONE_ASSOCIATED",
            opened_at="1970-01-01T07:00:10",  # raises OSError in datetime.timestamp() on Windows UTC+7
            severity="LOW",
            score=10.0,
        )
    )
    res = client.get(f"/api/sessions/{sess['session_id']}/events")
    assert res.status_code == 200
    assert {e["event_id"] for e in res.json()} == {"ev_ok", "ev_epoch_edge"}
