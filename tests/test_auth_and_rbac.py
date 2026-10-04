"""
Comprehensive test suite for ExamGuard Authentication, RBAC, Sessions, CSRF, and Rate Limiting.
"""

import os
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.persistence.database import DatabaseManager
from src.persistence.migrations import run_migrations
from src.persistence.service import PersistenceService
from src.security.models import Role, Permission
from src.security.permissions import has_permission
from src.security.password import hash_password, verify_password, validate_password_policy
from src.security.csrf import generate_csrf_token, verify_csrf_token


@pytest.fixture
def auth_test_env(tmp_path):
    db_file = tmp_path / "auth_test.sqlite3"
    ps = PersistenceService(db_path=str(db_file))
    app = create_app(persistence_service=ps, enforce_auth=True)
    client = TestClient(app)
    yield client, ps, tmp_path
    ps.db.close()


def test_password_policy():
    """Verify password length and policy requirements."""
    # Under 12 characters rejected
    ok, err = validate_password_policy("short")
    assert not ok
    assert "12" in err

    ok, err = validate_password_policy("12345678901")
    assert not ok

    # 12+ characters accepted
    ok, err = validate_password_policy("StrongPassword2026!")
    assert ok
    assert err is None


def test_argon2id_hashing():
    """Verify Argon2id password hashing and constant-time verification."""
    pw = "SuperSecureExamGuard2026@"
    h1 = hash_password(pw)
    h2 = hash_password(pw)

    # Hash does not equal plaintext
    assert h1 != pw
    # Different salts produce distinct hashes
    assert h1 != h2
    assert "$argon2id$" in h1

    # Verification
    assert verify_password(pw, h1) is True
    assert verify_password(pw, h2) is True
    assert verify_password("wrong_password", h1) is False


def test_first_run_admin_setup(auth_test_env):
    """Verify initial admin account creation flow when 0 users exist."""
    client, ps, _ = auth_test_env

    # 1. Zero users initially
    assert not ps.has_users()

    # 2. Setup page accessible
    res_get = client.get("/setup")
    assert res_get.status_code == 200
    assert "Khởi tạo Quản trị viên" in res_get.text

    # 3. Create initial admin
    res_post = client.post(
        "/api/setup",
        json={
            "username": "admin_root",
            "password": "MasterAdminExam2026!",
            "display_name": "Quản trị viên Trưởng",
        },
    )
    assert res_post.status_code == 200
    assert "eg_session" in res_post.cookies
    assert ps.has_users()

    # 4. Subsequent setup attempts must be rejected with 403
    res_post_repeat = client.post(
        "/api/setup",
        json={
            "username": "intruder",
            "password": "IntruderPassword2026!",
            "display_name": "Kẻ xâm nhập",
        },
    )
    assert res_post_repeat.status_code == 403


def test_login_rate_limiting_and_lockout(auth_test_env):
    """Verify lockout after 5 consecutive failed login attempts."""
    client, ps, _ = auth_test_env

    # Create user
    ps.create_initial_admin(
        username="proctor_van_a",
        password="ValidPassword123456!",
        display_name="Cán bộ Nguyễn Văn A",
    )

    # 4 failed attempts
    for _ in range(4):
        res = client.post(
            "/api/auth/login",
            json={"username": "proctor_van_a", "password": "WrongPassword!"},
        )
        assert res.status_code == 401

    # 5th failed attempt locks the account
    res5 = client.post(
        "/api/auth/login",
        json={"username": "proctor_van_a", "password": "WrongPassword!"},
    )
    assert res5.status_code == 423
    assert "khóa" in res5.json()["detail"].lower()

    # Even correct password fails while locked
    res_locked = client.post(
        "/api/auth/login",
        json={"username": "proctor_van_a", "password": "ValidPassword123456!"},
    )
    assert res_locked.status_code == 423


def test_session_token_and_logout(auth_test_env):
    """Verify server-side sessions, SHA-256 token hashing, and logout revocation."""
    client, ps, _ = auth_test_env

    user = ps.create_initial_admin(
        username="admin_session",
        password="ValidPassword123456!",
        display_name="Quản trị phiên",
    )

    # Login
    login_res = client.post(
        "/api/auth/login",
        json={"username": "admin_session", "password": "ValidPassword123456!"},
    )
    assert login_res.status_code == 200
    raw_token = login_res.cookies.get("eg_session")
    assert raw_token is not None

    # Check raw token is NOT in database (only hashed token stored)
    with ps.db.cursor() as cur:
        cur.execute("SELECT token_hash FROM auth_sessions;")
        rows = cur.fetchall()
        assert len(rows) >= 1
        stored_hashes = [r["token_hash"] for r in rows]
        assert raw_token not in stored_hashes

    # Validate session via /api/auth/me
    me_res = client.get("/api/auth/me", cookies={"eg_session": raw_token})
    assert me_res.status_code == 200
    assert me_res.json()["user"]["username"] == "admin_session"

    # Logout
    logout_res = client.post("/api/auth/logout", cookies={"eg_session": raw_token})
    assert logout_res.status_code == 200

    # Subsequent access with revoked session is rejected
    after_res = client.get("/api/auth/me", cookies={"eg_session": raw_token})
    assert after_res.status_code == 401


def test_csrf_protection_on_modifying_actions(auth_test_env):
    """Verify CSRF tokens required on state-changing requests."""
    client, ps, _ = auth_test_env

    ps.create_initial_admin(
        username="admin_csrf",
        password="ValidPassword123456!",
        display_name="Quản trị viên CSRF",
    )

    # Login to obtain session & CSRF token
    login_res = client.post(
        "/api/auth/login",
        json={"username": "admin_csrf", "password": "ValidPassword123456!"},
    )
    cookies = {"eg_session": login_res.cookies.get("eg_session")}
    csrf_token = login_res.json()["csrf_token"]

    # 1. State-changing request without CSRF token -> 403 Rejected
    no_csrf = client.post(
        "/api/users",
        json={"username": "test_inv", "password": "Password123456!", "display_name": "Test Inv"},
        cookies=cookies,
    )
    assert no_csrf.status_code == 403

    # 2. State-changing request with wrong CSRF token -> 403 Rejected
    wrong_csrf = client.post(
        "/api/users",
        json={"username": "test_inv", "password": "Password123456!", "display_name": "Test Inv"},
        cookies=cookies,
        headers={"X-CSRF-Token": "invalid_csrf_token_fake"},
    )
    assert wrong_csrf.status_code == 403

    # 3. State-changing request with correct CSRF token -> 200 Succeeded
    valid_req = client.post(
        "/api/users",
        json={"username": "test_inv", "password": "Password123456!", "display_name": "Test Inv", "role": "INVIGILATOR"},
        cookies=cookies,
        headers={"X-CSRF-Token": csrf_token},
    )
    assert valid_req.status_code == 200
    assert valid_req.json()["username"] == "test_inv"


def test_rbac_authorization_matrix(auth_test_env):
    """Verify RBAC role capabilities for ADMIN, INVIGILATOR, REVIEWER, and ANONYMOUS."""
    client, ps, _ = auth_test_env

    # 1. Create ADMIN and regular users
    admin = ps.create_initial_admin("admin_rbac", "ValidPassword123456!", "Admin User")
    invigilator = ps.create_user("invigilator_1", "ValidPassword123456!", "Giám thị 1", Role.INVIGILATOR)
    reviewer = ps.create_user("reviewer_1", "ValidPassword123456!", "Người rà soát 1", Role.REVIEWER)

    # Login sessions
    def get_user_session(uname):
        r = client.post("/api/auth/login", json={"username": uname, "password": "ValidPassword123456!"})
        return {"eg_session": r.cookies.get("eg_session")}, r.json()["csrf_token"]

    admin_cookies, admin_csrf = get_user_session("admin_rbac")
    inv_cookies, inv_csrf = get_user_session("invigilator_1")
    rev_cookies, rev_csrf = get_user_session("reviewer_1")

    # ANONYMOUS: Cannot access sensitive routes
    client.cookies.clear()
    assert client.get("/api/events").status_code == 401
    assert client.get("/api/sessions").status_code == 401
    assert client.get("/api/users").status_code == 401
    assert client.get("/api/system/security").status_code == 401

    # ADMIN: Can access all
    assert client.get("/api/events", cookies=admin_cookies).status_code == 200
    assert client.get("/api/users", cookies=admin_cookies).status_code == 200
    assert client.get("/api/system/security", cookies=admin_cookies).status_code == 200

    # INVIGILATOR: Can view events & current session, but CANNOT manage users or view security
    assert client.get("/api/events", cookies=inv_cookies).status_code == 200
    assert client.get("/api/users", cookies=inv_cookies).status_code == 403
    assert client.get("/api/system/security", cookies=inv_cookies).status_code == 403

    # REVIEWER: Can view history, but CANNOT configure cameras or manage users
    assert client.get("/api/sessions", cookies=rev_cookies).status_code == 200
    assert client.get("/api/users", cookies=rev_cookies).status_code == 403
    assert client.get("/api/cameras/config", cookies=rev_cookies).status_code == 403


def test_review_attribution_server_side(auth_test_env):
    """Verify human review decisions attribute reviewer_id and reviewer_name from session."""
    from src.persistence.models import PersistedEvent

    client, ps, _ = auth_test_env

    # Create admin and invigilator
    ps.create_initial_admin("admin_u", "ValidPassword123456!", "Admin User")
    ps.create_user("proctor_mai", "ValidPassword123456!", "Giám thị Mai", Role.INVIGILATOR)

    # Create synthetic event in DB
    sess = ps.initialize_runtime_session()
    ev = PersistedEvent(
        event_id="ev_attr_01",
        session_id=sess.session_id,
        camera_id="cam_main",
        track_id=2,
        event_type="PHONE_ASSOCIATED",
        opened_at="2026-10-04T12:00:00",
        severity="HIGH",
        score=95.0,
        lifecycle_status="active",
        review_status="awaiting",
    )
    ps.events.upsert_event(ev)

    # Login as invigilator
    login_res = client.post("/api/auth/login", json={"username": "proctor_mai", "password": "ValidPassword123456!"})
    cookies = {"eg_session": login_res.cookies.get("eg_session")}
    csrf = login_res.json()["csrf_token"]

    # Invigilator confirms event
    patch_res = client.patch(
        "/api/events/ev_attr_01",
        json={"status": "confirmed", "reviewer_notes": "Xác nhận nhìn thấy điện thoại dưới gầm bàn"},
        cookies=cookies,
        headers={"X-CSRF-Token": csrf},
    )
    assert patch_res.status_code == 200

    # Verify attribution persisted in DB
    review = ps.reviews.get_latest_review("ev_attr_01")
    assert review is not None
    assert review.decision == "CONFIRMED"
    assert review.reviewer_name == "Giám thị Mai"
    assert review.reviewer_id is not None
