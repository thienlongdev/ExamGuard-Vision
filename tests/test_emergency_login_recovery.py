"""
Focused test suite for ExamGuard Vision Emergency Login & Forgot Password Recovery.
Covers all 12 mandatory validation points:
1. Repeated incorrect passwords trigger lockout
2. Correct password cannot bypass active lockout
3. Authorized reset clears lockout
4. Reset clears failed-attempt counter
5. Old password fails after reset
6. New password succeeds after reset
7. Existing sessions revoked after reset
8. Unauthorized user cannot reset another user
9. Login page contains visible "Quên mật khẩu?"
10. Forgot-password route/modal loads
11. Password is never returned by API
12. Audit record exists
"""

import sys
import subprocess
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.persistence.service import PersistenceService
from src.security.models import Role


@pytest.fixture
def recovery_test_env(tmp_path):
    db_file = tmp_path / "recovery_test.sqlite3"
    ps = PersistenceService(db_path=str(db_file))
    app = create_app(persistence_service=ps, enforce_auth=True)
    client = TestClient(app)
    yield client, ps, db_file
    ps.db.close()


def test_lockout_trigger_and_cannot_bypass(recovery_test_env):
    """
    Points 1 & 2:
    - 5 incorrect attempts trigger lockout (423)
    - Correct password cannot bypass active lockout
    """
    client, ps, _ = recovery_test_env

    ps.create_initial_admin(
        username="admin_user",
        password="ValidPassword2026!@",
        display_name="Admin Test",
    )
    user = ps.create_user(
        username="invigilator_lock",
        password="CorrectPassword123!",
        display_name="Cán bộ Coi thi",
        role=Role.INVIGILATOR,
    )

    # 4 incorrect attempts -> 401
    for i in range(4):
        res = client.post(
            "/api/auth/login",
            json={"username": "invigilator_lock", "password": "WrongPassword!"},
        )
        assert res.status_code == 401

    # 5th attempt -> 423 Locked
    res5 = client.post(
        "/api/auth/login",
        json={"username": "invigilator_lock", "password": "WrongPassword!"},
    )
    assert res5.status_code == 423
    assert "khóa" in res5.json()["detail"].lower()
    assert "quên mật khẩu" in res5.json()["detail"].lower()

    # Point 2: Correct password CANNOT bypass active lockout
    res_correct_while_locked = client.post(
        "/api/auth/login",
        json={"username": "invigilator_lock", "password": "CorrectPassword123!"},
    )
    assert res_correct_while_locked.status_code == 423
    assert "khóa" in res_correct_while_locked.json()["detail"].lower()


def test_authorized_reset_clears_lockout_and_counters(recovery_test_env):
    """
    Points 3 & 4:
    - Authorized admin reset clears lockout (locked_until = None)
    - Reset clears failed-attempt counter (failed_login_count = 0)
    """
    client, ps, _ = recovery_test_env

    admin = ps.create_initial_admin(
        username="admin_root",
        password="AdminPassword2026!",
        display_name="Root Admin",
    )
    user = ps.create_user(
        username="target_user",
        password="InitialPassword123!",
        display_name="Target User",
        role=Role.INVIGILATOR,
    )

    # Lockout user with 5 failures
    for _ in range(5):
        client.post(
            "/api/auth/login",
            json={"username": "target_user", "password": "BadPassword!"},
        )

    # Verify locked in DB
    u_locked = ps.users.get_user_by_username("target_user")
    assert u_locked.failed_login_count == 5
    assert u_locked.locked_until is not None

    # Authorized admin reset
    admin_login = client.post(
        "/api/auth/login",
        json={"username": "admin_root", "password": "AdminPassword2026!"},
    )
    admin_cookies = {"eg_session": admin_login.cookies.get("eg_session")}
    admin_csrf = admin_login.json()["csrf_token"]

    reset_res = client.post(
        f"/api/users/{user.user_id}/reset-password",
        json={"new_password": "NewSecretPassword2026!"},
        cookies=admin_cookies,
        headers={"X-CSRF-Token": admin_csrf},
    )
    assert reset_res.status_code == 200

    # Point 3 & 4 verification:
    u_after = ps.users.get_user_by_username("target_user")
    assert u_after.failed_login_count == 0
    assert u_after.locked_until is None


def test_old_password_fails_and_new_password_succeeds(recovery_test_env):
    """
    Points 5 & 6:
    - Old password fails after reset (401)
    - New password succeeds after reset (200)
    """
    client, ps, _ = recovery_test_env

    admin = ps.create_initial_admin(
        username="admin_sys",
        password="AdminPassword2026!",
        display_name="System Admin",
    )
    user = ps.create_user(
        username="proctor_b",
        password="OldSecurePassword2026!",
        display_name="Proctor B",
        role=Role.INVIGILATOR,
    )

    # Admin resets password
    ok = ps.reset_password(user.user_id, "BrandNewPassword2026!@", admin_user_id=admin.user_id)
    assert ok is True

    # Point 5: Old password fails
    old_res = client.post(
        "/api/auth/login",
        json={"username": "proctor_b", "password": "OldSecurePassword2026!"},
    )
    assert old_res.status_code == 401

    # Point 6: New password succeeds
    new_res = client.post(
        "/api/auth/login",
        json={"username": "proctor_b", "password": "BrandNewPassword2026!@"},
    )
    assert new_res.status_code == 200
    assert "eg_session" in new_res.cookies


def test_existing_sessions_revoked_after_reset(recovery_test_env):
    """
    Point 7:
    - Existing sessions revoked after reset
    """
    client, ps, _ = recovery_test_env

    admin = ps.create_initial_admin(
        username="admin_sess",
        password="AdminPassword2026!",
        display_name="Admin Session",
    )
    user = ps.create_user(
        username="victim_user",
        password="InitialPassword2026!",
        display_name="Victim User",
        role=Role.INVIGILATOR,
    )

    # User logs in and gets active session
    u_login = client.post(
        "/api/auth/login",
        json={"username": "victim_user", "password": "InitialPassword2026!"},
    )
    user_session_token = u_login.cookies.get("eg_session")
    assert user_session_token is not None

    # Validate active session
    me_before = client.get("/api/auth/me", cookies={"eg_session": user_session_token})
    assert me_before.status_code == 200

    # Admin resets user's password
    ps.reset_password(user.user_id, "CompletelyNewPassword2026!", admin_user_id=admin.user_id)

    # Point 7: Previous session must now be rejected
    me_after = client.get("/api/auth/me", cookies={"eg_session": user_session_token})
    assert me_after.status_code == 401


def test_unauthorized_user_cannot_reset_another_user(recovery_test_env):
    """
    Point 8:
    - Unauthorized user cannot reset another user's password
    """
    client, ps, _ = recovery_test_env

    ps.create_initial_admin("admin_boss", "AdminPassword2026!", "Admin Boss")
    user1 = ps.create_user("invigilator_1", "Password123456!", "Giám thị 1", Role.INVIGILATOR)
    user2 = ps.create_user("invigilator_2", "Password123456!", "Giám thị 2", Role.INVIGILATOR)

    # 1. Anonymous attempt -> 401
    anon_res = client.post(
        f"/api/users/{user2.user_id}/reset-password",
        json={"new_password": "HackedPassword2026!"},
    )
    assert anon_res.status_code == 401

    # 2. Invigilator attempt on another user -> 403 Forbidden
    login_inv1 = client.post(
        "/api/auth/login",
        json={"username": "invigilator_1", "password": "Password123456!"},
    )
    inv1_cookies = {"eg_session": login_inv1.cookies.get("eg_session")}
    inv1_csrf = login_inv1.json()["csrf_token"]

    forbidden_res = client.post(
        f"/api/users/{user2.user_id}/reset-password",
        json={"new_password": "HackedPassword2026!"},
        cookies=inv1_cookies,
        headers={"X-CSRF-Token": inv1_csrf},
    )
    assert forbidden_res.status_code == 403


def test_login_page_ui_and_recovery_flow(recovery_test_env):
    """
    Points 9 & 10:
    - Login page contains visible "Quên mật khẩu?" link
    - Forgot-password route loads and modal structure is present
    """
    client, ps, _ = recovery_test_env

    ps.create_initial_admin("admin_ui", "AdminPassword2026!", "Admin UI")

    # Point 9: Login page has visible "Quên mật khẩu?"
    res_login = client.get("/login")
    assert res_login.status_code == 200
    assert "Quên mật khẩu?" in res_login.text
    assert 'id="forgot-password-link"' in res_login.text
    assert 'id="recovery-modal"' in res_login.text

    # Point 10: Dedicated forgot-password route loads offline recovery guide
    res_forgot = client.get("/forgot-password")
    assert res_forgot.status_code == 200
    assert "Khôi phục tài khoản" in res_forgot.text
    assert "Offline-First" in res_forgot.text
    assert "reset_admin_password.py" in res_forgot.text


def test_passwords_never_returned_by_api(recovery_test_env):
    """
    Point 11:
    - Passwords and password hashes are NEVER returned in API responses
    """
    client, ps, _ = recovery_test_env

    admin = ps.create_initial_admin("admin_sec", "AdminPassword2026!", "Admin Security")
    login_res = client.post(
        "/api/auth/login",
        json={"username": "admin_sec", "password": "AdminPassword2026!"},
    )
    cookies = {"eg_session": login_res.cookies.get("eg_session")}
    csrf = login_res.json()["csrf_token"]

    def assert_no_passwords(data):
        if isinstance(data, dict):
            assert "password" not in data
            assert "password_hash" not in data
            for v in data.values():
                assert_no_passwords(v)
        elif isinstance(data, list):
            for item in data:
                assert_no_passwords(item)

    # 1. Login response
    assert_no_passwords(login_res.json())

    # 2. /api/auth/me response
    me_res = client.get("/api/auth/me", cookies=cookies)
    assert_no_passwords(me_res.json())

    # 3. Create user response
    create_res = client.post(
        "/api/users",
        json={"username": "new_proctor", "password": "NewProctorPassword2026!", "display_name": "New Proctor"},
        cookies=cookies,
        headers={"X-CSRF-Token": csrf},
    )
    assert_no_passwords(create_res.json())

    # 4. List users response
    list_res = client.get("/api/users", cookies=cookies)
    assert_no_passwords(list_res.json())


def test_audit_records_exist_for_lockout_and_reset(recovery_test_env):
    """
    Point 12:
    - Audit records exist for failed login, lockout, unlock, and password change
    """
    client, ps, db_file = recovery_test_env

    admin = ps.create_initial_admin("admin_audit", "AdminPassword2026!", "Admin Audit")
    user = ps.create_user("audit_user", "AuditPassword2026!", "Audit User", Role.INVIGILATOR)

    # Trigger failure and lockout
    for _ in range(5):
        client.post(
            "/api/auth/login",
            json={"username": "audit_user", "password": "WrongPassword!"},
        )

    # Admin unlocks user
    ps.unlock_user(user.user_id, admin_user_id=admin.user_id)

    # Admin resets password
    ps.reset_password(user.user_id, "NewAuditPassword2026!", admin_user_id=admin.user_id)

    # Verify audit actions recorded
    with ps.db.cursor() as cur:
        cur.execute("SELECT action, actor_id, details_json FROM audit_logs ORDER BY created_at ASC;")
        rows = cur.fetchall()
        actions = [r["action"] for r in rows]

        assert "USER_CREATED" in actions
        assert "LOGIN_FAILED" in actions
        assert "LOGIN_LOCKED" in actions
        assert "USER_UNLOCKED" in actions
        assert "PASSWORD_CHANGED" in actions


def test_local_admin_cli_tool_unlock_and_reset(recovery_test_env):
    """
    Verify tools/security/reset_admin_password.py CLI operations:
    - --unlock-only clears lockout without changing password
    - --password sets new Argon2id password and clears lockout
    """
    client, ps, db_file = recovery_test_env

    user = ps.create_initial_admin("cli_admin", "OriginalPassword2026!", "CLI Admin")

    # Lock account with 5 failed attempts
    for _ in range(5):
        client.post("/api/auth/login", json={"username": "cli_admin", "password": "WrongPassword!"})

    u_locked = ps.users.get_user_by_username("cli_admin")
    assert u_locked.failed_login_count == 5
    assert u_locked.locked_until is not None

    cli_path = Path(__file__).resolve().parent.parent / "tools" / "security" / "reset_admin_password.py"

    # 1. Test CLI --unlock-only
    cmd_unlock = [
        sys.executable,
        str(cli_path),
        "--username",
        "cli_admin",
        "--unlock-only",
        "--db-path",
        str(db_file),
    ]
    res_cli_unlock = subprocess.run(
        cmd_unlock,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    assert "[SUCCESS]" in res_cli_unlock.stdout

    u_unlocked = ps.users.get_user_by_username("cli_admin")
    assert u_unlocked.failed_login_count == 0
    assert u_unlocked.locked_until is None

    # Original password still works
    res_orig = client.post("/api/auth/login", json={"username": "cli_admin", "password": "OriginalPassword2026!"})
    assert res_orig.status_code == 200

    # 2. Test CLI password reset
    cmd_reset = [
        sys.executable,
        str(cli_path),
        "--username",
        "cli_admin",
        "--password",
        "NewCliPassword2026!@",
        "--db-path",
        str(db_file),
    ]
    res_cli_reset = subprocess.run(
        cmd_reset,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    assert "[SUCCESS]" in res_cli_reset.stdout

    # Old password fails
    res_old = client.post("/api/auth/login", json={"username": "cli_admin", "password": "OriginalPassword2026!"})
    assert res_old.status_code == 401

    # New password succeeds
    res_new = client.post("/api/auth/login", json={"username": "cli_admin", "password": "NewCliPassword2026!@"})
    assert res_new.status_code == 200
