#!/usr/bin/env python3
"""
tools/security/reset_admin_password.py
ExamGuard Vision — Local Administrative Password Reset & Account Unlock Tool.

Emergency offline utility to:
- Identify local account
- Clear temporary lockout
- Reset failed attempt counters
- Optionally set a new Argon2id password interactively
- Revoke all active sessions
- Record audit event
- Never reveal old password or store plaintext passwords
"""

import sys
import os
import argparse
import getpass
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reconfigure stdout/stderr for Unicode/Vietnamese support on Windows terminals
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.persistence.service import PersistenceService
from src.persistence.database import DEFAULT_DB_PATH
from src.security.password import validate_password_policy


def main():
    parser = argparse.ArgumentParser(
        description="ExamGuard Vision — Local Admin Account Unlock & Password Reset Utility"
    )
    parser.add_argument(
        "--username",
        "-u",
        type=str,
        required=False,
        help="Username of account to unlock / reset (e.g. B24DCCN370)",
    )
    parser.add_argument(
        "--unlock-only",
        action="store_true",
        help="Only clear lockout and reset failed attempts counter without modifying password",
    )
    parser.add_argument(
        "--password",
        "-p",
        type=str,
        default=None,
        help="New password (warning: CLI arguments may be visible in process lists; prefer interactive prompt)",
    )
    parser.add_argument(
        "--password-stdin",
        action="store_true",
        help="Read new password from standard input (useful for automation)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to ExamGuard SQLite database (default: {DEFAULT_DB_PATH})",
    )

    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    if not db_path.exists():
        print(f"[ERROR] Database file not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    ps = PersistenceService(db_path=str(db_path))

    username = args.username
    if not username:
        username = input("Nhập tên đăng nhập cần xử lý: ").strip()

    if not username:
        print("[ERROR] Tên đăng nhập không được để trống.", file=sys.stderr)
        sys.exit(1)

    user = ps.users.get_user_by_username(username)
    if not user:
        print(f"[ERROR] Không tìm thấy tài khoản với tên đăng nhập: '{username}'", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("EXAMGUARD VISION — QUẢN TRỊ BẢO MẬT TÀI KHOẢN CỤC BỘ")
    print("=" * 60)
    print(f"Mã người dùng:        {user.user_id}")
    print(f"Tên đăng nhập:        {user.username}")
    print(f"Tên hiển thị:         {user.display_name}")
    print(f"Vai trò:              {user.role_value if hasattr(user, 'role_value') else user.role}")
    print(f"Trạng thái hoạt động: {'Hoạt động' if user.is_active else 'Bị vô hiệu hóa'}")
    print(f"Số lần nhập sai:      {user.failed_login_count}")
    print(f"Khóa tạm thời:        {user.locked_until or 'Không bị khóa'}")
    print("-" * 60)

    # Determine whether to reset password or unlock only
    if args.unlock_only:
        # Unlock only
        ps.unlock_user(user.user_id, admin_user_id="LOCAL_CLI")
        print("[SUCCESS] Đã mở khóa tài khoản thành công.")
        print("  - Trạng thái khóa: ĐÃ XÓA (None)")
        print("  - Số lần nhập sai: ĐÃ ĐẶT LẠI (0)")
        print("  - Mật khẩu hiện tại: GIỮ NGUYÊN")
        print("  - Nhật ký kiểm toán: ĐÃ GHI NHẬN (USER_UNLOCKED)")
        sys.exit(0)

    new_password = None
    if args.password:
        new_password = args.password
    elif args.password_stdin:
        new_password = sys.stdin.read().strip()
    else:
        # Interactive mode
        print("Tùy chọn thao tác:")
        print("  - Nhập mật khẩu mới để vừa mở khóa vừa đặt lại mật khẩu.")
        print("  - Hoặc bấm Enter (để trống) nếu chỉ muốn MỞ KHÓA và giữ nguyên mật khẩu cũ.")
        try:
            p1 = getpass.getpass("Mật khẩu mới (tối thiểu 12 ký tự, Enter để chỉ mở khóa): ")
        except (EOFError, KeyboardInterrupt):
            print("\n[ABORTED] Đã hủy thao tác.")
            sys.exit(1)

        if not p1.strip():
            # User wants unlock only
            ps.unlock_user(user.user_id, admin_user_id="LOCAL_CLI")
            print("[SUCCESS] Đã mở khóa tài khoản thành công (không đổi mật khẩu).")
            print("  - Trạng thái khóa: ĐÃ XÓA (None)")
            print("  - Số lần nhập sai: ĐÃ ĐẶT LẠI (0)")
            print("  - Nhật ký kiểm toán: ĐÃ GHI NHẬN (USER_UNLOCKED)")
            sys.exit(0)

        p2 = getpass.getpass("Xác nhận mật khẩu mới: ")
        if p1 != p2:
            print("[ERROR] Mật khẩu xác nhận không khớp!", file=sys.stderr)
            sys.exit(1)
        new_password = p1

    # Validate policy
    is_valid, err_msg = validate_password_policy(new_password)
    if not is_valid:
        print(f"[ERROR] Mật khẩu không đạt chính sách bảo mật: {err_msg}", file=sys.stderr)
        sys.exit(1)

    # Perform safe reset
    ok = ps.reset_password(
        user_id=user.user_id,
        new_password=new_password,
        admin_user_id="LOCAL_CLI",
        must_change_password=False,
    )
    if not ok:
        print("[ERROR] Không thể cập nhật mật khẩu trong cơ sở dữ liệu.", file=sys.stderr)
        sys.exit(1)

    print("[SUCCESS] Đã đặt lại mật khẩu và khôi phục tài khoản thành công.")
    print("  - Thuật toán băm: Argon2id (RFC 9106)")
    print("  - Trạng thái khóa: ĐÃ XÓA (None)")
    print("  - Số lần nhập sai: ĐÃ ĐẶT LẠI (0)")
    print("  - Phiên làm việc cũ: ĐÃ THU HỒI TOÀN BỘ (REVOKED)")
    print("  - Nhật ký kiểm toán: ĐÃ GHI NHẬN (PASSWORD_CHANGED)")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
