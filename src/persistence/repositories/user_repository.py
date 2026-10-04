"""
Repository for users and auth_sessions database operations.
"""

from datetime import datetime, timedelta
import logging
from typing import Optional, List, Tuple, Any, Union
from src.persistence.database import DatabaseManager
from src.security.models import User, AuthSession, Role

logger = logging.getLogger(__name__)


class UserRepository:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def count_users(self) -> int:
        """Return total count of registered users."""
        sql = "SELECT COUNT(*) FROM users;"
        with self.db.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Fetch user by case-insensitive username."""
        sql = "SELECT * FROM users WHERE LOWER(username) = LOWER(?);"
        with self.db.cursor() as cur:
            cur.execute(sql, (username.strip(),))
            row = cur.fetchone()
            return self._row_to_user(row) if row else None

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Fetch user by user_id."""
        sql = "SELECT * FROM users WHERE user_id = ?;"
        with self.db.cursor() as cur:
            cur.execute(sql, (user_id,))
            row = cur.fetchone()
            return self._row_to_user(row) if row else None

    def create_user(self, user: User) -> User:
        """Insert a new user record."""
        sql = """
        INSERT INTO users (
            user_id, username, password_hash, display_name, role,
            is_active, must_change_password, failed_login_count,
            locked_until, last_login_at, created_at, updated_at, created_by
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.db.transaction() as cur:
            cur.execute(
                sql,
                (
                    user.user_id,
                    user.username.strip(),
                    user.password_hash,
                    user.display_name.strip(),
                    user.role,
                    user.is_active,
                    user.must_change_password,
                    user.failed_login_count,
                    user.locked_until,
                    user.last_login_at,
                    user.created_at,
                    user.updated_at,
                    user.created_by,
                ),
            )
        return user

    def update_user(
        self,
        user_id: Union[str, User],
        display_name: Optional[str] = None,
        role: Optional[str] = None,
        is_active: Optional[int] = None,
        must_change_password: Optional[int] = None,
    ) -> bool:
        """Update user metadata."""
        if hasattr(user_id, "user_id"):
            u = user_id
            user_id = u.user_id
            if display_name is None:
                display_name = u.display_name
            if role is None:
                role = u.role_value if hasattr(u, "role_value") else str(u.role)
            if is_active is None:
                is_active = 1 if u.is_active else 0
            if must_change_password is None:
                must_change_password = int(u.must_change_password) if hasattr(u, "must_change_password") else 0

        fields = []
        values = []
        if display_name is not None:
            fields.append("display_name = ?")
            values.append(display_name.strip())
        if role is not None:
            fields.append("role = ?")
            values.append(role)
        if is_active is not None:
            fields.append("is_active = ?")
            values.append(is_active)
        if must_change_password is not None:
            fields.append("must_change_password = ?")
            values.append(must_change_password)

        if not fields:
            return False

        fields.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(user_id)

        sql = f"UPDATE users SET {', '.join(fields)} WHERE user_id = ?;"
        with self.db.transaction() as cur:
            cur.execute(sql, tuple(values))
            return cur.rowcount > 0

    def update_password(self, user_id: str, password_hash: str, must_change_password: int = 0) -> bool:
        """Update password hash, reset failed login count, and clear lockout."""
        now_iso = datetime.now().isoformat()
        sql = """
        UPDATE users SET
            password_hash = ?,
            must_change_password = ?,
            failed_login_count = 0,
            locked_until = NULL,
            updated_at = ?
        WHERE user_id = ?;
        """
        with self.db.transaction() as cur:
            cur.execute(sql, (password_hash, must_change_password, now_iso, user_id))
            return cur.rowcount > 0

    def unlock_user(self, user_id: str) -> bool:
        """Clear lockout and reset failed login count without modifying password."""
        now_iso = datetime.now().isoformat()
        sql = """
        UPDATE users SET
            failed_login_count = 0,
            locked_until = NULL,
            updated_at = ?
        WHERE user_id = ?;
        """
        with self.db.transaction() as cur:
            cur.execute(sql, (now_iso, user_id))
            return cur.rowcount > 0

    def record_login_success(self, user_id: str) -> None:
        """Reset failed login count and update last_login_at timestamp."""
        now_iso = datetime.now().isoformat()
        sql = """
        UPDATE users SET
            failed_login_count = 0,
            locked_until = NULL,
            last_login_at = ?,
            updated_at = ?
        WHERE user_id = ?;
        """
        with self.db.transaction() as cur:
            cur.execute(sql, (now_iso, now_iso, user_id))

    def record_login_failure(
        self,
        username: str,
        max_attempts: int = 5,
        lockout_minutes: int = 15,
    ) -> Tuple[int, Optional[str]]:
        """
        Increment failed login count. If count >= max_attempts, set locked_until timestamp.
        Returns (new_failed_count, locked_until_iso).
        """
        user = self.get_user_by_username(username)
        if not user:
            return 0, None

        now = datetime.now()
        new_count = user.failed_login_count + 1
        locked_until_iso = None

        if new_count >= max_attempts:
            locked_until_dt = now + timedelta(minutes=lockout_minutes)
            locked_until_iso = locked_until_dt.isoformat()

        sql = """
        UPDATE users SET
            failed_login_count = ?,
            locked_until = ?,
            updated_at = ?
        WHERE user_id = ?;
        """
        with self.db.transaction() as cur:
            cur.execute(sql, (new_count, locked_until_iso, now.isoformat(), user.user_id))

        return new_count, locked_until_iso

    def list_users(self) -> List[User]:
        """List all users ordered by creation date."""
        sql = "SELECT * FROM users ORDER BY created_at ASC;"
        results = []
        with self.db.cursor() as cur:
            cur.execute(sql)
            for row in cur.fetchall():
                results.append(self._row_to_user(row))
        return results

    # =========================================================================
    # Auth Sessions Operations
    # =========================================================================

    def create_auth_session(self, session: AuthSession) -> AuthSession:
        """Store new authenticated session with hashed token."""
        sql = """
        INSERT INTO auth_sessions (
            auth_session_id, user_id, token_hash, created_at,
            expires_at, last_seen_at, revoked_at, user_agent_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.db.transaction() as cur:
            cur.execute(
                sql,
                (
                    session.auth_session_id,
                    session.user_id,
                    session.token_hash,
                    session.created_at,
                    session.expires_at,
                    session.last_seen_at,
                    session.revoked_at,
                    session.user_agent_hash,
                ),
            )
        return session

    def get_auth_session_by_token_hash(self, token_hash: str) -> Optional[AuthSession]:
        """Lookup active session by token hash."""
        sql = "SELECT * FROM auth_sessions WHERE token_hash = ? AND revoked_at IS NULL;"
        with self.db.cursor() as cur:
            cur.execute(sql, (token_hash,))
            row = cur.fetchone()
            if not row:
                return None
            sess = self._row_to_session(row)
            return sess if sess.is_valid else None

    def touch_auth_session(self, auth_session_id: str) -> None:
        """Update last_seen_at timestamp."""
        now_iso = datetime.now().isoformat()
        sql = "UPDATE auth_sessions SET last_seen_at = ? WHERE auth_session_id = ?;"
        with self.db.transaction() as cur:
            cur.execute(sql, (now_iso, auth_session_id))

    def revoke_auth_session(self, auth_session_id: str) -> None:
        """Revoke a specific session."""
        now_iso = datetime.now().isoformat()
        sql = "UPDATE auth_sessions SET revoked_at = ? WHERE auth_session_id = ?;"
        with self.db.transaction() as cur:
            cur.execute(sql, (now_iso, auth_session_id))

    def revoke_all_user_sessions(self, user_id: str) -> None:
        """Revoke all sessions for a user (e.g. after password reset or role change)."""
        now_iso = datetime.now().isoformat()
        sql = "UPDATE auth_sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL;"
        with self.db.transaction() as cur:
            cur.execute(sql, (now_iso, user_id))

    def count_active_sessions(self) -> int:
        """Count valid, unexpired, non-revoked sessions."""
        now_iso = datetime.now().isoformat()
        sql = "SELECT COUNT(*) FROM auth_sessions WHERE revoked_at IS NULL AND expires_at > ?;"
        with self.db.cursor() as cur:
            cur.execute(sql, (now_iso,))
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def _row_to_user(self, row: Any) -> User:
        return User(
            user_id=row["user_id"],
            username=row["username"],
            password_hash=row["password_hash"],
            display_name=row["display_name"],
            role=row["role"],
            is_active=int(row["is_active"]),
            must_change_password=int(row["must_change_password"]),
            failed_login_count=int(row["failed_login_count"]),
            locked_until=row["locked_until"],
            last_login_at=row["last_login_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            created_by=row["created_by"],
        )

    def _row_to_session(self, row: Any) -> AuthSession:
        return AuthSession(
            auth_session_id=row["auth_session_id"],
            user_id=row["user_id"],
            token_hash=row["token_hash"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            last_seen_at=row["last_seen_at"],
            revoked_at=row["revoked_at"],
            user_agent_hash=row["user_agent_hash"],
        )
