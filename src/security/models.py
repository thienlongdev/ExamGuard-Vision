"""
Security and RBAC domain models for ExamGuard Vision.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List


class Role(str, Enum):
    ADMIN = "ADMIN"
    INVIGILATOR = "INVIGILATOR"
    REVIEWER = "REVIEWER"
    VIEWER = "VIEWER"

    @property
    def display_name_vi(self) -> str:
        mapping = {
            Role.ADMIN: "Quản trị viên",
            Role.INVIGILATOR: "Giám thị",
            Role.REVIEWER: "Người rà soát",
            Role.VIEWER: "Chỉ xem",
        }
        return mapping.get(self, str(self.value))

    @property
    def display_name(self) -> str:
        return self.display_name_vi

    @classmethod
    def from_str(cls, val: str) -> "Role":
        try:
            return cls(str(val).upper())
        except Exception:
            return cls.VIEWER


class Permission(str, Enum):
    MONITOR_VIEW = "monitor:view"
    CAMERAS_VIEW = "cameras:view"
    CAMERAS_MANAGE = "cameras:manage"
    CAMERA_CONFIGURE = "cameras:manage"
    EVENTS_ADJUDICATE = "events:adjudicate"
    EVENT_CONFIRM = "events:adjudicate"
    EVENT_DISMISS = "events:adjudicate"
    EVIDENCE_VIEW = "evidence:view"
    HISTORY_VIEW = "history:view"
    SESSION_EDIT = "session:edit"
    USERS_MANAGE = "users:manage"
    USER_MANAGE = "users:manage"
    BACKUP_CREATE = "backup:create"
    BACKUP_VERIFY = "backup:verify"
    AUDIT_VIEW = "audit:view"
    SECURITY_ADMIN = "security:admin"


@dataclass
class User:
    user_id: str
    username: str
    password_hash: str
    display_name: str
    role: str  # Role enum string
    is_active: int = 1
    must_change_password: int = 0
    failed_login_count: int = 0
    locked_until: Optional[str] = None
    last_login_at: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: Optional[str] = None

    @property
    def role_enum(self) -> Role:
        if isinstance(self.role, Role):
            return self.role
        return Role.from_str(str(self.role))

    @property
    def role_value(self) -> str:
        return self.role_enum.value

    @property
    def role_display(self) -> str:
        return self.role_enum.display_name_vi

    def to_dict(self, safe: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        if safe:
            d.pop("password_hash", None)
        d["role"] = self.role_value
        d["role_display"] = self.role_display
        d["role_display_vi"] = self.role_display
        return d


@dataclass
class AuthSession:
    auth_session_id: str
    user_id: str
    token_hash: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    expires_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_seen_at: str = field(default_factory=lambda: datetime.now().isoformat())
    revoked_at: Optional[str] = None
    user_agent_hash: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        if self.revoked_at is not None:
            return False
        try:
            exp = datetime.fromisoformat(self.expires_at)
            return datetime.now() < exp
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
