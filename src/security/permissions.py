"""
Centralized Role-Based Access Control (RBAC) resolver for ExamGuard Vision.
"""

import logging
from typing import Set, Union
from src.security.models import Role, Permission, User

logger = logging.getLogger(__name__)

ROLE_PERMISSIONS: dict[Role, Set[Permission]] = {
    Role.ADMIN: {
        Permission.MONITOR_VIEW,
        Permission.CAMERAS_VIEW,
        Permission.CAMERAS_MANAGE,
        Permission.EVENTS_ADJUDICATE,
        Permission.EVIDENCE_VIEW,
        Permission.HISTORY_VIEW,
        Permission.SESSION_EDIT,
        Permission.USERS_MANAGE,
        Permission.BACKUP_CREATE,
        Permission.BACKUP_VERIFY,
        Permission.AUDIT_VIEW,
        Permission.SECURITY_ADMIN,
    },
    Role.INVIGILATOR: {
        Permission.MONITOR_VIEW,
        Permission.CAMERAS_VIEW,
        Permission.EVENTS_ADJUDICATE,
        Permission.EVIDENCE_VIEW,
        Permission.HISTORY_VIEW,
        Permission.SESSION_EDIT,
    },
    Role.REVIEWER: {
        Permission.HISTORY_VIEW,
        Permission.EVIDENCE_VIEW,
        Permission.EVENTS_ADJUDICATE,
    },
    Role.VIEWER: {
        Permission.MONITOR_VIEW,
        Permission.CAMERAS_VIEW,
        Permission.HISTORY_VIEW,
        Permission.EVIDENCE_VIEW,
    },
}


class PermissionDeniedError(Exception):
    """Raised when user role lacks required permission."""
    def __init__(self, permission: Permission, role: Union[Role, str]):
        self.permission = permission
        self.role = role
        super().__init__(f"Quyền '{permission.value}' bị từ chối đối với vai trò '{role}'.")


def get_role_permissions(role: Union[Role, str]) -> Set[Permission]:
    """Retrieve full set of permissions for given role."""
    r = Role.from_str(role) if isinstance(role, str) else role
    return ROLE_PERMISSIONS.get(r, set())


def has_permission(role: Union[Role, str], permission: Permission) -> bool:
    """Return True if role grants given permission."""
    r = Role.from_str(role) if isinstance(role, str) else role
    return permission in ROLE_PERMISSIONS.get(r, set())


def check_user_permission(user: User, permission: Permission) -> bool:
    """Check if active user has permission."""
    if not user.is_active:
        return False
    return has_permission(user.role, permission)


def require_permission(user: User, permission: Permission) -> None:
    """Assert active user has permission, raising PermissionDeniedError if not."""
    if not check_user_permission(user, permission):
        raise PermissionDeniedError(permission, user.role)
