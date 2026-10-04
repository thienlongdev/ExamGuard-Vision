"""
ExamGuard Vision Security and Access Control Package.
"""

from src.security.models import Role, Permission, User, AuthSession
from src.security.password import (
    hash_password,
    verify_password,
    validate_password_policy,
    MIN_PASSWORD_LENGTH,
)
from src.security.key_provider import (
    KeyProvider,
    WindowsDPAPIKeyProvider,
    InMemoryKeyProvider,
    EnvironmentKeyProvider,
    get_key_provider,
    set_key_provider,
)
from src.security.crypto import (
    canonicalize_aad,
    encrypt_evidence_bytes,
    decrypt_evidence_bytes,
    ENVELOPE_MAGIC,
)
from src.security.permissions import (
    ROLE_PERMISSIONS,
    PermissionDeniedError,
    get_role_permissions,
    has_permission,
    check_user_permission,
    require_permission,
)
from src.security.csrf import generate_csrf_token, validate_csrf_token
from src.security.redactor import redact_sensitive_text, redact_sensitive_dict

__all__ = [
    "Role",
    "Permission",
    "User",
    "AuthSession",
    "hash_password",
    "verify_password",
    "validate_password_policy",
    "MIN_PASSWORD_LENGTH",
    "KeyProvider",
    "WindowsDPAPIKeyProvider",
    "InMemoryKeyProvider",
    "EnvironmentKeyProvider",
    "get_key_provider",
    "set_key_provider",
    "canonicalize_aad",
    "encrypt_evidence_bytes",
    "decrypt_evidence_bytes",
    "ENVELOPE_MAGIC",
    "ROLE_PERMISSIONS",
    "PermissionDeniedError",
    "get_role_permissions",
    "has_permission",
    "check_user_permission",
    "require_permission",
    "generate_csrf_token",
    "validate_csrf_token",
    "redact_sensitive_text",
    "redact_sensitive_dict",
]
