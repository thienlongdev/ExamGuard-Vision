"""
Cross-Site Request Forgery (CSRF) protection for ExamGuard state-changing routes.
"""

import hmac
import hashlib
import secrets
from typing import Optional


_DEFAULT_CSRF_KEY = "examguard-session-csrf-key-v1"


def generate_csrf_token(session_id: str, csrf_secret: str = _DEFAULT_CSRF_KEY) -> str:
    """Generate deterministic HMAC-SHA256 CSRF token bound to auth session."""
    key = (csrf_secret or _DEFAULT_CSRF_KEY).encode("utf-8")
    msg = f"csrf:{session_id}".encode("utf-8")
    sig = hmac.new(key, msg, hashlib.sha256).hexdigest()
    return f"{session_id}:{sig}"


def validate_csrf_token(token: Optional[str], expected_session_id: str, csrf_secret: str = _DEFAULT_CSRF_KEY) -> bool:
    """Validate CSRF token matches active session in constant time."""
    if not token or not expected_session_id:
        return False
    parts = token.split(":", 1)
    if len(parts) != 2:
        return False
    sess_id, sig = parts
    if not hmac.compare_digest(sess_id, expected_session_id):
        return False
    expected_token = generate_csrf_token(sess_id, csrf_secret)
    return hmac.compare_digest(token, expected_token)


verify_csrf_token = validate_csrf_token
