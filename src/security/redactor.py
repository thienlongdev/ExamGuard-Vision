"""
Secret and credential redactor for ExamGuard Vision logging and diagnostics.
"""

import re
from typing import Any, Dict


URL_CREDENTIAL_PATTERN = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)([^:/@]+):([^@]+)@")
KEY_VALUE_PATTERN = re.compile(
    r'(?i)(password|passwd|token|secret|master_key|recovery_passphrase|api_key|authorization)\s*[:=]\s*["\']?([^"\'\s,;]+)["\']?'
)
COOKIE_PATTERN = re.compile(r"(?i)(eg_session)=([a-zA-Z0-9_-]+)")


def redact_sensitive_text(text: str) -> str:
    """Redact passwords, RTSP credentials, session tokens, and keys from diagnostic text."""
    if not text:
        return text
    # 1. Redact credentials inside URLs (e.g. rtsp://user:pass@host)
    redacted = URL_CREDENTIAL_PATTERN.sub(r"\1\2:••••••••@", text)
    # 2. Redact key-value pairs (e.g. password="foo")
    redacted = KEY_VALUE_PATTERN.sub(r"\1=••••••••", redacted)
    # 3. Redact cookies
    redacted = COOKIE_PATTERN.sub(r"\1=••••••••", redacted)
    return redacted


def redact_sensitive_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return shallow/nested copy of dict with sensitive fields redacted."""
    sensitive_keys = {
        "password",
        "password_hash",
        "password_confirm",
        "token",
        "token_hash",
        "secret",
        "master_key",
        "recovery_passphrase",
        "authorization",
    }
    result = {}
    for k, v in d.items():
        if k.lower() in sensitive_keys:
            result[k] = "••••••••"
        elif isinstance(v, dict):
            result[k] = redact_sensitive_dict(v)
        elif isinstance(v, str):
            result[k] = redact_sensitive_text(v)
        else:
            result[k] = v
    return result
