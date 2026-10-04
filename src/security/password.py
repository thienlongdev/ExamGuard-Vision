"""
Password hashing and policy enforcement for ExamGuard Vision.
Uses Argon2id with memory-hard parameters and constant-time verification.
"""

import logging
from typing import Tuple, Optional
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHashError

logger = logging.getLogger(__name__)

# Argon2id hasher configuration
# time_cost=2, memory_cost=64MB (65536 KiB), parallelism=2 threads, hash_len=32, salt_len=16
_hasher = PasswordHasher(
    time_cost=2,
    memory_cost=65536,
    parallelism=2,
    hash_len=32,
    salt_len=16,
)

MIN_PASSWORD_LENGTH = 12


def hash_password(password: str) -> str:
    """Hash plaintext password using Argon2id with random 16-byte salt."""
    if not password:
        raise ValueError("Mật khẩu không được để trống.")
    return _hasher.hash(password)


def verify_password(arg1: str, arg2: str) -> bool:
    """Verify password against Argon2id hash in constant time. Accepts either (hash, password) or (password, hash)."""
    if not arg1 or not arg2:
        return False
    if arg1.startswith("$argon2"):
        hash_val, candidate_password = arg1, arg2
    elif arg2.startswith("$argon2"):
        hash_val, candidate_password = arg2, arg1
    else:
        return False

    try:
        return _hasher.verify(hash_val, candidate_password)
    except (VerifyMismatchError, InvalidHashError):
        return False
    except Exception as e:
        logger.warning(f"Unexpected error during password verification: {e}")
        return False


def validate_password_policy(password: str) -> Tuple[bool, Optional[str]]:
    """
    Enforce password policy:
    - Minimum 12 characters
    - Non-empty
    Returns (is_valid, localized_vi_error_message).
    """
    if not password or len(password.strip()) == 0:
        return False, "Mật khẩu không được để trống."
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Mật khẩu phải có độ dài tối thiểu {MIN_PASSWORD_LENGTH} ký tự."
    return True, None
