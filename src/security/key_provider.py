"""
KeyProvider abstraction and Windows DPAPI master key management.
"""

from abc import ABC, abstractmethod
import ctypes
from ctypes import wintypes
import hashlib
import logging
import os
from pathlib import Path
import secrets
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_KEY_BLOB_PATH = "storage/security/master-key.dpapi"


class KeyProvider(ABC):
    """Abstract interface for master encryption key provision."""

    @abstractmethod
    def get_key(self, key_id: Optional[str] = None) -> bytes:
        """Return 32-byte AES-256 master key for given key_id (or default key)."""
        pass

    @abstractmethod
    def get_current_key(self) -> Tuple[str, bytes]:
        """Return (key_id, 32-byte key) for newly encrypted artifacts."""
        pass


class DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_byte)),
    ]


def _dpapi_protect(plaintext: bytes) -> bytes:
    """Protect binary data using Windows DPAPI CryptProtectData (CurrentUser scope)."""
    p_in = DATA_BLOB(
        len(plaintext),
        ctypes.cast(ctypes.create_string_buffer(plaintext, len(plaintext)), ctypes.POINTER(ctypes.c_byte)),
    )
    p_out = DATA_BLOB()
    res = ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(p_in),
        "ExamGuardMasterKey",
        None,
        None,
        None,
        0,
        ctypes.byref(p_out),
    )
    if not res:
        err = ctypes.GetLastError()
        raise OSError(f"CryptProtectData failed with error code {err}")
    try:
        raw = ctypes.string_at(p_out.pbData, p_out.cbData)
        return raw
    finally:
        ctypes.windll.kernel32.LocalFree(p_out.pbData)


def _dpapi_unprotect(ciphertext: bytes) -> bytes:
    """Unprotect binary data using Windows DPAPI CryptUnprotectData (CurrentUser scope)."""
    p_in = DATA_BLOB(
        len(ciphertext),
        ctypes.cast(ctypes.create_string_buffer(ciphertext, len(ciphertext)), ctypes.POINTER(ctypes.c_byte)),
    )
    p_out = DATA_BLOB()
    res = ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(p_in),
        None,
        None,
        None,
        None,
        0,
        ctypes.byref(p_out),
    )
    if not res:
        err = ctypes.GetLastError()
        raise OSError(f"CryptUnprotectData failed with error code {err}")
    try:
        raw = ctypes.string_at(p_out.pbData, p_out.cbData)
        return raw
    finally:
        ctypes.windll.kernel32.LocalFree(p_out.pbData)


class WindowsDPAPIKeyProvider(KeyProvider):
    """
    Production Windows KeyProvider protecting the 256-bit master key with DPAPI.
    Blob is stored in storage/security/master-key.dpapi.
    Raw key is NEVER stored on disk in plaintext.
    """

    def __init__(self, key_path: Optional[str] = None):
        if key_path is None:
            repo_root = Path(__file__).resolve().parent.parent.parent
            self.key_path = repo_root / DEFAULT_KEY_BLOB_PATH
        else:
            self.key_path = Path(key_path)

        self._cached_key: Optional[bytes] = None
        self._cached_key_id: Optional[str] = None
        self._ensure_key_initialized()

    def _ensure_key_initialized(self) -> None:
        """Load and unprotect key from disk or generate and DPAPI-protect a new one."""
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        if self.key_path.is_file():
            try:
                blob = self.key_path.read_bytes()
                raw_key = _dpapi_unprotect(blob)
                if len(raw_key) != 32:
                    raise ValueError(f"Decrypted master key has invalid length: {len(raw_key)}")
                self._cached_key = raw_key
                self._cached_key_id = f"mk_{hashlib.sha256(raw_key).hexdigest()[:12]}"
                logger.debug(f"Loaded existing master key {self._cached_key_id} via DPAPI.")
                return
            except Exception as e:
                logger.error(f"Failed to unprotect DPAPI master key from {self.key_path}: {e}")
                raise

        # Generate new 32-byte key
        new_key = secrets.token_bytes(32)
        protected_blob = _dpapi_protect(new_key)
        self.key_path.write_bytes(protected_blob)
        self._cached_key = new_key
        self._cached_key_id = f"mk_{hashlib.sha256(new_key).hexdigest()[:12]}"
        logger.info(f"Initialized new DPAPI master key {self._cached_key_id} at {self.key_path}.")

    def get_key(self, key_id: Optional[str] = None) -> bytes:
        if self._cached_key is None:
            self._ensure_key_initialized()
        return self._cached_key

    def get_current_key(self) -> Tuple[str, bytes]:
        if self._cached_key is None or self._cached_key_id is None:
            self._ensure_key_initialized()
        return self._cached_key_id, self._cached_key


class InMemoryKeyProvider(KeyProvider):
    """Isolated in-memory key provider for unit testing without touching OS DPAPI."""

    def __init__(self, key: Optional[bytes] = None, key_id: Optional[str] = None):
        self.key = key or secrets.token_bytes(32)
        self.key_id = key_id or f"mk_{hashlib.sha256(self.key).hexdigest()[:12]}"

    def get_key(self, key_id: Optional[str] = None) -> bytes:
        return self.key

    def get_current_key(self) -> Tuple[str, bytes]:
        return self.key_id, self.key


class EnvironmentKeyProvider(KeyProvider):
    """Key provider reading 32-byte master key from environment variable (hex/urlsafe)."""

    def __init__(self, env_var: str = "EXAMGUARD_MASTER_KEY"):
        val = os.environ.get(env_var)
        if not val:
            raise ValueError(f"Environment variable '{env_var}' is not set.")
        val_clean = val.strip()
        try:
            if len(val_clean) == 64:
                self.key = bytes.fromhex(val_clean)
            else:
                import base64
                self.key = base64.urlsafe_b64decode(val_clean + "==")
        except Exception as e:
            raise ValueError(f"Failed to decode key from {env_var}: {e}")
        if len(self.key) != 32:
            raise ValueError(f"Decoded key must be exactly 32 bytes, got {len(self.key)}")
        self.key_id = f"mk_{hashlib.sha256(self.key).hexdigest()[:12]}"

    def get_key(self, key_id: Optional[str] = None) -> bytes:
        return self.key

    def get_current_key(self) -> Tuple[str, bytes]:
        return self.key_id, self.key


_GLOBAL_KEY_PROVIDER: Optional[KeyProvider] = None


def get_key_provider(provider_type: Optional[str] = None) -> KeyProvider:
    """Return default or configured KeyProvider instance."""
    global _GLOBAL_KEY_PROVIDER
    if _GLOBAL_KEY_PROVIDER is not None and provider_type is None:
        return _GLOBAL_KEY_PROVIDER

    ptype = (provider_type or os.environ.get("EXAMGUARD_KEY_PROVIDER", "")).lower()

    if ptype == "memory" or ptype == "test" or os.environ.get("EXAMGUARD_TEST_MODE") == "1":
        kp = InMemoryKeyProvider()
    elif ptype == "env" or os.environ.get("EXAMGUARD_MASTER_KEY"):
        kp = EnvironmentKeyProvider()
    elif os.name == "nt":
        kp = WindowsDPAPIKeyProvider()
    else:
        logger.warning("Non-Windows OS without EXAMGUARD_MASTER_KEY: using fallback InMemoryKeyProvider.")
        kp = InMemoryKeyProvider()

    if provider_type is None:
        _GLOBAL_KEY_PROVIDER = kp
    return kp


def set_key_provider(provider: KeyProvider) -> None:
    """Explicitly override global KeyProvider (useful in test fixtures)."""
    global _GLOBAL_KEY_PROVIDER
    _GLOBAL_KEY_PROVIDER = provider
