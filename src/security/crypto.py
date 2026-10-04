"""
Cryptographic envelope and AES-256-GCM authenticated encryption for ExamGuard evidence.
Binary envelope format (EGE1):
- Magic: b'EGE1' (4 bytes)
- Key ID length: 1 byte
- Key ID: UTF-8 string (1..255 bytes)
- Nonce: 12 bytes
- Payload: Ciphertext + 16-byte GCM authentication tag
"""

import hashlib
import json
import logging
import os
from typing import Dict, Any, Tuple, Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

from src.security.key_provider import KeyProvider, get_key_provider

logger = logging.getLogger(__name__)

ENVELOPE_MAGIC = b"EGE1"


def canonicalize_aad(aad_dict: Dict[str, Any]) -> bytes:
    """Produce deterministic canonical JSON representation of AAD metadata."""
    # Ensure minimal normalized keys
    normalized = {
        "evidence_id": str(aad_dict.get("evidence_id", "")),
        "evidence_type": str(aad_dict.get("evidence_type", "")).upper(),
        "event_id": str(aad_dict.get("event_id", "")),
        "session_id": str(aad_dict.get("session_id", "")),
    }
    return json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")


def encrypt_evidence_bytes(
    data: bytes,
    aad_dict: Dict[str, Any],
    key: bytes,
    key_id: str,
) -> Tuple[bytes, str]:
    """
    Encrypt plaintext bytes using AES-256-GCM bound to canonical AAD.
    Returns (envelope_bytes, envelope_sha256).
    """
    if len(key) != 32:
        raise ValueError(f"AES-256 key must be exactly 32 bytes, got {len(key)}")

    kid_bytes = key_id.encode("utf-8")
    if len(kid_bytes) > 255:
        raise ValueError(f"Key ID too long: {len(kid_bytes)} bytes (max 255)")

    nonce = os.urandom(12)
    aad_bytes = canonicalize_aad(aad_dict)

    aesgcm = AESGCM(key)
    ciphertext_and_tag = aesgcm.encrypt(nonce, data, aad_bytes)

    # Pack envelope: [MAGIC 4B] [KID_LEN 1B] [KID_BYTES] [NONCE 12B] [CIPHERTEXT+TAG]
    envelope = ENVELOPE_MAGIC + bytes([len(kid_bytes)]) + kid_bytes + nonce + ciphertext_and_tag
    sha256 = hashlib.sha256(envelope).hexdigest()
    return envelope, sha256


def decrypt_evidence_bytes(
    envelope: bytes,
    aad_dict: Dict[str, Any],
    key_provider: Optional[KeyProvider] = None,
) -> bytes:
    """
    Parse EGE1 envelope, look up key, and decrypt in-memory.
    Raises InvalidTag if tampered or AAD mismatch.
    """
    if len(envelope) < 4 + 1 + 12 + 16:
        raise ValueError("Invalid envelope: payload shorter than minimum header length.")

    magic = envelope[:4]
    if magic != ENVELOPE_MAGIC:
        raise ValueError(f"Unknown envelope format magic: {magic}")

    kid_len = envelope[4]
    kid_end = 5 + kid_len
    if len(envelope) < kid_end + 12 + 16:
        raise ValueError("Malformed envelope: header truncated.")

    key_id = envelope[5:kid_end].decode("utf-8")
    nonce = envelope[kid_end : kid_end + 12]
    payload = envelope[kid_end + 12 :]

    kp = key_provider or get_key_provider()
    key = kp.get_key(key_id)
    if not key or len(key) != 32:
        raise ValueError(f"Unable to retrieve 32-byte key for key_id '{key_id}'")

    aad_bytes = canonicalize_aad(aad_dict)
    aesgcm = AESGCM(key)

    # Will raise InvalidTag if tag mismatch or AAD mismatch
    plaintext = aesgcm.decrypt(nonce, payload, aad_bytes)
    return plaintext
