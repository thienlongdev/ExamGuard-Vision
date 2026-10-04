"""Tests for evidence encryption, AES-256-GCM envelope, AAD binding, and authorized evidence serving."""

import io
import os
import shutil
import tempfile
import pytest
from cryptography.exceptions import InvalidTag
from starlette.testclient import TestClient

from src.security.crypto import (
    encrypt_evidence_bytes,
    decrypt_evidence_bytes,
    canonicalize_aad,
    ENVELOPE_MAGIC,
)
from src.security.key_provider import InMemoryKeyProvider, get_key_provider, set_key_provider
from src.security.models import Role
from src.persistence.service import PersistenceService
from src.persistence.models import PersistedEvent
from src.api.main import create_app
from src.behavior.event_manager import EventManager


@pytest.fixture
def test_crypto_env():
    """Setup isolated crypto and persistence environment."""
    temp_dir = tempfile.mkdtemp()
    kp = InMemoryKeyProvider()
    set_key_provider(kp)
    db_path = os.path.join(temp_dir, "test_crypto.db")

    ps = PersistenceService(db_path=db_path)
    ev_manager = EventManager()

    app = create_app(
        persistence_service=ps,
        event_manager=ev_manager,
        enforce_auth=True,
    )
    client = TestClient(app)

    yield client, ps, kp, temp_dir

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_aes_gcm_envelope_encrypt_decrypt_roundtrip(test_crypto_env):
    """Verify AES-256-GCM envelope roundtrip preserves identical bytes."""
    _, _, kp, _ = test_crypto_env
    kid, key = kp.get_current_key()

    raw_data = b"ExamGuard-Synthetic-Snapshot-Payload-1234567890"
    aad = {
        "session_id": "ses_1",
        "event_id": "ev_1",
        "evidence_id": "evi_1",
        "evidence_type": "snapshot",
    }

    encrypted, sha = encrypt_evidence_bytes(raw_data, aad, key, key_id=kid)

    # Envelope structure checks
    assert encrypted.startswith(ENVELOPE_MAGIC)
    assert len(encrypted) > len(raw_data) + 12 + 16  # Nonce + tag + header overhead

    decrypted = decrypt_evidence_bytes(encrypted, aad, key_provider=kp)
    assert decrypted == raw_data


def test_tampered_ciphertext_fails_integrity(test_crypto_env):
    """Verify any bit tampering in ciphertext or tag raises InvalidTag."""
    _, _, kp, _ = test_crypto_env
    kid, key = kp.get_current_key()

    raw_data = b"Video-Clip-Critical-Evidence-Bytes"
    aad = {
        "session_id": "ses_1",
        "event_id": "ev_1",
        "evidence_id": "evi_1",
        "evidence_type": "clip",
    }
    encrypted, _ = encrypt_evidence_bytes(raw_data, aad, key, key_id=kid)
    tampered = bytearray(encrypted)

    # Tamper with the ciphertext (last byte)
    tampered[-1] ^= 0xFF

    with pytest.raises(InvalidTag):
        decrypt_evidence_bytes(bytes(tampered), aad, key_provider=kp)


def test_wrong_aad_fails_decryption(test_crypto_env):
    """Verify AAD mismatch prevents substituting encrypted blobs across events."""
    _, _, kp, _ = test_crypto_env
    kid, key = kp.get_current_key()

    raw_data = b"Snapshot-For-Track-1"
    aad_correct = {
        "session_id": "ses_1",
        "event_id": "ev_1",
        "evidence_id": "evi_1",
        "evidence_type": "snapshot",
    }
    aad_wrong_event = {
        "session_id": "ses_1",
        "event_id": "ev_2",
        "evidence_id": "evi_1",
        "evidence_type": "snapshot",
    }
    aad_wrong_session = {
        "session_id": "ses_99",
        "event_id": "ev_1",
        "evidence_id": "evi_1",
        "evidence_type": "snapshot",
    }

    encrypted, _ = encrypt_evidence_bytes(raw_data, aad_correct, key, key_id=kid)

    # Decrypt with wrong event_id -> InvalidTag
    with pytest.raises(InvalidTag):
        decrypt_evidence_bytes(encrypted, aad_wrong_event, key_provider=kp)

    # Decrypt with wrong session_id -> InvalidTag
    with pytest.raises(InvalidTag):
        decrypt_evidence_bytes(encrypted, aad_wrong_session, key_provider=kp)


def test_nonce_uniqueness(test_crypto_env):
    """Verify two encryptions of the same plaintext produce different ciphertexts."""
    _, _, kp, _ = test_crypto_env
    kid, key = kp.get_current_key()

    data = b"Same-Plaintext-Different-Encryptions"
    aad = {
        "session_id": "ses_1",
        "event_id": "ev_1",
        "evidence_id": "evi_1",
        "evidence_type": "snapshot",
    }

    enc1, _ = encrypt_evidence_bytes(data, aad, key, key_id=kid)
    enc2, _ = encrypt_evidence_bytes(data, aad, key, key_id=kid)

    assert enc1 != enc2
    assert decrypt_evidence_bytes(enc1, aad, key_provider=kp) == data
    assert decrypt_evidence_bytes(enc2, aad, key_provider=kp) == data


def test_evidence_route_authorization_and_serving(test_crypto_env):
    """Test authorized evidence serving, range requests, and unauthorized block."""
    client, ps, kp, temp_dir = test_crypto_env

    # Setup admin, reviewer
    admin = ps.create_initial_admin("admin_evi", "ValidPassword123456!", "Admin User")
    reviewer = ps.create_user("rev_evi", "ValidPassword123456!", "Reviewer", Role.REVIEWER)

    # Create synthetic session & event
    session = ps.initialize_runtime_session(
        name="Exam Room Session 1",
        room="Room 402",
        invigilator_name="Test Proctor",
    )

    clip_bytes = b"EXAMGUARD_SYNTHETIC_MP4_VIDEO_HEADER_CONTENT_BYTES_PADDED_TO_TEST_RANGE_REQUESTS_1234567890"
    event_id = "test_ev_evi_01"

    persisted_ev = PersistedEvent(
        event_id=event_id,
        session_id=session.session_id,
        camera_id="cam01",
        track_id=1,
        event_type="phone_detected",
        opened_at="2026-10-04T12:00:00",
        severity="HIGH",
        score=0.92,
        lifecycle_status="active",
        review_status="awaiting",
    )
    ps.events.upsert_event(persisted_ev)

    raw_clip_file = os.path.join(temp_dir, "test_clip.mp4")
    with open(raw_clip_file, "wb") as f:
        f.write(clip_bytes)

    evidence_item = ps.record_evidence_file(
        event_id=event_id,
        evidence_type="clip",
        file_path=raw_clip_file,
        mime_type="video/mp4",
        encrypt=True,
    )
    assert evidence_item is not None
    assert evidence_item.encryption_state == "ENCRYPTED_V1"
    # Ensure plaintext file was cleaned up and .enc exists
    assert not os.path.exists(raw_clip_file)
    assert os.path.exists(raw_clip_file + ".enc")

    # 1. ANONYMOUS: Cannot access evidence
    client.cookies.clear()
    r_anon = client.get(f"/api/evidence/{evidence_item.evidence_id}")
    assert r_anon.status_code == 401

    # 2. Path traversal attempt: must be rejected (401/403/404)
    r_traversal = client.get("/api/evidence/..%2F..%2F..%2Fetc%2Fpasswd")
    assert r_traversal.status_code in [401, 403, 404]

    # 3. AUTHORIZED REVIEWER: Login and retrieve full evidence
    r_login = client.post("/api/auth/login", json={"username": "rev_evi", "password": "ValidPassword123456!"})
    assert r_login.status_code == 200

    r_auth = client.get(f"/api/evidence/{evidence_item.evidence_id}")
    assert r_auth.status_code == 200
    assert r_auth.content == clip_bytes
    assert r_auth.headers.get("content-type") == "video/mp4"
    assert r_auth.headers.get("accept-ranges") == "bytes"

    # 4. HTTP RANGE REQUEST: HTML5 Video scrub/streaming
    headers = {"Range": "bytes=0-19"}
    r_range = client.get(
        f"/api/evidence/{evidence_item.evidence_id}",
        headers=headers,
    )
    assert r_range.status_code == 206
    assert r_range.content == clip_bytes[0:20]
    assert r_range.headers.get("content-range") == f"bytes 0-19/{len(clip_bytes)}"
    assert int(r_range.headers.get("content-length")) == 20

    # 5. Non-existent evidence ID returns 404
    r_notfound = client.get("/api/evidence/non_existent_id")
    assert r_notfound.status_code == 404
