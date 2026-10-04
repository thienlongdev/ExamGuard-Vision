"""Tests for encrypted backup creation, verification, and recovery key unwrapping."""

import os
import shutil
import tempfile
import pytest
from pathlib import Path

from src.security.key_provider import InMemoryKeyProvider, set_key_provider
from src.persistence.service import PersistenceService
from src.persistence.backup import (
    create_encrypted_backup,
    verify_backup_directory,
    BACKUP_FORMAT_V2_ENCRYPTED,
)
from src.persistence.models import PersistedEvent


@pytest.fixture
def backup_test_env():
    """Isolated test environment for encrypted backup and restore verification."""
    temp_dir = tempfile.mkdtemp()
    kp = InMemoryKeyProvider()
    set_key_provider(kp)

    db_path = os.path.join(temp_dir, "test_backup.db")
    ps = PersistenceService(db_path=db_path)

    # Initialize a session and an event with encrypted evidence
    session = ps.initialize_runtime_session(
        name="Backup Test Session",
        room="Room 101",
        invigilator_name="Proctor Alpha",
    )

    event_id = "ev_backup_001"
    persisted_ev = PersistedEvent(
        event_id=event_id,
        session_id=session.session_id,
        camera_id="cam01",
        track_id=42,
        event_type="suspicious_posture",
        opened_at="2026-10-04T12:00:00",
        severity="AMBER",
        score=0.75,
        lifecycle_status="active",
        review_status="awaiting",
    )
    ps.events.upsert_event(persisted_ev)

    raw_clip_file = os.path.join(temp_dir, "sample_clip.mp4")
    with open(raw_clip_file, "wb") as f:
        f.write(b"SYNTHETIC_CLIP_PAYLOAD_FOR_BACKUP_TEST")

    ps.record_evidence_file(
        event_id=event_id,
        evidence_type="clip",
        file_path=raw_clip_file,
        mime_type="video/mp4",
        encrypt=True,
    )

    backup_dir = os.path.join(temp_dir, "backups")
    os.makedirs(backup_dir, exist_ok=True)

    yield ps, backup_dir, temp_dir

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_encrypted_backup_creation_and_manifest(backup_test_env):
    """Verify encrypted backup generates encrypted DB and valid v2 manifest."""
    ps, backup_dir, temp_dir = backup_test_env
    passphrase = "SecureBackupPassphrase2026!"

    res = create_encrypted_backup(
        persistence_service=ps,
        recovery_passphrase=passphrase,
        backup_base_dir=backup_dir,
    )

    assert res["success"] is True
    assert res["encrypted"] is True
    assert res["backup_id"].startswith("backup_")

    target_path = Path(res["backup_path"])
    if not target_path.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        target_path = repo_root / target_path

    # Verify encrypted database exists, unencrypted database does NOT exist in backup
    assert (target_path / "examguard.sqlite3.enc").is_file()
    assert not (target_path / "examguard.sqlite3").exists()

    # Raw database content must NOT start with SQLite format header
    enc_db_bytes = (target_path / "examguard.sqlite3.enc").read_bytes()
    assert not enc_db_bytes.startswith(b"SQLite format 3")

    # Verify manifest fields
    manifest_file = target_path / "backup-manifest.json"
    assert manifest_file.is_file()
    import json
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["format_version"] == BACKUP_FORMAT_V2_ENCRYPTED
    assert manifest["encryption"]["algorithm"] == "AES-256-GCM"
    assert manifest["encryption"]["kdf"] == "scrypt"
    assert "salt_hex" in manifest["encryption"]["kdf_params"]
    assert "wrapped_key_hex" in manifest["encryption"]
    # NEVER store plaintext passphrase
    assert passphrase not in json.dumps(manifest)


def test_encrypted_backup_verification_correct_passphrase(backup_test_env):
    """Verify backup verification succeeds with the correct recovery passphrase."""
    ps, backup_dir, temp_dir = backup_test_env
    passphrase = "CorrectPassword123!"

    res = create_encrypted_backup(
        persistence_service=ps,
        recovery_passphrase=passphrase,
        backup_base_dir=backup_dir,
    )
    target_path = Path(res["backup_path"])
    if not target_path.is_absolute():
        target_path = Path(__file__).resolve().parent.parent / target_path

    is_valid, msg, details = verify_backup_directory(str(target_path), recovery_passphrase=passphrase)
    assert is_valid is True
    assert details["database_verified"] is True
    assert details["encrypted"] is True

    # Verify no decrypted temp files leaked in target directory
    assert not (target_path / "examguard.sqlite3").exists()


def test_encrypted_backup_verification_wrong_passphrase_fails(backup_test_env):
    """Verify backup verification fails cleanly when wrong recovery passphrase is provided."""
    ps, backup_dir, temp_dir = backup_test_env
    passphrase = "CorrectPassword123!"
    wrong_passphrase = "WrongPassword999!"

    res = create_encrypted_backup(
        persistence_service=ps,
        recovery_passphrase=passphrase,
        backup_base_dir=backup_dir,
    )
    target_path = Path(res["backup_path"])
    if not target_path.is_absolute():
        target_path = Path(__file__).resolve().parent.parent / target_path

    is_valid, msg, details = verify_backup_directory(str(target_path), recovery_passphrase=wrong_passphrase)
    assert is_valid is False
    assert "Mật khẩu" in msg or "giải mã" in msg.lower() or "tag" in msg.lower()


def test_encrypted_backup_tampered_fails(backup_test_env):
    """Verify modifying a byte in the encrypted database causes verification to fail."""
    ps, backup_dir, temp_dir = backup_test_env
    passphrase = "AnotherSecurePassword123!"

    res = create_encrypted_backup(
        persistence_service=ps,
        recovery_passphrase=passphrase,
        backup_base_dir=backup_dir,
    )
    target_path = Path(res["backup_path"])
    if not target_path.is_absolute():
        target_path = Path(__file__).resolve().parent.parent / target_path

    enc_db = target_path / "examguard.sqlite3.enc"
    data = bytearray(enc_db.read_bytes())
    # Tamper with byte in ciphertext
    data[-1] ^= 0xFF
    enc_db.write_bytes(data)

    is_valid, msg, details = verify_backup_directory(str(target_path), recovery_passphrase=passphrase)
    assert is_valid is False


def test_short_recovery_passphrase_rejected(backup_test_env):
    """Verify recovery passphrases under 8 characters are rejected."""
    ps, backup_dir, temp_dir = backup_test_env

    with pytest.raises(ValueError):
        create_encrypted_backup(
            persistence_service=ps,
            recovery_passphrase="short",
            backup_base_dir=backup_dir,
        )
