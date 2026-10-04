"""Tests for encrypted backup creation, verification, and recovery key unwrapping."""

import os
import shutil
import tempfile
import pytest
from pathlib import Path

from src.security.key_provider import InMemoryKeyProvider, set_key_provider, get_key_provider
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


def test_no_plaintext_backup_db_on_disk(backup_test_env):
    """
    Automated filesystem monitoring test:
    Verify that NO plaintext SQLite file (header 'SQLite format 3' or plaintext DB marker)
    is ever written to disk during encrypted backup creation.
    """
    ps, backup_dir, temp_dir = backup_test_env
    passphrase = "UltraSecurePassphrase2026!"
    synthetic_marker = b"PLAINTEXT_SECRET_MARKER_CERTIFICATION_998877"

    # Insert synthetic marker into production database
    with ps.db.transaction() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS cert_secrets (id INT, secret_blob BLOB);")
        cur.execute("INSERT INTO cert_secrets VALUES (1, ?);", (synthetic_marker,))

    # Snapshot directory state before backup
    before_files = set(Path(temp_dir).rglob("*"))

    res = create_encrypted_backup(
        persistence_service=ps,
        recovery_passphrase=passphrase,
        backup_base_dir=backup_dir,
    )
    assert res["success"] is True

    # Inspect all newly created files in the entire test environment
    after_files = set(Path(temp_dir).rglob("*"))
    new_files = [p for p in (after_files - before_files) if p.is_file()]

    sqlite_header = b"SQLite format 3\x00"

    for p in new_files:
        content = p.read_bytes()
        # 1. No backup file written to disk may have an unencrypted SQLite header
        if p.name != "test_backup.db" and not p.name.endswith(".db-wal") and not p.name.endswith(".db-shm"):
            assert not content.startswith(sqlite_header), f"Found plaintext SQLite database at {p}"
            # 2. No backup file may reveal the synthetic unencrypted marker
            assert synthetic_marker not in content, f"Plaintext secret marker leaked into disk file: {p}"

    # Confirm encrypted database exists and does NOT contain plaintext header
    target_path = Path(res["backup_path"])
    if not target_path.is_absolute():
        target_path = Path(__file__).resolve().parent.parent / target_path
    enc_db = target_path / "examguard.sqlite3.enc"
    assert enc_db.is_file()
    assert not enc_db.read_bytes().startswith(sqlite_header)


def test_portable_master_key_wrapping_and_cross_machine_recovery(backup_test_env):
    """
    Verify portable backup contains wrapped master key that allows full evidence recovery
    on another machine with a different/blank KeyProvider using only the recovery passphrase.
    """
    from src.persistence.backup import recover_wrapped_master_key
    from src.security.crypto import decrypt_evidence_bytes
    from cryptography.exceptions import InvalidTag

    ps, backup_dir, temp_dir = backup_test_env
    passphrase = "CrossMachinePassphrase2026!"
    wrong_passphrase = "WrongCrossMachinePassphrase999!"

    # Get original master key
    kp_orig = get_key_provider()
    orig_key_id, orig_master_key = kp_orig.get_current_key()

    res = create_encrypted_backup(
        persistence_service=ps,
        recovery_passphrase=passphrase,
        backup_base_dir=backup_dir,
    )
    assert res["success"] is True

    target_path = Path(res["backup_path"])
    if not target_path.is_absolute():
        target_path = Path(__file__).resolve().parent.parent / target_path

    # 1. Recover wrapped key with correct passphrase
    rec_key_id, rec_master_key = recover_wrapped_master_key(target_path, passphrase)
    assert rec_master_key == orig_master_key
    assert rec_key_id == orig_key_id

    # 2. Wrong passphrase fails with InvalidTag
    with pytest.raises(InvalidTag):
        recover_wrapped_master_key(target_path, wrong_passphrase)

    # 3. Simulate destination clean machine: create new isolated KeyProvider with recovered key
    dest_kp = InMemoryKeyProvider(key=rec_master_key, key_id=rec_key_id)

    # 4. Decrypt synthetic evidence file copied in the backup using destination KeyProvider
    manifest_file = target_path / "backup-manifest.json"
    import json
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert len(manifest["evidence_files"]) > 0
    ev_info = manifest["evidence_files"][0]
    ev_file = target_path / ev_info["relative_path"]
    assert ev_file.is_file()

    # Decrypt with destination key provider using the evidence record's canonical AAD
    ev_record = ps.evidence.list_all_evidence()[0]
    aad = json.loads(ev_record.aad_json)
    decrypted_bytes = decrypt_evidence_bytes(ev_file.read_bytes(), aad, key_provider=dest_kp)
    assert decrypted_bytes == b"SYNTHETIC_CLIP_PAYLOAD_FOR_BACKUP_TEST"


def test_backup_stale_temp_cleanup(backup_test_env):
    """Verify startup stale backup temp artifact cleanup removes only ExamGuard-owned temp files."""
    from src.persistence.backup import cleanup_stale_backup_temp_artifacts, BACKUP_TEMP_SUFFIX

    ps, backup_dir, temp_dir = backup_test_env

    # Create stale temp file matching naming policy
    stale_file = Path(backup_dir) / f"examguard-partial{BACKUP_TEMP_SUFFIX}"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_bytes(b"PARTIAL_INTERRUPTED_BACKUP_BYTES")

    # Create non-ExamGuard temp file that must NOT be touched
    unrelated_file = Path(backup_dir) / "notes.txt"
    unrelated_file.write_text("Do not delete me", encoding="utf-8")

    cleaned_count = cleanup_stale_backup_temp_artifacts(backup_base_dir=backup_dir, persistence_service=ps)
    assert cleaned_count >= 1
    assert not stale_file.exists()
    assert unrelated_file.exists()

    # Verify audit log was emitted
    logs = ps.audit.list_all_logs(limit=20)
    cleaned_logs = [l for l in logs if l.action == "BACKUP_STALE_TEMP_CLEANED"]
    assert len(cleaned_logs) > 0


def test_backup_unsupported_format_version_rejected(backup_test_env):
    """Verify backup verification strictly rejects unknown or unsupported format versions."""
    ps, backup_dir, temp_dir = backup_test_env
    passphrase = "FormatVersionPassphrase2026!"

    res = create_encrypted_backup(
        persistence_service=ps,
        recovery_passphrase=passphrase,
        backup_base_dir=backup_dir,
    )
    target_path = Path(res["backup_path"])
    if not target_path.is_absolute():
        target_path = Path(__file__).resolve().parent.parent / target_path

    manifest_file = target_path / "backup-manifest.json"
    import json
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Set unsupported format version
    manifest["format_version"] = "99.0-unsupported"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    is_valid, msg, details = verify_backup_directory(str(target_path), recovery_passphrase=passphrase)
    assert is_valid is False
    assert "không được hỗ trợ" in msg


def test_backup_passphrase_and_keys_never_logged(backup_test_env, caplog):
    """Verify recovery passphrase and raw keys are NEVER logged during success or error paths."""
    import logging
    ps, backup_dir, temp_dir = backup_test_env
    secret_passphrase = "DO-NOT-LOG-THIS-TEST-PASSPHRASE-999!"

    caplog.set_level(logging.DEBUG)

    # 1. Success path
    res = create_encrypted_backup(
        persistence_service=ps,
        recovery_passphrase=secret_passphrase,
        backup_base_dir=backup_dir,
    )
    target_path = Path(res["backup_path"])
    if not target_path.is_absolute():
        target_path = Path(__file__).resolve().parent.parent / target_path

    # 2. Error path (wrong passphrase)
    verify_backup_directory(str(target_path), recovery_passphrase="WRONG_SECRET_PASSPHRASE_888!")

    # Check all captured log text
    all_logs = caplog.text
    assert secret_passphrase not in all_logs
    assert "WRONG_SECRET_PASSPHRASE_888!" not in all_logs


def test_backup_manifest_privacy_no_student_identifiers(backup_test_env):
    """Verify unencrypted backup-manifest.json contains NO student identifiers or event IDs."""
    ps, backup_dir, temp_dir = backup_test_env
    passphrase = "PrivacyVerificationPassphrase2026!"

    res = create_encrypted_backup(
        persistence_service=ps,
        recovery_passphrase=passphrase,
        backup_base_dir=backup_dir,
    )
    target_path = Path(res["backup_path"])
    if not target_path.is_absolute():
        target_path = Path(__file__).resolve().parent.parent / target_path

    manifest_file = target_path / "backup-manifest.json"
    import json
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Event ID and student room must not be in evidence manifest list
    for ev_item in manifest.get("evidence_files", []):
        assert "event_id" not in ev_item, f"Leaked event_id in unencrypted manifest: {ev_item}"
        assert "student" not in str(ev_item).lower()

    # Manifest must not contain session name
    manifest_str = json.dumps(manifest)
    assert "Backup Test Session" not in manifest_str
    assert "Proctor Alpha" not in manifest_str
