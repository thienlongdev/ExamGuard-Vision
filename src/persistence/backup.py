"""
Local backup and verification service for ExamGuard SQLite database and evidence storage.
Supports both portable encrypted backups (with user recovery passphrase) and standard local snapshots.
"""

from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
import tempfile
import time
from typing import Dict, Any, Tuple, Optional, List
import uuid

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

from src.persistence.service import PersistenceService, compute_file_sha256

logger = logging.getLogger(__name__)

BACKUP_FORMAT_V2_ENCRYPTED = "2.0-encrypted"
BACKUP_FORMAT_V1 = "1.0-plaintext"


def derive_backup_wrapping_key(passphrase: str, salt: bytes) -> bytes:
    """Derive 32-byte key from passphrase using scrypt."""
    return hashlib.scrypt(
        passphrase.encode("utf-8"),
        salt=salt,
        n=16384,
        r=8,
        p=1,
        maxmem=64 * 1024 * 1024,
        dklen=32,
    )


def create_encrypted_backup(
    persistence_service: PersistenceService,
    recovery_passphrase: str,
    backup_base_dir: str = "backups",
    actor_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform a portable encrypted backup of the SQLite database and evidence files.
    - User recovery passphrase derives a wrapping key via scrypt.
    - Database is encrypted at rest using AES-256-GCM.
    - Evidence files are preserved in their encrypted state.
    - Manifest records cryptographic parameters (no plaintext passwords).
    """
    if not recovery_passphrase or len(recovery_passphrase.strip()) < 8:
        raise ValueError("Mật khẩu khôi phục sao lưu phải có ít nhất 8 ký tự.")

    repo_root = Path(__file__).resolve().parent.parent.parent
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    backup_id = f"backup_{timestamp_str}_{uuid.uuid4().hex[:6]}"

    base = Path(backup_base_dir)
    target_base = base if base.is_absolute() else (repo_root / base)
    target_dir = target_base / date_str / f"examguard-backup-{timestamp_str}"
    target_dir.mkdir(parents=True, exist_ok=True)

    evidence_backup_dir = target_dir / "evidence"
    evidence_backup_dir.mkdir(parents=True, exist_ok=True)

    # 1. Consistent SQLite standalone backup via VACUUM INTO
    temp_db_fd, temp_db_path = tempfile.mkstemp(suffix=".sqlite3")
    os.close(temp_db_fd)
    try:
        src_conn = persistence_service.db.get_connection()
        norm_path = temp_db_path.replace("\\", "/")
        src_conn.execute(f"VACUUM INTO '{norm_path}'")
        raw_db_bytes = Path(temp_db_path).read_bytes()
        raw_db_hash = hashlib.sha256(raw_db_bytes).hexdigest()
    finally:
        if os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
            except Exception:
                pass

    # 2. Encrypt database with random payload key
    salt = secrets.token_bytes(16)
    wrapping_key = derive_backup_wrapping_key(recovery_passphrase, salt)

    payload_key = secrets.token_bytes(32)
    wrap_nonce = secrets.token_bytes(12)
    aes_wrap = AESGCM(wrapping_key)
    wrapped_payload_key = aes_wrap.encrypt(wrap_nonce, payload_key, b"EXAMGUARD_BACKUP_KEY_WRAP")

    db_nonce = secrets.token_bytes(12)
    aes_db = AESGCM(payload_key)
    encrypted_db_payload = aes_db.encrypt(db_nonce, raw_db_bytes, b"EXAMGUARD_SQLITE_BACKUP")

    # Write encrypted database: [NONCE 12B] [CIPHERTEXT+TAG]
    enc_db_file = target_dir / "examguard.sqlite3.enc"
    enc_db_file.write_bytes(db_nonce + encrypted_db_payload)
    enc_db_hash = compute_file_sha256(str(enc_db_file))
    enc_db_size = enc_db_file.stat().st_size

    # 3. Copy evidence files
    evidence_records = persistence_service.evidence.list_all_evidence(limit=5000)
    evidence_manifest_list = []

    for ev in evidence_records:
        src_path = Path(ev.relative_path) if os.path.isabs(ev.relative_path) else (repo_root / ev.relative_path)
        if src_path.is_file():
            if os.path.isabs(ev.relative_path):
                rel_sub = Path(ev.relative_path).relative_to(Path(ev.relative_path).anchor)
            else:
                rel_sub = Path(ev.relative_path)

            dest_file = target_dir / "evidence" / rel_sub
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            if dest_file.resolve() != src_path.resolve():
                shutil.copy2(src_path, dest_file)

            file_hash = compute_file_sha256(str(dest_file))
            file_size = dest_file.stat().st_size
            evidence_manifest_list.append({
                "evidence_id": ev.evidence_id,
                "event_id": ev.event_id,
                "relative_path": str(Path("evidence") / rel_sub).replace("\\", "/"),
                "sha256": file_hash,
                "size_bytes": file_size,
                "encryption_state": getattr(ev, "encryption_state", "LEGACY_PLAINTEXT"),
            })

    # 4. Construct manifest
    manifest_data = {
        "format_version": BACKUP_FORMAT_V2_ENCRYPTED,
        "backup_id": backup_id,
        "created_at": now.isoformat(),
        "app_version": "2.0.0-orchestration",
        "schema_version": 2,
        "encryption": {
            "algorithm": "AES-256-GCM",
            "kdf": "scrypt",
            "kdf_params": {"n": 16384, "r": 8, "p": 1, "salt_hex": salt.hex()},
            "wrapped_key_nonce_hex": wrap_nonce.hex(),
            "wrapped_key_hex": wrapped_payload_key.hex(),
        },
        "database": {
            "filename": "examguard.sqlite3.enc",
            "ciphertext_sha256": enc_db_hash,
            "plaintext_sha256": raw_db_hash,
            "size_bytes": enc_db_size,
        },
        "evidence_files": evidence_manifest_list,
        "evidence_count": len(evidence_manifest_list),
        "verification_status": "PENDING",
    }

    manifest_path = target_dir / "backup-manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    # 5. Immediate verification
    is_valid, msg, verif_details = verify_backup_directory(str(target_dir), recovery_passphrase=recovery_passphrase)
    manifest_data["verification_status"] = "PASSED" if is_valid else "FAILED"
    manifest_data["verification_details"] = verif_details

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    # 6. Audit logging
    action_type = "BACKUP_CREATED" if is_valid else "BACKUP_VERIFICATION_FAILED"
    try:
        display_path = str(target_dir.relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        display_path = str(target_dir).replace("\\", "/")

    persistence_service.audit.log_action(
        audit_id=f"aud_{uuid.uuid4().hex[:12]}",
        action=action_type,
        actor_type="USER" if actor_id else "SYSTEM",
        actor_id=actor_id,
        details={
            "backup_id": backup_id,
            "backup_dir": display_path,
            "format": BACKUP_FORMAT_V2_ENCRYPTED,
            "evidence_count": len(evidence_manifest_list),
            "verified": is_valid,
        },
    )

    return {
        "success": is_valid,
        "backup_id": backup_id,
        "backup_path": display_path,
        "database_sha256": enc_db_hash,
        "evidence_count": len(evidence_manifest_list),
        "encrypted": True,
        "message": "Sao lưu mã hóa thành công và đã xác minh toàn vẹn." if is_valid else f"Sao lưu không hoàn tất: {msg}",
    }


def create_local_backup(
    persistence_service: PersistenceService,
    backup_base_dir: str = "backups",
    recovery_passphrase: Optional[str] = None,
    actor_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Standard or encrypted local backup creator.
    If recovery_passphrase is provided, performs encrypted backup.
    Otherwise creates legacy local backup for backward compatibility with automated tests.
    """
    if recovery_passphrase:
        return create_encrypted_backup(
            persistence_service,
            recovery_passphrase=recovery_passphrase,
            backup_base_dir=backup_base_dir,
            actor_id=actor_id,
        )

    # Legacy unencrypted backup for test fixtures
    repo_root = Path(__file__).resolve().parent.parent.parent
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    backup_id = f"backup_{timestamp_str}_{uuid.uuid4().hex[:6]}"

    base = Path(backup_base_dir)
    target_base = base if base.is_absolute() else (repo_root / base)
    target_dir = target_base / date_str / f"examguard-backup-{timestamp_str}"
    target_dir.mkdir(parents=True, exist_ok=True)

    evidence_backup_dir = target_dir / "evidence"
    evidence_backup_dir.mkdir(parents=True, exist_ok=True)

    backup_db_path = target_dir / "examguard.sqlite3"
    src_conn = persistence_service.db.get_connection()
    dest_conn = sqlite3.connect(str(backup_db_path))
    try:
        with dest_conn:
            src_conn.backup(dest_conn, pages=250, sleep=0.01)
    finally:
        dest_conn.close()

    db_sha256 = compute_file_sha256(str(backup_db_path))
    db_size = backup_db_path.stat().st_size

    evidence_records = persistence_service.evidence.list_all_evidence(limit=5000)
    evidence_manifest_list = []

    for ev in evidence_records:
        src_path = Path(ev.relative_path) if os.path.isabs(ev.relative_path) else (repo_root / ev.relative_path)
        if src_path.is_file():
            if os.path.isabs(ev.relative_path):
                rel_sub = Path(ev.relative_path).relative_to(Path(ev.relative_path).anchor)
            else:
                rel_sub = Path(ev.relative_path)

            dest_file = target_dir / "evidence" / rel_sub
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            if dest_file.resolve() != src_path.resolve():
                shutil.copy2(src_path, dest_file)

            file_hash = compute_file_sha256(str(dest_file))
            file_size = dest_file.stat().st_size
            evidence_manifest_list.append({
                "evidence_id": ev.evidence_id,
                "event_id": ev.event_id,
                "relative_path": str(Path("evidence") / rel_sub).replace("\\", "/"),
                "sha256": file_hash,
                "size_bytes": file_size,
            })

    manifest_data = {
        "format_version": BACKUP_FORMAT_V1,
        "backup_id": backup_id,
        "created_at": now.isoformat(),
        "database": {
            "filename": "examguard.sqlite3",
            "sha256": db_sha256,
            "size_bytes": db_size,
        },
        "evidence_files": evidence_manifest_list,
        "evidence_count": len(evidence_manifest_list),
        "app_version": "2.0.0-orchestration",
        "verification_status": "PENDING",
    }

    manifest_path = target_dir / "backup-manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    is_valid, msg, verif_details = verify_backup_directory(str(target_dir))
    manifest_data["verification_status"] = "PASSED" if is_valid else "FAILED"
    manifest_data["verification_details"] = verif_details

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    action_type = "BACKUP_CREATED" if is_valid else "BACKUP_VERIFICATION_FAILED"
    try:
        display_path = str(target_dir.relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        display_path = str(target_dir).replace("\\", "/")

    persistence_service.audit.log_action(
        audit_id=f"aud_{uuid.uuid4().hex[:12]}",
        action=action_type,
        actor_type="USER" if actor_id else "SYSTEM",
        actor_id=actor_id,
        details={
            "backup_id": backup_id,
            "backup_dir": display_path,
            "evidence_count": len(evidence_manifest_list),
            "db_sha256": db_sha256,
            "verified": is_valid,
        },
    )

    return {
        "success": is_valid,
        "backup_id": backup_id,
        "backup_path": display_path,
        "database_sha256": db_sha256,
        "evidence_count": len(evidence_manifest_list),
        "encrypted": False,
        "message": "Sao lưu thành công và đã xác minh toàn vẹn." if is_valid else f"Sao lưu không hoàn tất: {msg}",
    }


def verify_backup_directory(
    backup_dir_path: str,
    recovery_passphrase: Optional[str] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Verify integrity of a backup package:
    - If encrypted: unwraps key with recovery_passphrase, decrypts DB to temp, verifies integrity.
    - If unencrypted: verifies DB directly.
    - Verifies all evidence file hashes.
    - Cleans up any decrypted temp files immediately.
    """
    bdir = Path(backup_dir_path).resolve()
    manifest_file = bdir / "backup-manifest.json"
    if not manifest_file.is_file():
        return False, "Thiếu tệp backup-manifest.json.", {}

    try:
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        return False, f"Không thể đọc backup-manifest.json: {e}", {}

    format_ver = manifest.get("format_version", BACKUP_FORMAT_V1)

    # Encrypted backup verification
    if format_ver == BACKUP_FORMAT_V2_ENCRYPTED:
        if not recovery_passphrase:
            return False, "Cần cung cấp mật khẩu khôi phục để xác minh bản sao lưu mã hóa.", {}

        enc_meta = manifest.get("encryption", {})
        kdf_params = enc_meta.get("kdf_params", {})
        salt_hex = kdf_params.get("salt_hex", "")
        if not salt_hex:
            return False, "Thiếu tham số salt trong manifest sao lưu.", {}
        salt = bytes.fromhex(salt_hex)

        wrapping_key = derive_backup_wrapping_key(recovery_passphrase, salt)
        wrap_nonce = bytes.fromhex(enc_meta.get("wrapped_key_nonce_hex", ""))
        wrapped_key = bytes.fromhex(enc_meta.get("wrapped_key_hex", ""))

        aes_wrap = AESGCM(wrapping_key)
        try:
            payload_key = aes_wrap.decrypt(wrap_nonce, wrapped_key, b"EXAMGUARD_BACKUP_KEY_WRAP")
        except InvalidTag:
            return False, "Mật khẩu khôi phục không đúng (không thể giải mã khóa sao lưu).", {}
        except Exception as e:
            return False, f"Lỗi giải mã khóa sao lưu: {e}", {}

        # Decrypt database to temporary location and test
        db_meta = manifest.get("database", {})
        enc_db_file = bdir / db_meta.get("filename", "examguard.sqlite3.enc")
        if not enc_db_file.is_file():
            return False, f"Thiếu tệp CSDL mã hóa '{enc_db_file.name}'.", {}

        # Verify ciphertext SHA-256
        ct_hash = compute_file_sha256(str(enc_db_file))
        if ct_hash.lower() != db_meta.get("ciphertext_sha256", "").lower():
            return False, "Mã băm tệp CSDL mã hóa không khớp.", {}

        enc_db_bytes = enc_db_file.read_bytes()
        db_nonce = enc_db_bytes[:12]
        db_ciphertext = enc_db_bytes[12:]

        aes_db = AESGCM(payload_key)
        try:
            raw_db_bytes = aes_db.decrypt(db_nonce, db_ciphertext, b"EXAMGUARD_SQLITE_BACKUP")
        except InvalidTag:
            return False, "Tệp CSDL mã hóa đã bị can thiệp hoặc sai khóa giải mã.", {}
        except Exception as e:
            return False, f"Lỗi giải mã CSDL: {e}", {}

        # Verify plaintext hash
        calc_pt_hash = hashlib.sha256(raw_db_bytes).hexdigest()
        if calc_pt_hash.lower() != db_meta.get("plaintext_sha256", "").lower():
            return False, "Mã băm CSDL sau giải mã không khớp với bản gốc.", {}

        # In-memory SQLite query test to verify decrypted database schema without leaking files to disk
        try:
            test_conn = sqlite3.connect(":memory:")
            try:
                test_conn.deserialize(raw_db_bytes)
                test_cur = test_conn.cursor()
                test_cur.execute("SELECT COUNT(*) FROM exam_sessions;")
                test_cur.close()
            finally:
                test_conn.close()
        except Exception as e:
            return False, f"Tệp CSDL sau giải mã bị lỗi cấu trúc: {e}", {}

    else:
        # Legacy unencrypted verification
        db_meta = manifest.get("database", {})
        db_file = bdir / db_meta.get("filename", "examguard.sqlite3")
        if not db_file.is_file():
            return False, f"Tệp CSDL '{db_file.name}' không tồn tại trong thư mục sao lưu.", {}

        db_calc_hash = compute_file_sha256(str(db_file))
        if db_calc_hash.lower() != db_meta.get("sha256", "").lower():
            return False, f"Mã băm CSDL không khớp (kỳ vọng: {db_meta.get('sha256')}, thực tế: {db_calc_hash}).", {}

        try:
            test_conn = sqlite3.connect(str(db_file))
            test_cur = test_conn.cursor()
            test_cur.execute("SELECT COUNT(*) FROM exam_sessions;")
            test_conn.close()
        except Exception as e:
            return False, f"Tệp CSDL sao lưu bị lỗi cấu trúc: {e}", {}

    # Verify evidence files
    evidence_list = manifest.get("evidence_files", [])
    verified_files = 0
    for item in evidence_list:
        ev_file = bdir / item.get("relative_path", "")
        if not ev_file.is_file():
            return False, f"Thiếu tệp bằng chứng '{item.get('relative_path')}' trong bản sao lưu.", {}
        ev_calc_hash = compute_file_sha256(str(ev_file))
        if ev_calc_hash.lower() != item.get("sha256", "").lower():
            return False, f"Mã băm tệp bằng chứng '{ev_file.name}' không khớp.", {}
        verified_files += 1

    details = {
        "verified_evidence_files": verified_files,
        "database_verified": True,
        "encrypted": (format_ver == BACKUP_FORMAT_V2_ENCRYPTED),
        "verified_at": datetime.now().isoformat(),
    }
    return True, "Xác minh bản sao lưu thành công.", details


def preview_retention_candidates(
    persistence_service: PersistenceService,
    dismissed_days: int = 30,
    confirmed_days: int = 90,
) -> Dict[str, Any]:
    """
    Preview what records and evidence files would be deleted under retention policy.
    Non-destructive preview: does NOT delete anything.
    """
    now = datetime.now()
    dismissed_candidates = []
    confirmed_candidates = []

    all_events = persistence_service.events.list_recent_events(limit=2000)
    for ev in all_events:
        try:
            ev_time = datetime.fromisoformat(ev.opened_at)
            age_days = (now - ev_time).days
            if ev.review_status == "dismissed" and age_days >= dismissed_days:
                dismissed_candidates.append({
                    "event_id": ev.event_id,
                    "session_id": ev.session_id,
                    "age_days": age_days,
                    "status": ev.review_status,
                })
            elif ev.review_status == "confirmed" and age_days >= confirmed_days:
                confirmed_candidates.append({
                    "event_id": ev.event_id,
                    "session_id": ev.session_id,
                    "age_days": age_days,
                    "status": ev.review_status,
                })
        except Exception:
            continue

    return {
        "policy": {
            "retention_enabled_default": False,
            "dismissed_threshold_days": dismissed_days,
            "confirmed_threshold_days": confirmed_days,
        },
        "preview_summary": {
            "dismissed_candidates_count": len(dismissed_candidates),
            "confirmed_candidates_count": len(confirmed_candidates),
            "total_candidates": len(dismissed_candidates) + len(confirmed_candidates),
        },
        "candidates": dismissed_candidates + confirmed_candidates,
    }
