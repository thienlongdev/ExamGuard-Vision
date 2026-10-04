"""
Local backup and verification service for ExamGuard SQLite database and evidence storage.
"""

from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import time
from typing import Dict, Any, Tuple, Optional, List
import uuid

from src.persistence.service import PersistenceService, compute_file_sha256

logger = logging.getLogger(__name__)


def create_local_backup(
    persistence_service: PersistenceService,
    backup_base_dir: str = "backups",
) -> Dict[str, Any]:
    """
    Perform a safe, transactional local backup of the SQLite database and evidence files.
    Calculates SHA-256 for all artifacts, writes backup-manifest.json, and verifies consistency.
    """
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

    # 1. Safe transactional SQLite backup via sqlite3.backup API
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

    # 2. Copy evidence files
    evidence_records = persistence_service.evidence.list_all_evidence(limit=5000)
    evidence_manifest_list = []

    for ev in evidence_records:
        src_path = Path(ev.relative_path) if os.path.isabs(ev.relative_path) else (repo_root / ev.relative_path)
        if src_path.is_file():
            # Keep relative path under evidence backup
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

    # 3. Create backup manifest
    manifest_data = {
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

    # 4. Immediate self-verification
    is_valid, msg, verif_details = verify_backup_directory(str(target_dir))
    manifest_data["verification_status"] = "PASSED" if is_valid else "FAILED"
    manifest_data["verification_details"] = verif_details

    # Re-save manifest with verification result
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    # 5. Record audit entry
    action_type = "BACKUP_CREATED" if is_valid else "BACKUP_VERIFICATION_FAILED"
    try:
        display_path = str(target_dir.relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        display_path = str(target_dir).replace("\\", "/")

    persistence_service.audit.log_action(
        audit_id=f"aud_{uuid.uuid4().hex[:12]}",
        action=action_type,
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
        "message": "Sao lưu thành công và đã xác minh toàn vẹn." if is_valid else f"Sao lưu không hoàn tất: {msg}",
    }


def verify_backup_directory(backup_dir_path: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Verify the integrity of a backup directory:
    - Checks backup-manifest.json exists and is valid JSON
    - Checks database file exists, verifies SHA-256, and ensures SQLite opens cleanly
    - Verifies all evidence file existences and their SHA-256 hashes
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

    db_meta = manifest.get("database", {})
    db_file = bdir / db_meta.get("filename", "examguard.sqlite3")
    if not db_file.is_file():
        return False, f"Tệp CSDL '{db_file.name}' không tồn tại trong thư mục sao lưu.", {}

    db_calc_hash = compute_file_sha256(str(db_file))
    if db_calc_hash.lower() != db_meta.get("sha256", "").lower():
        return False, f"Mã băm CSDL không khớp (kỳ vọng: {db_meta.get('sha256')}, thực tế: {db_calc_hash}).", {}

    # Verify database can be queried
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
    repo_root = Path(__file__).resolve().parent.parent.parent
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
