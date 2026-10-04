"""
Migrate Legacy Plaintext Evidence to AES-256-GCM Encrypted Storage.
===================================================================
Scans database for event_evidence records with encryption_state = 'LEGACY_PLAINTEXT'.
Safely encrypts each file into a temporary .enc artifact, verifies decryption roundtrip,
atomically swaps the file, updates database metadata, and unlinks plaintext only after success.

Usage:
    python tools/security/migrate_evidence_encryption.py            # Defaults to --dry-run
    python tools/security/migrate_evidence_encryption.py --apply    # Executes migration
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.persistence.database import DatabaseManager
from src.persistence.service import PersistenceService, compute_file_sha256
from src.security.key_provider import get_key_provider
from src.security.crypto import encrypt_evidence_bytes, decrypt_evidence_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migrate_evidence")


def migrate_evidence(apply: bool = False, db_path: str = None) -> dict:
    ps = PersistenceService.get_instance(db_path=db_path)
    records = ps.evidence.list_all_evidence(limit=10000)

    legacy_candidates = [
        r for r in records
        if getattr(r, "encryption_state", "LEGACY_PLAINTEXT") == "LEGACY_PLAINTEXT"
    ]

    logger.info(f"Discovered {len(legacy_candidates)} legacy plaintext evidence records.")
    results = {
        "total_legacy": len(legacy_candidates),
        "migrated": 0,
        "failed": 0,
        "skipped_missing": 0,
        "applied": apply,
    }

    if not legacy_candidates:
        logger.info("Zero legacy evidence records require migration.")
        return results

    if not apply:
        logger.info("DRY-RUN mode active. No files or database records will be modified.")
        for r in legacy_candidates:
            full_path = (REPO_ROOT / r.relative_path).resolve()
            status = "FOUND" if full_path.is_file() else "FILE_MISSING"
            logger.info(f"  [DRY-RUN] {r.evidence_id} ({r.evidence_type}): {r.relative_path} [{status}]")
        return results

    # Apply migration
    kp = get_key_provider()
    key_id, master_key = kp.get_current_key()
    logger.info(f"Encrypting with Master Key: {key_id}")

    for r in legacy_candidates:
        full_path = (REPO_ROOT / r.relative_path).resolve()
        if not full_path.is_file():
            logger.warning(f"Skipping {r.evidence_id}: physical file missing at {full_path}")
            results["skipped_missing"] += 1
            continue

        temp_enc_path = full_path.with_name(full_path.name + ".migrating.enc")
        final_enc_path = full_path.with_name(full_path.name + ".enc")

        try:
            # 1. Read plaintext
            raw_bytes = full_path.read_bytes()

            # 2. Canonical AAD
            aad_dict = {
                "session_id": "",
                "event_id": r.event_id,
                "evidence_id": r.evidence_id,
                "evidence_type": r.evidence_type.upper(),
            }

            # 3. Encrypt into temporary file
            envelope, ct_sha256 = encrypt_evidence_bytes(raw_bytes, aad_dict, master_key, key_id)
            temp_enc_path.write_bytes(envelope)

            # 4. Immediate self-verification: test decryption
            decrypted = decrypt_evidence_bytes(envelope, aad_dict, key_provider=kp)
            if decrypted != raw_bytes:
                raise ValueError("Verification failed: decrypted bytes did not match original plaintext.")

            # 5. Atomically move temporary enc to final enc
            temp_enc_path.replace(final_enc_path)

            # 6. Update database record
            try:
                rel_enc_path = str(final_enc_path.relative_to(REPO_ROOT)).replace("\\", "/")
            except ValueError:
                rel_enc_path = str(final_enc_path).replace("\\", "/")

            updated = ps.evidence.update_encryption_metadata(
                evidence_id=r.evidence_id,
                relative_path=rel_enc_path,
                sha256=ct_sha256,
                size_bytes=len(envelope),
                encryption_state="ENCRYPTED_V1",
                key_id=key_id,
                aad_json=json.dumps(aad_dict),
            )
            if not updated:
                raise RuntimeError("Failed to update evidence metadata in database.")

            # 7. Unlink plaintext original only after successful DB update
            full_path.unlink()
            results["migrated"] += 1
            logger.info(f"MIGRATED: {r.evidence_id} -> {rel_enc_path} (SHA: {ct_sha256[:16]}...)")

        except Exception as e:
            logger.error(f"Failed to migrate {r.evidence_id}: {e}. Original plaintext preserved.")
            if temp_enc_path.exists():
                temp_enc_path.unlink()
            results["failed"] += 1

    logger.info(
        f"Migration complete: {results['migrated']} migrated, {results['failed']} failed, {results['skipped_missing']} missing."
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="ExamGuard Evidence Encryption Migration Utility")
    parser.add_argument("--apply", action="store_true", help="Apply encryption migration (default: dry-run)")
    parser.add_argument("--db-path", type=str, default=None, help="Custom database path")
    args = parser.parse_args()

    migrate_evidence(apply=args.apply, db_path=args.db_path)


if __name__ == "__main__":
    main()
