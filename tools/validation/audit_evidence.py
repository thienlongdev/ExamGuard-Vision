"""
ExamGuard Vision — Evidence Forensic Audit & Verification Utility (Workstream 45 & 46)
Read-only inspection and forensic validation of stored snapshots, video clips, AES-GCM crypto tags,
and media containers. Reports numerical categories only without outputting sensitive personal data.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
from src.persistence.service import PersistenceService
from src.persistence.database import DEFAULT_DB_PATH


def audit_evidence(db_path: str = None, repair_mode: bool = False):
    svc = PersistenceService(db_path=db_path)
    db_file = Path(svc.db.db_path)
    storage_root = REPO_ROOT / "storage"

    print("\n" + "=" * 76)
    print("      EXAMGUARD VISION — EVIDENCE FORENSIC AUDIT (WORKSTREAM 45/46)")
    print("=" * 76)
    print(f" Database:     {db_file}")
    print(f" Storage Root: {storage_root}")
    print(f" Mode:         {'REPAIR' if repair_mode else 'READ-ONLY AUDIT'}")
    print("=" * 76)

    # 1. Fetch all evidence records from DB
    with svc.db.get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT 
                ee.evidence_id, e.session_id, ee.event_id, e.camera_id, ee.evidence_type,
                ee.relative_path, ee.sha256, ee.size_bytes,
                ee.encryption_state, ee.artifact_state, ee.codec, ee.container,
                ee.duration_sec, ee.frame_count, ee.created_at
            FROM event_evidence ee
            LEFT JOIN events e ON ee.event_id = e.event_id
            ORDER BY ee.created_at ASC
            """
        )
        rows = cursor.fetchall()

    total_records = len(rows)
    print(f"\n[1] Total Evidence Records in Database: {total_records}")

    stats = {
        "snapshots": {
            "total": 0,
            "valid": 0,
            "missing": 0,
            "crypto_invalid": 0,
            "media_invalid": 0,
            "legacy_unencrypted": 0,
        },
        "videos": {
            "total": 0,
            "valid": 0,
            "zero_duration": 0,
            "missing": 0,
            "crypto_invalid": 0,
            "media_invalid": 0,
            "legacy_unencrypted": 0,
        },
    }

    for row in rows:
        (
            ev_id, sess_id, event_id, cam_id, ev_type,
            rel_path, expected_hash, fsize,
            enc_state, art_state, codec, container,
            dur, frames, created_at
        ) = row

        is_snap = (ev_type.upper() == "SNAPSHOT")
        cat = "snapshots" if is_snap else "videos"
        stats[cat]["total"] += 1

        # Attempt to load and decrypt using production service
        try:
            decrypted_bytes, mime_type = svc.load_and_decrypt_evidence(ev_id)
        except FileNotFoundError:
            stats[cat]["missing"] += 1
            if repair_mode:
                with svc.db.get_connection() as conn:
                    conn.execute("UPDATE event_evidence SET artifact_state = 'FAILED', error_message = 'Evidence file missing from disk' WHERE evidence_id = ?", (ev_id,))
            continue
        except Exception as exc:
            stats[cat]["crypto_invalid"] += 1
            if repair_mode:
                with svc.db.get_connection() as conn:
                    conn.execute("UPDATE event_evidence SET artifact_state = 'FAILED', error_message = ? WHERE evidence_id = ?", (f"Decryption failure: {str(exc)[:100]}", ev_id))
            continue

        if enc_state != "ENCRYPTED_V1":
            stats[cat]["legacy_unencrypted"] += 1

        # Verify media validity
        if is_snap:
            # Snapshot decode test
            nparr = np.frombuffer(decrypted_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                stats["snapshots"]["valid"] += 1
                if repair_mode and art_state != "READY":
                    with svc.db.get_connection() as conn:
                        conn.execute(
                            "UPDATE event_evidence SET artifact_state = 'READY', error_message = NULL WHERE evidence_id = ?",
                            (ev_id,)
                        )
            else:
                stats["snapshots"]["media_invalid"] += 1
                if repair_mode:
                    with svc.db.get_connection() as conn:
                        conn.execute(
                            "UPDATE event_evidence SET artifact_state = 'FAILED', error_message = 'Image decoding failed' WHERE evidence_id = ?",
                            (ev_id,)
                        )
        else:
            # Video decode & duration test
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                tf_path = Path(tf.name)
                try:
                    tf.write(decrypted_bytes)
                    tf.flush()
                    tf.close()

                    cap = cv2.VideoCapture(str(tf_path))
                    is_opened = cap.isOpened()
                    cap_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_opened else 0
                    cap_fps = cap.get(cv2.CAP_PROP_FPS) if is_opened else 0.0
                    cap_dur = cap_frames / cap_fps if (cap_fps and cap_fps > 0) else 0.0
                    cap.release()

                    if is_opened and cap_frames > 0 and cap_dur > 0:
                        stats["videos"]["valid"] += 1
                        if repair_mode and (art_state != "READY" or (dur is None or dur <= 0)):
                            with svc.db.get_connection() as conn:
                                conn.execute(
                                    """
                                    UPDATE event_evidence 
                                    SET artifact_state = 'READY', duration_sec = ?, frame_count = ?, error_message = NULL
                                    WHERE evidence_id = ?
                                    """,
                                    (cap_dur, cap_frames, ev_id)
                                )
                    elif cap_dur <= 0 or cap_frames <= 0:
                        stats["videos"]["zero_duration"] += 1
                        if repair_mode:
                            with svc.db.get_connection() as conn:
                                conn.execute(
                                    """
                                    UPDATE event_evidence 
                                    SET artifact_state = 'FAILED', duration_sec = 0.0, frame_count = 0, error_message = 'Legacy zero-duration clip (missing moov atom)'
                                    WHERE evidence_id = ?
                                    """,
                                    (ev_id,)
                                )
                    else:
                        stats["videos"]["media_invalid"] += 1
                        if repair_mode:
                            with svc.db.get_connection() as conn:
                                conn.execute(
                                    """
                                    UPDATE event_evidence 
                                    SET artifact_state = 'FAILED', error_message = 'Corrupt media container'
                                    WHERE evidence_id = ?
                                    """,
                                    (ev_id,)
                                )
                finally:
                    if tf_path.exists():
                        tf_path.unlink()

    print("\n[2] Forensic Summary by Artifact Type:")
    print("  --- SNAPSHOT EVIDENCE ---")
    print(f"  • Total Registered:     {stats['snapshots']['total']}")
    print(f"  • Valid & Decryptable:  {stats['snapshots']['valid']}")
    print(f"  • File Missing:         {stats['snapshots']['missing']}")
    print(f"  • Crypto Invalid:       {stats['snapshots']['crypto_invalid']}")
    print(f"  • Media Decode Invalid: {stats['snapshots']['media_invalid']}")
    print(f"  • Unencrypted (Legacy): {stats['snapshots']['legacy_unencrypted']}")

    print("\n  --- VIDEO EVIDENCE ---")
    print(f"  • Total Registered:     {stats['videos']['total']}")
    print(f"  • Valid & Playable:     {stats['videos']['valid']}")
    print(f"  • Zero Duration (0:00): {stats['videos']['zero_duration']}")
    print(f"  • File Missing:         {stats['videos']['missing']}")
    print(f"  • Crypto Invalid:       {stats['videos']['crypto_invalid']}")
    print(f"  • Media Decode Invalid: {stats['videos']['media_invalid']}")
    print(f"  • Unencrypted (Legacy): {stats['videos']['legacy_unencrypted']}")

    print("\n" + "=" * 76)
    print("      EVIDENCE AUDIT VERIFICATION COMPLETE")
    print("=" * 76 + "\n")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ExamGuard Evidence Forensic Audit")
    parser.add_argument("--db-path", type=str, default=None, help="Database path")
    parser.add_argument("--repair", action="store_true", help="Repair recoverable metadata")
    args = parser.parse_args()

    audit_evidence(db_path=args.db_path, repair_mode=args.repair)
