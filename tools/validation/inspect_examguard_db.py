"""
ExamGuard Vision — Read-Only Database Inspection Utility
Inspects sessions, events, reviews, and evidence integrity without modifying data.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.persistence.service import PersistenceService, compute_file_sha256
from src.persistence.database import DEFAULT_DB_PATH


def main():
    parser = argparse.ArgumentParser(description="Read-only SQLite inspection tool for ExamGuard Vision")
    parser.add_argument("--db-path", type=str, default=None, help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--limit", type=int, default=10, help="Number of records to display (default: 10)")
    args = parser.parse_args()

    svc = PersistenceService(db_path=args.db_path)
    db_file = Path(svc.db.db_path)

    print("\n" + "=" * 76)
    print("         EXAMGUARD VISION — DATABASE INSPECTION UTILITY (READ-ONLY)")
    print("=" * 76)
    print(f" Database: {db_file}")
    print(f" Size:     {db_file.stat().st_size if db_file.exists() else 0} bytes")
    print(f" Health:   {'HEALTHY' if svc.db.check_health() else 'UNHEALTHY'}")
    print("=" * 76)

    # 1. Sessions
    sessions = svc.sessions.list_sessions(limit=args.limit)
    print(f"\n[1] Recent Sessions (Count: {len(sessions)}):")
    for s in sessions:
        summary = svc.sessions.get_session_summary(s.session_id)
        print(f"  • [{s.status}] ID: {s.session_id} | Tên: {s.name} | Phòng: {s.room}")
        print(f"    Bắt đầu: {s.started_at} | Kết thúc: {s.ended_at or 'Đang diễn ra'}")
        print(f"    Sự kiện: {summary['total_events']} (Cao: {summary['high_risk_count']}, Chờ: {summary['awaiting_count']}, Đã xác nhận: {summary['confirmed_count']}, Đã bỏ qua: {summary['dismissed_count']})")

    # 2. Events
    recent_events = svc.events.list_recent_events(limit=args.limit)
    print(f"\n[2] Recent Events (Count: {len(recent_events)}):")
    for ev in recent_events:
        print(f"  • [{ev.severity}] ID: {ev.event_id} | Thí sinh #{ev.track_id} | Loại: {ev.event_type} | Duyệt: {ev.review_status}")

    # 3. Evidence Integrity
    all_evidence = svc.evidence.list_all_evidence(limit=args.limit)
    print(f"\n[3] Evidence Integrity Check (Inspected: {len(all_evidence)}):")
    valid_count = 0
    missing_count = 0
    mismatch_count = 0

    for item in all_evidence:
        res = svc.verify_evidence_integrity(item.evidence_id)
        st = res["status"]
        if st == "VALID":
            valid_count += 1
        elif st == "FILE_MISSING":
            missing_count += 1
            print(f"  ! [THIẾU] {item.relative_path}")
        elif st == "HASH_MISMATCH":
            mismatch_count += 1
            print(f"  ! [MÃ BĂM KHÔNG KHỚP] {item.relative_path}")

    print(f"  --> Hợp lệ: {valid_count} | Thiếu file: {missing_count} | Sai mã băm: {mismatch_count}")

    # 4. Audit Chain Verification
    chain_valid, chain_err = svc.audit.verify_audit_chain()
    print(f"\n[4] Tamper-Evident Audit Chain: {'✓ HỢP LỆ (VERIFIED)' if chain_valid else f'✗ LỖI: {chain_err}'}")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
