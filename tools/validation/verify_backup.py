"""
ExamGuard Vision — Read-Only Backup Verification Utility
Verifies backup-manifest.json, database hash, and evidence files integrity.
Exits with code 0 on success, or code 1 on mismatch/corruption.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.persistence.backup import verify_backup_directory


def main():
    parser = argparse.ArgumentParser(description="Read-only backup verification utility for ExamGuard Vision")
    parser.add_argument("backup_dir", type=str, help="Path to backup directory to verify")
    args = parser.parse_args()

    backup_path = Path(args.backup_dir).resolve()
    print("\n" + "=" * 76)
    print("         EXAMGUARD VISION — BACKUP INTEGRITY VERIFICATION")
    print("=" * 76)
    print(f" Checking backup directory: {backup_path}")

    if not backup_path.is_dir():
        print(f" [LỖI] Thư mục không tồn tại: {backup_path}")
        sys.exit(1)

    is_valid, message, details = verify_backup_directory(str(backup_path))
    if is_valid:
        print(f"\n [THÀNH CÔNG] {message}")
        print(f"   • Database:      ✓ Hợp lệ (SHA-256 khớp và SQLite mở thành công)")
        print(f"   • Evidence:      ✓ {details.get('verified_evidence_files', 0)} tệp bằng chứng hợp lệ")
        print(f"   • Thời điểm kt:  {details.get('verified_at')}")
        print("=" * 76 + "\n")
        sys.exit(0)
    else:
        print(f"\n [THẤT BẠI] {message}")
        print("=" * 76 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
