"""
Verify All Protected Checkpoint Hashes After Stage 2 Integrity Repair
====================================================================
Computes SHA-256 hashes for all 6 protected model checkpoints, validates exact equality
with expected frozen hashes, and saves runs/stage2_integrity/checkpoint_hashes_after.json.
"""

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_HASHES = {
    "models/trained/stage1_best.pt": "6d713808f0bc670e8bf06e01b006fac73d8d6e5db7130584cec1658b003ce98a",
    "models/trained/stage1_5_best.pt": "68690cf82715dc6dad2eb35a08711531170e935eb7dbe1c30fafabae341e2c2c",
    "models/trained/v4_posture_best.pt": "529a23f96ebec6051ad29279fba3cd5279e7aa8fe2e46292ab4bb9c9523ca180",
    "models/trained/v4_headpose_yaw_best.pt": "5d15eec5941cfc8d2468d09c27bff8eca2014e62bb12fc9a95c0c6239fbbca55",
    "models/fallback/posture_320/best_model.pt": "070a2e328a161b1c957576ec13647d65846a073f9dd35488c7c6edc847f0f4cf",
    "models/fallback/headpose_resnet18/best_model.pt": "bc31d46cfea007ebddfb6a8d5845641a32fd1cc018a831c4c8ae8b61e579a7d9",
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("Verifying protected checkpoint hashes after execution...")
    results = {}
    all_matched = True

    for rel_path, expected in EXPECTED_HASHES.items():
        full_path = REPO_ROOT / rel_path
        if not full_path.exists():
            print(f"ERROR: Checkpoint file missing: {rel_path}")
            sys.exit(1)
        actual = compute_sha256(full_path)
        matched = (actual == expected)
        results[rel_path] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "intact": matched,
        }
        if not matched:
            all_matched = False
            print(f"HARD STOP: Hash mismatch on {rel_path}!\n  Expected: {expected}\n  Actual:   {actual}")
        else:
            print(f"MATCH: {rel_path} ({actual[:16]}...)")

    out_file = REPO_ROOT / "runs" / "stage2_integrity" / "checkpoint_hashes_after.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "all_checkpoints_intact": all_matched,
            "checkpoints": results
        }, f, indent=2)

    if not all_matched:
        sys.exit(1)
    print("ALL 6 PROTECTED CHECKPOINTS INTACT. ZERO MUTATIONS.")


if __name__ == "__main__":
    main()
