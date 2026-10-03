"""
Pre-Audit Inventory Generator for V4C Final Full Verification
Inventories all artifacts in runs/v4c, models/trained, reports/v4c, configs, src/models, tests.
Computes size, modification time, and SHA-256 for physical traceability.
"""

import os
import sys
import json
import hashlib
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def scan_dir(base_dir: Path):
    items = []
    if not base_dir.exists():
        return items
    for p in sorted(base_dir.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
            # Skip huge temporary cache files if any or focus on relevant
            try:
                stat = p.stat()
                items.append({
                    "path": rel,
                    "size_bytes": stat.st_size,
                    "modified_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)),
                    "sha256": get_sha256(p),
                })
            except Exception as e:
                items.append({
                    "path": rel,
                    "error": str(e),
                })
    return items


def main():
    target_dirs = [
        PROJECT_ROOT / "models/trained",
        PROJECT_ROOT / "runs/v4c",
        PROJECT_ROOT / "reports/v4c",
        PROJECT_ROOT / "configs",
        PROJECT_ROOT / "src/models",
        PROJECT_ROOT / "tests",
    ]

    all_inventory = {}
    for d in target_dirs:
        key = str(d.relative_to(PROJECT_ROOT)).replace("\\", "/")
        print(f"Scanning {key}...")
        all_inventory[key] = scan_dir(d)

    out_json = PROJECT_ROOT / "reports/v4c/final_verification/PRE_AUDIT_INVENTORY.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_inventory, f, indent=2)
    print(f"Saved {out_json}")

    # Generate Markdown Summary
    out_md = PROJECT_ROOT / "reports/v4c/final_verification/PRE_AUDIT_INVENTORY.md"
    lines = [
        "# V4C Final Verification: Pre-Audit Artifact Inventory",
        "",
        "**Generated At**: " + time.strftime("%Y-%m-%d %H:%M:%S"),
        "**Status**: Complete Forensic Artifact Inventory",
        "",
        "---",
        "",
        "## 1. Executive Checkpoint & Baseline Inventory",
        "",
        "| Directory / Resource | Artifact Path | Size (Bytes) | SHA-256 Hash | Modified Time |",
        "| :--- | :--- | :---: | :--- | :---: |",
    ]

    # Priority checkpoints
    key_files = [
        "models/trained/stage1_best.pt",
        "models/trained/stage1_5_best.pt",
        "models/trained/v4_posture_best.pt",
        "models/trained/v4_headpose_yaw_best.pt",
    ]

    # Map all scanned items by path
    by_path = {}
    for d_key, items in all_inventory.items():
        for it in items:
            if "path" in it:
                by_path[it["path"]] = it

    for kf in key_files:
        if kf in by_path:
            it = by_path[kf]
            lines.append(f"| `models/trained` | `{it['path']}` | {it['size_bytes']:,} | `{it['sha256']}` | {it['modified_time']} |")
        else:
            lines.append(f"| `models/trained` | `{kf}` | `MISSING` | `N/A` | `N/A` |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Training Run Checkpoint & Metric Inventory (`runs/v4c/`)",
        "",
        "| Experiment Directory | Key Artifact | Size (Bytes) | SHA-256 Hash |",
        "| :--- | :--- | :---: | :--- |",
    ])

    for it in all_inventory.get("runs/v4c", []):
        p = it["path"]
        name = Path(p).name
        if name in ["best_model.pt", "last_checkpoint.pt", "metrics.json", "headpose_metrics.json", "training_history.csv"]:
            lines.append(f"| `{Path(p).parent}` | `{name}` | {it['size_bytes']:,} | `{it['sha256']}` |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Inventory Totals by Directory",
        "",
        "| Directory | Total Files | Total Size (MB) |",
        "| :--- | :---: | :---: |",
    ])

    for d_key, items in all_inventory.items():
        total_size_mb = sum(it.get("size_bytes", 0) for it in items) / (1024 * 1024)
        lines.append(f"| `{d_key}` | {len(items)} | {total_size_mb:.2f} MB |")

    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
