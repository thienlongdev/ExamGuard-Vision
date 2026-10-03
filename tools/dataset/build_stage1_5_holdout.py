"""Build and freeze the Stage 1.5 comparison holdout from original Stage 1 train groups."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys
import yaml
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_V3 = ROOT_DIR / "datasets" / "processed_v3"
HOLDOUT_DIR = ROOT_DIR / "datasets" / "stage1_5_holdout"
REPORTS_DIR = ROOT_DIR / "reports" / "stage1_5"
HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Selected groups from original Stage 1 TRAIN (15 groups = 9.93% of 151 train groups)
HOLDOUT_GROUPS = {
    # 10 CCTV Oblique High-Angle groups with strong weak-class representation
    'grp_137', 'grp_9', 'grp_135', 'grp_131', 'grp_0014', 'grp_0011', 'grp_22', 'grp_0002', 'grp_2', 'grp_0003',
    # 5 Frontal Classroom groups with normal writing/reading postures
    'grp_0600', 'grp_0601', 'grp_0800', 'grp_1000', 'grp_1121'
}

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("BUILDING STAGE 1.5 COMPARISON HOLDOUT")
    print("=" * 60)

    with open(PROCESSED_V3 / "manifest_v3.json", "r", encoding="utf-8") as f:
        v3_manifest = json.load(f)

    train_entries = v3_manifest["splits"]["train"]
    classes = v3_manifest["classes"]

    holdout_entries = [e for e in train_entries if e["group_id"] in HOLDOUT_GROUPS]
    print(f"Total holdout entries extracted: {len(holdout_entries)} from {len(HOLDOUT_GROUPS)} groups")

    # Setup directories
    img_out = HOLDOUT_DIR / "images"
    lbl_out = HOLDOUT_DIR / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    holdout_manifest_entries = []
    class_box_counts = Counter()
    group_box_counts = defaultdict(lambda: Counter())
    group_img_counts = Counter()
    group_viewpoints = {}
    group_provenances = defaultdict(set)

    for entry in holdout_entries:
        gid = entry["group_id"]
        group_img_counts[gid] += 1
        src_img = ROOT_DIR / entry["image_path"]
        src_lbl = ROOT_DIR / entry["label_path"]
        dst_img = img_out / src_img.name
        dst_lbl = lbl_out / src_lbl.name

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)

        # Classify viewpoint
        subsets = set()
        for ann in entry.get("annotations", []):
            cname = ann["canonical_class_name"]
            class_box_counts[cname] += 1
            group_box_counts[gid][cname] += 1
            for s in ann.get("sources", []):
                subsets.add(s.get("source_subset", ""))
                group_provenances[gid].add(f"{s.get('source_dataset')}/{s.get('source_subset')}")

        is_oblique = any(k in s for s in subsets for k in ['Stand', 'Bow', 'Turn', 'Discuss'])
        vp = "cctv_oblique_high_angle" if is_oblique else "frontal_classroom"
        group_viewpoints[gid] = vp

        holdout_manifest_entries.append({
            "image_name": src_img.name,
            "image_path": str(dst_img.relative_to(ROOT_DIR)).replace("\\", "/"),
            "label_path": str(dst_lbl.relative_to(ROOT_DIR)).replace("\\", "/"),
            "image_hash": entry["image_hash"],
            "group_id": gid,
            "viewpoint": vp,
            "annotations": entry["annotations"]
        })

    # Save manifest.json
    holdout_manifest = {
        "classes": classes,
        "total_images": len(holdout_manifest_entries),
        "total_groups": len(HOLDOUT_GROUPS),
        "groups": sorted(list(HOLDOUT_GROUPS)),
        "class_box_counts": dict(class_box_counts),
        "entries": holdout_manifest_entries
    }
    with open(HOLDOUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(holdout_manifest, f, indent=2)

    # Save dataset.yaml for evaluation
    yaml_content = {
        "path": str(HOLDOUT_DIR.resolve()).replace("\\", "/"),
        "train": "images",
        "val": "images",
        "nc": len(classes),
        "names": classes
    }
    with open(HOLDOUT_DIR / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"Saved dataset.yaml and manifest.json to {HOLDOUT_DIR}")

    # Generate Markdown Report: reports/stage1_5/STAGE1_5_HOLDOUT_MANIFEST.md
    md_lines = [
        "# Stage 1.5 Comparison Holdout Manifest",
        "",
        "**Date**: 2026-10-02  ",
        f"**Dataset Location**: `{HOLDOUT_DIR.relative_to(ROOT_DIR)}/`  ",
        f"**Source Origin**: Strictly derived from Stage 1 `train` split only (zero leakage with original test/val)  ",
        f"**Total Images**: {len(holdout_manifest_entries):,}  ",
        f"**Total Groups**: {len(HOLDOUT_GROUPS)} groups (9.93% of 151 Stage 1 train groups)  ",
        f"**Total Bounding Boxes**: {sum(class_box_counts.values()):,}  ",
        "",
        "## 1. Class Distribution in Frozen Holdout",
        "",
        "| Class ID | Canonical Class | Box Count | % of Holdout Boxes |",
        "| :---: | :--- | :---: | :---: |",
    ]
    total_boxes = sum(class_box_counts.values())
    for cid, cname in enumerate(classes):
        cnt = class_box_counts[cname]
        pct = (cnt / total_boxes * 100.0) if total_boxes else 0.0
        md_lines.append(f"| {cid} | **{cname}** | {cnt:,} | {pct:.2f}% |")

    md_lines.extend([
        "",
        "## 2. Group Breakdown and Viewpoint Provenance",
        "",
        "| Group ID | Images | Viewpoint | Provenance Subsets | HD | TH | Normal | Discuss | Stand | Total Boxes |",
        "| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for gid in sorted(HOLDOUT_GROUPS):
        b = group_box_counts[gid]
        prov = "<br>".join(sorted(group_provenances[gid]))
        md_lines.append(
            f"| `{gid}` | {group_img_counts[gid]} | `{group_viewpoints[gid]}` | {prov} | "
            f"{b['head_down']} | {b['turn_head']} | {b['normal']} | {b['discuss']} | {b['stand']} | {sum(b.values())} |"
        )

    md_lines.extend([
        "",
        "## 3. Viewpoint Summary",
        "",
        "| Viewpoint Category | Groups | Images | Boxes |",
        "| :--- | :---: | :---: | :---: |",
    ])
    vp_counts = defaultdict(lambda: {"groups": 0, "images": 0, "boxes": 0})
    for gid in HOLDOUT_GROUPS:
        vp = group_viewpoints[gid]
        vp_counts[vp]["groups"] += 1
        vp_counts[vp]["images"] += group_img_counts[gid]
        vp_counts[vp]["boxes"] += sum(group_box_counts[gid].values())

    for vp, d in vp_counts.items():
        md_lines.append(f"| `{vp}` | {d['groups']} | {d['images']} | {d['boxes']} |")

    report_path = REPORTS_DIR / "STAGE1_5_HOLDOUT_MANIFEST.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Holdout manifest written to {report_path}")

    # Now evaluate original stage1_best.pt on this holdout immediately!
    stage1_weights = ROOT_DIR / "models" / "trained" / "stage1_best.pt"
    print(f"\nEvaluating Stage 1 baseline ({stage1_weights}) on frozen Stage 1.5 holdout...")
    model = YOLO(str(stage1_weights))
    val_results = model.val(
        data=str(HOLDOUT_DIR / "dataset.yaml"),
        split="val",
        imgsz=768,
        batch=8,
        device=0,
        plots=False,
        save_json=False,
        verbose=True
    )

    p_overall = float(val_results.results_dict.get("metrics/precision(B)", 0.0))
    r_overall = float(val_results.results_dict.get("metrics/recall(B)", 0.0))
    map50_overall = float(val_results.results_dict.get("metrics/mAP50(B)", 0.0))
    map5095_overall = float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0))

    per_class_metrics = {}
    class_p = val_results.box.p
    class_r = val_results.box.r
    class_ap50 = val_results.box.ap50
    class_ap = val_results.box.ap

    for i, cname in enumerate(classes):
        per_class_metrics[cname] = {
            "precision": float(class_p[i]) if i < len(class_p) else 0.0,
            "recall": float(class_r[i]) if i < len(class_r) else 0.0,
            "mAP50": float(class_ap50[i]) if i < len(class_ap50) else 0.0,
            "mAP50-95": float(class_ap[i]) if i < len(class_ap) else 0.0,
        }

    baseline_metrics = {
        "model": "models/trained/stage1_best.pt",
        "holdout": "datasets/stage1_5_holdout",
        "overall": {
            "precision": round(p_overall, 4),
            "recall": round(r_overall, 4),
            "mAP50": round(map50_overall, 4),
            "mAP50-95": round(map5095_overall, 4)
        },
        "per_class": per_class_metrics
    }

    baseline_path = REPORTS_DIR / "STAGE1_BASELINE_ON_STAGE1_5_HOLDOUT.json"
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline_metrics, f, indent=2)
    print(f"Saved STAGE1_BASELINE_ON_STAGE1_5_HOLDOUT to {baseline_path}")

    # Also append baseline table to the holdout report or create STAGE1_BASELINE_ON_STAGE1_5_HOLDOUT.md
    bl_lines = [
        "# Stage 1 Baseline on Stage 1.5 Frozen Holdout",
        "",
        "**Model**: `models/trained/stage1_best.pt`  ",
        "**Holdout**: `datasets/stage1_5_holdout/`  ",
        "**Image Size**: 768px  ",
        "**Batch Size**: 8  ",
        "",
        "## Overall Metrics",
        f"- **Precision**: `{p_overall:.4f}`",
        f"- **Recall**: `{r_overall:.4f}`",
        f"- **mAP50**: `{map50_overall:.4f}`",
        f"- **mAP50-95**: `{map5095_overall:.4f}`",
        "",
        "## Per-Class Breakdown",
        "",
        "| Class | Precision | Recall | mAP50 | mAP50-95 |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for cname in classes:
        m = per_class_metrics[cname]
        bl_lines.append(f"| **{cname}** | {m['precision']:.4f} | {m['recall']:.4f} | {m['mAP50']:.4f} | {m['mAP50-95']:.4f} |")

    with open(REPORTS_DIR / "STAGE1_BASELINE_ON_STAGE1_5_HOLDOUT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(bl_lines) + "\n")
    print(f"Written baseline markdown to {REPORTS_DIR / 'STAGE1_BASELINE_ON_STAGE1_5_HOLDOUT.md'}")

if __name__ == "__main__":
    main()
