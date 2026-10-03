"""Evaluate final Stage 1.5 candidate checkpoint across Holdout, V3 Val, and Reference Test."""
import json
from pathlib import Path
import sys
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
BEST_CKPT = ROOT_DIR / "runs" / "stage1_5" / "v3_5_refinement" / "weights" / "best.pt"
CLASS_NAMES = ["normal", "head_down", "turn_head", "discuss", "stand"]

def evaluate_splits():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("EVALUATING FINAL STAGE 1.5 CANDIDATE")
    print("=" * 60)

    if not BEST_CKPT.exists():
        raise FileNotFoundError(f"Best checkpoint missing: {BEST_CKPT}")

    print(f"Loading checkpoint: {BEST_CKPT}")
    model = YOLO(str(BEST_CKPT))

    # 1. Evaluate on Stage 1.5 Frozen Holdout
    print("\n[1/3] Evaluating on Stage 1.5 Frozen Holdout (datasets/stage1_5_holdout/)...")
    res_holdout = model.val(
        data=str(ROOT_DIR / "datasets" / "stage1_5_holdout" / "dataset.yaml"),
        imgsz=768,
        batch=4,
        device=0,
        plots=False,
        verbose=False
    )
    holdout_metrics = {
        "precision": float(res_holdout.box.mp),
        "recall": float(res_holdout.box.mr),
        "map50": float(res_holdout.box.map50),
        "map50_95": float(res_holdout.box.map),
        "per_class": {}
    }
    for i, cname in enumerate(CLASS_NAMES):
        holdout_metrics["per_class"][cname] = {
            "precision": float(res_holdout.box.p[i]),
            "recall": float(res_holdout.box.r[i]),
            "map50": float(res_holdout.box.ap50[i]),
            "map50_95": float(res_holdout.box.ap[i])
        }

    # 2. Evaluate on Original V3 Validation split
    print("\n[2/3] Evaluating on Original V3 Val split (datasets/processed_v3/)...")
    res_v3_val = model.val(
        data=str(ROOT_DIR / "datasets" / "processed_v3" / "dataset.yaml"),
        split="val",
        imgsz=768,
        batch=4,
        device=0,
        plots=False,
        verbose=False
    )
    v3_val_metrics = {
        "precision": float(res_v3_val.box.mp),
        "recall": float(res_v3_val.box.mr),
        "map50": float(res_v3_val.box.map50),
        "map50_95": float(res_v3_val.box.map),
        "per_class": {}
    }
    for i, cname in enumerate(CLASS_NAMES):
        v3_val_metrics["per_class"][cname] = {
            "precision": float(res_v3_val.box.p[i]),
            "recall": float(res_v3_val.box.r[i]),
            "map50": float(res_v3_val.box.ap50[i]),
            "map50_95": float(res_v3_val.box.ap[i])
        }

    # 3. Evaluate on Reference Test split (NOT PRISTINE FOR TUNING)
    print("\n[3/3] Evaluating on Reference Test split (datasets/processed_v3/)...")
    res_v3_test = model.val(
        data=str(ROOT_DIR / "datasets" / "processed_v3" / "dataset.yaml"),
        split="test",
        imgsz=768,
        batch=4,
        device=0,
        plots=False,
        verbose=False
    )
    v3_test_metrics = {
        "precision": float(res_v3_test.box.mp),
        "recall": float(res_v3_test.box.mr),
        "map50": float(res_v3_test.box.map50),
        "map50_95": float(res_v3_test.box.map),
        "per_class": {}
    }
    for i, cname in enumerate(CLASS_NAMES):
        v3_test_metrics["per_class"][cname] = {
            "precision": float(res_v3_test.box.p[i]),
            "recall": float(res_v3_test.box.r[i]),
            "map50": float(res_v3_test.box.ap50[i]),
            "map50_95": float(res_v3_test.box.ap[i])
        }

    out_json = {
        "checkpoint": str(BEST_CKPT),
        "holdout": holdout_metrics,
        "v3_val": v3_val_metrics,
        "v3_test_reference": v3_test_metrics
    }
    json_path = ROOT_DIR / "reports" / "stage1_5" / "STAGE1_5_EVALUATION_METRICS.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved structured evaluation metrics to {json_path}")

    # Generate Markdown Summary
    md_lines = [
        "# Stage 1.5 Final Candidate Evaluation Metrics",
        "",
        f"**Checkpoint**: `{BEST_CKPT.name}` (from `runs/stage1_5/v3_5_refinement/weights/best.pt`)  ",
        "- **Resolution**: 768px  ",
        "- **Batch Size**: 4  ",
        "- **Device**: CUDA:0 (NVIDIA RTX 5070)  ",
        "",
        "---",
        "",
        "## 1. Primary Comparison: Frozen Stage 1.5 Holdout (Unseen Holdout Groups)",
        "",
        f"- **Overall Precision**: `{holdout_metrics['precision']:.4f}`",
        f"- **Overall Recall**: `{holdout_metrics['recall']:.4f}`",
        f"- **Overall mAP50**: `{holdout_metrics['map50']:.4f}`",
        f"- **Overall mAP50-95**: `{holdout_metrics['map50_95']:.4f}`",
        "",
        "| Class | Precision | Recall | mAP50 | mAP50-95 |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for cname in CLASS_NAMES:
        m = holdout_metrics["per_class"][cname]
        md_lines.append(f"| **{cname}** | {m['precision']:.4f} | {m['recall']:.4f} | {m['map50']:.4f} | {m['map50_95']:.4f} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Generalization Audit: Original V3 Validation Split (Completely Unseen Groups)",
        "",
        f"- **Overall Precision**: `{v3_val_metrics['precision']:.4f}`",
        f"- **Overall Recall**: `{v3_val_metrics['recall']:.4f}`",
        f"- **Overall mAP50**: `{v3_val_metrics['map50']:.4f}`",
        f"- **Overall mAP50-95**: `{v3_val_metrics['map50_95']:.4f}`",
        "",
        "| Class | Precision | Recall | mAP50 | mAP50-95 |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for cname in CLASS_NAMES:
        m = v3_val_metrics["per_class"][cname]
        md_lines.append(f"| **{cname}** | {m['precision']:.4f} | {m['recall']:.4f} | {m['map50']:.4f} | {m['map50_95']:.4f} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Reference Test Split (REFERENCE ONLY — NOT PRISTINE FOR STAGE 1.5)",
        "",
        "> [!NOTE]",
        "> Per Requirement 19, the original V3 test set is strictly a post-selection reference test,",
        "> as its Stage 1 baseline numbers were already known and it was not used for tuning.",
        "",
        f"- **Overall Precision**: `{v3_test_metrics['precision']:.4f}`",
        f"- **Overall Recall**: `{v3_test_metrics['recall']:.4f}`",
        f"- **Overall mAP50**: `{v3_test_metrics['map50']:.4f}`",
        f"- **Overall mAP50-95**: `{v3_test_metrics['map50_95']:.4f}`",
        "",
        "| Class | Precision | Recall | mAP50 | mAP50-95 |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for cname in CLASS_NAMES:
        m = v3_test_metrics["per_class"][cname]
        md_lines.append(f"| **{cname}** | {m['precision']:.4f} | {m['recall']:.4f} | {m['map50']:.4f} | {m['map50_95']:.4f} |")

    md_path = ROOT_DIR / "reports" / "stage1_5" / "STAGE1_5_CANDIDATE_ON_HOLDOUT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved evaluation markdown summary to {md_path}")

if __name__ == "__main__":
    evaluate_splits()
