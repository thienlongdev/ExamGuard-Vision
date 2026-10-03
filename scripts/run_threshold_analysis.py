"""Threshold and PR Curve Analysis for Stage 1.5 Weak Behavior Classes."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORT_FILE = ROOT_DIR / "reports" / "stage1_5" / "THRESHOLD_ANALYSIS.md"
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["normal", "head_down", "turn_head", "discuss", "stand"]

def analyze_thresholds():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("STARTING THRESHOLD AND PR CURVE ANALYSIS")
    print("=" * 60)

    ckpt_candidates = [
        ROOT_DIR / "runs" / "stage1_5" / "v3_5_refinement" / "weights" / "best.pt",
        ROOT_DIR / "models" / "trained" / "stage1_5_best.pt",
        ROOT_DIR / "models" / "trained" / "stage1_best.pt"
    ]
    model_path = None
    for cand in ckpt_candidates:
        if cand.exists():
            model_path = cand
            break

    print(f"Analyzing model: {model_path}")
    model = YOLO(str(model_path))

    # Evaluate validation metrics across thresholds
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
    results_by_thresh = {}

    holdout_yaml = ROOT_DIR / "datasets" / "stage1_5_holdout" / "dataset.yaml"

    for th in thresholds:
        print(f"Evaluating at conf={th:.2f}...")
        val_res = model.val(
            data=str(holdout_yaml),
            conf=th,
            imgsz=768,
            batch=4,
            device=0,
            plots=False,
            verbose=False
        )
        p = val_res.box.p.tolist() if hasattr(val_res.box.p, 'tolist') else list(val_res.box.p)
        r = val_res.box.r.tolist() if hasattr(val_res.box.r, 'tolist') else list(val_res.box.r)
        f1 = [2 * pi * ri / (pi + ri + 1e-6) for pi, ri in zip(p, r)]
        results_by_thresh[th] = {
            "p": p,
            "r": r,
            "f1": f1,
            "map50": float(val_res.box.map50),
            "map50_95": float(val_res.box.map)
        }

    # Find optimal threshold per class
    optimal_thresholds = {}
    for c_idx, c_name in enumerate(CLASS_NAMES):
        best_f1 = -1.0
        best_th = 0.25
        best_p, best_r = 0.0, 0.0
        for th in thresholds:
            cur_f1 = results_by_thresh[th]["f1"][c_idx]
            if cur_f1 > best_f1:
                best_f1 = cur_f1
                best_th = th
                best_p = results_by_thresh[th]["p"][c_idx]
                best_r = results_by_thresh[th]["r"][c_idx]
        optimal_thresholds[c_name] = {
            "threshold": best_th,
            "f1": best_f1,
            "precision": best_p,
            "recall": best_r
        }

    print("\nOptimal Operating Thresholds (maximizing F1):")
    for c_name, v in optimal_thresholds.items():
        print(f"  {c_name:10s}: conf={v['threshold']:.2f} | F1={v['f1']:.4f} (P={v['precision']:.4f}, R={v['recall']:.4f})")

    # Generate Markdown Report
    lines = [
        "# Stage 1.5 Operating Threshold and PR Curve Analysis",
        "",
        f"- **Model**: `{model_path.name}`",
        f"- **Holdout Dataset**: `datasets/stage1_5_holdout/`",
        "- **Resolution**: 768px",
        "- **Evaluation Sweep**: Confidence thresholds from 0.05 to 0.70",
        "",
        "## 1. Optimal F1 Operating Thresholds per Class",
        "",
        "| Class | Optimal Conf Threshold | Max F1 Score | Precision @ Opt | Recall @ Opt | Standard @ 0.25 F1 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for c_idx, c_name in enumerate(CLASS_NAMES):
        opt = optimal_thresholds[c_name]
        std_f1 = results_by_thresh[0.25]["f1"][c_idx]
        lines.append(f"| **{c_name}** | {opt['threshold']:.2f} | {opt['f1']:.4f} | {opt['precision']:.4f} | {opt['recall']:.4f} | {std_f1:.4f} |")

    lines.extend([
        "",
        "## 2. Weak-Class Confidence Sweep Breakdown",
        "",
        "### `head_down` Precision-Recall vs Confidence",
        "| Confidence | Precision | Recall | F1 Score |",
        "| :---: | :---: | :---: | :---: |",
    ])
    hd_idx = 1
    for th in thresholds:
        d = results_by_thresh[th]
        lines.append(f"| {th:.2f} | {d['p'][hd_idx]:.4f} | {d['r'][hd_idx]:.4f} | {d['f1'][hd_idx]:.4f} |")

    lines.extend([
        "",
        "### `turn_head` Precision-Recall vs Confidence",
        "| Confidence | Precision | Recall | F1 Score |",
        "| :---: | :---: | :---: | :---: |",
    ])
    th_idx = 2
    for th in thresholds:
        d = results_by_thresh[th]
        lines.append(f"| {th:.2f} | {d['p'][th_idx]:.4f} | {d['r'][th_idx]:.4f} | {d['f1'][th_idx]:.4f} |")

    # Diagnose separation vs threshold failure
    hd_r_low = results_by_thresh[0.05]["r"][hd_idx]
    th_r_low = results_by_thresh[0.05]["r"][th_idx]

    lines.extend([
        "",
        "## 3. Failure Mode Diagnosis: Separation Failure vs Threshold Failure",
        "",
        "### Physical Diagnostic Rules",
        "- **Threshold Failure**: If recall jumps dramatically (> 0.20 boost) when lowering confidence to 0.05-0.10 while maintaining acceptable precision, the model separated features but the default 0.25 threshold was calibrated too aggressively.",
        "- **Model-Separation Failure**: If lowering confidence to 0.05 does NOT substantially increase recall or causes precision to collapse near zero, the detector backbone/head lacks discriminative spatial features to separate the subtle posture differences (e.g. slight head tilt vs reading) from single 2D bounding boxes.",
        "",
        "### Diagnostic Findings",
        f"1. **`head_down`**: At conf=0.05, Recall is {hd_r_low:.4f} (vs {results_by_thresh[0.25]['r'][hd_idx]:.4f} at conf=0.25).",
        f"2. **`turn_head`**: At conf=0.05, Recall is {th_r_low:.4f} (vs {results_by_thresh[0.25]['r'][th_idx]:.4f} at conf=0.25).",
        "",
        "### Recommendation for Temporal Rule Engine",
        f"- Recommended detector confidence threshold for `head_down`: **{optimal_thresholds['head_down']['threshold']:.2f}**.",
        f"- Recommended detector confidence threshold for `turn_head`: **{optimal_thresholds['turn_head']['threshold']:.2f}**.",
        f"- Recommended detector confidence threshold for `normal`: **{optimal_thresholds['normal']['threshold']:.2f}**.",
        "- In the multi-stage exam monitoring system, single-frame false positives at lower confidence thresholds are filtered out by the **ByteTrack temporal buffer** requiring persistent behavior across consecutive frames (e.g. 5+ frames) and cooldown hysteresis.",
        "",
    ])

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report written to {REPORT_FILE}")

if __name__ == "__main__":
    analyze_thresholds()
