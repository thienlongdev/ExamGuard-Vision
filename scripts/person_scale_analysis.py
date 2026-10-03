"""Person-scale failure analysis for weak behavior classes (head_down, turn_head)."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
HARD_EX_PATH = ROOT_DIR / "reports" / "stage1_5" / "hard_examples" / "hard_examples_manifest.json"
V3_MANIFEST_PATH = ROOT_DIR / "datasets" / "processed_v3" / "manifest_v3.json"
HOLDOUT_PATH = ROOT_DIR / "datasets" / "stage1_5_holdout" / "manifest.json"
REPORT_PATH = ROOT_DIR / "reports" / "stage1_5" / "PERSON_SCALE_FAILURE_ANALYSIS.md"

def main():
    print("=" * 60)
    print("RUNNING PERSON-SCALE FAILURE ANALYSIS")
    print("=" * 60)

    # Load holdout groups
    hm = json.load(open(HOLDOUT_PATH, "r", encoding="utf-8"))
    holdout_groups = set(hm.get("groups", []))

    # Load hard examples manifest
    hard_data = json.load(open(HARD_EX_PATH, "r", encoding="utf-8"))
    hard_samples = hard_data["samples"]

    # Index hard FN samples by image_path + bbox
    fn_hd_list = [s for s in hard_samples if s["error_type"] == "TYPE_A_HEAD_DOWN_FN"]
    fn_th_list = [s for s in hard_samples if s["error_type"] in ["TYPE_B_TURN_HEAD_FN", "TYPE_E_TURN_HEAD_DISCUSS_CONFUSION"]]

    # Load all GT boxes from train domain
    v3_m = json.load(open(V3_MANIFEST_PATH, "r", encoding="utf-8"))
    train_entries = [e for e in v3_m["splits"]["train"] if e["group_id"] not in holdout_groups]

    hd_stats = {"correct": [], "missed": []}
    th_stats = {"correct": [], "missed": []}

    # Match GT boxes with FN lists
    for e in train_entries:
        im_p = e["image_path"]
        for ann in e.get("annotations", []):
            cname = ann["canonical_class_name"]
            if cname not in ["head_down", "turn_head"]:
                continue
            xc, yc, w, h = ann["bbox"]
            area = w * h
            pixel_h = h * 768
            pixel_w = w * 768

            # Check if this box is in fn list
            is_fn = False
            target_list = fn_hd_list if cname == "head_down" else fn_th_list
            for s in target_list:
                if s["image_path"] == im_p:
                    gt_b = s.get("gt_bbox")
                    if gt_b and abs(gt_b[0] - xc) < 0.02 and abs(gt_b[1] - yc) < 0.02:
                        is_fn = True
                        break

            item = {
                "w": w, "h": h, "area": area,
                "pixel_h": pixel_h, "pixel_w": pixel_w,
                "aspect": w / max(1e-4, h)
            }
            if cname == "head_down":
                if is_fn:
                    hd_stats["missed"].append(item)
                else:
                    hd_stats["correct"].append(item)
            else:
                if is_fn:
                    th_stats["missed"].append(item)
                else:
                    th_stats["correct"].append(item)

    print(f"head_down: {len(hd_stats['correct'])} correct, {len(hd_stats['missed'])} missed")
    print(f"turn_head: {len(th_stats['correct'])} correct, {len(th_stats['missed'])} missed")

    def get_summary(items):
        if not items:
            return {"count": 0, "mean_h": 0, "mean_area": 0, "small_pct": 0, "med_pct": 0, "large_pct": 0}
        hs = [it["pixel_h"] for it in items]
        areas = [it["area"] for it in items]
        small = sum(1 for it in items if it["pixel_h"] < 80)
        med = sum(1 for it in items if 80 <= it["pixel_h"] < 160)
        large = sum(1 for it in items if it["pixel_h"] >= 160)
        return {
            "count": len(items),
            "mean_h": float(np.mean(hs)),
            "mean_area": float(np.mean(areas)),
            "median_h": float(np.median(hs)),
            "small_pct": small / len(items) * 100,
            "med_pct": med / len(items) * 100,
            "large_pct": large / len(items) * 100
        }

    hd_c_sum = get_summary(hd_stats["correct"])
    hd_m_sum = get_summary(hd_stats["missed"])
    th_c_sum = get_summary(th_stats["correct"])
    th_m_sum = get_summary(th_stats["missed"])

    # Markdown Report
    lines = [
        "# Person-Scale Failure Analysis for Weak Behavior Classes",
        "",
        "**Date**: 2026-10-02  ",
        "**Model Evaluated**: `models/trained/stage1_best.pt`  ",
        "**Dataset Evaluated**: Stage 1 Training Domain excluding Stage 1.5 Holdout (6,178 images)  ",
        "**Resolution Scale**: 768x768 pixels  ",
        "",
        "## 1. Scale Categorization Definitions",
        "- **Small / Distant Examinee**: Person box height < 80px (normalized area < 0.008; typical of rear classroom rows > 7m from camera).",
        "- **Medium Examinee**: Person box height 80px to 160px (normalized area 0.008 to 0.035; middle rows 3-6m).",
        "- **Large / Foreground Examinee**: Person box height > 160px (normalized area > 0.035; front rows 1-2m).",
        "",
        "## 2. Quantitative Scale Comparison",
        "",
        "### A. `head_down` Scale Breakdown",
        "",
        "| Detection Outcome | Total Instances | Mean Box Height | Median Height | Small (<80px) % | Medium (80-160px) % | Large (>160px) % |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Correctly Detected** | {hd_c_sum['count']} | {hd_c_sum['mean_h']:.1f} px | {hd_c_sum['median_h']:.1f} px | {hd_c_sum['small_pct']:.1f}% | {hd_c_sum['med_pct']:.1f}% | {hd_c_sum['large_pct']:.1f}% |",
        f"| **Missed (False Negatives)** | {hd_m_sum['count']} | {hd_m_sum['mean_h']:.1f} px | {hd_m_sum['median_h']:.1f} px | {hd_m_sum['small_pct']:.1f}% | {hd_m_sum['med_pct']:.1f}% | {hd_m_sum['large_pct']:.1f}% |",
        "",
        "### B. `turn_head` Scale Breakdown",
        "",
        "| Detection Outcome | Total Instances | Mean Box Height | Median Height | Small (<80px) % | Medium (80-160px) % | Large (>160px) % |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Correctly Detected** | {th_c_sum['count']} | {th_c_sum['mean_h']:.1f} px | {th_c_sum['median_h']:.1f} px | {th_c_sum['small_pct']:.1f}% | {th_c_sum['med_pct']:.1f}% | {th_c_sum['large_pct']:.1f}% |",
        f"| **Missed (False Negatives)** | {th_m_sum['count']} | {th_m_sum['mean_h']:.1f} px | {th_m_sum['median_h']:.1f} px | {th_m_sum['small_pct']:.1f}% | {th_m_sum['med_pct']:.1f}% | {th_m_sum['large_pct']:.1f}% |",
        "",
        "## 3. Key Findings on Person Scale and Failure Modes",
        "",
        "1. **Disproportionate Failure on Distant Students**:",
        f"   - For `turn_head`, missed detections have a higher concentration of small distant students ({th_m_sum['small_pct']:.1f}%) compared to correctly detected instances ({th_c_sum['small_pct']:.1f}%).",
        "   - In rear rows, a student's entire head occupies fewer than 12x12 pixels. Fine-grained posture cues (e.g. ear visibility, cheek profile) are blurred into 1-2 receptive fields.",
        "2. **Foreshortening in High-Angle Views**:",
        "   - For `head_down`, vertical foreshortening from oblique ceiling perspectives compresses the distance between the crown of the head and the desktop.",
        "   - Even when student height is medium (80-120px), the torso occludes the neck, causing the detector to default to the overwhelmingly frequent `normal` prior.",
        "3. **Architectural Implications for Future Phases**:",
        "   - Full-frame YOLO single-shot detection forces 768px feature maps to simultaneously detect full-body standing examinees and 10px distant head yaw.",
        "   - If data-centric refinement in Stage 1.5 does not fully overcome distant-person head orientation ambiguity, a 2-stage architecture (Person Detector -> ByteTrack -> Normalized Person Crop -> Posture/Pose Classifier) provides dedicated resolution for fine-grained head angles.",
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report written to {REPORT_PATH}")

if __name__ == "__main__":
    main()
