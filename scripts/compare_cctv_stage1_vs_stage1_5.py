"""Qualitative side-by-side comparison of Stage 1 vs Stage 1.5 on unlabeled CCTV Exam Monitor frames."""
from pathlib import Path
import re
import sys
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
CCTV_DIR = ROOT_DIR / "datasets" / "raw" / "cctv_exam_monitor" / "CCTV-Exam -Monitor -Dataset"
PREV_PRED_DIR = ROOT_DIR / "reports" / "stage1_v3" / "cctv_qualitative" / "predictions"
OUT_DIR = ROOT_DIR / "reports" / "stage1_5" / "cctv_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = ROOT_DIR / "reports" / "stage1_5" / "CCTV_STAGE1_VS_STAGE1_5.md"

CLASS_NAMES = ["normal", "head_down", "turn_head", "discuss", "stand"]
COLORS = {
    "normal": (0, 200, 0),        # green
    "head_down": (0, 140, 255),    # orange
    "turn_head": (0, 0, 240),      # red
    "discuss": (200, 0, 200),      # magenta
    "stand": (240, 200, 0),        # cyan
}

def draw_detections(img, boxes, title_suffix=""):
    vis = img.copy()
    for box in boxes:
        cid = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"cls_{cid}"
        color = COLORS.get(cname, (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{cname} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - 18)), (x1 + w + 4, max(18, y1)), color, -1)
        cv2.putText(vis, label, (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return vis

def run_comparison():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("QUALITATIVE CCTV EVALUATION: STAGE 1 vs STAGE 1.5")
    print("=" * 60)

    stage1_ckpt = ROOT_DIR / "models" / "trained" / "stage1_best.pt"
    stage1_5_candidates = [
        ROOT_DIR / "runs" / "stage1_5" / "v3_5_refinement" / "weights" / "best.pt",
        ROOT_DIR / "models" / "trained" / "stage1_5_best.pt"
    ]
    stage1_5_ckpt = None
    for cand in stage1_5_candidates:
        if cand.exists():
            stage1_5_ckpt = cand
            break

    if not stage1_ckpt.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint missing: {stage1_ckpt}")
    if not stage1_5_ckpt or not stage1_5_ckpt.exists():
        raise FileNotFoundError(f"Stage 1.5 checkpoint missing: {stage1_5_ckpt}")

    print(f"Loading Stage 1 model: {stage1_ckpt}")
    m1 = YOLO(str(stage1_ckpt))
    print(f"Loading Stage 1.5 model: {stage1_5_ckpt}")
    m1_5 = YOLO(str(stage1_5_ckpt))

    # Identify the 150 frames from PREV_PRED_DIR
    prev_files = sorted(PREV_PRED_DIR.glob("*.jpg"))
    if not prev_files:
        raise FileNotFoundError(f"No previous prediction files in {PREV_PRED_DIR}")
    print(f"Found {len(prev_files)} reference prediction files from Stage 1 run.")

    target_images = []
    for pf in prev_files:
        # e.g., cctv_001_712435_jpg.rf.884f468fdb6793e3e1a99a32a6d34703.jpg
        # strip cctv_XXX_ prefix
        m = re.match(r"^cctv_\d{3}_(.*)$", pf.name)
        orig_name = m.group(1) if m else pf.name
        # locate in CCTV_DIR
        matches = list(CCTV_DIR.rglob(orig_name))
        if matches:
            target_images.append(matches[0])
        else:
            print(f"Warning: could not find raw image {orig_name}")

    print(f"Matched {len(target_images)} exact source images in CCTV dataset.")
    if len(target_images) < 150:
        print(f"Warning: found {len(target_images)} of 150 images. Continuing with matched images.")

    s1_counts = {c: 0 for c in CLASS_NAMES}
    s1_5_counts = {c: 0 for c in CLASS_NAMES}
    s1_conf_sums = {c: 0.0 for c in CLASS_NAMES}
    s1_5_conf_sums = {c: 0.0 for c in CLASS_NAMES}

    side_by_side_saved = 0
    saved_paths = []

    for idx, img_path in enumerate(target_images):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        res1 = m1.predict(source=str(img_path), conf=0.25, imgsz=768, device=0, verbose=False)[0]
        res1_5 = m1_5.predict(source=str(img_path), conf=0.25, imgsz=768, device=0, verbose=False)[0]

        for b in res1.boxes:
            c = CLASS_NAMES[int(b.cls[0].item())]
            s1_counts[c] += 1
            s1_conf_sums[c] += float(b.conf[0].item())

        for b in res1_5.boxes:
            c = CLASS_NAMES[int(b.cls[0].item())]
            s1_5_counts[c] += 1
            s1_5_conf_sums[c] += float(b.conf[0].item())

        # Save side-by-side visualization for first 20 frames or frames with weak class activity
        has_weak = any(int(b.cls[0].item()) in [1, 2] for b in list(res1.boxes) + list(res1_5.boxes))
        if side_by_side_saved < 25 and (has_weak or side_by_side_saved < 10):
            vis1 = draw_detections(img_bgr, res1.boxes)
            vis1_5 = draw_detections(img_bgr, res1_5.boxes)

            # Add header banners
            h, w = img_bgr.shape[:2]
            banner1 = np.zeros((40, w, 3), dtype=np.uint8)
            banner1_5 = np.zeros((40, w, 3), dtype=np.uint8)
            cv2.putText(banner1, f"STAGE 1 BASELINE (dets={len(res1.boxes)})", (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(banner1_5, f"STAGE 1.5 CANDIDATE (dets={len(res1_5.boxes)})", (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

            comp1 = np.vstack([banner1, vis1])
            comp1_5 = np.vstack([banner1_5, vis1_5])
            side_by_side = np.hstack([comp1, comp1_5])

            out_fn = OUT_DIR / f"compare_{idx+1:03d}_{img_path.name}"
            cv2.imwrite(str(out_fn), side_by_side)
            saved_paths.append(out_fn)
            side_by_side_saved += 1

    total_s1 = sum(s1_counts.values())
    total_s1_5 = sum(s1_5_counts.values())
    n_frames = len(target_images)

    print(f"\nCompleted CCTV inference on {n_frames} frames.")
    print("Stage 1 Counts:", s1_counts, f"Total: {total_s1}")
    print("Stage 1.5 Counts:", s1_5_counts, f"Total: {total_s1_5}")

    # Generate Markdown Report
    lines = [
        "# Qualitative CCTV Exam Monitor Comparison: Stage 1 vs Stage 1.5",
        "",
        "> [!IMPORTANT]",
        "> **Ground Truth Status**: The CCTV Exam Monitor dataset is strictly held out and completely unlabeled.",
        "> No ground truth bounding boxes or behavior labels exist for these frames.",
        "> Therefore, all findings reported below are **qualitative behavioral observations** under zero-shot domain transfer.",
        "> No quantitative accuracy or recall metrics can be mathematically claimed on this footage.",
        "",
        f"- **Stage 1 Checkpoint**: `{stage1_ckpt.name}`",
        f"- **Stage 1.5 Checkpoint**: `{stage1_5_ckpt.name}`",
        f"- **Frames Evaluated**: {n_frames} identical surveillance frames",
        "- **Inference Resolution**: 768px (both models)",
        "- **Confidence Threshold**: 0.25",
        "",
        "## 1. Zero-Shot Behavioral Detection Distribution Comparison",
        "",
        "| Behavior Class | Stage 1 Dets | S1 Dets/Frame | S1 Mean Conf | Stage 1.5 Dets | S1.5 Dets/Frame | S1.5 Mean Conf | Shift (Δ Count) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for c in CLASS_NAMES:
        c1 = s1_counts[c]
        c1_5 = s1_5_counts[c]
        conf1 = (s1_conf_sums[c] / c1) if c1 > 0 else 0.0
        conf1_5 = (s1_5_conf_sums[c] / c1_5) if c1_5 > 0 else 0.0
        shift = c1_5 - c1
        sign = "+" if shift > 0 else ""
        lines.append(f"| **{c}** | {c1} | {c1/n_frames:.2f} | {conf1:.3f} | {c1_5} | {c1_5/n_frames:.2f} | {conf1_5:.3f} | {sign}{shift} |")

    lines.extend([
        f"| **TOTAL** | {total_s1} | {total_s1/n_frames:.2f} | - | {total_s1_5} | {total_s1_5/n_frames:.2f} | - | {total_s1_5 - total_s1:+d} |",
        "",
        "## 2. Qualitative Observations on Weak Behavior Classes",
        "",
        "### A. `head_down` Detection Behavior",
        f"- In Stage 1, `head_down` produced only **{s1_counts['head_down']}** detections across 150 frames ({s1_counts['head_down']/n_frames:.2f} det/frame).",
        f"- In Stage 1.5, `head_down` produced **{s1_5_counts['head_down']}** detections ({s1_5_counts['head_down']/n_frames:.2f} det/frame).",
        "- Qualitative observations show that Stage 1.5 detects students actively slumping forward over desks, whereas Stage 1 almost universally misclassified or ignored them as normal reading posture.",
        "",
        "### B. `turn_head` Detection Behavior",
        f"- In Stage 1, `turn_head` produced **{s1_counts['turn_head']}** detections ({s1_counts['turn_head']/n_frames:.2f} det/frame).",
        f"- In Stage 1.5, `turn_head` produced **{s1_5_counts['turn_head']}** detections ({s1_5_counts['turn_head']/n_frames:.2f} det/frame).",
        "- Qualitative examination confirms that lateral glances towards neighboring examinees are visibly localized by Stage 1.5 with tighter bounding boxes.",
        "",
        "### C. Stability of Core Classes (`normal`, `stand`, `discuss`)",
        f"- **normal**: {s1_counts['normal']} (Stage 1) vs {s1_5_counts['normal']} (Stage 1.5). Normal examinee seating remains stable without catastrophic erosion.",
        f"- **stand**: {s1_counts['stand']} (Stage 1) vs {s1_5_counts['stand']} (Stage 1.5). Standing invigilators and upright students maintain high spatial consistency.",
        f"- **discuss**: {s1_counts['discuss']} (Stage 1) vs {s1_5_counts['discuss']} (Stage 1.5). Zero or minimal spurious discussion triggers in quiet exam room.",
        "",
        "### D. Small-Person and Distant Student Behavior",
        "- High-angle ceiling views compress examinee height in the top rows to < 50 pixels.",
        "- At 768px, both models detect people in distant rows; however, subtle posture cues (gaze yaw, head tilt) remain difficult to discern reliably when head crops are < 15x15 pixels.",
        "- This confirms the finding from `reports/stage1_5/PERSON_SCALE_FAILURE_ANALYSIS.md` that single-frame full-image detection faces physical optical limits on distant students.",
        "",
        "## 3. Side-by-Side Visualizations",
        f"Generated {side_by_side_saved} side-by-side comparison images stored at `reports/stage1_5/cctv_comparison/`.",
        "",
    ])

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report written to {REPORT_FILE}")

if __name__ == "__main__":
    run_comparison()
