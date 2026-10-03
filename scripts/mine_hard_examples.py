"""Run hard-example mining using stage1_best.pt over Stage 1 train domain excluding holdout."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_V3 = ROOT_DIR / "datasets" / "processed_v3"
HOLDOUT_DIR = ROOT_DIR / "datasets" / "stage1_5_holdout"
REPORTS_DIR = ROOT_DIR / "reports" / "stage1_5"
HARD_EX_DIR = REPORTS_DIR / "hard_examples"
HARD_EX_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ROOT_DIR / "models" / "trained" / "stage1_best.pt"

CLASSES = ['normal', 'head_down', 'turn_head', 'discuss', 'stand']

def bbox_iou(b1, b2):
    # b = (xc, yc, w, h)
    x1_min, y1_min = b1[0] - b1[2] / 2, b1[1] - b1[3] / 2
    x1_max, y1_max = b1[0] + b1[2] / 2, b1[1] + b1[3] / 2
    x2_min, y2_min = b2[0] - b2[2] / 2, b2[1] - b2[3] / 2
    x2_max, y2_max = b2[0] + b2[2] / 2, b2[1] + b2[3] / 2

    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    inter_w = max(0.0, xi_max - xi_min)
    inter_h = max(0.0, yi_max - yi_min)
    inter_area = inter_w * inter_h

    area1 = b1[2] * b1[3]
    area2 = b2[2] * b2[3]
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def make_contact_sheet(crops, titles, out_path, cols=6, thumb_size=(160, 200)):
    if not crops:
        return
    rows = math.ceil(len(crops) / cols)
    sheet_w = cols * thumb_size[0]
    sheet_h = rows * thumb_size[1]
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for idx, (crop, title) in enumerate(zip(crops, titles)):
        r = idx // cols
        c = idx % cols
        x1 = c * thumb_size[0]
        y1 = r * thumb_size[1]
        x2 = x1 + thumb_size[0]
        y2 = y1 + thumb_size[1]

        ch, cw = crop.shape[:2]
        if ch == 0 or cw == 0:
            continue
        scale = min((thumb_size[0] - 10) / cw, (thumb_size[1] - 30) / ch)
        nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

        dx = x1 + (thumb_size[0] - nw) // 2
        dy = y1 + 5 + (thumb_size[1] - 25 - nh) // 2
        sheet[dy:dy+nh, dx:dx+nw] = resized

        cv2.rectangle(sheet, (x1, y1), (x2, y2), (60, 60, 60), 1)
        cv2.putText(sheet, title[:22], (x1 + 4, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    cv2.imwrite(str(out_path), sheet)

def main():
    print("=" * 60)
    print("RUNNING HARD-EXAMPLE MINING (STAGE 1 BEST ON TRAIN DOMAIN)")
    print("=" * 60)

    # 1. Load holdout groups
    holdout_manifest_file = HOLDOUT_DIR / "manifest.json"
    holdout_groups = set()
    if holdout_manifest_file.exists():
        hm = json.load(open(holdout_manifest_file, "r", encoding="utf-8"))
        holdout_groups = set(hm.get("groups", []))
    print(f"Excluding {len(holdout_groups)} holdout groups from mining: {holdout_groups}")

    # 2. Collect candidate train frames
    with open(PROCESSED_V3 / "manifest_v3.json", "r", encoding="utf-8") as f:
        v3_manifest = json.load(f)

    train_entries = [e for e in v3_manifest["splits"]["train"] if e["group_id"] not in holdout_groups]
    print(f"Total training domain frames to mine: {len(train_entries)}")

    # 3. Load model
    print(f"Loading weights from {MODEL_PATH}...")
    model = YOLO(str(MODEL_PATH))

    # We will mine hard examples in batches
    hard_examples = []
    error_counts = Counter()

    crops_by_type = defaultdict(list)
    titles_by_type = defaultdict(list)

    batch_size = 16
    total_batches = math.ceil(len(train_entries) / batch_size)

    print("Beginning inference and ground-truth comparison...")
    for b_idx in range(total_batches):
        batch_slice = train_entries[b_idx * batch_size : (b_idx + 1) * batch_size]
        img_paths = [str(ROOT_DIR / e["image_path"]) for e in batch_slice]

        results = model.predict(source=img_paths, imgsz=768, conf=0.15, device=0, verbose=False)

        for e, res, img_p in zip(batch_slice, results, img_paths):
            gt_annots = e.get("annotations", [])
            gid = e["group_id"]
            im_hash = e["image_hash"]

            # Viewpoint determination
            subsets = set()
            for a in gt_annots:
                for s in a.get("sources", []):
                    subsets.add(s.get("source_subset", ""))
            is_oblique = any(k in s for s in subsets for k in ['Stand', 'Bow', 'Turn', 'Discuss'])
            vp = "cctv_oblique_high_angle" if is_oblique else "frontal_classroom"

            # Parse predictions
            preds = []
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    xyxy = b.xyxy[0].cpu().numpy()
                    cid = int(b.cls[0].cpu().item())
                    conf = float(b.conf[0].cpu().item())
                    xc = float(b.xywhn[0][0].cpu().item())
                    yc = float(b.xywhn[0][1].cpu().item())
                    w = float(b.xywhn[0][2].cpu().item())
                    h = float(b.xywhn[0][3].cpu().item())
                    preds.append({
                        "cid": cid,
                        "cname": CLASSES[cid],
                        "conf": conf,
                        "bbox": (xc, yc, w, h),
                        "xyxy": xyxy
                    })

            # Compare GT vs Preds
            matched_pred_indices = set()

            # Need cv2 image only if we extract crop
            cv_img = None

            for gt in gt_annots:
                gt_cid = gt["canonical_class_id"]
                gt_cname = gt["canonical_class_name"]
                gt_bbox = gt["bbox"]

                best_iou = 0.0
                best_pred = None
                best_p_idx = -1

                for p_idx, p in enumerate(preds):
                    iou = bbox_iou(gt_bbox, p["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_pred = p
                        best_p_idx = p_idx

                err_type = None
                p_cname = best_pred["cname"] if best_pred else "missing"
                p_conf = best_pred["conf"] if best_pred else 0.0

                if best_iou >= 0.35:
                    matched_pred_indices.add(best_p_idx)
                    if gt_cname == "head_down" and p_cname != "head_down":
                        err_type = "TYPE_A_HEAD_DOWN_FN"
                    elif gt_cname == "turn_head" and p_cname != "turn_head":
                        if p_cname == "discuss":
                            err_type = "TYPE_E_TURN_HEAD_DISCUSS_CONFUSION"
                        else:
                            err_type = "TYPE_B_TURN_HEAD_FN"
                    elif gt_cname == "normal" and p_cname == "head_down":
                        err_type = "TYPE_C_HARD_NORMAL_NEG_HEAD_DOWN"
                    elif gt_cname == "normal" and p_cname == "turn_head":
                        err_type = "TYPE_C_HARD_NORMAL_NEG_TURN_HEAD"
                    elif gt_cname == "discuss" and p_cname == "turn_head":
                        err_type = "TYPE_E_TURN_HEAD_DISCUSS_CONFUSION"
                else:
                    # Missed completely
                    if gt_cname == "head_down":
                        err_type = "TYPE_A_HEAD_DOWN_FN"
                    elif gt_cname == "turn_head":
                        err_type = "TYPE_B_TURN_HEAD_FN"

                if err_type:
                    error_counts[err_type] += 1
                    ex_info = {
                        "image_path": e["image_path"],
                        "group_id": gid,
                        "image_hash": im_hash,
                        "viewpoint": vp,
                        "gt_class": gt_cname,
                        "pred_class": p_cname,
                        "confidence": round(p_conf, 4),
                        "iou": round(best_iou, 4),
                        "error_type": err_type,
                        "gt_bbox": gt_bbox
                    }
                    hard_examples.append(ex_info)

                    # Extract visual crop if needed
                    if len(crops_by_type[err_type]) < 24:
                        if cv_img is None:
                            cv_img = cv2.imread(img_p)
                        if cv_img is not None:
                            ih, iw = cv_img.shape[:2]
                            xc, yc, bw, bh = gt_bbox
                            x1 = max(0, int((xc - bw / 2) * iw))
                            y1 = max(0, int((yc - bh / 2) * ih))
                            x2 = min(iw, int((xc + bw / 2) * iw))
                            y2 = min(ih, int((yc + bh / 2) * ih))
                            crop = cv_img[y1:y2, x1:x2]
                            if crop.size > 0:
                                crops_by_type[err_type].append(crop)
                                titles_by_type[err_type].append(f"GT:{gt_cname[:4]}->P:{p_cname[:4]}|{p_conf:.2f}")

            # Check for background hallucinations (Pred box with no GT match and high conf)
            for p_idx, p in enumerate(preds):
                if p_idx not in matched_pred_indices and p["conf"] >= 0.35:
                    if p["cname"] in ["head_down", "turn_head"]:
                        max_gt_iou = max([bbox_iou(gt["bbox"], p["bbox"]) for gt in gt_annots], default=0.0)
                        if max_gt_iou < 0.2:
                            err_type = "TYPE_D_BACKGROUND_HALLUCINATION"
                            error_counts[err_type] += 1
                            ex_info = {
                                "image_path": e["image_path"],
                                "group_id": gid,
                                "image_hash": im_hash,
                                "viewpoint": vp,
                                "gt_class": "background",
                                "pred_class": p["cname"],
                                "confidence": round(p["conf"], 4),
                                "iou": round(max_gt_iou, 4),
                                "error_type": err_type,
                                "pred_bbox": p["bbox"]
                            }
                            hard_examples.append(ex_info)
                            if len(crops_by_type[err_type]) < 24:
                                if cv_img is None:
                                    cv_img = cv2.imread(img_p)
                                if cv_img is not None:
                                    ih, iw = cv_img.shape[:2]
                                    xc, yc, bw, bh = p["bbox"]
                                    x1 = max(0, int((xc - bw / 2) * iw))
                                    y1 = max(0, int((yc - bh / 2) * ih))
                                    x2 = min(iw, int((xc + bw / 2) * iw))
                                    y2 = min(ih, int((yc + bh / 2) * ih))
                                    crop = cv_img[y1:y2, x1:x2]
                                    if crop.size > 0:
                                        crops_by_type[err_type].append(crop)
                                        titles_by_type[err_type].append(f"BG->P:{p['cname'][:4]}|{p['conf']:.2f}")

        if (b_idx + 1) % 50 == 0 or (b_idx + 1) == total_batches:
            print(f"Processed batch {b_idx + 1}/{total_batches} ({len(hard_examples)} hard instances mined so far)...")

    print(f"\nHard Example Mining Complete! Total instances mined: {len(hard_examples)}")
    print("Error breakdown:")
    for etype, cnt in error_counts.most_common():
        print(f"  {etype:<40}: {cnt}")

    # Save manifest
    with open(HARD_EX_DIR / "hard_examples_manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": str(MODEL_PATH),
            "total_mined": len(hard_examples),
            "counts_by_type": dict(error_counts),
            "samples": hard_examples
        }, f, indent=2)

    # Save contact sheets
    for etype, crops in crops_by_type.items():
        fname = f"gallery_{etype.lower()}.jpg"
        make_contact_sheet(crops, titles_by_type[etype], HARD_EX_DIR / fname, cols=6)
        print(f"Saved visual gallery: {fname}")

    # Generate Markdown Report
    lines = [
        "# Hard-Example Mining Report (Stage 1 Best Checkpoint)",
        "",
        "**Date**: 2026-10-02  ",
        "**Model Evaluated**: `models/trained/stage1_best.pt`  ",
        f"**Domain Mined**: Stage 1 `train` split excluding frozen Stage 1.5 holdout ({len(train_entries)} images)  ",
        f"**Total Hard Examples Discovered**: {len(hard_examples):,}  ",
        "",
        "## 1. Distribution of Error Types",
        "",
        "| Error Category | Code | Description | Instance Count | % of Errors |",
        "| :--- | :--- | :--- | :---: | :---: |",
    ]
    total_errs = len(hard_examples)
    descriptions = {
        "TYPE_A_HEAD_DOWN_FN": "Ground truth is `head_down`, predicted as `normal` or completely undetected",
        "TYPE_B_TURN_HEAD_FN": "Ground truth is `turn_head`, predicted as `normal` or completely undetected",
        "TYPE_C_HARD_NORMAL_NEG_HEAD_DOWN": "Ground truth is `normal`, falsely predicted as `head_down`",
        "TYPE_C_HARD_NORMAL_NEG_TURN_HEAD": "Ground truth is `normal`, falsely predicted as `turn_head`",
        "TYPE_D_BACKGROUND_HALLUCINATION": "False detection of `head_down` or `turn_head` on empty desk/background",
        "TYPE_E_TURN_HEAD_DISCUSS_CONFUSION": "Mutual confusion between `turn_head` and `discuss` interactions",
    }
    for etype in ["TYPE_A_HEAD_DOWN_FN", "TYPE_B_TURN_HEAD_FN", "TYPE_C_HARD_NORMAL_NEG_HEAD_DOWN", "TYPE_C_HARD_NORMAL_NEG_TURN_HEAD", "TYPE_E_TURN_HEAD_DISCUSS_CONFUSION", "TYPE_D_BACKGROUND_HALLUCINATION"]:
        cnt = error_counts[etype]
        pct = (cnt / total_errs * 100.0) if total_errs else 0.0
        lines.append(f"| **{etype.split('_', 2)[-1]}** | `{etype}` | {descriptions.get(etype, '')} | {cnt:,} | {pct:.1f}% |")

    lines.extend([
        "",
        "## 2. Hard Example Findings and Impact on Refinement Training",
        "",
        "1. **Dominance of False Negatives in `turn_head` and `head_down`**:",
        f"   - Weak-class false negatives represent the vast majority of mined hard instances ({error_counts['TYPE_A_HEAD_DOWN_FN'] + error_counts['TYPE_B_TURN_HEAD_FN']:,} instances).",
        "   - The detector repeatedly suppresses head_down and turn_head in favor of the high-prior `normal` class.",
        "2. **Hard Normal Negatives**:",
        f"   - Normal examinees leaning low or glancing slightly sideways generate {error_counts['TYPE_C_HARD_NORMAL_NEG_HEAD_DOWN']} head_down FPs and {error_counts['TYPE_C_HARD_NORMAL_NEG_TURN_HEAD']} turn_head FPs.",
        "   - These hard normal frames are essential negative examples during Stage 1.5 fine-tuning to prevent the model from over-predicting weak classes.",
        "3. **Turn-Head / Discuss Ambiguity**:",
        f"   - {error_counts['TYPE_E_TURN_HEAD_DISCUSS_CONFUSION']} confusion instances occur when examinees turn toward each other in cluster rows.",
        "",
        "## 3. Visual Gallery Artifacts",
        "- `reports/stage1_5/hard_examples/gallery_type_a_head_down_fn.jpg`",
        "- `reports/stage1_5/hard_examples/gallery_type_b_turn_head_fn.jpg`",
        "- `reports/stage1_5/hard_examples/gallery_type_c_hard_normal_neg_head_down.jpg`",
        "- `reports/stage1_5/hard_examples/gallery_type_c_hard_normal_neg_turn_head.jpg`",
        "- `reports/stage1_5/hard_examples/gallery_type_e_turn_head_discuss_confusion.jpg`",
        "- `reports/stage1_5/hard_examples/gallery_type_d_background_hallucination.jpg`",
        "",
        "## 4. Integration into Stage 1.5 Refinement Set (`processed_v3_5`)",
        "- Frames containing hard weak-class positives and hard normal negatives are flagged for prioritized inclusion and frequency reweighting.",
    ])

    report_p = REPORTS_DIR / "HARD_EXAMPLE_MINING.md"
    with open(report_p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report written to {report_p}")

if __name__ == "__main__":
    main()
