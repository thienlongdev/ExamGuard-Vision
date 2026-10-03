#!/usr/bin/env python3
"""
scripts/generate_v4b_contact_sheets.py
Generates visual contact sheets (10x10 grids) from the final V4B person-crop dataset manifest.
Covers primary posture classes, small persons, high angle views, hard negatives, and quarantined samples.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path

def create_grid(image_paths, grid_size=(10, 10), thumb_size=(120, 120), title=""):
    rows, cols = grid_size
    target_count = rows * cols
    
    pad_top = 40
    grid_img = np.zeros((pad_top + rows * thumb_size[1], cols * thumb_size[0], 3), dtype=np.uint8)
    grid_img[:] = (30, 30, 30) # dark gray background
    
    # Title
    cv2.putText(grid_img, title, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    for idx in range(min(len(image_paths), target_count)):
        r = idx // cols
        c = idx % cols
        p = Path(image_paths[idx])
        if p.exists():
            im = cv2.imread(str(p))
            if im is not None:
                im_res = cv2.resize(im, thumb_size)
                # Border
                cv2.rectangle(im_res, (0, 0), (thumb_size[0]-1, thumb_size[1]-1), (70, 70, 70), 1)
                y_offset = pad_top + r * thumb_size[1]
                x_offset = c * thumb_size[0]
                grid_img[y_offset:y_offset + thumb_size[1], x_offset:x_offset + thumb_size[0]] = im_res
                
    return grid_img

def main():
    root = Path(".")
    manifest_path = root / "datasets/v4_crop/manifest.jsonl"
    out_dir = root / "reports/v4b/final_contact_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading manifest from {manifest_path}...")
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            
    print(f"Loaded {len(records)} crop records.")
    
    # Selection categories
    categories = {
        "normal_upright": {
            "title": "V4B Final Crops: NORMAL_UPRIGHT (100 samples)",
            "filter": lambda r: r["ontology_label"] == "NORMAL_UPRIGHT" and "QUARANTINED" not in r["quality_flags"],
            "filename": "contact_sheet_normal_upright.jpg"
        },
        "normal_read_write": {
            "title": "V4B Final Crops: NORMAL_READ_WRITE (Hard Negatives vs Head Down) (100 samples)",
            "filter": lambda r: r["ontology_label"] == "NORMAL_READ_WRITE" and "QUARANTINED" not in r["quality_flags"],
            "filename": "contact_sheet_normal_read_write.jpg"
        },
        "head_down_sleep": {
            "title": "V4B Final Crops: HEAD_REST_SLEEP / HEAD_DOWN_DEEP (100 samples)",
            "filter": lambda r: r["ontology_label"] in ["HEAD_REST_SLEEP", "HEAD_DOWN_DEEP"] and "QUARANTINED" not in r["quality_flags"],
            "filename": "contact_sheet_head_down_sleep.jpg"
        },
        "turn_head_clear": {
            "title": "V4B Final Crops: TURN_HEAD_CLEAR (100 samples)",
            "filter": lambda r: r["ontology_label"] == "TURN_HEAD_CLEAR" and "QUARANTINED" not in r["quality_flags"],
            "filename": "contact_sheet_turn_head_clear.jpg"
        },
        "small_person": {
            "title": "V4B Final Crops: Small-Person Scale (PERSON_SMALL & VERY_SMALL) (100 samples)",
            "filter": lambda r: r["scale_bucket"] in ["PERSON_SMALL", "PERSON_VERY_SMALL"],
            "filename": "contact_sheet_small_person.jpg"
        },
        "high_angle": {
            "title": "V4B Final Crops: High-Angle 4K Surveillance Ceiling View (100 samples)",
            "filter": lambda r: r["viewpoint"] == "4K_CEILING_HIGH_ANGLE",
            "filename": "contact_sheet_high_angle.jpg"
        },
        "hard_negatives": {
            "title": "V4B Final Crops: Desk Interaction Hard Negatives (100 samples)",
            "filter": lambda r: r["original_label"] in ["read", "write", "writing"] and "QUARANTINED" not in r["quality_flags"],
            "filename": "contact_sheet_hard_negatives.jpg"
        },
        "excluded_ambiguous": {
            "title": "V4B Excluded & Quarantined Ambiguous Candidates (100 samples)",
            "filter": lambda r: "QUARANTINED" in r["quality_flags"] or r["ontology_label"] in ["AMBIGUOUS_LOOKUP", "TALKING_CONTEXT", "PHONE_INTERACTION_CONTEXT"],
            "filename": "contact_sheet_excluded_ambiguous.jpg"
        }
    }
    
    # Deterministic sampling
    rng = np.random.RandomState(42)
    
    for key, spec in categories.items():
        matched = [r["normalized_224_path"] for r in records if spec["filter"](r)]
        print(f"Category '{key}': {len(matched)} matched crops.")
        if matched:
            indices = np.arange(len(matched))
            rng.shuffle(indices)
            selected = [matched[i] for i in indices[:100]]
            sheet_img = create_grid(selected, grid_size=(10, 10), thumb_size=(120, 120), title=spec["title"])
            out_path = out_dir / spec["filename"]
            cv2.imwrite(str(out_path), sheet_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"  Saved contact sheet to {out_path}")
            
    print("All contact sheets generated successfully!")

if __name__ == "__main__":
    main()
