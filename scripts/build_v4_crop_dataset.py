#!/usr/bin/env python3
"""
scripts/build_v4_crop_dataset.py
Constructs the clean, leakage-safe V4 Person-Crop Dataset from SCBehavior High-Res
and EduAction video sources.
Generates tight, context, and upper-body crop representations, normalized 224x224
and 320x320 versions, quality metadata, perceptual hashes, and multi-evaluation splits.
"""

import os
import cv2
import json
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image

def compute_sha256(data_bytes):
    return hashlib.sha256(data_bytes).hexdigest()

def compute_dhash(image_gray, hash_size=8):
    """Computes difference hash (dHash) as a hex string."""
    resized = cv2.resize(image_gray, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return hex(int("".join(["1" if b else "0" for b in diff.flatten()]), 2))[2:].zfill(16)

def compute_quality_metrics(img_bgr, bbox_w, bbox_h):
    """Computes blur (Laplacian variance), brightness, and assigns quality flags."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(np.mean(gray))
    
    # Scale bucket
    if bbox_h >= 300:
        scale_bucket = "PERSON_LARGE"
    elif bbox_h >= 180:
        scale_bucket = "PERSON_MEDIUM"
    elif bbox_h >= 120:
        scale_bucket = "PERSON_SMALL"
    else:
        scale_bucket = "PERSON_VERY_SMALL"
        
    flags = []
    if bbox_h < 80 or bbox_w < 50:
        flags.append("VERY_SMALL")
    elif bbox_h < 120:
        flags.append("SMALL")
        
    if laplacian_var < 35.0:
        flags.append("BLURRY")
        
    if mean_brightness < 35.0 or mean_brightness > 235.0:
        flags.append("EXTREME_LIGHTING")
        
    if bbox_h < 60 and laplacian_var < 20.0:
        flags.append("UNRESOLVABLE")
        
    if not flags:
        flags.append("GOOD")
        
    return {
        "blur_score": round(laplacian_var, 2),
        "brightness": round(mean_brightness, 2),
        "scale_bucket": scale_bucket,
        "quality_flags": flags,
        "dhash": compute_dhash(gray)
    }

def main():
    root = Path(".")
    raw_scb = root / "datasets/raw_v4/scbehavior_highres/repo"
    raw_edu = root / "datasets/raw_v4/other_candidates/eduaction"
    
    out_dir = root / "datasets/v4_crop"
    raw_crops_dir = out_dir / "raw_crops"
    norm_224_dir = out_dir / "normalized_224"
    norm_320_dir = out_dir / "normalized_320"
    splits_dir = out_dir / "splits"
    
    for d in [raw_crops_dir, norm_224_dir, norm_320_dir, splits_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    print("=== Starting V4B Person-Crop Dataset Construction ===")
    
    manifest_records = []
    
    # ---------------------------------------------------------
    # 1. SCBehavior High-Res Split Assignment & Crop Extraction
    # ---------------------------------------------------------
    print("\n--- Processing SCBehavior High-Res ---")
    scb_ann_train = json.loads((raw_scb / "SCBehavior_COCO/coco/annotations/instances_train2017.json").read_text(encoding="utf-8"))
    scb_ann_val = json.loads((raw_scb / "SCBehavior_COCO/coco/annotations/instances_val2017.json").read_text(encoding="utf-8"))
    
    cats = {c["id"]: c["name"] for c in scb_ann_train["categories"]}
    
    # Map images: identify viewpoint and assign splits at image level
    all_scb_imgs = scb_ann_train["images"] + scb_ann_val["images"]
    all_scb_anns = scb_ann_train["annotations"] + scb_ann_val["annotations"]
    
    print(f"Total SCBehavior images: {len(all_scb_imgs)}, Total annotations: {len(all_scb_anns)}")
    
    # Image ID to filepath mapping
    img_id_to_path = {}
    for im in all_scb_imgs:
        f_name = im["file_name"]
        p1 = raw_scb / "SCBehavior_COCO/coco/train2017" / f_name
        p2 = raw_scb / "SCBehavior_COCO/coco/val2017" / f_name
        if p1.exists():
            img_id_to_path[im["id"]] = (im, p1)
        elif p2.exists():
            img_id_to_path[im["id"]] = (im, p2)
            
    # Deterministic split assignment for SCBehavior images
    # Resolution (2560, 1440) -> QHD_FRONT_OBLIQUE (232 images)
    # Resolution (3840, 2160) -> 4K_CEILING_HIGH_ANGLE (168 images)
    rng = np.random.RandomState(42)
    
    qhd_ids = [im["id"] for im in all_scb_imgs if im["width"] == 2560]
    uhd_ids = [im["id"] for im in all_scb_imgs if im["width"] == 3840]
    
    rng.shuffle(qhd_ids)
    rng.shuffle(uhd_ids)
    
    # UHD high angle: hold out 40 images for high_angle_holdout!
    uhd_holdout_ids = set(uhd_ids[:40])
    uhd_val_ids = set(uhd_ids[40:65])
    uhd_train_ids = set(uhd_ids[65:])
    
    qhd_val_ids = set(qhd_ids[:35])
    qhd_train_ids = set(qhd_ids[35:])
    
    scb_img_splits = {}
    for iid in uhd_holdout_ids:
        scb_img_splits[iid] = "high_angle_holdout"
    for iid in uhd_val_ids:
        scb_img_splits[iid] = "same_domain_val"
    for iid in uhd_train_ids:
        scb_img_splits[iid] = "train"
        
    for iid in qhd_val_ids:
        scb_img_splits[iid] = "same_domain_val"
    for iid in qhd_train_ids:
        scb_img_splits[iid] = "train"
        
    print(f"SCBehavior Image Split Allocation:")
    print(f"  Train:              {len([i for i, s in scb_img_splits.items() if s == 'train'])} images")
    print(f"  Same-Domain Val:    {len([i for i, s in scb_img_splits.items() if s == 'same_domain_val'])} images")
    print(f"  High-Angle Holdout: {len([i for i, s in scb_img_splits.items() if s == 'high_angle_holdout'])} images")
    
    # Cache loaded images to avoid reading from disk repeatedly
    img_cache = {}
    
    # Process annotations
    for idx, ann in enumerate(all_scb_anns):
        img_id = ann["image_id"]
        im_meta, img_path = img_id_to_path[img_id]
        split_name = scb_img_splits[img_id]
        
        cname = cats[ann["category_id"]]
        
        # Determine ontology label and future canonical target
        if cname == "turn_head":
            ont_label = "TURN_HEAD_CLEAR"
            canonical_target = "turn_head"
            quarantine = False
        elif cname in ["read", "write"]:
            ont_label = "NORMAL_READ_WRITE"
            canonical_target = "normal"
            quarantine = False
        elif cname == "lookup":
            # Physically inspect box size and aspect ratio
            bx, by, bw, bh = [int(v) for v in ann["bbox"]]
            if bh >= 160:
                ont_label = "NORMAL_UPRIGHT"
                canonical_target = "normal"
                quarantine = False
            else:
                ont_label = "AMBIGUOUS_LOOKUP"
                canonical_target = "UNDECIDED"
                quarantine = True
        elif cname == "raise_hand":
            ont_label = "NORMAL_UPRIGHT"
            canonical_target = "normal"
            quarantine = False
        elif cname == "discuss":
            ont_label = "DISCUSS_PAIR"
            canonical_target = "discuss"
            quarantine = True # Macro full-frame behavior
        elif cname == "stand":
            ont_label = "STAND_MACRO"
            canonical_target = "stand"
            quarantine = True # Macro mobility behavior
        else:
            ont_label = "AMBIGUOUS"
            canonical_target = "UNDECIDED"
            quarantine = True
            
        # Load source image
        if img_id not in img_cache:
            if len(img_cache) > 20: # keep memory low
                img_cache.clear()
            img_bgr = cv2.imread(str(img_path))
            img_cache[img_id] = img_bgr
        else:
            img_bgr = img_cache[img_id]
            
        H_img, W_img = img_bgr.shape[:2]
        
        # Bbox coordinates [x, y, w, h]
        bx, by, bw, bh = [int(round(v)) for v in ann["bbox"]]
        # Clamp to bounds
        bx = max(0, min(W_img - 1, bx))
        by = max(0, min(H_img - 1, by))
        bw = max(1, min(W_img - bx, bw))
        bh = max(1, min(H_img - by, bh))
        
        if bw < 10 or bh < 10:
            continue
            
        viewpoint = "4K_CEILING_HIGH_ANGLE" if W_img == 3840 else "QHD_FRONT_OBLIQUE"
        
        # 1. Representation A: TIGHT_PERSON_CROP (3% minimal boundary padding)
        pad_t_x = int(round(0.03 * bw))
        pad_t_y = int(round(0.03 * bh))
        tx1 = max(0, bx - pad_t_x)
        ty1 = max(0, by - pad_t_y)
        tx2 = min(W_img, bx + bw + pad_t_x)
        ty2 = min(H_img, by + bh + pad_t_y)
        tight_crop = img_bgr[ty1:ty2, tx1:tx2]
        
        # 2. Representation B: CONTEXT_PERSON_CROP (15% context padding)
        pad_c_x = int(round(0.15 * bw))
        pad_c_y = int(round(0.15 * bh))
        cx1 = max(0, bx - pad_c_x)
        cy1 = max(0, by - pad_c_y)
        cx2 = min(W_img, bx + bw + pad_c_x)
        cy2 = min(H_img, by + bh + pad_c_y)
        context_crop = img_bgr[cy1:cy2, cx1:cx2]
        
        # 3. Representation C: UPPER_BODY_CROP (approx top 60% of person box)
        ux1 = max(0, bx - int(round(0.05 * bw)))
        uy1 = max(0, by - int(round(0.03 * bh)))
        ux2 = min(W_img, bx + bw + int(round(0.05 * bw)))
        uy2 = min(H_img, by + int(round(0.60 * bh)))
        upper_body_crop = img_bgr[uy1:uy2, ux1:ux2]
        upper_body_available = (uy2 - uy1) >= 40 and (ux2 - ux1) >= 30
        
        # Quality metrics
        q_metrics = compute_quality_metrics(tight_crop, bw, bh)
        if quarantine:
            q_metrics["quality_flags"].append("QUARANTINED")
            
        sample_base_id = f"scb_{im_meta['file_name'].replace('.jpg', '')}_ann{ann['id']:05d}"
        
        # Generate tight crop files
        tight_fname = f"{sample_base_id}_tight.jpg"
        raw_tight_path = raw_crops_dir / tight_fname
        norm224_tight_path = norm_224_dir / tight_fname
        norm320_tight_path = norm_320_dir / tight_fname
        
        cv2.imwrite(str(raw_tight_path), tight_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Normalized 224x224 and 320x320
        norm224 = cv2.resize(tight_crop, (224, 224), interpolation=cv2.INTER_AREA if min(tight_crop.shape[:2]) >= 224 else cv2.INTER_CUBIC)
        norm320 = cv2.resize(tight_crop, (320, 320), interpolation=cv2.INTER_AREA if min(tight_crop.shape[:2]) >= 320 else cv2.INTER_CUBIC)
        cv2.imwrite(str(norm224_tight_path), norm224, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(norm320_tight_path), norm320, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Compute SHA256 of saved raw crop
        raw_sha = compute_sha256(raw_tight_path.read_bytes())
        
        rec_tight = {
            "sample_id": f"{sample_base_id}_tight",
            "image_path": str(raw_tight_path.relative_to(root)).replace("\\", "/"),
            "normalized_224_path": str(norm224_tight_path.relative_to(root)).replace("\\", "/"),
            "normalized_320_path": str(norm320_tight_path.relative_to(root)).replace("\\", "/"),
            "source_dataset": "SCBehavior-HighRes",
            "source_file": im_meta["file_name"],
            "source_image_id": im_meta["id"],
            "source_clip_id": None,
            "source_frame_index": None,
            "timestamp": None,
            "recording_group": f"scb_view_{viewpoint.lower()}",
            "camera_id": viewpoint,
            "viewpoint": viewpoint,
            "original_label": cname,
            "ontology_label": ont_label,
            "future_canonical_label": canonical_target,
            "split": split_name,
            "bbox_original": [bx, by, bw, bh],
            "bbox_padded": [tx1, ty1, tx2 - tx1, ty2 - ty1],
            "crop_type": "TIGHT_PERSON_CROP",
            "person_width": bw,
            "person_height": bh,
            "crop_width": tx2 - tx1,
            "crop_height": ty2 - ty1,
            "blur_score": q_metrics["blur_score"],
            "brightness": q_metrics["brightness"],
            "scale_bucket": q_metrics["scale_bucket"],
            "quality_flags": q_metrics["quality_flags"],
            "head_crop_available": bh >= 140,
            "upper_body_crop_available": upper_body_available,
            "face_crop_available": bh >= 200,
            "license_status": "ACADEMIC_ONLY",
            "sha256": raw_sha,
            "dhash": q_metrics["dhash"]
        }
        manifest_records.append(rec_tight)
        
        # Save context crop
        context_fname = f"{sample_base_id}_context.jpg"
        raw_context_path = raw_crops_dir / context_fname
        norm224_context_path = norm_224_dir / context_fname
        norm320_context_path = norm_320_dir / context_fname
        
        cv2.imwrite(str(raw_context_path), context_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        norm224_c = cv2.resize(context_crop, (224, 224), interpolation=cv2.INTER_AREA if min(context_crop.shape[:2]) >= 224 else cv2.INTER_CUBIC)
        norm320_c = cv2.resize(context_crop, (320, 320), interpolation=cv2.INTER_AREA if min(context_crop.shape[:2]) >= 320 else cv2.INTER_CUBIC)
        cv2.imwrite(str(norm224_context_path), norm224_c, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(norm320_context_path), norm320_c, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        rec_context = dict(rec_tight)
        rec_context["sample_id"] = f"{sample_base_id}_context"
        rec_context["image_path"] = str(raw_context_path.relative_to(root)).replace("\\", "/")
        rec_context["normalized_224_path"] = str(norm224_context_path.relative_to(root)).replace("\\", "/")
        rec_context["normalized_320_path"] = str(norm320_context_path.relative_to(root)).replace("\\", "/")
        rec_context["bbox_padded"] = [cx1, cy1, cx2 - cx1, cy2 - cy1]
        rec_context["crop_type"] = "CONTEXT_PERSON_CROP"
        rec_context["crop_width"] = cx2 - cx1
        rec_context["crop_height"] = cy2 - cy1
        rec_context["sha256"] = compute_sha256(raw_context_path.read_bytes())
        manifest_records.append(rec_context)
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(all_scb_anns)} SCBehavior annotations...")
            
    print(f"Completed SCBehavior: generated {len(manifest_records)} crop entries (tight + context).")
    
    # ---------------------------------------------------------
    # 2. EduAction Temporal Sampling & Crop Generation
    # ---------------------------------------------------------
    print("\n--- Processing EduAction Video Clips ---")
    edu_categories = {
        "sleeping": ("HEAD_REST_SLEEP", "head_down", False),
        "writing": ("NORMAL_READ_WRITE", "normal", False),
        "lecture": ("NORMAL_UPRIGHT", "normal", False),
        "talking": ("TALKING_CONTEXT", "discuss", True), # Quarantined from posture class
        "play_phone": ("PHONE_INTERACTION_CONTEXT", "OBJECT_RULE", True), # Quarantined from posture class
        "drinking": ("DRINKING_CONTEXT", "normal", True),
        "watch_computer": ("COMPUTER_CONTEXT", "normal", True)
    }
    
    edu_records_count = 0
    
    for cat_name, (ont_label, canon_target, is_quarantined) in edu_categories.items():
        cat_dir = raw_edu / cat_name
        mp4s = sorted(cat_dir.glob("*.mp4"))
        assert len(mp4s) == 50, f"Expected 50 clips for {cat_name}, got {len(mp4s)}"
        
        # Partition 50 clips:
        # 30 clips -> train
        # 8 clips  -> same_domain_val
        # 6 clips  -> cross_source_holdout (external generalization test)
        # 6 clips  -> temporal_holdout (completely unseen temporal evaluation)
        rng_cat = np.random.RandomState(100 + len(cat_name))
        clip_indices = np.arange(50)
        rng_cat.shuffle(clip_indices)
        
        train_clip_ids = set(clip_indices[:30])
        val_clip_ids = set(clip_indices[30:38])
        cross_clip_ids = set(clip_indices[38:44])
        temporal_clip_ids = set(clip_indices[44:50])
        
        print(f"Sampling {cat_name} (50 clips, 30 train, 8 val, 6 cross-source, 6 temporal)...")
        
        for c_idx, mp4_path in enumerate(mp4s):
            if c_idx in train_clip_ids:
                clip_split = "train"
            elif c_idx in val_clip_ids:
                clip_split = "same_domain_val"
            elif c_idx in cross_clip_ids:
                clip_split = "cross_source_holdout"
            elif c_idx in temporal_clip_ids:
                clip_split = "temporal_holdout"
            else:
                clip_split = "unassigned"
                
            cap = cv2.VideoCapture(str(mp4_path))
            raw_frames = []
            frame_i = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                raw_frames.append((frame_i, frame))
                frame_i += 1
            cap.release()
            
            n_frames = len(raw_frames)
            if n_frames == 0:
                continue
                
            stride = max(5, n_frames // 12)
            last_gray = None
            last_accepted_idx = -999
            
            clean_clip_name = mp4_path.name.replace(" ", "_").replace("(", "").replace(")", "").replace(".mp4", "")
            
            for f_idx, frame in raw_frames[::stride]:
                gray_small = cv2.cvtColor(cv2.resize(frame, (112, 112)), cv2.COLOR_BGR2GRAY)
                accept = False
                if last_gray is None:
                    accept = True
                else:
                    diff = float(np.mean(cv2.absdiff(gray_small, last_gray)) / 255.0)
                    if diff >= 0.003 or (f_idx - last_accepted_idx) >= 25:
                        accept = True
                        
                if accept:
                    last_gray = gray_small
                    last_accepted_idx = f_idx
                    
                    sample_id = f"edu_{cat_name}_{clean_clip_name}_f{f_idx:04d}"
                    raw_fname = f"{sample_id}.jpg"
                    
                    raw_fpath = raw_crops_dir / raw_fname
                    norm224_fpath = norm_224_dir / raw_fname
                    norm320_fpath = norm_320_dir / raw_fname
                    
                    # EduAction frames are already 224x224 person crops
                    cv2.imwrite(str(raw_fpath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    cv2.imwrite(str(norm224_fpath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    norm320_frame = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(str(norm320_fpath), norm320_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    q_metrics = compute_quality_metrics(frame, 224, 224)
                    if is_quarantined:
                        q_metrics["quality_flags"].append("QUARANTINED")
                        
                    raw_sha = compute_sha256(raw_fpath.read_bytes())
                    
                    rec = {
                        "sample_id": sample_id,
                        "image_path": str(raw_fpath.relative_to(root)).replace("\\", "/"),
                        "normalized_224_path": str(norm224_fpath.relative_to(root)).replace("\\", "/"),
                        "normalized_320_path": str(norm320_fpath.relative_to(root)).replace("\\", "/"),
                        "source_dataset": "EduAction",
                        "source_file": mp4_path.name,
                        "source_image_id": None,
                        "source_clip_id": mp4_path.name,
                        "source_frame_index": f_idx,
                        "timestamp": round(f_idx / 25.0, 3),
                        "recording_group": f"edu_{cat_name}_{clean_clip_name}",
                        "camera_id": "EDU_CROP_CAMERA",
                        "viewpoint": "FRONTAL_DESK_LEVEL",
                        "original_label": cat_name,
                        "ontology_label": ont_label,
                        "future_canonical_label": canon_target,
                        "split": clip_split,
                        "bbox_original": [0, 0, 224, 224],
                        "bbox_padded": [0, 0, 224, 224],
                        "crop_type": "PRE_ISOLATED_PERSON_CROP",
                        "person_width": 224,
                        "person_height": 224,
                        "crop_width": 224,
                        "crop_height": 224,
                        "blur_score": q_metrics["blur_score"],
                        "brightness": q_metrics["brightness"],
                        "scale_bucket": q_metrics["scale_bucket"],
                        "quality_flags": q_metrics["quality_flags"],
                        "head_crop_available": True,
                        "upper_body_crop_available": True,
                        "face_crop_available": True,
                        "license_status": "ACADEMIC_ONLY",
                        "sha256": raw_sha,
                        "dhash": q_metrics["dhash"]
                    }
                    manifest_records.append(rec)
                    edu_records_count += 1
                    
    print(f"Completed EduAction: sampled {edu_records_count} diverse frames across 350 clips.")
    
    # ---------------------------------------------------------
    # 3. Write Manifest & Partition Splits
    # ---------------------------------------------------------
    manifest_path = out_dir / "manifest.jsonl"
    print(f"\nWriting canonical manifest to {manifest_path} ({len(manifest_records)} records)...")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in manifest_records:
            f.write(json.dumps(r) + "\n")
            
    print("Partitioning split files...")
    splits_data = {
        "train": [],
        "same_domain_val": [],
        "cross_source_holdout": [],
        "high_angle_holdout": [],
        "temporal_holdout": []
    }
    
    for r in manifest_records:
        s = r["split"]
        if s in splits_data:
            splits_data[s].append(r)
            
    for s_name, recs in splits_data.items():
        s_path = splits_dir / f"{s_name}.jsonl"
        with open(s_path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"  {s_name:22}: {len(recs):6d} crops")
        
    print("\n=== V4B Person-Crop Dataset Construction Finished Successfully ===")

if __name__ == "__main__":
    main()
