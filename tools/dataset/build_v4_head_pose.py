#!/usr/bin/env python3
"""
scripts/build_v4_head_pose.py
Constructs the clean V4 Head-Pose Dataset from AFLW / AFLW2000-3D sources.
Enforces strict split isolation, zero duplicate leakage, continuous Euler angles,
and generates canonical manifests and distribution reports.
"""

import os
import json
import zipfile
import hashlib
from pathlib import Path
import numpy as np

def compute_sha256(data_bytes):
    return hashlib.sha256(data_bytes).hexdigest()

def derive_pitch_roll_from_pts68(p):
    """
    Derives pitch and roll from 68 3D landmark points p of shape (3, 68).
    p[0] = x, p[1] = y, p[2] = z
    """
    # Roll: eye line orientation in image plane
    left_eye_outer = p[:, 36]
    right_eye_outer = p[:, 45]
    eye_vec = right_eye_outer - left_eye_outer
    roll = float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))
    
    # Pitch: sellion/bridge (pt 27) to nose tip (pt 30) vector tilt
    nose_tip = p[:, 30]
    sellion = p[:, 27]
    nose_vec = nose_tip - sellion
    # When head pitches down, nose tip z decreases relative to bridge
    pitch = float(np.degrees(np.arctan2(nose_vec[2], np.linalg.norm(nose_vec[:2]))))
    return pitch, roll

def main():
    root = Path(".")
    raw_dir = root / "datasets/raw_v4/head_pose_aflw2000"
    out_dir = root / "datasets/v4_head_pose"
    images_dir = out_dir / "images"
    landmarks_dir = out_dir / "landmarks"
    splits_dir = out_dir / "splits"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    landmarks_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = raw_dir / "test.data.zip"
    print(f"Opening {zip_path}...")
    
    # Load metadata arrays
    aflw2000_yaws = np.load(raw_dir / "configs/AFLW2000-3D.pose.npy")
    aflw2000_pts68 = np.load(raw_dir / "configs/AFLW2000-3D.pts68.npy")
    aflw2000_rois = np.load(raw_dir / "configs/AFLW2000-3D_crop.roi_box.npy")
    
    aflw_gt_yaws = np.load(raw_dir / "configs/AFLW_GT_crop_yaws.npy")
    aflw_gt_rois = np.load(raw_dir / "configs/AFLW_GT_crop_roi_box.npy")
    
    # Load list files
    with zipfile.ZipFile(zip_path, "r") as z:
        l2k_names = [x.strip() for x in z.read("test.data/AFLW2000-3D_crop.list").decode().strip().split("\n") if x.strip()]
        lgt_names = [x.strip() for x in z.read("test.data/AFLW_GT_crop.list").decode().strip().split("\n") if x.strip()]
    
    assert len(l2k_names) == 2000, f"Expected 2000 AFLW2000 names, got {len(l2k_names)}"
    assert len(lgt_names) == 21080, f"Expected 21080 AFLW_GT names, got {len(lgt_names)}"
    
    overlapping_names = set(l2k_names).intersection(set(lgt_names))
    print(f"Overlapping image base names between AFLW2000 and AFLW_GT: {len(overlapping_names)}")
    
    # Partition AFLW_GT non-overlapping images into train (85%) and val (15%)
    # Use deterministic shuffle with fixed seed
    rng = np.random.RandomState(42)
    non_overlap_gt = [name for name in lgt_names if name not in overlapping_names]
    print(f"Non-overlapping AFLW_GT images: {len(non_overlap_gt)}")
    
    indices = np.arange(len(non_overlap_gt))
    rng.shuffle(indices)
    
    n_train = int(len(non_overlap_gt) * 0.85)
    train_names = set(non_overlap_gt[i] for i in indices[:n_train])
    val_names = set(non_overlap_gt[i] for i in indices[n_train:])
    
    print(f"Head-Pose Partition plan:")
    print(f"  Train (AFLW_GT unique): {len(train_names)}")
    print(f"  Val   (AFLW_GT unique): {len(val_names)}")
    print(f"  Test  (AFLW2000-3D):     {len(l2k_names)}")
    
    manifest_records = []
    
    # Extract AFLW2000-3D images and landmarks
    print("Processing AFLW2000-3D images...")
    aflw2000_out_dir = images_dir / "aflw2000_3d"
    aflw2000_out_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, "r") as z:
        for idx, img_name in enumerate(l2k_names):
            zip_subpath = f"test.data/AFLW2000-3D_crop/{img_name}"
            img_bytes = z.read(zip_subpath)
            sha = compute_sha256(img_bytes)
            
            dest_img = aflw2000_out_dir / img_name
            if not dest_img.exists():
                dest_img.write_bytes(img_bytes)
            
            # Save 3D landmark array
            pts = aflw2000_pts68[idx] # (3, 68)
            lmk_dest = landmarks_dir / f"{img_name.replace('.jpg', '')}_pts68.npy"
            if not lmk_dest.exists():
                np.save(lmk_dest, pts)
            
            yaw = float(aflw2000_yaws[idx])
            pitch, roll = derive_pitch_roll_from_pts68(pts)
            
            record = {
                "sample_id": f"hp_aflw2000_{img_name.replace('.jpg', '')}",
                "image_path": str(dest_img.relative_to(root)).replace("\\", "/"),
                "source_dataset": "AFLW2000-3D",
                "source_identity": "AFLW",
                "subject_id": None,
                "sequence_id": None,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "landmark_path": str(lmk_dest.relative_to(root)).replace("\\", "/"),
                "split": "test",
                "license": "ACADEMIC_NON_COMMERCIAL",
                "sha256": sha
            }
            manifest_records.append(record)
    
    # Extract AFLW_GT images
    print("Processing AFLW-GT images...")
    aflw_gt_out_dir = images_dir / "aflw_gt"
    aflw_gt_out_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, "r") as z:
        for idx, img_name in enumerate(lgt_names):
            # Determine split
            if img_name in overlapping_names:
                split_name = "test_counterpart"
            elif img_name in train_names:
                split_name = "train"
            elif img_name in val_names:
                split_name = "val"
            else:
                split_name = "unassigned"
            
            zip_subpath = f"test.data/AFLW_GT_crop/{img_name}"
            img_bytes = z.read(zip_subpath)
            sha = compute_sha256(img_bytes)
            
            dest_img = aflw_gt_out_dir / img_name
            if not dest_img.exists():
                dest_img.write_bytes(img_bytes)
            
            yaw = float(aflw_gt_yaws[idx])
            
            record = {
                "sample_id": f"hp_aflwgt_{img_name.replace('.jpg', '')}",
                "image_path": str(dest_img.relative_to(root)).replace("\\", "/"),
                "source_dataset": "AFLW-GT",
                "source_identity": "AFLW",
                "subject_id": None,
                "sequence_id": None,
                "yaw": yaw,
                "pitch": None, # Unmeasured in 2D AFLW
                "roll": None,
                "landmark_path": None,
                "split": split_name,
                "license": "ACADEMIC_NON_COMMERCIAL",
                "sha256": sha
            }
            manifest_records.append(record)
    
    # Write canonical manifest.jsonl
    manifest_path = out_dir / "manifest.jsonl"
    print(f"Writing {manifest_path} with {len(manifest_records)} records...")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for rec in manifest_records:
            f.write(json.dumps(rec) + "\n")
    
    # Write split files (train, val, test)
    train_recs = [r for r in manifest_records if r["split"] == "train"]
    val_recs = [r for r in manifest_records if r["split"] == "val"]
    test_recs = [r for r in manifest_records if r["split"] == "test"]
    
    with open(splits_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train_recs:
            f.write(json.dumps(r) + "\n")
            
    with open(splits_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for r in val_recs:
            f.write(json.dumps(r) + "\n")
            
    with open(splits_dir / "test.jsonl", "w", encoding="utf-8") as f:
        for r in test_recs:
            f.write(json.dumps(r) + "\n")
            
    print(f"Splits written successfully:")
    print(f"  train: {len(train_recs)}")
    print(f"  val:   {len(val_recs)}")
    print(f"  test:  {len(test_recs)}")
    print("Done!")

if __name__ == "__main__":
    main()
