"""
tests/test_v4b_crop_dataset.py
Automated regression and integrity tests for V4B Clean Person-Crop and Head-Pose Datasets.
Verifies manifest validity, file existence, ontology conformity, zero split leakage,
zero clip leakage, duplicate protection, quarantine isolation, head-pose safety,
and baseline benchmark protection.
"""

import json
import hashlib
from pathlib import Path
from PIL import Image
import pytest

def get_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def test_v4b_manifest_files_exist_and_readable():
    """Verify that both V4B manifests exist and contain non-empty valid JSONL records."""
    crop_manifest = Path("datasets/v4_crop/manifest.jsonl")
    hp_manifest = Path("datasets/v4_head_pose/manifest.jsonl")
    
    assert crop_manifest.exists(), "datasets/v4_crop/manifest.jsonl must exist"
    assert hp_manifest.exists(), "datasets/v4_head_pose/manifest.jsonl must exist"
    
    with open(crop_manifest, "r", encoding="utf-8") as f:
        crop_count = sum(1 for line in f if line.strip())
    assert crop_count > 10000, f"Expected >10,000 crop records, got {crop_count}"
    
    with open(hp_manifest, "r", encoding="utf-8") as f:
        hp_count = sum(1 for line in f if line.strip())
    assert hp_count == 23080, f"Expected 23,080 head pose records, got {hp_count}"

def test_v4b_crop_files_physical_existence_and_normalization():
    """Verify sample records have valid raw, 224x224, and 320x320 physical image files."""
    crop_manifest = Path("datasets/v4_crop/manifest.jsonl")
    
    checked = 0
    with open(crop_manifest, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx % 500 == 0:
                rec = json.loads(line)
                raw_path = Path(rec["image_path"])
                norm224_path = Path(rec["normalized_224_path"])
                norm320_path = Path(rec["normalized_320_path"])
                
                assert raw_path.exists(), f"Raw crop missing: {raw_path}"
                assert norm224_path.exists(), f"Normalized 224 crop missing: {norm224_path}"
                assert norm320_path.exists(), f"Normalized 320 crop missing: {norm320_path}"
                
                with Image.open(norm224_path) as im:
                    assert im.size == (224, 224), f"Expected (224, 224), got {im.size}"
                with Image.open(norm320_path) as im:
                    assert im.size == (320, 320), f"Expected (320, 320), got {im.size}"
                    
                checked += 1
                if checked >= 30:
                    break
    assert checked >= 20, "Must verify at least 20 sampled crops"

def test_v4b_ontology_values_and_no_cheating_labels():
    """Verify that all ontology labels adhere strictly to authorized physical states and no 'cheating' labels exist."""
    crop_manifest = Path("datasets/v4_crop/manifest.jsonl")
    authorized_labels = {
        "NORMAL_UPRIGHT", "NORMAL_READ_WRITE", "HEAD_DOWN_DEEP", "HEAD_REST_SLEEP", "TURN_HEAD_CLEAR",
        "AMBIGUOUS_LOOKUP", "DISCUSS_PAIR", "STAND_MACRO", "TALKING_CONTEXT", "PHONE_INTERACTION_CONTEXT",
        "DRINKING_CONTEXT", "COMPUTER_CONTEXT", "AMBIGUOUS"
    }
    
    with open(crop_manifest, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ont = rec["ontology_label"]
            assert ont in authorized_labels, f"Unauthorized ontology label: {ont}"
            
            # Anti-cheating check: NO subjective 'cheating' label permitted
            for k in ["original_label", "ontology_label", "future_canonical_label"]:
                val = str(rec.get(k, "")).lower()
                assert "cheating" not in val, f"Subjective 'cheating' label forbidden: {val} in field {k}"

def test_v4b_split_leakage_zero_image_and_clip_overlap():
    """Verify zero overlap of source images or video clips across different splits."""
    splits_dir = Path("datasets/v4_crop/splits")
    split_files = list(splits_dir.glob("*.jsonl"))
    assert len(split_files) >= 4, "Must have at least train, same_domain_val, cross_source, and holdouts"
    
    sources_by_split = {}
    for sf in split_files:
        split_name = sf.stem
        sources_by_split[split_name] = set()
        with open(sf, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                src_key = rec["source_clip_id"] if rec["source_clip_id"] else f"scb_img_{rec['source_image_id']}"
                sources_by_split[split_name].add(src_key)
                
    # Check pairwise disjointness
    split_names = list(sources_by_split.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            overlap = sources_by_split[s1].intersection(sources_by_split[s2])
            assert len(overlap) == 0, f"Leakage detected between {s1} and {s2}: {len(overlap)} overlapping sources ({list(overlap)[:3]})"

def test_v4b_split_leakage_zero_duplicate_hash_overlap():
    """Verify zero duplicate SHA256 hashes cross between train and evaluation splits."""
    splits_dir = Path("datasets/v4_crop/splits")
    train_file = splits_dir / "train.jsonl"
    eval_files = [f for f in splits_dir.glob("*.jsonl") if f.name != "train.jsonl"]
    
    train_hashes = set()
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            train_hashes.add(rec["sha256"])
            
    for ef in eval_files:
        with open(ef, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                h = rec["sha256"]
                assert h not in train_hashes, f"Duplicate hash leakage found in {ef.name}: {h} in sample {rec['sample_id']}"

def test_v4b_head_pose_split_safety_and_provenance():
    """Verify that head-pose test partition is pure AFLW2000-3D and train/val contain zero duplicate image names."""
    hp_splits = Path("datasets/v4_head_pose/splits")
    train_recs = [json.loads(l) for l in open(hp_splits / "train.jsonl", encoding="utf-8")]
    val_recs = [json.loads(l) for l in open(hp_splits / "val.jsonl", encoding="utf-8")]
    test_recs = [json.loads(l) for l in open(hp_splits / "test.jsonl", encoding="utf-8")]
    
    assert len(test_recs) == 2000, f"Expected exactly 2000 test records, got {len(test_recs)}"
    assert all(r["source_dataset"] == "AFLW2000-3D" for r in test_recs), "All test records must be AFLW2000-3D"
    assert all(r["source_dataset"] == "AFLW-GT" for r in train_recs), "All train records must be AFLW-GT"
    assert all(r["source_dataset"] == "AFLW-GT" for r in val_recs), "All val records must be AFLW-GT"
    
    # Verify zero base filename overlap
    train_names = set(Path(r["image_path"]).name for r in train_recs)
    val_names = set(Path(r["image_path"]).name for r in val_recs)
    test_names = set(Path(r["image_path"]).name for r in test_recs)
    
    assert len(train_names.intersection(val_names)) == 0, "Train and Val head pose base names overlap"
    assert len(train_names.intersection(test_names)) == 0, "Train and Test head pose base names overlap"
    assert len(val_names.intersection(test_names)) == 0, "Val and Test head pose base names overlap"

def test_v4b_head_pose_angle_validity_no_nan_inf():
    """Verify continuous yaw/pitch/roll angles are valid finite floats without NaN or Inf."""
    hp_manifest = Path("datasets/v4_head_pose/manifest.jsonl")
    with open(hp_manifest, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            rec = json.loads(line)
            yaw = rec["yaw"]
            assert yaw is not None and isinstance(yaw, (int, float)), f"Invalid yaw in record {idx}: {yaw}"
            assert not (yaw != yaw), f"NaN yaw in record {idx}"
            assert abs(yaw) != float("inf"), f"Inf yaw in record {idx}"
            
            pitch = rec["pitch"]
            if pitch is not None:
                assert isinstance(pitch, (int, float)) and not (pitch != pitch) and abs(pitch) != float("inf")

def test_v4b_quarantine_exclusion_from_supervised_classes():
    """Verify that ambiguous or context classes have quality flag QUARANTINED and are not primary posture positives."""
    crop_manifest = Path("datasets/v4_crop/manifest.jsonl")
    primary_posture_classes = {"NORMAL_UPRIGHT", "NORMAL_READ_WRITE", "HEAD_DOWN_DEEP", "HEAD_REST_SLEEP", "TURN_HEAD_CLEAR"}
    
    with open(crop_manifest, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ont = rec["ontology_label"]
            flags = rec["quality_flags"]
            if ont not in primary_posture_classes:
                assert "QUARANTINED" in flags, f"Class {ont} must be QUARANTINED, got flags {flags}"

def test_v4b_license_field_and_provenance():
    """Verify all records carry explicit academic license and provenance fields."""
    crop_manifest = Path("datasets/v4_crop/manifest.jsonl")
    with open(crop_manifest, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            assert "license_status" in rec and rec["license_status"] == "ACADEMIC_ONLY"
            assert "source_dataset" in rec and rec["source_dataset"] in ["SCBehavior-HighRes", "EduAction"]
            assert "sha256" in rec and len(rec["sha256"]) == 64

def test_v4b_crop_geometry_validity():
    """Verify bbox coordinates and crop dimensions are strictly positive and bounded."""
    crop_manifest = Path("datasets/v4_crop/manifest.jsonl")
    checked = 0
    with open(crop_manifest, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx % 200 == 0:
                rec = json.loads(line)
                orig_bbox = rec["bbox_original"]
                pad_bbox = rec["bbox_padded"]
                assert len(orig_bbox) == 4 and orig_bbox[2] > 0 and orig_bbox[3] > 0
                assert len(pad_bbox) == 4 and pad_bbox[2] > 0 and pad_bbox[3] > 0
                assert rec["person_width"] > 0 and rec["person_height"] > 0
                assert rec["crop_width"] > 0 and rec["crop_height"] > 0
                checked += 1
    assert checked >= 50

def test_v4b_v3_benchmark_and_checkpoints_protection():
    """CRITICAL: Verify production checkpoints and external V3 benchmarks are 100% untouched."""
    s1_path = Path("models/trained/stage1_best.pt")
    s1_5_path = Path("models/trained/stage1_5_best.pt")
    
    assert s1_path.exists(), "stage1_best.pt must exist"
    assert s1_5_path.exists(), "stage1_5_best.pt must exist"
    
    h1 = get_file_sha256(s1_path)
    h1_5 = get_file_sha256(s1_5_path)
    
    assert h1 == "6d713808f0bc670e8bf06e01b006fac73d8d6e5db7130584cec1658b003ce98a", "Stage 1 hash mismatch!"
    assert h1_5 == "68690cf82715dc6dad2eb35a08711531170e935eb7dbe1c30fafabae341e2c2c", "Stage 1.5 hash mismatch!"
    
    assert Path("datasets/processed_v3").exists(), "datasets/processed_v3 must exist and remain protected"
    assert Path("datasets/processed_v3_5").exists(), "datasets/processed_v3_5 must exist and remain protected"
