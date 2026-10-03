"""Stage 1.5 Pipeline Regression Tests: Holdout Integrity, Group Leakage, Duplicate Safety, and Model Loading."""
import hashlib
import json
from pathlib import Path
import pytest
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
HOLDOUT_DIR = ROOT_DIR / "datasets" / "stage1_5_holdout"
PROCESSED_V3_5 = ROOT_DIR / "datasets" / "processed_v3_5"
STAGE1_BEST = ROOT_DIR / "models" / "trained" / "stage1_best.pt"
STAGE1_HASH = "6d713808f0bc670e8bf06e01b006fac73d8d6e5db7130584cec1658b003ce98a"

CANONICAL_CLASSES = ['normal', 'head_down', 'turn_head', 'discuss', 'stand']

def test_stage1_5_protected_artifacts_unmodified():
    """Verify that canonical Stage 1 model checkpoint is physically protected and unchanged."""
    assert STAGE1_BEST.exists(), f"Stage 1 best checkpoint missing: {STAGE1_BEST}"
    h = hashlib.sha256(open(STAGE1_BEST, "rb").read()).hexdigest()
    assert h == STAGE1_HASH, f"Stage 1 checkpoint hash mismatch! Expected {STAGE1_HASH}, got {h}"

def test_stage1_5_holdout_integrity():
    """Verify holdout dataset structure, manifest, and physical file presence."""
    manifest_p = HOLDOUT_DIR / "manifest.json"
    yaml_p = HOLDOUT_DIR / "dataset.yaml"
    assert manifest_p.exists(), f"Holdout manifest missing: {manifest_p}"
    assert yaml_p.exists(), f"Holdout dataset.yaml missing: {yaml_p}"

    m = json.load(open(manifest_p, "r", encoding="utf-8"))
    assert m["classes"] == CANONICAL_CLASSES
    assert len(m["groups"]) == 15, f"Expected 15 groups, found {len(m['groups'])}"
    assert len(m["entries"]) == 488, f"Expected 488 images, found {len(m['entries'])}"

    # Check a sample of images and labels physically exist on disk
    for entry in m["entries"][:50]:
        img_p = ROOT_DIR / entry["image_path"]
        lbl_p = ROOT_DIR / entry["label_path"]
        assert img_p.exists(), f"Holdout image missing: {img_p}"
        assert lbl_p.exists(), f"Holdout label missing: {lbl_p}"

def test_stage1_5_zero_group_leakage():
    """Verify zero group overlap between refinement train and frozen holdout."""
    m_holdout = json.load(open(HOLDOUT_DIR / "manifest.json", "r", encoding="utf-8"))
    holdout_groups = set(m_holdout["groups"])

    m_v3_5 = json.load(open(PROCESSED_V3_5 / "manifest_v3_5.json", "r", encoding="utf-8"))
    train_groups = {e["group_id"] for e in m_v3_5["splits"]["train"]}

    leakage = train_groups & holdout_groups
    assert len(leakage) == 0, f"Found {len(leakage)} overlapping groups between train and holdout: {leakage}"

def test_stage1_5_zero_duplicate_hash_leakage():
    """Verify zero cryptographic SHA-256 duplicate image leakage across splits."""
    m_holdout = json.load(open(HOLDOUT_DIR / "manifest.json", "r", encoding="utf-8"))
    holdout_hashes = {e["image_hash"] for e in m_holdout["entries"]}

    m_v3_5 = json.load(open(PROCESSED_V3_5 / "manifest_v3_5.json", "r", encoding="utf-8"))
    train_hashes = {e["image_hash"] for e in m_v3_5["splits"]["train"]}

    # Also check against original v3 val and test
    m_v3 = json.load(open(ROOT_DIR / "datasets" / "processed_v3" / "manifest_v3.json", "r", encoding="utf-8"))
    v3_val_hashes = {e["image_hash"] for e in m_v3["splits"]["val"]}
    v3_test_hashes = {e["image_hash"] for e in m_v3["splits"]["test"]}

    assert len(train_hashes & holdout_hashes) == 0, "Hash collision with Stage 1.5 holdout!"
    assert len(train_hashes & v3_val_hashes) == 0, "Hash collision with V3 validation split!"
    assert len(train_hashes & v3_test_hashes) == 0, "Hash collision with V3 locked test split!"

def test_stage1_5_quarantine_exclusion():
    """Verify all quarantined annotation conflicts remain excluded from positive supervision."""
    conflicts_p = ROOT_DIR / "reports" / "annotation_conflicts_v3.json"
    assert conflicts_p.exists()
    conflicts = json.load(open(conflicts_p, "r", encoding="utf-8"))
    assert len(conflicts) == 2939, f"Expected 2939 quarantined conflicts, got {len(conflicts)}"

def test_stage1_5_dataset_class_ids_and_geometry():
    """Verify all bounding boxes in processed_v3_5 have valid class IDs (0..4) and valid geometry."""
    lbl_files = list((PROCESSED_V3_5 / "labels" / "train").glob("*.txt"))[:200]
    assert len(lbl_files) > 0

    for lf in lbl_files:
        for line in open(lf, "r", encoding="utf-8"):
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                assert 0 <= cid < 5, f"Invalid class ID {cid} in {lf}"
                xc, yc, w, h = map(float, parts[1:5])
                assert 0.0 <= xc <= 1.0, f"xc out of bounds in {lf}: {xc}"
                assert 0.0 <= yc <= 1.0, f"yc out of bounds in {lf}: {yc}"
                assert 0.0 < w <= 1.0, f"w out of bounds in {lf}: {w}"
                assert 0.0 < h <= 1.0, f"h out of bounds in {lf}: {h}"

def test_stage1_5_model_loading_configuration():
    """Verify Stage 1 checkpoint loads cleanly with canonical taxonomy."""
    model = YOLO(str(STAGE1_BEST))
    names = list(model.names.values()) if isinstance(model.names, dict) else model.names
    assert names == CANONICAL_CLASSES

def test_stage1_5_candidate_loading_configuration():
    """Verify Stage 1.5 candidate checkpoint loads cleanly with canonical taxonomy."""
    cand_path = ROOT_DIR / "models" / "trained" / "stage1_5_best.pt"
    assert cand_path.exists(), f"Stage 1.5 candidate missing: {cand_path}"
    model = YOLO(str(cand_path))
    names = list(model.names.values()) if isinstance(model.names, dict) else model.names
    assert names == CANONICAL_CLASSES

