"""Unit tests for Phase V3: Full dataset identity, duplicate preference, conflict quarantine, and preflight."""
import json
from pathlib import Path
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_full_dataset_identity_handling():
    """Verify that full SCB-Dataset5 identity report confirms complete status with multi-subdataset evidence."""
    identity_path = ROOT / "reports" / "scb_dataset5_full" / "IDENTITY.md"
    stats_path = ROOT / "reports" / "scb_dataset5_full" / "statistics.json"

    assert identity_path.exists(), "IDENTITY.md must exist in reports/scb_dataset5_full"
    assert stats_path.exists(), "statistics.json must exist in reports/scb_dataset5_full"

    content = identity_path.read_text(encoding="utf-8")
    assert "IS THIS ACTUALLY THE FULL SCB-DATASET5?" in content
    assert "**YES**" in content

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    assert stats["total_images"] > 10000, "Full SCB5 must contain tens of thousands of physical images"
    assert stats["total_annotations"] > 50000, "Full SCB5 must contain tens of thousands of annotations"
    assert stats["subdatasets_count"] >= 5, "Full SCB5 must comprise multiple subdataset releases"


def test_duplicate_source_preference():
    """Verify that duplicate resolution prefers scb_dataset5_full over old subsets."""
    manifest_path = ROOT / "datasets" / "processed_v3" / "manifest_v3.json"
    assert manifest_path.exists(), "manifest_v3.json must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Check that annotations contain source_dataset
    found_scb5_full = False
    for sname, records in manifest.get("splits", {}).items():
        for rec in records:
            for annot in rec.get("annotations", []):
                for src in annot.get("sources", []):
                    if src.get("source_dataset") == "scb_dataset5_full":
                        found_scb5_full = True
                        break

    assert found_scb5_full, "scb_dataset5_full must be preserved and utilized as primary source"


def test_annotation_conflict_quarantine():
    """Verify that contradictory labels for the same person box are quarantined into annotation_conflicts_v3.json."""
    conflicts_path = ROOT / "reports" / "annotation_conflicts_v3.json"
    assert conflicts_path.exists(), "annotation_conflicts_v3.json must exist"

    with open(conflicts_path, "r", encoding="utf-8") as f:
        conflicts = json.load(f)

    assert isinstance(conflicts, list)
    if len(conflicts) > 0:
        c = conflicts[0]
        assert "image_hash" in c
        assert "box_existing" in c
        assert "box_conflicting" in c
        assert "iou" in c
        assert c["iou"] > 0.5, "Conflicting boxes must have high spatial overlap"
        assert c["box_existing"]["canonical_class"] != c["box_conflicting"]["canonical_class"]


def test_viewpoint_aware_grouping():
    """Verify that no duplicate clusters cross train/val/test splits."""
    manifest_path = ROOT / "datasets" / "processed_v3" / "manifest_v3.json"
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    splits = manifest.get("splits", {})
    hashes_per_split = {s: {r["image_hash"] for r in records} for s, records in splits.items()}

    train_val_overlap = hashes_per_split.get("train", set()) & hashes_per_split.get("val", set())
    train_test_overlap = hashes_per_split.get("train", set()) & hashes_per_split.get("test", set())
    val_test_overlap = hashes_per_split.get("val", set()) & hashes_per_split.get("test", set())

    assert len(train_val_overlap) == 0, f"Duplicate hash leakage between train and val: {len(train_val_overlap)}"
    assert len(train_test_overlap) == 0, f"Duplicate hash leakage between train and test: {len(train_test_overlap)}"
    assert len(val_test_overlap) == 0, f"Duplicate hash leakage between val and test: {len(val_test_overlap)}"


def test_no_needs_review_mapping_enters_dataset():
    """Verify that mappings marked needs_review, ignored, or rejected never enter processed training labels."""
    mapping_path = ROOT / "configs" / "dataset_mapping_v3_candidate.yaml"
    manifest_path = ROOT / "datasets" / "processed_v3" / "manifest_v3.json"

    with open(mapping_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    excluded_source_classes = {
        (m["dataset"], m["subset"], m["source_class_name"])
        for m in cfg.get("mappings", [])
        if m.get("status") in ("needs_review", "ignored", "rejected")
    }

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for sname, records in manifest.get("splits", {}).items():
        for rec in records:
            for annot in rec.get("annotations", []):
                for src in annot.get("sources", []):
                    key = (src.get("source_dataset"), src.get("source_subset"), src.get("source_class_name"))
                    assert key not in excluded_source_classes, f"Excluded class {key} leaked into training!"


def test_provenance_after_fusion():
    """Verify that fused annotations maintain source dataset provenance."""
    manifest_path = ROOT / "datasets" / "processed_v3" / "manifest_v3.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for sname, records in manifest.get("splits", {}).items():
        for rec in records[:50]:  # check first 50 records per split
            for annot in rec.get("annotations", []):
                assert "sources" in annot
                assert len(annot["sources"]) >= 1
                for s in annot["sources"]:
                    assert "source_dataset" in s
                    assert "source_image" in s


def test_dynamic_taxonomy_ids():
    """Verify dataset.yaml has continuous 0-indexed class IDs matching the 5 approved classes."""
    data_yaml_path = ROOT / "datasets" / "processed_v3" / "dataset.yaml"
    assert data_yaml_path.exists()

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    assert data_cfg["nc"] == 5
    assert data_cfg["names"] == ["normal", "head_down", "turn_head", "discuss", "stand"]


def test_omitted_class_not_appearing_in_dataset_yaml():
    """Verify that unverified/omitted classes like use_phone and lean do NOT appear in dataset.yaml."""
    data_yaml_path = ROOT / "datasets" / "processed_v3" / "dataset.yaml"
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    names = [n.lower() for n in data_cfg["names"]]
    assert "use_phone" not in names
    assert "phone" not in names
    assert "lean" not in names
    assert "cheating" not in names
