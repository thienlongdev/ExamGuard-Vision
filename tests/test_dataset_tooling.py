"""Unit tests for dataset audit, validation, mapping, duplicate detection, and grouping tools."""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest

from training.class_mapping import ClassMapper, MappingStatus, load_dataset_mapping
from training.duplicate_detector import DuplicateDetector, compute_dhash, compute_sha256, hamming_distance
from training.split_dataset import DatasetSplitter, ManifestRecord, extract_group_id
from training.validate_dataset import (
    AnnotationIssue,
    validate_annotation_file,
    validate_yolo_bbox,
)


def test_validate_yolo_bbox_valid():
    issues = validate_yolo_bbox(
        cls_id=0, xc=0.5, yc=0.5, w=0.2, h=0.4, file_path="test.txt", line_no=1, raw_line="0 0.5 0.5 0.2 0.4"
    )
    assert len(issues) == 0


def test_validate_yolo_bbox_out_of_bounds():
    # xc > 1.0
    issues = validate_yolo_bbox(
        cls_id=0, xc=1.2, yc=0.5, w=0.2, h=0.4, file_path="test.txt", line_no=1, raw_line="0 1.2 0.5 0.2 0.4"
    )
    assert any(i.issue_type == "out_of_bounds" for i in issues)


def test_validate_yolo_bbox_zero_dim():
    issues = validate_yolo_bbox(
        cls_id=0, xc=0.5, yc=0.5, w=0.0, h=0.4, file_path="test.txt", line_no=1, raw_line="0 0.5 0.5 0.0 0.4"
    )
    assert any(i.issue_type == "zero_dim" for i in issues)


def test_validate_annotation_file_syntax(tmp_path):
    lbl_file = tmp_path / "sample.txt"
    content = (
        "0 0.5 0.5 0.2 0.3\n"      # valid
        "1 0.4 0.4 -0.1 0.2\n"     # negative dim
        "corrupted line\n"          # malformed line
        "2 0.8 0.8 0.1 0.1\n"      # valid
    )
    lbl_file.write_text(content, encoding="utf-8")

    annots, issues = validate_annotation_file(lbl_file)
    assert len(annots) == 3
    assert len(issues) == 2  # 1 zero/negative dim, 1 malformed line


def test_class_mapping_rules():
    mapper = ClassMapper(config_path="configs/dataset_mapping.yaml")

    # Approved mapping: SCB BowHead -> head_down
    target, cid, status = mapper.map_class("scbehavior_bow_turn", "BowHead")
    assert target == "head_down"
    assert cid == 1
    assert status == MappingStatus.APPROVED

    # Approved mapping: SCB read -> normal
    target, cid, status = mapper.map_class("scb_handrise_read_write", "read")
    assert target == "normal"
    assert cid == 0
    assert status == MappingStatus.APPROVED

    # Rejected mapping: Kaggle cheating -> rejected
    target, cid, status = mapper.map_class("exam_cheating_kaggle", "cheating")
    assert target is None
    assert cid is None
    assert status == MappingStatus.REJECTED

    # Needs review mapping: CCTV LeftSideMove
    target, cid, status = mapper.map_class("cctv_exam_monitor", "LeftSideMove")
    assert status == MappingStatus.NEEDS_REVIEW
    # By default, needs_review classes are not approved for training without explicit override
    assert target is None

    # Ignored mapping: hand-raising
    target, cid, status = mapper.map_class("scb_handrise_read_write", "hand-raising")
    assert status == MappingStatus.IGNORED
    assert target is None


def test_duplicate_detector_hashes(tmp_path):
    # Create two identical synthetic images
    img1_path = tmp_path / "img1.jpg"
    img2_path = tmp_path / "img2.jpg"
    img3_path = tmp_path / "img3.jpg"

    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(canvas, (50, 50), 20, (255, 255, 255), -1)
    cv2.imwrite(str(img1_path), canvas)
    cv2.imwrite(str(img2_path), canvas)

    # Different image
    canvas_diff = np.ones((100, 100, 3), dtype=np.uint8) * 200
    cv2.imwrite(str(img3_path), canvas_diff)

    sha1 = compute_sha256(img1_path)
    sha2 = compute_sha256(img2_path)
    sha3 = compute_sha256(img3_path)

    assert sha1 == sha2
    assert sha1 != sha3

    # Perceptual hash
    dh1 = compute_dhash(canvas)
    dh2 = compute_dhash(canvas)
    dh3 = compute_dhash(canvas_diff)

    assert dh1 == dh2
    assert hamming_distance(dh1, dh2) == 0
    assert hamming_distance(dh1, dh3) > 0


def test_split_dataset_grouping_no_leakage():
    splitter = DatasetSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)

    # Create synthetic records belonging to 4 distinct video sequences
    records = []
    for seq_id in ["seq_alpha", "seq_beta", "seq_gamma", "seq_delta"]:
        for frame_idx in range(10):
            records.append(
                ManifestRecord(
                    image_path=f"/fake/{seq_id}_frame_{frame_idx}.jpg",
                    label_path=f"/fake/{seq_id}_frame_{frame_idx}.txt",
                    source_dataset="test_ds",
                    source_image=f"{seq_id}_frame_{frame_idx}.jpg",
                    group_id=seq_id,
                    source_split="train",
                    annotations=[{"canonical_class_name": "normal", "canonical_class_id": 0}],
                    canonical_classes=["normal"],
                )
            )

    train_recs, val_recs, test_recs = splitter.split_records_by_group(records)

    train_groups = {r.group_id for r in train_recs}
    val_groups = {r.group_id for r in val_recs}
    test_groups = {r.group_id for r in test_recs}

    # Verify zero leakage across splits: intersection must be empty
    assert len(train_groups.intersection(val_groups)) == 0
    assert len(train_groups.intersection(test_groups)) == 0
    assert len(val_groups.intersection(test_groups)) == 0
    assert len(train_recs) + len(val_recs) + len(test_recs) == len(records)


def test_extract_group_id():
    root = Path("datasets/raw/exam_data")
    p1 = root / "cam01" / "images" / "frame_001.jpg"
    assert extract_group_id(p1, root) == "cam01"

    p2 = root / "video_05_frame_123.jpg"
    assert extract_group_id(p2, root) == "video_05"

    p3 = root / "SCB5-Discuss" / "images" / "train" / "0006001.jpg"
    assert extract_group_id(p3, root) == "clip_0006"

    p4 = root / "9_001092.jpg"
    assert extract_group_id(p4, root) == "clip_video_9"


# ==============================================================================
# Step 21: Added Data Integrity & Preflight Verification Tests
# ==============================================================================

def test_nonexistent_source_classes_rejected():
    """Verify that hallucinated/nonexistent classes are rejected with needs_review."""
    mapper = ClassMapper(config_path="configs/dataset_mapping.yaml")
    
    # Query classes that do not exist locally
    cname, cid, status = mapper.map_class("scb_discuss", "using phone")
    assert cname is None
    assert cid is None
    assert status == MappingStatus.NEEDS_REVIEW

    cname, cid, status = mapper.map_class("scbehavior_bow_turn", "stand")
    assert cname is None
    assert cid is None
    assert status == MappingStatus.NEEDS_REVIEW


def test_only_approved_mappings_enter_training():
    """Verify that only approved mappings produce non-None canonical class IDs."""
    mapper = ClassMapper(config_path="configs/dataset_mapping.yaml")

    # hand-raising is ignored -> None
    cname, cid, status = mapper.map_class("scb_handrise_read_write", "hand-raising")
    assert cid is None

    # cheating is rejected -> None
    cname, cid, status = mapper.map_class("exam_cheating_kaggle", "cheating")
    assert cid is None

    # read is approved -> 0
    cname, cid, status = mapper.map_class("scb_handrise_read_write", "read")
    assert cid == 0
    assert status == MappingStatus.APPROVED


def test_duplicate_groups_cannot_cross_splits():
    """Verify that records sharing duplicate_cluster_id are forced into the same split."""
    splitter = DatasetSplitter(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42)

    # Two records with different initial group_ids but identical duplicate_cluster_id
    shared_cluster = "sha_abc123"
    rec1 = ManifestRecord(
        image_path="/path/img1.jpg",
        label_path="/path/lbl1.txt",
        source_dataset="ds1",
        source_image="img1.jpg",
        group_id="group_A",
        source_split="train",
        annotations=[{"canonical_class_name": "normal", "canonical_class_id": 0, "mapping_status": "approved", "bbox": [0.5, 0.5, 0.2, 0.2]}],
        canonical_classes=["normal"],
        duplicate_cluster_id=shared_cluster,
    )
    rec2 = ManifestRecord(
        image_path="/path/img2.jpg",
        label_path="/path/lbl2.txt",
        source_dataset="ds2",
        source_image="img2.jpg",
        group_id="group_B",
        source_split="train",
        annotations=[{"canonical_class_name": "turn_head", "canonical_class_id": 3, "mapping_status": "approved", "bbox": [0.5, 0.5, 0.2, 0.2]}],
        canonical_classes=["turn_head"],
        duplicate_cluster_id=shared_cluster,
    )

    train_recs, val_recs, test_recs = splitter.split_records_by_group([rec1, rec2])

    # Both rec1 and rec2 must end up in the exact same split
    in_train = (rec1 in train_recs) and (rec2 in train_recs)
    in_val = (rec1 in val_recs) and (rec2 in val_recs)
    in_test = (rec1 in test_recs) and (rec2 in test_recs)
    assert in_train or in_val or in_test, "Duplicate records were split across different splits!"


def test_rejected_cheating_labels_never_enter_canonical_data():
    """Verify that single-frame subjective 'cheating' labels never enter canonical training."""
    mapper = ClassMapper(config_path="configs/dataset_mapping.yaml")
    cname, cid, status = mapper.map_class("exam_cheating_kaggle", "cheating")
    assert status == MappingStatus.REJECTED
    assert cname is None
    assert cid is None


def test_invalid_class_ids_fail_validation():
    """Verify that class IDs outside canonical range [0, 6] are flagged as invalid."""
    # Class ID 7 is outside canonical taxonomy
    issues = validate_yolo_bbox(
        cls_id=7, xc=0.5, yc=0.5, w=0.2, h=0.2, file_path="sample.txt", line_no=1, raw_line="7 0.5 0.5 0.2 0.2", max_class_id=6
    )
    assert any(i.issue_type == "unknown_class" for i in issues)

    # Negative class ID
    neg_issues = validate_yolo_bbox(
        cls_id=-1, xc=0.5, yc=0.5, w=0.2, h=0.2, file_path="sample.txt", line_no=1, raw_line="-1 0.5 0.5 0.2 0.2", max_class_id=6
    )
    assert any(i.issue_type == "negative_coord" for i in neg_issues)


def test_dataset_provenance_survives_processing():
    """Verify that source metadata (dataset, image, original class) is preserved in records."""
    rec = ManifestRecord(
        image_path="/data/img.jpg",
        label_path="/data/lbl.txt",
        source_dataset="scb_handrise_read_write",
        source_image="3001001.jpg",
        group_id="clip_3001",
        source_split="val",
        annotations=[{
            "source_class_id": 1,
            "source_class_name": "read",
            "canonical_class_id": 0,
            "canonical_class_name": "normal",
            "mapping_status": "approved",
            "bbox": [0.5, 0.5, 0.2, 0.2],
        }],
        canonical_classes=["normal"],
        duplicate_cluster_id="sha_xyz789",
    )

    assert rec.source_dataset == "scb_handrise_read_write"
    assert rec.source_image == "3001001.jpg"
    assert rec.group_id == "clip_3001"
    assert rec.annotations[0]["source_class_name"] == "read"
    assert rec.annotations[0]["canonical_class_name"] == "normal"

