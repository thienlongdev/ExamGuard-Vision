"""Orchestrator script: audits datasets, extracts visual class samples, detects duplicates, and builds grouped manifests."""

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.build_manifest import ProcessedDatasetBuilder
from training.class_mapping import ClassMapper
from training.dataset_downloader import DatasetDownloader
from training.duplicate_detector import DuplicateDetector
from training.inspect_dataset import DatasetInspector
from training.split_dataset import DatasetSplitter
from training.visual_sample_inspector import VisualSampleInspector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("dataset_audit")


def run_full_audit():
    logger.info("=" * 70)
    logger.info(" STARTING COMPREHENSIVE DATASET AUDIT & PREPARATION")
    logger.info("=" * 70)

    raw_dir = Path("datasets/raw")
    reports_dir = Path("reports")
    reports_datasets_dir = reports_dir / "datasets"
    reports_samples_dir = reports_dir / "class_samples"
    reports_duplicates_dir = reports_dir / "duplicates"
    reports_splits_dir = reports_dir / "splits"

    for d in (reports_datasets_dir, reports_samples_dir, reports_duplicates_dir, reports_splits_dir):
        d.mkdir(parents=True, exist_ok=True)

    mapper = ClassMapper()
    all_manifest_records = []

    # 1. Audit SCB-Dataset3
    scb3_path = raw_dir / "scb_dataset3"
    scb3_classes = ["discuss", "hand-raising", "read", "write"]
    logger.info(f"\n--- Auditing {scb3_path} ---")
    inspector_scb3 = DatasetInspector("scb_dataset3", str(scb3_path), class_names=scb3_classes)
    res_scb3 = inspector_scb3.inspect()
    inspector_scb3.generate_json_report(res_scb3, str(reports_datasets_dir / "scb_dataset3.json"))
    inspector_scb3.generate_markdown_report(res_scb3, str(reports_datasets_dir / "scb_dataset3.md"))

    # Visual inspection for SCB-Dataset3
    logger.info("Generating visual sample sheets for scb_dataset3...")
    vis_scb3 = VisualSampleInspector("scb_dataset3", str(scb3_path), class_names=scb3_classes, output_dir=str(reports_samples_dir))
    vis_scb3.inspect_all_classes(max_samples_per_class=25)

    # Duplicate detection for SCB-Dataset3
    logger.info("Detecting duplicates in scb_dataset3...")
    dup_scb3 = DuplicateDetector("scb_dataset3", str(scb3_path))
    res_dup_scb3 = dup_scb3.scan()
    dup_scb3.save_report(res_dup_scb3, str(reports_duplicates_dir / "scb_dataset3_duplicates.json"))

    # 2. Audit SCBehavior
    scb_beh_path = raw_dir / "scbehavior"
    scb_beh_classes = ["BowHead", "TurnHead"]
    logger.info(f"\n--- Auditing {scb_beh_path} ---")
    inspector_beh = DatasetInspector("scbehavior", str(scb_beh_path), class_names=scb_beh_classes)
    res_beh = inspector_beh.inspect()
    inspector_beh.generate_json_report(res_beh, str(reports_datasets_dir / "scbehavior.json"))
    inspector_beh.generate_markdown_report(res_beh, str(reports_datasets_dir / "scbehavior.md"))

    # Visual inspection for SCBehavior
    logger.info("Generating visual sample sheets for scbehavior...")
    vis_beh = VisualSampleInspector("scbehavior", str(scb_beh_path), class_names=scb_beh_classes, output_dir=str(reports_samples_dir))
    vis_beh.inspect_all_classes(max_samples_per_class=25)

    # Duplicate detection for SCBehavior
    logger.info("Detecting duplicates in scbehavior...")
    dup_beh = DuplicateDetector("scbehavior", str(scb_beh_path))
    res_dup_beh = dup_beh.scan()
    dup_beh.save_report(res_dup_beh, str(reports_duplicates_dir / "scbehavior_duplicates.json"))

    # 3. Build Grouped Splits & Manifests
    logger.info("\n--- Building Leakage-Safe Grouped Splits ---")
    splitter = DatasetSplitter(class_mapper=mapper, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    # Collect records
    scb3_recs = splitter.build_records_for_dataset("scb_dataset3", str(scb3_path), scb3_classes, allow_needs_review=False)
    beh_recs = splitter.build_records_for_dataset("scbehavior", str(scb_beh_path), scb_beh_classes, allow_needs_review=False)

    combined_recs = scb3_recs + beh_recs
    logger.info(f"Total mapped records across available datasets: {len(combined_recs)}")

    train_recs, val_recs, test_recs = splitter.split_records_by_group(combined_recs)
    manifest_paths = splitter.save_manifests(train_recs, val_recs, test_recs, output_dir="datasets/processed/manifests")
    splitter.generate_split_report(train_recs, val_recs, test_recs, output_path=str(reports_splits_dir / "split_report.md"))

    # 4. Export normalized YOLO processed structure
    logger.info("\n--- Exporting Processed YOLO Dataset Structure ---")
    builder = ProcessedDatasetBuilder(output_dir="datasets/processed")
    yolo_yaml = builder.export_yolo_dataset(train_recs, val_recs, test_recs, copy_images=False)
    logger.info(f"YOLO dataset definition created at: {yolo_yaml}")

    logger.info("=" * 70)
    logger.info(" AUDIT & DATASET PREPARATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_full_audit()
