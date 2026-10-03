"""Rebuild processed dataset and manifests with zero duplicate leakage."""

import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.build_manifest import ProcessedDatasetBuilder
from training.split_dataset import DatasetSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rebuild_dataset")

RAW_DATASETS = [
    {
        "name": "scb_discuss",
        "root": "datasets/raw/scb_dataset3/SCB5-Discuss-2024-9-17",
        "classes": ["discuss"],
    },
    {
        "name": "scb_handrise_read_write",
        "root": "datasets/raw/scb_dataset3/SCB5-Handrise-Read-write-2024-9-17",
        "classes": ["hand-raising", "read", "write"],
    },
    {
        "name": "scbehavior_bow_turn",
        "root": "datasets/raw/scbehavior/SCB_BowTurnHead_20250509",
        "classes": ["BowHead", "TurnHead"],
    },
]

def main():
    builder = ProcessedDatasetBuilder(
        output_dir="datasets/processed",
        mapping_config_path="configs/dataset_mapping.yaml",
    )
    splitter = builder.splitter

    all_records = []
    for d in RAW_DATASETS:
        logger.info(f"Scanning raw dataset: {d['name']} at {d['root']}")
        recs = splitter.build_records_for_dataset(
            dataset_name=d["name"],
            dataset_root=d["root"],
            source_class_names=d["classes"],
            allow_needs_review=False,
        )
        logger.info(f"Loaded {len(recs)} records from {d['name']}")
        all_records.extend(recs)

    logger.info(f"Total raw image records across all datasets: {len(all_records)}")

    # Split into train/val/test by group and connected components
    train_recs, val_recs, test_recs = splitter.split_records_by_group(all_records)
    logger.info(f"Group-split completed: train={len(train_recs)}, val={len(val_recs)}, test={len(test_recs)}")

    # Export normalized YOLO dataset
    dataset_yaml = builder.export_yolo_dataset(
        train_records=train_recs,
        val_records=val_recs,
        test_records=test_recs,
        copy_images=False,
    )
    logger.info(f"Successfully exported normalized dataset to {dataset_yaml}")

if __name__ == "__main__":
    main()
