import hashlib
import json
from pathlib import Path
import torch

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def get_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def check_file(path: Path, expected_hash: str) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False, "hash": None, "match": False}
    actual = get_sha256(path)
    return {"path": str(path), "exists": True, "hash": actual, "expected": expected_hash, "match": actual == expected_hash}

def count_jsonl(path: Path) -> int:
    if not path.exists():
        return -1
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def main():
    print("=======================================================")
    print("VERIFYING BASELINES & CRASH STATE")
    print("=======================================================")

    # 1. Protected Checkpoints
    s1 = check_file(PROJECT_ROOT / "models/trained/stage1_best.pt", "6d713808f0bc670e8bf06e01b006fac73d8d6e5db7130584cec1658b003ce98a")
    s1_5 = check_file(PROJECT_ROOT / "models/trained/stage1_5_best.pt", "68690cf82715dc6dad2eb35a08711531170e935eb7dbe1c30fafabae341e2c2c")
    print("Stage 1 Best:", s1["match"], s1["hash"])
    print("Stage 1.5 Best:", s1_5["match"], s1_5["hash"])

    # 2. V3 / V3.5 Protected Benchmarks
    v3_dir = PROJECT_ROOT / "datasets/processed_v3"
    v3_5_dir = PROJECT_ROOT / "datasets/processed_v3_5"
    print("V3 Benchmark exists:", v3_dir.exists(), "files:", len(list(v3_dir.glob("*"))))
    print("V3.5 Benchmark exists:", v3_5_dir.exists(), "files:", len(list(v3_5_dir.glob("*"))))

    # 3. V4 Crop Dataset Splits
    crop_splits = {
        "manifest.jsonl": (PROJECT_ROOT / "datasets/v4_crop/manifest.jsonl", 20490),
        "train.jsonl": (PROJECT_ROOT / "datasets/v4_crop/splits/train.jsonl", 14821),
        "same_domain_val.jsonl": (PROJECT_ROOT / "datasets/v4_crop/splits/same_domain_val.jsonl", 2973),
        "high_angle_holdout.jsonl": (PROJECT_ROOT / "datasets/v4_crop/splits/high_angle_holdout.jsonl", 1618),
        "cross_source_holdout.jsonl": (PROJECT_ROOT / "datasets/v4_crop/splits/cross_source_holdout.jsonl", 543),
        "temporal_holdout.jsonl": (PROJECT_ROOT / "datasets/v4_crop/splits/temporal_holdout.jsonl", 535),
    }

    crop_results = {}
    for name, (p, exp_cnt) in crop_splits.items():
        cnt = count_jsonl(p)
        crop_results[name] = {"count": cnt, "expected": exp_cnt, "match": cnt == exp_cnt}
        print(f"Crop {name}: {cnt} (Expected: {exp_cnt}, Match: {cnt == exp_cnt})")

    # 4. V4 Head Pose Dataset Splits
    hp_splits = {
        "manifest.jsonl": (PROJECT_ROOT / "datasets/v4_head_pose/manifest.jsonl", 23080),
        "train.jsonl": (PROJECT_ROOT / "datasets/v4_head_pose/splits/train.jsonl", 16218),
        "val.jsonl": (PROJECT_ROOT / "datasets/v4_head_pose/splits/val.jsonl", 2862),
        "test.jsonl": (PROJECT_ROOT / "datasets/v4_head_pose/splits/test.jsonl", 2000),
    }

    hp_results = {}
    for name, (p, exp_cnt) in hp_splits.items():
        cnt = count_jsonl(p)
        hp_results[name] = {"count": cnt, "expected": exp_cnt, "match": cnt == exp_cnt}
        print(f"HeadPose {name}: {cnt} (Expected: {exp_cnt}, Match: {cnt == exp_cnt})")

    # 5. Existing V4C Checkpoint Audit
    a1_ckpt_path = PROJECT_ROOT / "runs/v4c/A1_resnet18_cbam_tight_person_crop_224/best_model.pt"
    a1_status = {}
    if a1_ckpt_path.exists():
        a1_hash = get_sha256(a1_ckpt_path)
        ckpt = torch.load(a1_ckpt_path, map_location="cpu", weights_only=False)
        from src.models.posture.posture_classifier import create_posture_model
        model = create_posture_model("resnet18_cbam", num_classes=4, pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        finite = torch.isfinite(out).all().item()

        a1_status = {
            "path": str(a1_ckpt_path),
            "sha256": a1_hash,
            "epoch": ckpt.get("epoch"),
            "best_epoch": ckpt.get("extra_metadata", {}).get("best_epoch"),
            "metrics": ckpt.get("metrics"),
            "classes": ckpt.get("class_names"),
            "num_classes": ckpt.get("num_classes"),
            "has_optimizer": "optimizer_state" in ckpt,
            "has_scheduler": "scheduler_state" in ckpt,
            "smoke_test_finite": finite,
            "output_shape": list(out.shape),
        }
        print("\nA1 Checkpoint Audit:")
        print(f"  SHA256: {a1_hash}")
        print(f"  Epoch: {ckpt.get('epoch')}")
        print(f"  Classes: {ckpt.get('class_names')}")
        print(f"  Smoke Test Finite: {finite}")
        print(f"  Has Optimizer State: {a1_status['has_optimizer']}")
        print(f"  Has Scheduler State: {a1_status['has_scheduler']}")
    else:
        print("A1 Checkpoint does not exist.")

    summary = {
        "protected_stage1": s1,
        "protected_stage1_5": s1_5,
        "crop_splits": crop_results,
        "head_pose_splits": hp_results,
        "a1_checkpoint": a1_status,
    }

    with open(PROJECT_ROOT / "runs/v4c/audit_baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Baseline audit summary written to runs/v4c/audit_baseline_results.json")

if __name__ == "__main__":
    main()
