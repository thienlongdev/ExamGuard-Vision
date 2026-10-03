"""
V4C Posture Classifier Training & Comprehensive Benchmark Pipeline
Trains and evaluates candidate posture classifiers across the five canonical splits
and evaluation slices (scale, blur, temporal clips).
Conforms strictly to V4C experimental requirements and output formats.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple, Optional

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from src.models.posture.posture_classifier import (
    POSTURE_CLASSES,
    POSTURE_CLASS_TO_IDX,
    POSTURE_IDX_TO_CLASS,
    create_posture_model,
    save_posture_checkpoint,
    load_posture_checkpoint,
)


class PostureCropDataset(Dataset):
    """
    Dataset loader for clean normalized person crops.
    Loads pre-normalized images from normalized_224_path (or normalized_320_path).
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        resolution: int = 224,
        transform: Optional[transforms.Compose] = None,
        is_training: bool = False,
        representation: str = "TIGHT_PERSON_CROP",
    ):
        self.records = records
        self.resolution = resolution
        self.transform = transform
        self.is_training = is_training
        self.representation = representation
        self.norm_key = f"normalized_{resolution}_path"

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        rec = self.records[idx]
        img_path = rec.get(self.norm_key) or rec.get("image_path")
        p = Path(PROJECT_ROOT) / img_path if not Path(img_path).is_absolute() else Path(img_path)

        try:
            image = Image.open(p).convert("RGB")
            if self.representation == "UPPER_BODY_CROP":
                ub_h = max(int(round(image.height * 0.60)), 1)
                image = image.crop((0, 0, image.width, ub_h))
        except Exception as e:
            # Fallback to black image if corrupted/missing
            print(f"Warning: error opening {p}: {e}")
            image = Image.new("RGB", (self.resolution, self.resolution), (0, 0, 0))

        if self.transform is not None:
            tensor = self.transform(image)
        else:
            tensor = transforms.ToTensor()(image)

        label_name = rec["ontology_label"]
        label_idx = POSTURE_CLASS_TO_IDX[label_name]

        return tensor, label_idx, rec


def collate_train(batch):
    tensors = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return tensors, labels


def collate_eval(batch):
    tensors = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    recs = [item[2] for item in batch]
    return tensors, labels, recs


def get_transforms(resolution: int = 224):
    """Conservative data augmentation policy conforming to V4C Section 9."""
    train_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.RandomAffine(degrees=5, translate=(0.04, 0.04), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


def filter_representation_records(
    records: List[Dict[str, Any]],
    representation: str,
) -> List[Dict[str, Any]]:
    """
    Filters manifest records strictly for the target representation experiment.
    Quarantined samples are strictly excluded.
    """
    valid = []
    for r in records:
        if "QUARANTINED" in r.get("quality_flags", []):
            continue
        ont = r.get("ontology_label")
        if ont not in POSTURE_CLASS_TO_IDX:
            continue

        ctype = r.get("crop_type")
        if representation == "TIGHT_PERSON_CROP":
            if ctype in ["TIGHT_PERSON_CROP", "PRE_ISOLATED_PERSON_CROP"]:
                valid.append(r)
        elif representation == "CONTEXT_PERSON_CROP":
            if ctype in ["CONTEXT_PERSON_CROP", "PRE_ISOLATED_PERSON_CROP"]:
                valid.append(r)
        elif representation == "UPPER_BODY_CROP":
            if r.get("upper_body_crop_available", False) and ctype in ["TIGHT_PERSON_CROP", "PRE_ISOLATED_PERSON_CROP"]:
                valid.append(r)
        else:
            raise ValueError(f"Unknown representation: {representation}")
    return valid


def compute_class_weights(records: List[Dict[str, Any]], num_classes: int = 4) -> torch.Tensor:
    """Computes balanced inverse-frequency class weights."""
    counts = Counter(POSTURE_CLASS_TO_IDX[r["ontology_label"]] for r in records)
    total = len(records)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        cnt = counts.get(c, 0)
        weights[c] = total / (num_classes * max(cnt, 1))
    return weights


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    supported_classes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Calculates macro and per-class precision, recall, F1, balanced accuracy, and confusion matrix.
    Computes macro metrics strictly over physically supported classes.
    """
    if supported_classes is None:
        supported_classes = sorted(list(set(y_true)))

    if len(y_true) == 0:
        return {"count": 0, "macro_f1": 0.0, "balanced_acc": 0.0}

    # Confusion matrix
    num_all_classes = len(POSTURE_CLASSES)
    cm = [[0] * num_all_classes for _ in range(num_all_classes)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    per_class = {}
    recalls = []
    precisions = []
    f1s = []

    for c in supported_classes:
        c_name = POSTURE_CLASSES[c]
        tp = cm[c][c]
        fn = sum(cm[c][j] for j in range(num_all_classes) if j != c)
        fp = sum(cm[i][c] for i in range(num_all_classes) if i != c)
        support = tp + fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[c_name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_p = sum(precisions) / len(precisions) if precisions else 0.0
    macro_r = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    balanced_acc = macro_r

    # Overall accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = correct / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "count": len(y_true),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "supported_classes": [POSTURE_CLASSES[c] for c in supported_classes],
    }



@torch.no_grad()
def evaluate_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    supported_classes: Optional[List[int]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluates model over dataloader and returns metrics + detailed predictions."""
    model.eval()
    y_true = []
    y_pred = []
    all_details = []

    for tensors, labels, recs in dataloader:
        tensors = tensors.to(device)
        with torch.amp.autocast("cuda"):
            logits = model(tensors)
        probs = F.softmax(logits, dim=1).cpu()
        preds = probs.argmax(dim=1).tolist()
        targets = labels.tolist()

        y_true.extend(targets)
        y_pred.extend(preds)

        for i in range(len(preds)):
            r = recs[i]
            rec_dict = {
                "sample_id": r["sample_id"],
                "true_label": POSTURE_CLASSES[targets[i]],
                "pred_label": POSTURE_CLASSES[preds[i]],
                "confidence": float(probs[i][preds[i]]),
                "probs": [float(p) for p in probs[i]],
                "scale_bucket": r.get("scale_bucket", "UNKNOWN"),
                "quality_flags": r.get("quality_flags", []),
                "source_clip_id": r.get("source_clip_id"),
                "source_frame_index": r.get("source_frame_index"),
                "timestamp": r.get("timestamp"),
            }
            all_details.append(rec_dict)

    metrics = calculate_metrics(y_true, y_pred, supported_classes=supported_classes)
    return metrics, all_details


def update_heartbeat(
    active_task: str,
    experiment: str,
    current_epoch: int,
    max_epochs: int,
    last_completed_epoch: int,
    best_epoch: int,
    best_metric: float,
    status: str = "TRAINING",
):
    """Updates runs/v4c/V4C_EXECUTION_STATE.json with heartbeat telemetry."""
    state_file = PROJECT_ROOT / "runs/v4c/V4C_EXECUTION_STATE.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            state = {}
    state.update({
        "active_task": active_task,
        "experiment": experiment,
        "pid": os.getpid(),
        "current_epoch": current_epoch,
        "max_epochs": max_epochs,
        "last_completed_epoch": last_completed_epoch,
        "best_epoch": best_epoch,
        "best_metric": round(float(best_metric), 4) if best_metric is not None else None,
        "last_heartbeat": time.strftime("%Y-%m-%dT%H:%M:%S+07:00"),
        "status": status,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+07:00"),
    })
    tmp_path = state_file.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp_path.replace(state_file)


def evaluate_posture_model_all_splits(
    model: nn.Module,
    model_name: str,
    representation: str,
    resolution: int,
    output_dir: Path,
    batch_size: int = 64,
    num_workers: int = 0,
    best_epoch: int = -1,
    total_epochs: int = -1,
    training_duration_seconds: float = 0.0,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Exhaustively evaluates posture classifier across all 4 evaluation splits and slices."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    splits_dir = PROJECT_ROOT / "datasets/v4_crop/splits"
    _, eval_transform = get_transforms(resolution)

    # 1. Same-Domain Validation Full Evaluation
    val_records_raw = [json.loads(line) for line in open(splits_dir / "same_domain_val.jsonl", encoding="utf-8")]
    val_recs = filter_representation_records(val_records_raw, representation)
    val_ds = PostureCropDataset(val_recs, resolution=resolution, transform=eval_transform, representation=representation)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_eval)
    val_metrics, val_details = evaluate_dataloader(model, val_loader, device=device, supported_classes=[0, 1, 2, 3])

    # 2. High-Angle Holdout Evaluation (classes 0, 1, 3)
    high_angle_recs_raw = [json.loads(line) for line in open(splits_dir / "high_angle_holdout.jsonl", encoding="utf-8")]
    high_angle_recs = filter_representation_records(high_angle_recs_raw, representation)
    ha_ds = PostureCropDataset(high_angle_recs, resolution=resolution, transform=eval_transform, representation=representation)
    ha_loader = DataLoader(ha_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_eval)
    ha_metrics, ha_details = evaluate_dataloader(model, ha_loader, device=device, supported_classes=[0, 1, 3])

    # 3. Cross-Source Holdout Evaluation (classes 0, 1, 2)
    cross_recs_raw = [json.loads(line) for line in open(splits_dir / "cross_source_holdout.jsonl", encoding="utf-8")]
    cross_recs = filter_representation_records(cross_recs_raw, representation)
    cs_ds = PostureCropDataset(cross_recs, resolution=resolution, transform=eval_transform, representation=representation)
    cs_loader = DataLoader(cs_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_eval)
    cs_metrics, cs_details = evaluate_dataloader(model, cs_loader, device=device, supported_classes=[0, 1, 2])

    # 4. Temporal Holdout Evaluation (classes 0, 1, 2)
    temp_recs_raw = [json.loads(line) for line in open(splits_dir / "temporal_holdout.jsonl", encoding="utf-8")]
    temp_recs = filter_representation_records(temp_recs_raw, representation)
    temp_ds = PostureCropDataset(temp_recs, resolution=resolution, transform=eval_transform, representation=representation)
    temp_loader = DataLoader(temp_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_eval)
    temp_frame_metrics, temp_details = evaluate_dataloader(model, temp_loader, device=device, supported_classes=[0, 1, 2])

    # Temporal Clip-Level Aggregations & Flicker Analysis
    clips = defaultdict(list)
    for d in temp_details:
        cid = d["source_clip_id"] or "unspecified_clip"
        clips[cid].append(d)

    clip_maj_correct = 0
    clip_mean_correct = 0
    total_flickers = 0
    total_transitions = 0

    for cid, items in clips.items():
        items.sort(key=lambda x: (x["source_frame_index"] or 0, x["timestamp"] or 0))
        true_labels = [POSTURE_CLASS_TO_IDX[it["true_label"]] for it in items]
        pred_labels = [POSTURE_CLASS_TO_IDX[it["pred_label"]] for it in items]
        all_probs = [it["probs"] for it in items]

        clip_gt = Counter(true_labels).most_common(1)[0][0]
        maj_pred = Counter(pred_labels).most_common(1)[0][0]
        if maj_pred == clip_gt:
            clip_maj_correct += 1

        mean_p = [sum(p[c] for p in all_probs) / len(all_probs) for c in range(4)]
        prob_pred = int(torch.tensor(mean_p).argmax().item())
        if prob_pred == clip_gt:
            clip_mean_correct += 1

        for k in range(len(pred_labels) - 1):
            total_transitions += 1
            if pred_labels[k] != pred_labels[k + 1]:
                total_flickers += 1

    flicker_rate = total_flickers / max(total_transitions, 1)
    temporal_metrics = {
        "frame_level": temp_frame_metrics,
        "clip_count": len(clips),
        "clip_majority_accuracy": round(clip_maj_correct / max(len(clips), 1), 4),
        "clip_prob_mean_accuracy": round(clip_mean_correct / max(len(clips), 1), 4),
        "prediction_flicker_rate": round(flicker_rate, 4),
        "total_transitions": total_transitions,
        "total_flickers": total_flickers,
    }

    # 5. Slices: Scale Robustness (on same_domain_val + high_angle)
    combined_eval_details = val_details + ha_details
    scale_metrics = {}
    for scale in ["PERSON_LARGE", "PERSON_MEDIUM", "PERSON_SMALL", "PERSON_VERY_SMALL"]:
        slice_items = [it for it in combined_eval_details if it.get("scale_bucket") == scale]
        y_t = [POSTURE_CLASS_TO_IDX[it["true_label"]] for it in slice_items]
        y_p = [POSTURE_CLASS_TO_IDX[it["pred_label"]] for it in slice_items]
        scale_metrics[scale] = calculate_metrics(y_t, y_p)

    # 6. Slices: Blur Robustness
    blur_metrics = {}
    for blur_flag in ["GOOD", "BLURRY"]:
        if blur_flag == "BLURRY":
            slice_items = [it for it in combined_eval_details if "BLURRY" in it.get("quality_flags", [])]
        else:
            slice_items = [it for it in combined_eval_details if "BLURRY" not in it.get("quality_flags", [])]
        y_t = [POSTURE_CLASS_TO_IDX[it["true_label"]] for it in slice_items]
        y_p = [POSTURE_CLASS_TO_IDX[it["pred_label"]] for it in slice_items]
        blur_metrics[blur_flag] = calculate_metrics(y_t, y_p)

    best_ckpt_path = checkpoint_path or (output_dir / "best_model.pt")
    final_results = {
        "experiment_id": f"{model_name}_{representation.lower()}_{resolution}",
        "model_name": model_name,
        "representation": representation,
        "resolution": resolution,
        "best_epoch": best_epoch,
        "total_epochs": total_epochs,
        "training_duration_seconds": round(training_duration_seconds, 2),
        "checkpoint_path": str(best_ckpt_path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/"),
        "same_domain_val": val_metrics,
        "high_angle_holdout": ha_metrics,
        "cross_source_holdout": cs_metrics,
        "temporal_holdout": temporal_metrics,
        "scale_slices": scale_metrics,
        "blur_slices": blur_metrics,
    }

    # Save complete evaluation artifact atomically
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / "metrics.json"
    tmp_m = metrics_file.with_suffix(".tmp")
    with open(tmp_m, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    tmp_m.replace(metrics_file)

    # Save detailed prediction samples for FP/FN error analysis atomically
    predictions_file = output_dir / "eval_predictions.json"
    tmp_p = predictions_file.with_suffix(".tmp")
    with open(tmp_p, "w", encoding="utf-8") as f:
        json.dump(combined_eval_details + cs_details + temp_details, f, indent=2)
    tmp_p.replace(predictions_file)

    print(f"\nExperiment Results Saved to {metrics_file}")
    print(f"Summary: Val F1={val_metrics['macro_f1']} | HA F1={ha_metrics['macro_f1']} | CS F1={cs_metrics['macro_f1']} | Temp MajAcc={temporal_metrics['clip_majority_accuracy']}")
    return final_results


def train_posture_model(
    model_name: str,
    representation: str,
    resolution: int = 224,
    batch_size: int = 64,
    epochs: int = 30,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    patience: int = 6,
    num_workers: int = 0,
    output_dir: Optional[Path] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    """Full standardized, crash-safe, resumable training loop for one posture experiment."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"TRAINING EXPERIMENT: {model_name} | {representation} | {resolution}x{resolution}")
    print(f"Device: {device} | Batch Size: {batch_size} | Max Epochs: {epochs} | LR: {lr}")
    print(f"Workers: {num_workers} (Windows-Safe Deterministic)")
    print(f"=======================================================")

    splits_dir = PROJECT_ROOT / "datasets/v4_crop/splits"
    train_records_raw = [json.loads(line) for line in open(splits_dir / "train.jsonl", encoding="utf-8")]
    val_records_raw = [json.loads(line) for line in open(splits_dir / "same_domain_val.jsonl", encoding="utf-8")]

    train_recs = filter_representation_records(train_records_raw, representation)
    val_recs = filter_representation_records(val_records_raw, representation)

    print(f"Filtered Supervised Records -> Train: {len(train_recs)} | Same-Domain Val: {len(val_recs)}")

    train_transform, eval_transform = get_transforms(resolution)

    train_ds = PostureCropDataset(train_recs, resolution=resolution, transform=train_transform, is_training=True, representation=representation)
    val_ds = PostureCropDataset(val_recs, resolution=resolution, transform=eval_transform, is_training=False, representation=representation)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_eval,
    )

    # Class weights for weighted cross-entropy
    class_weights = compute_class_weights(train_recs).to(device)
    print(f"Effective Class Weights: { {POSTURE_CLASSES[i]: round(class_weights[i].item(), 4) for i in range(4)} }")

    model = create_posture_model(model_name=model_name, num_classes=4, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    if output_dir is None:
        output_dir = PROJECT_ROOT / "runs/v4c" / f"{model_name}_{representation.lower()}_{resolution}"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = output_dir / "best_model.pt"
    last_ckpt_path = output_dir / "last_checkpoint.pt"
    history_csv = output_dir / "training_history.csv"

    # Resumption handling
    start_epoch = 1
    best_macro_f1 = -1.0
    best_epoch = -1
    patience_counter = 0

    if not resume:
        for p_to_clean in [last_ckpt_path, best_ckpt_path, history_csv]:
            if p_to_clean.exists():
                try:
                    p_to_clean.unlink()
                except Exception:
                    pass

    if resume and last_ckpt_path.exists():
        try:
            ckpt_data = torch.load(last_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt_data["state_dict"])
            optimizer.load_state_dict(ckpt_data["optimizer_state"])
            scheduler.load_state_dict(ckpt_data["scheduler_state"])
            scaler.load_state_dict(ckpt_data["scaler_state"])
            start_epoch = ckpt_data["epoch"] + 1
            best_macro_f1 = ckpt_data.get("best_metric", -1.0)
            best_epoch = ckpt_data.get("best_epoch", -1)
            print(f">>> RESUMING from Epoch {start_epoch} (Previous Best: {best_macro_f1:.4f} at Epoch {best_epoch}) <<<")
        except Exception as e:
            print(f"Warning: Could not resume from {last_ckpt_path}: {e}. Starting fresh.")

    # Initialize CSV header if not exists
    if not history_csv.exists():
        with open(history_csv, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,val_macro_f1,best_metric,best_epoch,timestamp\n")

    start_time = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.perf_counter()

        for tensors, labels in train_loader:
            tensors, labels = tensors.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                logits = model(tensors)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(labels)

        scheduler.step()
        train_loss /= len(train_recs)
        epoch_sec = time.perf_counter() - t0

        # Evaluate on same-domain validation
        val_metrics, _ = evaluate_dataloader(model, val_loader, device=device, supported_classes=[0, 1, 2, 3])
        val_macro_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]

        # Log epoch summary
        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_sec:.1f}s] - "
            f"Train Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f} - "
            f"Val Macro F1: {val_macro_f1:.4f} (BalAcc: {val_metrics['balanced_accuracy']:.4f}) - "
            f"Sleep Rec: {val_metrics['per_class']['HEAD_REST_SLEEP']['recall']:.4f} - "
            f"Turn Rec: {val_metrics['per_class']['TURN_HEAD_CLEAR']['recall']:.4f}"
        )

        # Check for best
        is_best = False
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            patience_counter = 0
            is_best = True

            save_posture_checkpoint(
                model=model,
                filepath=best_ckpt_path,
                model_name=model_name,
                epoch=epoch,
                metrics=val_metrics,
                optimizer_state=optimizer.state_dict(),
                extra_metadata={
                    "representation": representation,
                    "resolution": resolution,
                    "best_epoch": best_epoch,
                },
            )
            print(f"  -> Saved new best checkpoint to {best_ckpt_path} (Macro F1: {best_macro_f1:.4f})")
        else:
            patience_counter += 1

        # Append to training_history.csv
        timestamp_str = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(history_csv, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.4f},0.0,{val_macro_f1:.4f},{best_macro_f1:.4f},{best_epoch},{timestamp_str}\n")

        # Save last_checkpoint.pt atomically every epoch
        last_ckpt_payload = {
            "model_name": model_name,
            "representation": representation,
            "resolution": resolution,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_metric": best_macro_f1,
            "best_epoch": best_epoch,
            "training_config": {
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "max_epochs": epochs,
            },
            "class_mapping": POSTURE_CLASS_TO_IDX,
        }
        tmp_last = last_ckpt_path.with_suffix(".tmp")
        torch.save(last_ckpt_payload, tmp_last)
        tmp_last.replace(last_ckpt_path)

        # Update heartbeat
        update_heartbeat(
            active_task=f"Training {model_name} ({representation} / {resolution})",
            experiment=f"{model_name}_{representation.lower()}_{resolution}",
            current_epoch=epoch,
            max_epochs=epochs,
            last_completed_epoch=epoch,
            best_epoch=best_epoch,
            best_metric=best_macro_f1,
            status="TRAINING",
        )

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (Patience: {patience})")
            break

    total_training_sec = time.perf_counter() - start_time
    print(f"Training completed in {total_training_sec:.1f}s. Best Epoch: {best_epoch} with Same-Domain Val Macro F1: {best_macro_f1:.4f}")

    # Reload best model for exhaustive multi-split evaluation
    best_model, _ = load_posture_checkpoint(best_ckpt_path, device=device)
    final_results = evaluate_posture_model_all_splits(
        model=best_model,
        model_name=model_name,
        representation=representation,
        resolution=resolution,
        output_dir=output_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        best_epoch=best_epoch,
        total_epochs=epoch,
        training_duration_seconds=total_training_sec,
        checkpoint_path=best_ckpt_path,
    )
    update_heartbeat(
        active_task=f"Finished {model_name} ({representation} / {resolution})",
        experiment=f"{model_name}_{representation.lower()}_{resolution}",
        current_epoch=epoch,
        max_epochs=epochs,
        last_completed_epoch=epoch,
        best_epoch=best_epoch,
        best_metric=best_macro_f1,
        status="COMPLETE",
    )
    return final_results

    return final_results


def main():
    parser = argparse.ArgumentParser(description="Train V4C Posture Classifier Candidate")
    parser.add_argument("--model", type=str, required=True, choices=["resnet18_cbam", "resnet50_cbam", "mobilenet_v3_small"])
    parser.add_argument("--representation", type=str, required=True, choices=["TIGHT_PERSON_CROP", "CONTEXT_PERSON_CROP", "UPPER_BODY_CROP"])
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--fresh", action="store_true", help="Start training fresh ignoring existing checkpoints")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None
    train_posture_model(
        model_name=args.model,
        representation=args.representation,
        resolution=args.resolution,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        num_workers=args.num_workers,
        output_dir=out_dir,
        resume=not args.fresh,
    )


if __name__ == "__main__":
    main()
