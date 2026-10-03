"""
V4C Head-Pose (Yaw Regression) Training & Evaluation Pipeline
Trains HopeNet-Yaw (and ResNet18-Yaw baseline) on AFLW-GT train (16,218 images),
validates on AFLW-GT val (2,862 images), and evaluates on AFLW2000-3D test (2,000 images).
Generates reports/v4c/HEAD_POSE_MODEL_COMPARISON.md
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
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

from src.models.headpose.headpose_estimator import (
    create_headpose_model,
    save_headpose_checkpoint,
    load_headpose_checkpoint,
    canonicalize_yaw,
    shortest_angular_difference,
    circular_mae,
    encode_yaw_sin_cos,
    decode_yaw_sin_cos,
)


class HeadPoseDataset(Dataset):
    """Dataset for face/head crops with ground-truth Euler yaw angles."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        resolution: int = 224,
        transform: Optional[transforms.Compose] = None,
    ):
        self.records = records
        self.resolution = resolution
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        rec = self.records[idx]
        img_path = rec["image_path"]
        p = PROJECT_ROOT / img_path if not Path(img_path).is_absolute() else Path(img_path)

        try:
            image = Image.open(p).convert("RGB")
        except Exception as e:
            image = Image.new("RGB", (self.resolution, self.resolution), (0, 0, 0))

        if self.transform is not None:
            tensor = self.transform(image)
        else:
            tensor = transforms.ToTensor()(image)

        yaw = canonicalize_yaw(float(rec["yaw"]))
        return tensor, yaw, rec


def collate_hp_train(batch):
    tensors = torch.stack([item[0] for item in batch], dim=0)
    yaws = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    return tensors, yaws


def collate_hp_eval(batch):
    tensors = torch.stack([item[0] for item in batch], dim=0)
    yaws = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    recs = [item[2] for item in batch]
    return tensors, yaws, recs


def get_hp_transforms(resolution: int = 224):
    # Note: NO Horizontal Flip to preserve yaw sign integrity!
    train_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
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


def compute_slice_metrics(errors_slice: np.ndarray) -> Dict[str, Any]:
    if len(errors_slice) == 0:
        return {"n": 0, "mae_deg": 0.0, "median_ae_deg": 0.0, "p75_deg": 0.0, "p90_deg": 0.0}
    return {
        "n": int(len(errors_slice)),
        "mae_deg": round(float(np.mean(errors_slice)), 2),
        "median_ae_deg": round(float(np.median(errors_slice)), 2),
        "p75_deg": round(float(np.percentile(errors_slice, 75)), 2),
        "p90_deg": round(float(np.percentile(errors_slice, 90)), 2),
    }


def compute_angle_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Computes shortest circular angular metrics: MAE, Median, P50, P75, P90,
    and distribution slices according to V4C Head-Pose Safety Specifications.
    """
    errors = np.asarray(shortest_angular_difference(y_pred, y_true, signed=False), dtype=np.float64)

    mae = float(np.mean(errors))
    median_error = float(np.median(errors))
    p50 = float(np.percentile(errors, 50))
    p75 = float(np.percentile(errors, 75))
    p90 = float(np.percentile(errors, 90))

    # Audit bins required by V4C Section 7:
    abs_true = np.abs(y_true)
    c_lt15 = int(np.sum(abs_true < 15.0))
    c_15_45 = int(np.sum((abs_true >= 15.0) & (abs_true < 45.0)))
    c_45_90 = int(np.sum((abs_true >= 45.0) & (abs_true < 90.0)))
    c_90_120 = int(np.sum((abs_true >= 90.0) & (abs_true < 120.0)))
    c_120_150 = int(np.sum((abs_true >= 120.0) & (abs_true < 150.0)))
    c_ge150 = int(np.sum(abs_true >= 150.0))

    # Profile slices:
    frontal_mask = abs_true < 30.0
    moderate_mask = (abs_true >= 30.0) & (abs_true < 45.0)
    large_mask = (abs_true >= 45.0) & (abs_true < 90.0)
    extreme_mask = abs_true >= 90.0

    # Classroom Reference Slices (V4C Section 4):
    # 1. FRONTAL_MODERATE_REFERENCE_SLICE: |yaw| <= 45.0 deg
    frontal_mod_mask = abs_true <= 45.0
    # 2. CLEAR_TURN_REFERENCE_SLICE: 35.0 <= |yaw| < 90.0 deg
    clear_turn_mask = (abs_true >= 35.0) & (abs_true < 90.0)
    # 3. ONTOLOGY_TURN_REFERENCE_SLICE: 35.0 <= |yaw| <= 75.0 deg
    ontology_turn_mask = (abs_true >= 35.0) & (abs_true <= 75.0)

    # HopeNet Supported Range Subset [-99.0, +99.0) (half-open native support):
    hopenet_supported_mask = (y_true >= -99.0) & (y_true < 99.0)

    return {
        "count": len(y_true),
        "mae_deg": round(mae, 2),
        "median_error_deg": round(median_error, 2),
        "p50_deg": round(p50, 2),
        "p75_deg": round(p75, 2),
        "p90_deg": round(p90, 2),
        "distribution_audit_counts": {
            "lt15": c_lt15,
            "15_to_45": c_15_45,
            "45_to_90": c_45_90,
            "90_to_120": c_90_120,
            "120_to_150": c_120_150,
            "ge150": c_ge150,
        },
        "slices": {
            "frontal_lt15": compute_slice_metrics(errors[abs_true < 15.0]),
            "frontal_lt15_mae": round(float(np.mean(errors[abs_true < 15.0])), 2) if np.any(abs_true < 15.0) else 0.0,
            "large_ge45_mae": round(float(np.mean(errors[abs_true >= 45.0])), 2) if np.any(abs_true >= 45.0) else 0.0,
            "frontal_lt30": compute_slice_metrics(errors[frontal_mask]),
            "moderate_30_45": compute_slice_metrics(errors[moderate_mask]),
            "large_45_90": compute_slice_metrics(errors[large_mask]),
            "extreme_profile_ge90": compute_slice_metrics(errors[extreme_mask]),
            "frontal_moderate_reference_slice": compute_slice_metrics(errors[frontal_mod_mask]),
            "clear_turn_reference_slice": compute_slice_metrics(errors[clear_turn_mask]),
            "ontology_turn_reference_slice": compute_slice_metrics(errors[ontology_turn_mask]),
            "hopenet_common_support_slice": compute_slice_metrics(errors[hopenet_supported_mask]),
        },
    }


@torch.no_grad()
def evaluate_hp_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    y_true_list = []
    y_pred_list = []
    details = []

    for tensors, yaws, recs in dataloader:
        tensors = tensors.to(device)
        with torch.amp.autocast("cuda"):
            _, pred_yaw = model(tensors)

        preds = pred_yaw.cpu().numpy()
        targets = yaws.numpy()

        y_true_list.extend(targets)
        y_pred_list.extend(preds)

        for i in range(len(preds)):
            r = recs[i]
            details.append({
                "sample_id": r["sample_id"],
                "source_dataset": r["source_dataset"],
                "true_yaw": float(targets[i]),
                "pred_yaw": float(preds[i]),
                "abs_error": float(shortest_angular_difference(preds[i], targets[i], signed=False)),
            })

    metrics = compute_angle_metrics(np.array(y_true_list), np.array(y_pred_list))
    return metrics, details


def update_hp_heartbeat(
    model_name: str,
    current_epoch: int,
    max_epochs: int,
    last_completed_epoch: int,
    best_epoch: int,
    best_mae: float,
    status: str = "TRAINING",
):
    """Updates runs/v4c/V4C_EXECUTION_STATE.json with head-pose heartbeat telemetry."""
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
        "active_task": f"Training Head-Pose {model_name}",
        "experiment": f"headpose_{model_name}",
        "pid": os.getpid(),
        "current_epoch": current_epoch,
        "max_epochs": max_epochs,
        "last_completed_epoch": last_completed_epoch,
        "best_epoch": best_epoch,
        "best_metric": round(float(best_mae), 2) if best_mae is not None and best_mae != float("inf") else None,
        "last_heartbeat": time.strftime("%Y-%m-%dT%H:%M:%S+07:00"),
        "status": status,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+07:00"),
    })
    tmp_path = state_file.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp_path.replace(state_file)


def train_headpose_model(
    model_name: str = "hopenet_yaw",
    batch_size: int = 64,
    epochs: int = 15,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    patience: int = 5,
    num_workers: int = 0,
    output_dir: Optional[Path] = None,
    resume: bool = True,
) -> Tuple[Dict[str, Any], nn.Module]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"HEAD-POSE TRAINING: {model_name} (Yaw Regression)")
    print(f"Device: {device} | Batch Size: {batch_size} | Epochs: {epochs} | LR: {lr}")
    print(f"Workers: {num_workers} (Windows-Safe Deterministic)")
    print(f"=======================================================")

    splits_dir = PROJECT_ROOT / "datasets/v4_head_pose/splits"
    train_recs = [json.loads(line) for line in open(splits_dir / "train.jsonl", encoding="utf-8")]
    val_recs = [json.loads(line) for line in open(splits_dir / "val.jsonl", encoding="utf-8")]
    test_recs = [json.loads(line) for line in open(splits_dir / "test.jsonl", encoding="utf-8")]

    # HopeNet range compatibility resolution (Strategy B):
    # If HopeNet: filter training and validation records to half-open supported range [-99.0, +99.0).
    # Log exact counts of excluded samples with zero silent clipping.
    is_hopenet = "hopenet" in model_name.lower()
    if is_hopenet:
        orig_train_len = len(train_recs)
        orig_val_len = len(val_recs)
        train_recs = [r for r in train_recs if -99.0 <= canonicalize_yaw(float(r["yaw"])) < 99.0]
        val_recs = [r for r in val_recs if -99.0 <= canonicalize_yaw(float(r["yaw"])) < 99.0]
        print(f"HopeNet Strategy B Native Support Range [-99°, +99°):")
        print(f"  Train: {len(train_recs)} retained, {orig_train_len - len(train_recs)} excluded (outside [-99°, +99°))")
        print(f"  Val:   {len(val_recs)} retained, {orig_val_len - len(val_recs)} excluded (outside [-99°, +99°))")
    else:
        print(f"ResNet18-Yaw Strategy: Full Range [-180°, +180°) circular sin/cos regression")
        print(f"  Train: {len(train_recs)} retained (100%), Val: {len(val_recs)} retained (100%)")

    print(f"AFLW-GT Train: {len(train_recs)} | AFLW-GT Val: {len(val_recs)} | AFLW2000-3D Test: {len(test_recs)}")

    train_transform, eval_transform = get_hp_transforms(224)

    train_ds = HeadPoseDataset(train_recs, resolution=224, transform=train_transform)
    val_ds = HeadPoseDataset(val_recs, resolution=224, transform=eval_transform)
    test_ds = HeadPoseDataset(test_recs, resolution=224, transform=eval_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=False, collate_fn=collate_hp_train
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False, collate_fn=collate_hp_eval
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False, collate_fn=collate_hp_eval
    )

    model = create_headpose_model(model_name=model_name, pretrained=True).to(device)

    ce_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    if output_dir is None:
        output_dir = PROJECT_ROOT / "runs/v4c" / f"headpose_{model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = output_dir / "best_model.pt"
    last_ckpt_path = output_dir / "last_checkpoint.pt"
    history_csv = output_dir / "training_history.csv"

    start_epoch = 1
    best_val_mae = float("inf")
    best_epoch = -1
    patience_counter = 0

    if resume and last_ckpt_path.exists():
        try:
            ckpt_data = torch.load(last_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt_data["state_dict"])
            optimizer.load_state_dict(ckpt_data["optimizer_state"])
            scheduler.load_state_dict(ckpt_data["scheduler_state"])
            scaler.load_state_dict(ckpt_data["scaler_state"])
            start_epoch = ckpt_data["epoch"] + 1
            best_val_mae = ckpt_data.get("best_val_mae", float("inf"))
            best_epoch = ckpt_data.get("best_epoch", -1)
            print(f">>> RESUMING Head-Pose from Epoch {start_epoch} (Best MAE: {best_val_mae:.2f}° at Epoch {best_epoch}) <<<")
        except Exception as e:
            print(f"Warning: Could not resume from {last_ckpt_path}: {e}. Starting fresh.")

    if not history_csv.exists():
        with open(history_csv, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_mae,best_val_mae,best_epoch,timestamp\n")

    start_time = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.perf_counter()

        for tensors, yaws in train_loader:
            tensors = tensors.to(device, non_blocking=True)
            yaws = yaws.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                if is_hopenet:
                    logits, pred_yaw = model(tensors)
                    bin_targets = model.angle_to_bin(yaws, strict=True)
                    loss_ce = ce_criterion(logits, bin_targets)
                    ang_diff = torch.remainder(pred_yaw - yaws + 180.0, 360.0) - 180.0
                    loss_mse = torch.mean(ang_diff ** 2)
                    loss = loss_ce + 0.5 * loss_mse
                else:
                    raw_vec, pred_yaw = model(tensors)
                    target_rad = yaws * (torch.pi / 180.0)
                    target_vec = torch.stack([torch.sin(target_rad), torch.cos(target_rad)], dim=-1)

                    pred_norm = torch.norm(raw_vec, p=2, dim=1, keepdim=True)
                    unit_vec = raw_vec / (pred_norm + 1e-7)

                    loss_mse = F.mse_loss(raw_vec, target_vec)
                    cos_sim = torch.sum(unit_vec * target_vec, dim=1)
                    loss_cos = 1.0 - torch.mean(cos_sim)
                    loss = loss_mse + loss_cos

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(yaws)

        scheduler.step()
        train_loss /= len(train_recs)
        epoch_sec = time.perf_counter() - t0

        val_metrics, _ = evaluate_hp_dataloader(model, val_loader, device=device)
        val_mae = val_metrics["mae_deg"]

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_sec:.1f}s] - Train Loss: {train_loss:.4f} - "
            f"Val MAE: {val_mae:.2f}° (Med: {val_metrics['median_error_deg']:.2f}°, P90: {val_metrics['p90_deg']:.2f}°) - "
            f"FrontalMod: {val_metrics['slices']['frontal_moderate_reference_slice']['mae_deg']:.2f}°, ClearTurn: {val_metrics['slices']['clear_turn_reference_slice']['mae_deg']:.2f}°"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0

            save_headpose_checkpoint(
                model=model,
                filepath=best_ckpt_path,
                model_name=model_name,
                epoch=epoch,
                metrics=val_metrics,
                optimizer_state=optimizer.state_dict(),
                extra_metadata={"best_epoch": best_epoch, "val_mae": val_mae},
            )
            print(f"  -> Saved new best checkpoint (Val MAE: {best_val_mae:.2f}°)")
        else:
            patience_counter += 1

        # History CSV
        timestamp_str = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(history_csv, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.4f},{val_mae:.2f},{best_val_mae:.2f},{best_epoch},{timestamp_str}\n")

        # Atomic last checkpoint
        last_ckpt_payload = {
            "model_name": model_name,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_mae": best_val_mae,
            "best_epoch": best_epoch,
        }
        tmp_last = last_ckpt_path.with_suffix(".tmp")
        torch.save(last_ckpt_payload, tmp_last)
        tmp_last.replace(last_ckpt_path)

        # Heartbeat
        update_hp_heartbeat(
            model_name=model_name,
            current_epoch=epoch,
            max_epochs=epochs,
            last_completed_epoch=epoch,
            best_epoch=best_epoch,
            best_mae=best_val_mae,
            status="TRAINING",
        )

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    total_training_sec = time.perf_counter() - start_time
    print(f"Head-pose training completed in {total_training_sec:.1f}s. Best Epoch: {best_epoch} with Val MAE: {best_val_mae:.2f}°")

    # Reload best model for final evaluation
    best_model, _ = load_headpose_checkpoint(best_ckpt_path, device=device)

    # 1. Final AFLW-GT Val Evaluation
    final_val_metrics, val_details = evaluate_hp_dataloader(best_model, val_loader, device=device)

    # 2. Final AFLW2000-3D External Test Evaluation (held out until this moment!)
    final_test_metrics, test_details = evaluate_hp_dataloader(best_model, test_loader, device=device)

    print(f"\n=======================================================")
    print(f"FINAL EVALUATION RESULTS for {model_name}:")
    print(f"=======================================================")
    print(f"AFLW-GT Val MAE: {final_val_metrics['mae_deg']}° | Median: {final_val_metrics['median_error_deg']}° | P90: {final_val_metrics['p90_deg']}°")

    # COMPARISON A — COMMON SUPPORT ([-99.0, +99.0))
    comm_slice = final_test_metrics['slices']['hopenet_common_support_slice']
    print(f"\n[COMPARISON A — COMMON SUPPORT ([-99°, +99°), N={comm_slice['n']}]:")
    print(f"  MAE: {comm_slice['mae_deg']}° | Med: {comm_slice['median_ae_deg']}° | P75: {comm_slice['p75_deg']}° | P90: {comm_slice['p90_deg']}°")

    # COMPARISON B — FULL DOMAIN ([-180.0, +180.0))
    print(f"\n[COMPARISON B — FULL DOMAIN (N={final_test_metrics['count']})]:")
    if is_hopenet:
        print(f"  Status: NOT_SUPPORTED_OUTSIDE_NATIVE_RANGE (5 samples outside [-99°, +99°) unpredicted)")
        print(f"  Supported Subset Only: MAE = {comm_slice['mae_deg']}°")
    else:
        print(f"  Full Test MAE: {final_test_metrics['mae_deg']}° | Med: {final_test_metrics['median_error_deg']}° | P90: {final_test_metrics['p90_deg']}°")

    # Classroom Reference Slices
    fm_slice = final_test_metrics['slices']['frontal_moderate_reference_slice']
    ct_slice = final_test_metrics['slices']['clear_turn_reference_slice']
    ot_slice = final_test_metrics['slices']['ontology_turn_reference_slice']
    print(f"\n[CLASSROOM-RELEVANT REFERENCE SLICES]:")
    print(f"  Frontal-Moderate (|yaw| <= 45°, N={fm_slice['n']}): MAE={fm_slice['mae_deg']}° | Med={fm_slice['median_ae_deg']}° | P90={fm_slice['p90_deg']}°")
    print(f"  Clear Turn (35° <= |yaw| < 90°, N={ct_slice['n']}): MAE={ct_slice['mae_deg']}° | Med={ct_slice['median_ae_deg']}° | P90={ct_slice['p90_deg']}°")
    print(f"  Ontology Turn (35° <= |yaw| <= 75°, N={ot_slice['n']}): MAE={ot_slice['mae_deg']}° | Med={ot_slice['median_ae_deg']}° | P90={ot_slice['p90_deg']}°")

    results = {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "training_duration_sec": round(total_training_sec, 2),
        "checkpoint_path": str(best_ckpt_path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/"),
        "provenance": {
            "aflw_gt": "DIRECT SOURCE TARGET: Euler signed yaw ONLY (AFLW_GT_crop_yaws.npy). Pitch and roll are physically absent.",
            "aflw2000_3d": "EXTERNAL TEST: Euler signed yaw ONLY (AFLW2000-3D.pose.npy, 1D array). Pitch and roll in manifest are derived from 3D landmarks (AFLW2000-3D.pts68.npy).",
            "primary_task": "SIGNED YAW. Pitch and roll are not required for V4C classroom monitoring success.",
        },
        "aflw_gt_val": final_val_metrics,
        "aflw2000_3d_test": final_test_metrics,
        "comparison_a_common_support": comm_slice,
        "comparison_b_full_domain": {
            "status": "NOT_SUPPORTED_OUTSIDE_NATIVE_RANGE" if is_hopenet else "EVALUATED_FULL_DOMAIN",
            "full_test_mae": None if is_hopenet else final_test_metrics['mae_deg'],
        },
    }

    # Save metrics JSON
    with open(output_dir / "headpose_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results, best_model



def main():
    parser = argparse.ArgumentParser(description="Train Head-Pose Yaw Regression Model")
    parser.add_argument("--model", type=str, default="hopenet_yaw", choices=["hopenet_yaw", "resnet18_yaw"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    train_headpose_model(
        model_name=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
