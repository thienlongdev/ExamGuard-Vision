"""
V4C A1 Confirmation Run Runner
Executes ONE hardened confirmation run of A1 (ResNet18+CBAM Tight 224) per Section 9.
Compares confirmed A1 with recovered A1 and C1, freezes final winner,
updates reports, runs error analysis, and updates master report.
"""

import sys
import os
import json
import time
import shutil
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tools.research.train_posture_classifier import train_posture_model
    from tools.benchmark.run_posture_benchmark import generate_posture_reports
    from tools.research.generate_posture_error_analysis import generate_error_analysis
    from tools.research.generate_v4c_master_report import generate_master_report
except ImportError:
    from scripts.train_posture_classifier import train_posture_model
    from scripts.run_posture_benchmark import generate_posture_reports
    from scripts.generate_posture_error_analysis import generate_error_analysis
    from scripts.generate_v4c_master_report import generate_master_report

STATE_FILE = PROJECT_ROOT / "runs/v4c/V4C_EXECUTION_STATE.json"
LOG_FILE = PROJECT_ROOT / "runs/v4c/V4C_RECOVERY_LOG.md"


def get_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def append_log(message: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n## [{timestamp}] {message}\n")


def run_a1_confirmation():
    print("======================================================================")
    print("STARTING V4C A1 HARDENED CONFIRMATION RUN (Section 9 Case B)")
    print("======================================================================")

    out_dir = PROJECT_ROOT / "runs/v4c/A1_resnet18_cbam_tight_person_crop_224_confirmed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Update state
    state = json.load(open(STATE_FILE, "r", encoding="utf-8")) if STATE_FILE.exists() else {}
    state["active_task"] = "A1 Hardened Confirmation Run"
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S+07:00")
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    t0 = time.perf_counter()
    metrics = train_posture_model(
        model_name="resnet18_cbam",
        representation="TIGHT_PERSON_CROP",
        resolution=224,
        batch_size=64,
        epochs=30,
        lr=3e-4,
        weight_decay=1e-2,
        patience=6,
        num_workers=0,
        output_dir=out_dir,
    )
    duration = time.perf_counter() - t0
    metrics["exp_id"] = "A1_Confirmed"
    metrics["display_name"] = "ResNet18+CBAM (Tight / 224) [Confirmed]"

    ckpt_file = out_dir / "best_model.pt"
    ckpt_hash = get_sha256(ckpt_file)

    log_msg = (
        f"Completed A1 Confirmation Run:\n"
        f"- Best Epoch: {metrics['best_epoch']}\n"
        f"- Duration: {duration:.1f}s\n"
        f"- Checkpoint: {ckpt_file.name} (SHA256: {ckpt_hash})\n"
        f"- Same-Domain Val Macro F1: {metrics['same_domain_val']['macro_f1']:.4f}\n"
        f"- High-Angle Macro F1: {metrics['high_angle_holdout']['macro_f1']:.4f}\n"
        f"- Cross-Source Macro F1: {metrics['cross_source_holdout']['macro_f1']:.4f}\n"
        f"- Temporal Clip MajAcc: {metrics['temporal_holdout']['clip_majority_accuracy']:.4f}\n"
        f"- Sleep Recall: {metrics['same_domain_val']['per_class']['HEAD_REST_SLEEP']['recall']:.4f}\n"
        f"- Turn Recall: {metrics['same_domain_val']['per_class']['TURN_HEAD_CLEAR']['recall']:.4f}"
    )
    append_log(log_msg)
    print(log_msg)

    # Load matrix summary and update
    matrix_path = PROJECT_ROOT / "runs/v4c/matrix_summary.json"
    with open(matrix_path, "r", encoding="utf-8") as f:
        matrix = json.load(f)

    # Score function
    def score_posture(m):
        ha = m["high_angle_holdout"]["macro_f1"]
        cs = m["cross_source_holdout"]["macro_f1"]
        sv = m["same_domain_val"]["macro_f1"]
        th = m["temporal_holdout"]["clip_majority_accuracy"]
        return 0.30 * ha + 0.30 * cs + 0.20 * sv + 0.20 * th

    # Compare confirmed A1 with recovered A1
    rec_a1 = next((m for m in matrix if m.get("exp_id") == "A1"), None)
    score_conf = score_posture(metrics)
    score_rec = score_posture(rec_a1) if rec_a1 else 0.0

    print(f"\nA1 Comparison: Confirmed Score = {score_conf:.4f} vs Recovered Epoch 22 Score = {score_rec:.4f}")

    # Determine whether confirmed A1 or recovered A1 represents A1
    if score_conf >= score_rec:
        # Update A1 entry in matrix
        for i, m in enumerate(matrix):
            if m.get("exp_id") == "A1":
                matrix[i] = metrics
                matrix[i]["exp_id"] = "A1"
                matrix[i]["display_name"] = "ResNet18+CBAM (Tight / 224)"
                break
        best_a1 = metrics
        chosen_ckpt = ckpt_file
        chosen_hash = ckpt_hash
    else:
        best_a1 = rec_a1
        chosen_ckpt = PROJECT_ROOT / rec_a1["checkpoint_path"]
        chosen_hash = get_sha256(chosen_ckpt)

    # Save updated matrix summary
    with open(matrix_path, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)

    # Regenerate posture reports
    generate_posture_reports(matrix)
    append_log("Regenerated posture benchmark reports with confirmed A1 results.")

    # Select overall posture winner
    sorted_candidates = sorted(matrix, key=score_posture, reverse=True)
    winner_posture = sorted_candidates[0]
    print(f"\n>>> FINAL POSTURE WINNER: {winner_posture['exp_id']} ({winner_posture['display_name']}) Score: {score_posture(winner_posture):.4f} <<<")

    # Freeze final winner
    winner_src = PROJECT_ROOT / winner_posture["checkpoint_path"]
    final_posture_dest = PROJECT_ROOT / "models/trained/v4_posture_best.pt"
    shutil.copy2(winner_src, final_posture_dest)
    final_hash = get_sha256(final_posture_dest)
    append_log(f"FROZEN FINAL POSTURE WINNER -> models/trained/v4_posture_best.pt (SHA256: {final_hash})")

    # Regenerate Error Analysis for final winner
    winner_preds = (PROJECT_ROOT / winner_posture["checkpoint_path"]).parent / "eval_predictions.json"
    generate_error_analysis(predictions_file=winner_preds, output_dir=PROJECT_ROOT / "reports/v4c")
    append_log("Regenerated Posture Error Analysis for final winner.")

    # Regenerate Master Report
    generate_master_report()
    append_log("REGENERATED MASTER REPORT -> reports/V4C_SPECIALIZED_MODELS_FINAL.md")

    # Update execution state
    state["active_task"] = None
    state["next_task"] = "AWAITING_USER_AUTHORIZATION"
    state["completed_tasks"].append("A1 Hardened Confirmation Complete")
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    print("\n======================================================================")
    print("A1 CONFIRMATION AND FINAL POSTURE WINNER SELECTION COMPLETE!")
    print("======================================================================")


if __name__ == "__main__":
    run_a1_confirmation()
