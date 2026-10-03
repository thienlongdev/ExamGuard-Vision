#!/usr/bin/env python3
"""
scripts/analyze_eduaction_redundancy.py
Analyzes temporal frame-to-frame correlation and redundancy across EduAction video clips.
Measures motion diversity and establishes an evidence-based temporal sampling policy.
"""

import cv2
import json
import numpy as np
from pathlib import Path

def compute_frame_diff(f1, f2):
    """Normalized mean absolute pixel difference in [0, 1]."""
    diff = cv2.absdiff(f1, f2)
    return float(np.mean(diff) / 255.0)

def compute_clip_temporal_stats(video_path, sample_lags=(1, 2, 5, 10, 15, 20, 25)):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize to 112x112 grayscale for rapid, robust difference analysis
        gray = cv2.cvtColor(cv2.resize(frame, (112, 112)), cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()
    
    total_frames = len(frames)
    if total_frames < 2:
        return None
    
    # Compute mean difference at each lag
    lag_diffs = {}
    for lag in sample_lags:
        if lag < total_frames:
            diffs = [compute_frame_diff(frames[i], frames[i + lag]) for i in range(total_frames - lag)]
            lag_diffs[lag] = float(np.mean(diffs))
        else:
            lag_diffs[lag] = None
            
    # Compute consecutive frame differences (lag 1)
    consec_diffs = [compute_frame_diff(frames[i], frames[i + 1]) for i in range(total_frames - 1)]
    mean_motion = float(np.mean(consec_diffs))
    max_motion = float(np.max(consec_diffs))
    min_motion = float(np.min(consec_diffs))
    
    return {
        "clip_name": video_path.name,
        "total_frames": total_frames,
        "mean_consec_diff": mean_motion,
        "max_consec_diff": max_motion,
        "min_consec_diff": min_motion,
        "lag_diffs": lag_diffs
    }

def main():
    root = Path(".")
    edu_dir = root / "datasets/raw_v4/other_candidates/eduaction"
    
    categories = ["sleeping", "writing", "lecture", "talking", "play_phone", "drinking", "watch_computer"]
    
    results = {}
    
    print("Analyzing EduAction temporal correlation...")
    for cat in categories:
        cat_dir = edu_dir / cat
        mp4s = sorted(cat_dir.glob("*.mp4"))
        print(f"Analyzing {cat} ({len(mp4s)} clips)...")
        cat_stats = []
        for p in mp4s:
            s = compute_clip_temporal_stats(p)
            if s:
                cat_stats.append(s)
                
        # Aggregate stats
        total_raw_frames = sum(s["total_frames"] for s in cat_stats)
        mean_motion = float(np.mean([s["mean_consec_diff"] for s in cat_stats]))
        
        # Aggregate lag differences
        avg_lags = {}
        for lag in [1, 2, 5, 10, 15, 20, 25]:
            vals = [s["lag_diffs"][lag] for s in cat_stats if s["lag_diffs"].get(lag) is not None]
            avg_lags[lag] = float(np.mean(vals)) if vals else None
            
        results[cat] = {
            "num_clips": len(cat_stats),
            "total_raw_frames": total_raw_frames,
            "avg_frames_per_clip": total_raw_frames / max(1, len(cat_stats)),
            "mean_consecutive_diff": mean_motion,
            "avg_lag_differences": avg_lags
        }
        print(f"  {cat}: {total_raw_frames} frames, mean consec diff: {mean_motion:.4f}")
        for lag, diff_val in avg_lags.items():
            print(f"    lag {lag:2d} ({lag/25.0:.2f}s): mean diff = {diff_val:.4f}")

    out_file = root / "reports/v4b/eduaction_temporal_redundancy.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved redundancy analysis to {out_file}")

if __name__ == "__main__":
    main()
