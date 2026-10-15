"""Read-only, bounded sampling of an already ACTIVE ExamGuard process; no camera handle."""
import argparse
import json
from pathlib import Path
import statistics
import time
import psutil

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=45)
    args = parser.parse_args()
    path = ROOT / ".runtime/status.json"
    initial = json.loads(path.read_text(encoding="utf-8"))
    if initial.get("monitoring") != "ACTIVE":
        raise SystemExit("BLOCKED: Sign in and explicitly start a monitoring session first.")
    process = psutil.Process(initial["pid"])
    process.cpu_percent()
    samples = []
    start = time.monotonic()
    while time.monotonic()-start < args.seconds:
        time.sleep(2)
        sample = json.loads(path.read_text(encoding="utf-8"))
        if sample.get("pid") != initial["pid"] or sample.get("monitoring") != "ACTIVE" or time.time()-sample.get("updated_at", 0) > 10:
            raise RuntimeError("Live runtime changed or stopped producing fresh inference results")
        sample["cpu_percent_one_core_basis"] = process.cpu_percent()
        samples.append(sample)
    last = samples[-1]
    results = dict(status="PASS", duration_seconds=time.monotonic()-start, sample_count=len(samples),
        camera=last["camera"], ai_fps_mean=statistics.mean(s["ai_fps"] for s in samples),
        capture_fps_mean=statistics.mean(s["camera"]["capture_fps"] for s in samples),
        latency_p50_ms=last["latency_p50_ms"], latency_p95_ms=last["latency_p95_ms"],
        processed_frames_during_observation=last["frames_processed"]-initial["frames_processed"],
        intentional_skips_during_observation=last["skipped_frames"]-initial["skipped_frames"],
        rss_min_mb=min(s["rss_mb"] for s in samples), rss_max_mb=max(s["rss_mb"] for s in samples),
        peak_gpu_mb=last["peak_gpu_mb"], max_queue_depth=max(s["queue_depth"] for s in samples),
        cpu_percent_mean_one_core_basis=statistics.mean(s["cpu_percent_one_core_basis"] for s in samples),
        samples=samples)
    out = ROOT / "reports/final-live-observation.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k,v in results.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
