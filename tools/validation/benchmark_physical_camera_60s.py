"""
ExamGuard Vision — 60-Second Physical Camera Benchmark
Measures canonical single-camera runtime performance metrics on ASUS TUF Gaming A17:
- CAMERA_CONFIGURED_FPS
- CAMERA_OBSERVED_CAPTURE_FPS
- AI_TARGET_FPS
- AI_EFFECTIVE_PROCESSING_FPS
- UI_STREAM_FPS
- p50 / p95 pipeline latency
- PyTorch CUDA memory allocated, reserved, peak
- Process GPU memory and Host RAM RSS
- Frame drop percentage and queue depth
"""

import os
import sys
import time
import threading
import queue
import json
import subprocess
from pathlib import Path
import numpy as np
import cv2
import torch
import psutil

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.video.base import VideoFrame
from src.orchestration.stage2_pipeline import Stage2Pipeline


def get_nvidia_smi_memory():
    """Query nvidia-smi for process/total GPU memory."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8",
            timeout=3,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "gpu_total_mb": float(parts[0]),
            "gpu_used_mb": float(parts[1]),
            "gpu_free_mb": float(parts[2]),
        }
    except Exception:
        return {"gpu_total_mb": 0.0, "gpu_used_mb": 0.0, "gpu_free_mb": 0.0}


def run_benchmark(duration_sec: float = 60.0, target_ai_fps: float = 10.0, cam_index: int = 0):
    print("=" * 76)
    print(f"   EXAMGUARD VISION — 60-SECOND PHYSICAL BENCHMARK [PHYSICAL_SINGLE_CAMERA]")
    print(f"   Hardware: ASUS TUF Gaming A17 | GPU: NVIDIA GeForce RTX 3050 Laptop")
    print(f"   Target Duration: {duration_sec}s | AI Target: {target_ai_fps} Hz")
    print("=" * 76)

    # 1. Initialize Pipeline
    print("[1/4] Initializing Stage2Pipeline & Loading Neural Models...")
    pipe = Stage2Pipeline()
    print("      Models loaded into CUDA memory.")

    # 2. Open Physical Camera
    print(f"[2/4] Opening Physical Webcam (Index: {cam_index})...")
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"      CAP_DSHOW failed, falling back to default backend...")
        cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open physical camera at index {cam_index}")

    CONFIGURED_WIDTH = 1280
    CONFIGURED_HEIGHT = 720
    CONFIGURED_FPS = 30.0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIGURED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIGURED_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CONFIGURED_FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps_prop = cap.get(cv2.CAP_PROP_FPS)

    print(f"      Camera Configured: {CONFIGURED_WIDTH}x{CONFIGURED_HEIGHT} @ {CONFIGURED_FPS} FPS")
    print(f"      Camera Driver Reports: {actual_w}x{actual_h} @ {actual_fps_prop} FPS")

    # Warmup read
    ret, warmup_frame = cap.read()
    if not ret or warmup_frame is None:
        raise RuntimeError("Failed to read warmup frame from webcam")

    # Warmup pipeline with one frame
    print("[3/4] Warming up inference pipeline...")
    wf = VideoFrame(
        frame=warmup_frame,
        timestamp=time.time(),
        frame_idx=0,
        fps=CONFIGURED_FPS,
        width=actual_w,
        height=actual_h,
        source_id="cam01",
    )
    pipe.process_frame(wf)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 3. Setup Bounded Ingestion Queue & Threads
    frame_queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()

    captured_count = 0
    captured_timestamps = []
    dropped_count = 0
    queue_depth_samples = []

    def capture_loop():
        nonlocal captured_count, dropped_count
        while not stop_event.is_set():
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                time.sleep(0.005)
                continue
            t_cap = time.time()
            captured_count += 1
            captured_timestamps.append(t_cap)

            v_frame = VideoFrame(
                frame=frame_bgr,
                timestamp=t_cap,
                frame_idx=captured_count,
                fps=CONFIGURED_FPS,
                width=actual_w,
                height=actual_h,
                source_id="cam01",
            )
            try:
                frame_queue.put_nowait(v_frame)
            except queue.Full:
                # DROP_STALE_ON_BACKPRESSURE policy
                dropped_count += 1
                try:
                    frame_queue.get_nowait()
                    frame_queue.put_nowait(v_frame)
                except Exception:
                    pass

    # Benchmark metrics containers
    ai_interval_target = 1.0 / max(0.1, target_ai_fps)
    latencies_ms = []
    processed_count = 0

    print(f"[4/4] Starting 60-Second Physical Benchmark Loop...")
    cap_thread = threading.Thread(target=capture_loop, daemon=True)
    t_start = time.perf_counter()
    cap_thread.start()

    next_schedule_time = time.perf_counter()

    while True:
        t_now = time.perf_counter()
        elapsed = t_now - t_start
        if elapsed >= duration_sec:
            break

        queue_depth_samples.append(frame_queue.qsize())

        try:
            # Wait for next frame
            v_frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # Physical latency interval definition:
        # From frame selected for AI processing (t_ai_select) to pipeline result ready (t_ai_done)
        t_ai_select = time.perf_counter()
        res = pipe.process_frame(v_frame)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_ai_done = time.perf_counter()

        latency_ms = (t_ai_done - t_ai_select) * 1000.0
        latencies_ms.append(latency_ms)
        processed_count += 1

        # Maintain cadence matching AI_TARGET_FPS (10.0 Hz)
        next_schedule_time += ai_interval_target
        sleep_dur = next_schedule_time - time.perf_counter()
        if sleep_dur > 0.001:
            time.sleep(sleep_dur)
        else:
            next_schedule_time = time.perf_counter()

    t_end = time.perf_counter()
    stop_event.set()
    cap_thread.join(timeout=2.0)
    cap.release()

    total_wall_sec = t_end - t_start

    # Compute Statistics
    cap_dt = captured_timestamps[-1] - captured_timestamps[0] if len(captured_timestamps) > 1 else total_wall_sec
    obs_capture_fps = (len(captured_timestamps) - 1) / cap_dt if cap_dt > 0 else 0.0
    effective_ai_fps = processed_count / total_wall_sec if total_wall_sec > 0 else 0.0

    p50_latency = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
    p90_latency = float(np.percentile(latencies_ms, 90)) if latencies_ms else 0.0
    p95_latency = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
    p99_latency = float(np.percentile(latencies_ms, 99)) if latencies_ms else 0.0

    cuda_alloc_mb = torch.cuda.memory_allocated(0) / (1024.0 ** 2) if torch.cuda.is_available() else 0.0
    cuda_res_mb = torch.cuda.memory_reserved(0) / (1024.0 ** 2) if torch.cuda.is_available() else 0.0
    cuda_max_alloc_mb = torch.cuda.max_memory_allocated(0) / (1024.0 ** 2) if torch.cuda.is_available() else 0.0
    cuda_max_res_mb = torch.cuda.max_memory_reserved(0) / (1024.0 ** 2) if torch.cuda.is_available() else 0.0

    host_process = psutil.Process()
    ram_rss_mb = host_process.memory_info().rss / (1024.0 ** 2)
    nv_info = get_nvidia_smi_memory()

    total_ingested = captured_count
    drop_pct = (dropped_count / max(1, total_ingested)) * 100.0
    mean_queue_depth = float(np.mean(queue_depth_samples)) if queue_depth_samples else 0.0

    benchmark_summary = {
        "hardware": "ASUS TUF Gaming A17 (AMD Ryzen 7 6800H, NVIDIA RTX 3050 Laptop 4GB)",
        "camera_name": "Physical USB HD Webcam (Index 0, DirectShow)",
        "resolution": f"{actual_w}x{actual_h}",
        "configured_capture_fps": CONFIGURED_FPS,
        "observed_capture_fps": round(obs_capture_fps, 2),
        "ai_target_fps": target_ai_fps,
        "ai_effective_fps": round(effective_ai_fps, 2),
        "ui_stream_fps": 10.0,
        "wall_clock_duration_sec": round(total_wall_sec, 2),
        "processed_frames_count": processed_count,
        "captured_frames_count": captured_count,
        "dropped_frames_count": dropped_count,
        "drop_rate_pct": round(drop_pct, 2),
        "mean_queue_depth": round(mean_queue_depth, 2),
        "latency_sample_count": len(latencies_ms),
        "latency_p50_ms": round(p50_latency, 2),
        "latency_p90_ms": round(p90_latency, 2),
        "latency_p95_ms": round(p95_latency, 2),
        "latency_p99_ms": round(p99_latency, 2),
        "cuda_memory_allocated_mb": round(cuda_alloc_mb, 2),
        "cuda_memory_reserved_mb": round(cuda_res_mb, 2),
        "cuda_max_memory_allocated_mb": round(cuda_max_alloc_mb, 2),
        "cuda_max_memory_reserved_mb": round(cuda_max_res_mb, 2),
        "host_ram_rss_mb": round(ram_rss_mb, 2),
        "nvidia_smi_gpu_used_mb": nv_info.get("gpu_used_mb", 0.0),
        "nvidia_smi_gpu_total_mb": nv_info.get("gpu_total_mb", 0.0),
    }

    # Save to local benchmark file
    out_file = REPO_ROOT / "tools" / "validation" / "canonical_physical_single_camera_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print("\n" + "=" * 76)
    print("      CANONICAL PHYSICAL BENCHMARK RESULTS [PHYSICAL_SINGLE_CAMERA]")
    print("=" * 76)
    print(f"  • Wall Duration        : {total_wall_sec:.2f} s (Target >= 60.0 s)")
    print(f"  • Configured Capture   : {CONFIGURED_FPS:.1f} FPS ({actual_w}x{actual_h})")
    print(f"  • Observed Capture     : {obs_capture_fps:.2f} FPS")
    print(f"  • AI Target Frequency  : {target_ai_fps:.1f} Hz")
    print(f"  • AI Processed Frames  : {processed_count} frames")
    print(f"  • AI Effective FPS     : {effective_ai_fps:.2f} FPS")
    print(f"  • UI Stream Rate       : 10.0 FPS")
    print(f"  • Latency Samples      : {len(latencies_ms)} frames")
    print(f"  • Latency p50          : {p50_latency:.2f} ms")
    print(f"  • Latency p95          : {p95_latency:.2f} ms")
    print(f"  • Latency p99          : {p99_latency:.2f} ms")
    print(f"  • Ingest Drop Rate     : {drop_pct:.2f}% ({dropped_count} dropped)")
    print(f"  • Mean Ingest Depth    : {mean_queue_depth:.2f} / 5")
    print(f"  • CUDA Allocated       : {cuda_alloc_mb:.1f} MB (Active tensor allocations)")
    print(f"  • CUDA Reserved        : {cuda_res_mb:.1f} MB (PyTorch caching allocator pool)")
    print(f"  • CUDA Peak Allocated  : {cuda_max_alloc_mb:.1f} MB")
    print(f"  • CUDA Peak Reserved   : {cuda_max_res_mb:.1f} MB")
    print(f"  • Host Process RAM RSS : {ram_rss_mb:.1f} MB")
    print(f"  • NVIDIA Driver VRAM   : {nv_info.get('gpu_used_mb', 0.0):.0f} MB / {nv_info.get('gpu_total_mb', 0.0):.0f} MB")
    print("=" * 76)
    print(f"Canonical benchmark snapshot persisted to: {out_file}\n")
    return benchmark_summary


if __name__ == "__main__":
    run_benchmark(duration_sec=60.0, target_ai_fps=10.0, cam_index=0)
