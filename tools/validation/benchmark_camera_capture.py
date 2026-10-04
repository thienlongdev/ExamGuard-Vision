"""
ExamGuard Vision — Camera Capture Isolation Benchmark & Auto-Exposure Study
Isolates the camera hardware and driver from AI inference to determine
the exact physical cause of the ~7.5 FPS capture rate.
"""

import time
import argparse
import json
import numpy as np
import cv2

def decode_fourcc(val):
    if val is None or val == -1:
        return "NONE"
    try:
        val = int(val)
        return "".join([chr((val >> 8 * i) & 0xFF) for i in range(4)])
    except Exception:
        return str(val)

def benchmark_single_configuration(
    cam_index: int = 0,
    backend_name: str = "CAP_DSHOW",
    fourcc_str: str = "DEFAULT",
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    duration_sec: float = 10.0,
    exposure_val: float = None,
    auto_exposure_val: float = None,
):
    backend_flag = cv2.CAP_DSHOW if backend_name == "CAP_DSHOW" else (
        cv2.CAP_MSMF if backend_name == "CAP_MSMF" else cv2.CAP_ANY
    )

    t0_open = time.perf_counter()
    cap = cv2.VideoCapture(cam_index, backend_flag)
    open_latency_ms = (time.perf_counter() - t0_open) * 1000.0

    if not cap.isOpened():
        return {
            "backend": backend_name,
            "fourcc_requested": fourcc_str,
            "open_success": False,
            "error": "Failed to open camera"
        }

    # Set FourCC if requested BEFORE setting width/height
    if fourcc_str and fourcc_str != "DEFAULT":
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc_str)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Auto exposure / exposure adjustment if requested
    initial_auto_exp = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    initial_exp = cap.get(cv2.CAP_PROP_EXPOSURE)

    if auto_exposure_val is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure_val)
    if exposure_val is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure_val)

    actual_auto_exp = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    actual_exp = cap.get(cv2.CAP_PROP_EXPOSURE)

    negotiated_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    negotiated_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    driver_fps = cap.get(cv2.CAP_PROP_FPS)
    reported_fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))

    # Warmup read (discard first few frames for camera AGC/AEC settle)
    warmup_count = 5
    for _ in range(warmup_count):
        cap.read()

    frame_intervals = []
    read_latencies = []
    brightnesses = []
    total_frames = 0
    stall_count = 0
    t_start = time.perf_counter()
    t_prev = t_start

    while True:
        t_now = time.perf_counter()
        if t_now - t_start >= duration_sec:
            break

        t_read_start = time.perf_counter()
        ret, frame = cap.read()
        t_read_done = time.perf_counter()

        if not ret or frame is None:
            stall_count += 1
            time.sleep(0.005)
            continue

        read_latencies.append((t_read_done - t_read_start) * 1000.0)
        dt = t_read_done - t_prev
        t_prev = t_read_done
        if total_frames > 0:
            frame_intervals.append(dt * 1000.0)

        # Sample brightness every 5 frames
        if total_frames % 5 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightnesses.append(float(np.mean(gray)))

        total_frames += 1

    t_end = time.perf_counter()
    cap.release()

    total_time = t_end - t_start
    actual_fps = total_frames / max(0.001, total_time)
    p50_interval = float(np.percentile(frame_intervals, 50)) if frame_intervals else 0.0
    p95_interval = float(np.percentile(frame_intervals, 95)) if frame_intervals else 0.0
    p50_read_latency = float(np.percentile(read_latencies, 50)) if read_latencies else 0.0
    p95_read_latency = float(np.percentile(read_latencies, 95)) if read_latencies else 0.0
    mean_brightness = float(np.mean(brightnesses)) if brightnesses else 0.0

    return {
        "backend": backend_name,
        "fourcc_requested": fourcc_str,
        "fourcc_reported": reported_fourcc,
        "open_success": True,
        "open_latency_ms": round(open_latency_ms, 2),
        "requested_dims": f"{width}x{height}@{fps}",
        "negotiated_dims": f"{negotiated_w}x{negotiated_h}",
        "driver_reported_fps": driver_fps,
        "duration_sec": round(total_time, 2),
        "total_frames": total_frames,
        "actual_read_fps": round(actual_fps, 2),
        "interval_p50_ms": round(p50_interval, 2),
        "interval_p95_ms": round(p95_interval, 2),
        "read_latency_p50_ms": round(p50_read_latency, 2),
        "read_latency_p95_ms": round(p95_read_latency, 2),
        "stall_count": stall_count,
        "initial_auto_exposure": initial_auto_exp,
        "initial_exposure": initial_exp,
        "actual_auto_exposure": actual_auto_exp,
        "actual_exposure": actual_exp,
        "mean_brightness": round(mean_brightness, 2),
    }

def main():
    parser = argparse.ArgumentParser(description="Camera-only Benchmark & Exposure Study")
    parser.add_argument("--cam-index", type=int, default=0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--full-soak", action="store_true")
    args = parser.parse_args()

    dur = 60.0 if args.full_soak else args.duration

    print("=" * 80)
    print("   EXAMGUARD VISION — CAMERA-ONLY HARDWARE BENCHMARK & EXPOSURE STUDY")
    print(f"   Camera Index: {args.cam_index} | Duration per test: {dur}s")
    print("=" * 80)

    configs_to_test = [
        # 1. DirectShow Default (current production path)
        {"backend": "CAP_DSHOW", "fourcc": "DEFAULT", "desc": "DirectShow Default (YUY2 raw)"},
        # 2. DirectShow MJPG
        {"backend": "CAP_DSHOW", "fourcc": "MJPG", "desc": "DirectShow MJPG"},
        # 3. DirectShow YUY2 explicit
        {"backend": "CAP_DSHOW", "fourcc": "YUY2", "desc": "DirectShow YUY2 explicit"},
        # 4. MSMF Default
        {"backend": "CAP_MSMF", "fourcc": "DEFAULT", "desc": "Media Foundation Default"},
        # 5. MSMF MJPG
        {"backend": "CAP_MSMF", "fourcc": "MJPG", "desc": "Media Foundation MJPG"},
    ]

    results = []
    for cfg in configs_to_test:
        print(f"\n---> Testing: {cfg['desc']} ...")
        res = benchmark_single_configuration(
            cam_index=args.cam_index,
            backend_name=cfg["backend"],
            fourcc_str=cfg["fourcc"],
            duration_sec=dur,
        )
        results.append(res)
        if res.get("open_success"):
            print(f"     Negotiated: {res['negotiated_dims']} | FourCC: {res['fourcc_reported']}")
            print(f"     Actual Read FPS: {res['actual_read_fps']} FPS (Interval p50: {res['interval_p50_ms']} ms, p95: {res['interval_p95_ms']} ms)")
            print(f"     Read Latency p50: {res['read_latency_p50_ms']} ms | Mean Brightness: {res['mean_brightness']}")
            print(f"     Exposure props: auto={res['actual_auto_exposure']}, exp={res['actual_exposure']}")
        else:
            print(f"     Failed: {res.get('error')}")
        time.sleep(1.0)

    # Now test exposure variations under DirectShow (best/standard backend)
    print("\n" + "=" * 80)
    print("   AUTO-EXPOSURE & SHUTTER TEST (CAP_DSHOW)")
    print("=" * 80)
    # Check if we can test manual exposure values
    exposure_tests = [
        {"fourcc": "DEFAULT", "auto": None, "exp": None, "desc": "Default Auto Exposure"},
        {"fourcc": "MJPG", "auto": None, "exp": None, "desc": "MJPG Auto Exposure"},
        # On DirectShow, auto_exposure is often 0.75 or 1 (auto) vs 0.25 (manual) or cv2.CAP_PROP_AUTO_EXPOSURE = 1 or 3
        # cv2.CAP_PROP_EXPOSURE is typically -4 to -7 (-6 = 1/64s, -5 = 1/32s, -7 = 1/128s)
        {"fourcc": "DEFAULT", "auto": 0.25, "exp": -6.0, "desc": "YUY2 Manual Shorter Exposure (-6.0)"},
        {"fourcc": "MJPG", "auto": 0.25, "exp": -6.0, "desc": "MJPG Manual Shorter Exposure (-6.0)"},
    ]

    exp_results = []
    for et in exposure_tests:
        print(f"\n---> Testing: {et['desc']} ...")
        res = benchmark_single_configuration(
            cam_index=args.cam_index,
            backend_name="CAP_DSHOW",
            fourcc_str=et["fourcc"],
            auto_exposure_val=et["auto"],
            exposure_val=et["exp"],
            duration_sec=min(dur, 10.0),
        )
        exp_results.append(et | {"result": res})
        if res.get("open_success"):
            print(f"     FourCC: {res['fourcc_reported']} | Actual FPS: {res['actual_read_fps']} FPS")
            print(f"     Brightness: {res['mean_brightness']} | Read Latency p50: {res['read_latency_p50_ms']} ms")
            print(f"     Reported Exposure: auto={res['actual_auto_exposure']}, val={res['actual_exposure']}")
        else:
            print(f"     Failed: {res.get('error')}")
        time.sleep(1.0)

    # Save benchmark results
    out_path = "tools/validation/camera_capture_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"backends": results, "exposure_tests": exp_results}, f, indent=2)
    print(f"\n[DONE] Results saved to {out_path}")

if __name__ == "__main__":
    main()
