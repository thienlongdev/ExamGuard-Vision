"""Bounded physical acquisition + real CUDA/CPU model validation. Stop the application first."""
import argparse
import copy
import json
from pathlib import Path
import socket
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    import yaml
    from src.orchestration.adaptive_runtime import discover_hardware, save_json, validate_checkpoints, validate_model_execution
    from src.orchestration.model_registry import ModelRegistry
    from src.video.webcam import WebcamSource
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--skip-camera", action="store_true")
    args = parser.parse_args()
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            raise SystemExit("Stop ExamGuard first: never open competing physical camera handles.")
    results = {"hardware": discover_hardware(), "checkpoint_integrity": validate_checkpoints()}
    source = WebcamSource(args.camera_index, width=3840, height=2160, fps=30, fourcc="MJPG", preferred_backend="CAP_DSHOW")
    try:
        if args.skip_camera:
            source = None
        if source is not None:
            results["physical_camera"] = measure_camera(source, args.seconds)
    finally:
        if source is not None:
            source.release()
    save_json(ROOT / "reports/adaptive-workstation.json", results)
    torch.set_num_threads(8)
    config = yaml.safe_load((ROOT / ".runtime/auto-runtime.yaml").read_text(encoding="utf-8"))
    gpu = ModelRegistry(config)
    gpu.initialize_models()
    results["cuda_models"] = validate_model_execution(gpu)
    cpu_config = copy.deepcopy(config)
    cpu_config["device"] = "cpu"
    for model in cpu_config["models"].values():
        model["device"] = "cpu"
    cpu = ModelRegistry(cpu_config)
    cpu.initialize_models()
    results["cpu_models"] = validate_model_execution(cpu)
    x = torch.linspace(-1, 1, 3*224*224).reshape(1, 3, 224, 224)
    with torch.inference_mode():
        pos_cpu = cpu.posture_model(x)
        pos_gpu = gpu.posture_model(x.cuda()).cpu()
        hp_cpu = cpu.headpose_model(x)[1]
        hp_gpu = gpu.headpose_model(x.cuda())[1].cpu()
        results["synthetic_tensor_parity"] = dict(status="PASS" if torch.allclose(pos_cpu, pos_gpu, rtol=1e-3, atol=1e-3) and torch.allclose(hp_cpu, hp_gpu, rtol=1e-3, atol=0.1) else "FAIL",
             posture_class_cpu=pos_cpu.argmax(1).item(), posture_class_cuda=pos_gpu.argmax(1).item(), posture_logits_cpu=pos_cpu.tolist(), posture_logits_cuda=pos_gpu.tolist(),
             posture_max_absolute_error=(pos_cpu-pos_gpu).abs().max().item(), yaw_difference_deg=(hp_cpu-hp_gpu).abs().max().item(),
             tf32_cudnn=torch.backends.cudnn.allow_tf32,
             scope="FP32 CPU/CUDA numerical contracts; not a labeled classroom accuracy study")
    save_json(ROOT / "reports/adaptive-workstation.json", results)
    print(json.dumps(results, indent=2))
    assert results["synthetic_tensor_parity"]["status"] == "PASS", "CPU/CUDA numeric parity failed; details preserved in report"


def measure_camera(source, seconds):
        if not source.open():
            raise RuntimeError("Physical 4K MJPEG mode unavailable")
        for _ in range(15):
            assert source.read() is not None
        start = time.perf_counter()
        frames = 0
        intervals = []
        last = start
        while time.perf_counter() - start < seconds:
            frame = source.read()
            if frame is None:
                raise RuntimeError("Physical camera frame acquisition failed")
            now = time.perf_counter()
            intervals.append((now-last)*1000)
            last = now
            frames += 1
        elapsed = time.perf_counter()-start
        return dict(status="PASS", resolution=[source.width, source.height], fourcc=source.actual_fourcc,
                backend=source.backend_name, frames=frames, seconds=elapsed, capture_fps=frames/elapsed,
                interval_p50_ms=float(np.percentile(intervals, 50)), interval_p95_ms=float(np.percentile(intervals, 95)))


if __name__ == "__main__":
    main()
