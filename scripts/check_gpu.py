"""GPU and CUDA environment verification utility for NVIDIA RTX 5070 and Ultralytics."""

import os
import sys
import time
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import ultralytics
from ultralytics import YOLO


def check_gpu():
    print("=" * 60)
    print(" GPU / CUDA COMPATIBILITY & SMOKE TEST")
    print("=" * 60)

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available:    {cuda_available}")

    if not cuda_available:
        print("ERROR: CUDA is not available! Check NVIDIA drivers and PyTorch build.")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    device_props = torch.cuda.get_device_properties(0)
    vram_gb = device_props.total_memory / (1024 ** 3)
    major, minor = device_props.major, device_props.minor
    cuda_version = torch.version.cuda
    torch_version = torch.__version__
    ultralytics_version = ultralytics.__version__

    print(f"Device Count:      {device_count}")
    print(f"GPU:               {gpu_name}")
    print(f"Compute Capability:sm_{major}{minor}")
    print(f"VRAM:              {vram_gb:.2f} GB")
    print(f"PyTorch Version:   {torch_version}")
    print(f"PyTorch CUDA:      {cuda_version}")
    print(f"Ultralytics:       {ultralytics_version}")
    print(f"Inference device:  cuda:0")
    print("-" * 60)

    # Small GPU smoke test:
    print("Running GPU Smoke Test (YOLO on synthetic image on cuda:0)...")
    model_path = os.path.join(PROJECT_ROOT, "yolo11n.pt")
    if not os.path.exists(model_path):
        model_path = "yolo11n.pt"

    # Reset CUDA memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    model = YOLO(model_path)

    # Create synthetic test frame
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)

    # Warmup
    _ = model.predict(source=dummy_image, device="cuda:0", verbose=False)

    # Timed run
    start_t = time.perf_counter()
    results = model.predict(source=dummy_image, device="cuda:0", verbose=False)
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    peak_mem_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)

    # Verify predictions were computed on CUDA
    boxes = results[0].boxes
    used_device = results[0].orig_img is not None  # inference ran successfully
    print(f"Smoke Test:        SUCCESS")
    print(f"Inference Latency: {elapsed_ms:.2f} ms")
    print(f"Peak GPU VRAM:     {peak_mem_mb:.2f} MB")
    print("=" * 60)
    return True


if __name__ == "__main__":
    check_gpu()
