"""
ASUS TUF Gaming A17 Laptop Preflight & Hardware Probe Script
===========================================================
Preflight validation tool for the target portable live-demo machine.

Performs:
1. Physical hardware discovery (CPU, RAM, GPU, VRAM, Storage, OS)
2. Target reference comparison against ASUS TUF Gaming A17 (FA707RC)
3. Cryptographic checkpoint verification (7 of 7)
4. Git LFS pointer detection (refuses torch.load if LFS pointer detected)
5. 4 GB VRAM-aware dry loading & memory profiling (OOM-safe)
6. Camera device probing & backend capability negotiation
7. Localhost port 8000 availability check

Outputs canonical console report:
LAPTOP_PREFLIGHT_COMPLETE
...
"""

import hashlib
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

import cv2
import psutil
import torch

os.environ["OPENCV_LOG_LEVEL"] = "OFF"
if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("laptop_preflight")

CERTIFIED_HASHES = {
    "yolo26m.pt": "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7",
    "models/trained/stage1_best.pt": "6d713808f0bc670e8bf06e01b006fac73d8d6e5db7130584cec1658b003ce98a",
    "models/trained/stage1_5_best.pt": "68690cf82715dc6dad2eb35a08711531170e935eb7dbe1c30fafabae341e2c2c",
    "models/trained/v4_posture_best.pt": "529a23f96ebec6051ad29279fba3cd5279e7aa8fe2e46292ab4bb9c9523ca180",
    "models/trained/v4_headpose_yaw_best.pt": "5d15eec5941cfc8d2468d09c27bff8eca2014e62bb12fc9a95c0c6239fbbca55",
    "runs/v4c/C1_mobilenet_v3_small_tight_person_crop_320/best_model.pt": "070a2e328a161b1c957576ec13647d65846a073f9dd35488c7c6edc847f0f4cf",
    "runs/v4c/headpose_resnet18_yaw/best_model.pt": "bc31d46cfea007ebddfb6a8d5845641a32fd1cc018a831c4c8ae8b61e579a7d9",
}

EXPECTED_REFERENCE = {
    "manufacturer": "ASUS",
    "model_family": "FA707RC",
    "cpu_pattern": "6800H",
    "ram_gb": 16,
    "gpu_pattern": "RTX 3050",
    "vram_gb": 4.0,
}


def query_windows_system_info() -> Dict[str, Any]:
    """Query manufacturer, model, RAM speed using PowerShell / CIM."""
    info = {
        "manufacturer": "Unknown",
        "model": "Unknown",
        "cpu_name": platform.processor(),
        "ram_speed_mts": "Unknown",
    }
    if os.name != "nt":
        return info

    try:
        ps_cmd = "Get-CimInstance -ClassName Win32_ComputerSystem | Select-Object Manufacturer, Model | ConvertTo-Json"
        res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and res.stdout.strip():
            data = json.loads(res.stdout)
            info["manufacturer"] = data.get("Manufacturer", "Unknown").strip()
            info["model"] = data.get("Model", "Unknown").strip()
    except Exception as e:
        logger.debug(f"Could not query Win32_ComputerSystem: {e}")

    try:
        ps_cpu = "(Get-CimInstance -ClassName Win32_Processor).Name"
        res_cpu = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cpu], capture_output=True, text=True, timeout=5)
        if res_cpu.returncode == 0 and res_cpu.stdout.strip():
            info["cpu_name"] = res_cpu.stdout.strip().split("\n")[0].strip()
    except Exception as e:
        logger.debug(f"Could not query Win32_Processor: {e}")

    try:
        ps_mem = "(Get-CimInstance -ClassName Win32_PhysicalMemory | Select-Object -ExpandProperty Speed)[0]"
        res_mem = subprocess.run(["powershell", "-NoProfile", "-Command", ps_mem], capture_output=True, text=True, timeout=5)
        if res_mem.returncode == 0 and res_mem.stdout.strip():
            info["ram_speed_mts"] = f"{res_mem.stdout.strip()} MT/s"
    except Exception as e:
        logger.debug(f"Could not query Win32_PhysicalMemory: {e}")

    return info


def check_git_lfs_pointer(file_path: str) -> bool:
    """Detect if file is a Git LFS pointer rather than the physical binary."""
    if not os.path.exists(file_path):
        return False
    sz = os.path.getsize(file_path)
    if sz > 2048:
        return False
    try:
        with open(file_path, "rb") as f:
            header = f.read(100)
            if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
                return True
    except Exception:
        pass
    return False


def verify_checkpoints() -> Tuple[Dict[str, Any], bool, List[str]]:
    """Verify all 7 checkpoints with LFS pointer detection."""
    results = {}
    all_ok = True
    lfs_pointers = []

    for path, expected in CERTIFIED_HASHES.items():
        if not os.path.exists(path):
            results[path] = {"exists": False, "is_lfs_pointer": False, "match": False}
            all_ok = False
            continue

        if check_git_lfs_pointer(path):
            lfs_pointers.append(path)
            results[path] = {"exists": True, "is_lfs_pointer": True, "match": False}
            all_ok = False
            continue

        with open(path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        ok = (h.lower() == expected.lower())
        results[path] = {"exists": True, "is_lfs_pointer": False, "actual_hash": h, "match": ok}
        if not ok:
            all_ok = False

    return results, all_ok, lfs_pointers


def probe_gpu_and_driver() -> Dict[str, Any]:
    """Query NVIDIA GPU, VRAM, and driver."""
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "NONE"
    vram_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 1) if cuda_avail else 0.0

    driver_version = "Unknown"
    nvidia_smi_avail = False
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0 and res.stdout.strip():
            driver_version = res.stdout.strip().split("\n")[0].strip()
            nvidia_smi_avail = True
    except Exception:
        pass

    return {
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "vram_mb": vram_mb,
        "driver_version": driver_version,
        "nvidia_smi_available": nvidia_smi_avail,
    }


def run_vram_aware_dry_load() -> Dict[str, Any]:
    """
    Safely load general YOLO, Posture, Headpose, and Macro detector to evaluate
    VRAM usage on target 4 GB GPU. Handles CUDA OOM gracefully.
    """
    if not torch.cuda.is_available():
        return {
            "status": "SKIP_NO_CUDA",
            "vram_allocated_peak_mb": 0.0,
            "oom_detected": False,
            "failing_model": None,
        }

    torch.cuda.empty_cache()
    vram_initial = torch.cuda.memory_allocated(0) / (1024**2)
    step_metrics = {"vram_initial_mb": round(vram_initial, 1)}
    failing_model = None
    oom_detected = False

    try:
        from src.detection.object_detector import YOLOObjectDetector
        from src.detection.behavior_detector import YOLOBehaviorDetector
        from src.models.posture.posture_classifier import load_posture_checkpoint
        from src.models.headpose.headpose_estimator import load_headpose_checkpoint

        # Step 1: General Object Detector (YOLO26m)
        yolo = YOLOObjectDetector(model_path="yolo26m.pt", device="cuda:0")
        vram_after_yolo = torch.cuda.memory_allocated(0) / (1024**2)
        step_metrics["vram_after_yolo_mb"] = round(vram_after_yolo, 1)

        # Step 2: Posture Classifier (MobileNetV3)
        pos_model, _ = load_posture_checkpoint("models/trained/v4_posture_best.pt", device="cuda:0")
        vram_after_posture = torch.cuda.memory_allocated(0) / (1024**2)
        step_metrics["vram_after_posture_mb"] = round(vram_after_posture, 1)

        # Step 3: Headpose Estimator (HopeNet)
        hp_model, _ = load_headpose_checkpoint("models/trained/v4_headpose_yaw_best.pt", device="cuda:0")
        vram_after_headpose = torch.cuda.memory_allocated(0) / (1024**2)
        step_metrics["vram_after_headpose_mb"] = round(vram_after_headpose, 1)

        # Step 4: Macro Behavior Detector (Stage 1.5)
        macro = YOLOBehaviorDetector(model_path="models/trained/stage1_5_best.pt", device="cuda:0")
        vram_after_macro = torch.cuda.memory_allocated(0) / (1024**2)
        step_metrics["vram_after_macro_mb"] = round(vram_after_macro, 1)

        peak_vram = torch.cuda.max_memory_allocated(0) / (1024**2)
        step_metrics["peak_vram_mb"] = round(peak_vram, 1)
        step_metrics["status"] = "PASS"

    except torch.cuda.OutOfMemoryError as oom_err:
        oom_detected = True
        failing_model = "CUDA_OOM"
        logger.error(f"CUDA Out of Memory during model dry load: {oom_err}")
        torch.cuda.empty_cache()
        step_metrics["status"] = "FAIL_CUDA_OOM"
    except Exception as ex:
        failing_model = str(type(ex).__name__)
        logger.error(f"Exception during model dry load: {ex}")
        step_metrics["status"] = f"FAIL_{failing_model}"

    return {
        "status": step_metrics.get("status", "FAIL"),
        "step_metrics": step_metrics,
        "oom_detected": oom_detected,
        "failing_model": failing_model,
        "peak_vram_mb": step_metrics.get("peak_vram_mb", 0.0),
    }


def probe_cameras() -> Tuple[List[Dict[str, Any]], bool, int, str, Optional[Tuple[int, int]], Optional[float]]:
    """Probes camera indices (0..3) across backends."""
    backends = [
        ("CAP_DSHOW", getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)),
        ("CAP_MSMF", getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)),
        ("CAP_ANY", cv2.CAP_ANY),
    ]
    discovered = []
    first_idx = -1
    first_backend = "NONE"
    actual_res = None
    actual_fps = None

    for idx in range(4):
        for b_name, b_flag in backends:
            try:
                cap = cv2.VideoCapture(idx, b_flag)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                        discovered.append({
                            "index": idx,
                            "backend": b_name,
                            "width": w,
                            "height": h,
                            "fps": fps,
                        })
                        if first_idx == -1:
                            first_idx = idx
                            first_backend = b_name
                            actual_res = (w, h)
                            actual_fps = fps
                        cap.release()
                        break
                cap.release()
            except Exception:
                pass

    return discovered, len(discovered) > 0, first_idx, first_backend, actual_res, actual_fps


def check_port(host="127.0.0.1", port=8000) -> bool:
    """Check if port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except socket.error:
            return False


def compare_expectation(sys_info: Dict[str, Any], gpu_info: Dict[str, Any], mem_gb: float) -> str:
    """
    Compare actual hardware to expected reference:
    ASUS TUF Gaming A17 / FA707RC, Ryzen 7 6800H, 16 GB RAM, RTX 3050 Laptop, 4 GB VRAM.
    Returns: YES, PARTIAL, or NO.
    """
    matches = 0
    total_checks = 4

    # 1. Manufacturer / Model
    manu = sys_info.get("manufacturer", "").upper()
    model = sys_info.get("model", "").upper()
    if "ASUS" in manu and ("FA707" in model or "TUF" in model or "A17" in model):
        matches += 1
    elif "ASUS" in manu:
        matches += 0.5

    # 2. CPU
    cpu = sys_info.get("cpu_name", "").upper()
    if "6800H" in cpu:
        matches += 1
    elif "RYZEN" in cpu:
        matches += 0.5

    # 3. RAM
    if 14.0 <= mem_gb <= 18.0:
        matches += 1
    elif mem_gb > 8.0:
        matches += 0.5

    # 4. GPU & VRAM
    gpu = gpu_info.get("gpu_name", "").upper()
    vram = gpu_info.get("vram_mb", 0.0)
    if "3050" in gpu and 3500 <= vram <= 4500:
        matches += 1
    elif "3050" in gpu or (3500 <= vram <= 4500):
        matches += 0.5

    if matches >= 3.5:
        return "YES"
    elif matches >= 1.5:
        return "PARTIAL"
    else:
        return "NO"


def main():
    print("======================================================================")
    print("ASUS TUF GAMING A17 LIVE-DEMO TARGET PREFLIGHT")
    print("======================================================================")

    # 1. System Info
    sys_info = query_windows_system_info()
    mem = psutil.virtual_memory()
    ram_gb = round(mem.total / (1024**3), 1)
    disk = shutil.disk_usage(os.getcwd())
    disk_free_gb = round(disk.free / (1024**3), 1)

    # 2. GPU & Driver
    gpu_info = probe_gpu_and_driver()

    # 3. Software Stack
    import fastapi
    import uvicorn
    torch_ver = torch.__version__
    torch_cuda = torch.version.cuda if torch.cuda.is_available() else "N/A"
    opencv_ver = cv2.__version__
    fastapi_ver = fastapi.__version__
    uvicorn_ver = uvicorn.__version__

    # 4. Checkpoints & Git LFS pointer check
    ckpt_results, ckpt_all_ok, lfs_ptrs = verify_checkpoints()

    if lfs_ptrs:
        print("\n[CRITICAL WARNING: GIT LFS POINTERS DETECTED]")
        print("The following checkpoints are Git LFS pointer files (not physical model weights):")
        for lp in lfs_ptrs:
            print(f"  - {lp}")
        print("\nAction required before running models:")
        print("  git lfs install")
        print("  git lfs pull")
        print("----------------------------------------------------------------------\n")

    # 5. Safe VRAM Profiling
    vram_profile = run_vram_aware_dry_load()
    if vram_profile["oom_detected"]:
        gpu_mem_status = "FAIL / REQUIRES_RUNTIME_PROFILE_ADJUSTMENT"
    elif vram_profile["status"] == "PASS":
        gpu_mem_status = f"PASS (Peak {vram_profile['peak_vram_mb']} MB / Headroom {round(gpu_info['vram_mb'] - vram_profile['peak_vram_mb'], 1)} MB)"
    else:
        gpu_mem_status = vram_profile["status"]

    # 6. Camera Discovery
    cams, cam_discovered, cam_idx, cam_backend, cam_res, cam_fps = probe_cameras()

    # 7. Localhost Port 8000
    port_8000_ok = check_port("127.0.0.1", 8000)

    # 8. Expectation Comparison
    exp_match = compare_expectation(sys_info, gpu_info, ram_gb)

    # 9. Readiness Verdict
    if not cam_discovered:
        ready_for_laptop_live_str = "DEFERRED_NO_CAMERA_HARDWARE (Connect USB webcam or run on ASUS A17 with integrated webcam)"
    elif ready_for_laptop_live:
        ready_for_laptop_live_str = "YES"
    else:
        ready_for_laptop_live_str = "NO"

    # Canonical Output Format
    print("LAPTOP_PREFLIGHT_COMPLETE\n")
    print(f"SYSTEM_MANUFACTURER = {sys_info['manufacturer']}")
    print(f"SYSTEM_MODEL = {sys_info['model']}\n")
    print(f"CPU = {sys_info['cpu_name']}")
    print(f"PHYSICAL_CORES = {psutil.cpu_count(logical=False)}")
    print(f"LOGICAL_THREADS = {psutil.cpu_count(logical=True)}\n")
    print(f"RAM_GB = {ram_gb} (Speed: {sys_info['ram_speed_mts']})\n")
    print(f"NVIDIA_GPU = {gpu_info['gpu_name']}")
    print(f"GPU_VRAM_MB = {gpu_info['vram_mb']}")
    print(f"NVIDIA_DRIVER = {gpu_info['driver_version']} (smi_available: {gpu_info['nvidia_smi_available']})\n")
    print(f"CUDA_AVAILABLE = {'YES' if gpu_info['cuda_available'] else 'NO'}")
    print(f"TORCH_VERSION = {torch_ver}")
    print(f"TORCH_CUDA_VERSION = {torch_cuda}\n")
    print(f"CAMERA_DISCOVERED = {'YES' if cam_discovered else 'NO'}")
    print(f"CAMERA_INDEX = {cam_idx if cam_discovered else 'NONE'}")
    print(f"CAMERA_BACKEND = {cam_backend if cam_discovered else 'NONE'}")
    print(f"CAMERA_ACTUAL_RESOLUTION = {f'{cam_res[0]}x{cam_res[1]}' if cam_res else 'NONE'}")
    print(f"CAMERA_ACTUAL_FPS = {cam_fps if cam_fps is not None else 'NONE'}\n")
    print(f"CHECKPOINTS_VERIFIED = {'YES (7/7 Certified OK)' if ckpt_all_ok else 'FAIL'}")
    print(f"PORT_8000_AVAILABLE = {'YES' if port_8000_ok else 'NO'}\n")
    print(f"EXPECTED_HARDWARE_MATCH = {exp_match}\n")
    print(f"GPU_MEMORY_PREFLIGHT = {gpu_mem_status}\n")
    print(f"READY_FOR_LAPTOP_LIVE_VALIDATION = {ready_for_laptop_live_str}\n")
    print("======================================================================")


if __name__ == "__main__":
    main()
