"""
ASUS TUF Gaming A17 Laptop Preflight & Hardware Probe Script
===========================================================
Preflight validation tool for the target portable live-demo machine.

Performs:
1. Physical hardware discovery (CPU, RAM, GPU, VRAM, Storage, OS) dynamically
2. Target reference comparison against ASUS TUF Gaming A17 (FA707RC)
3. Cryptographic checkpoint verification (7 of 7)
4. Git LFS pointer detection (refuses torch.load if LFS pointer detected)
5. 4 GB VRAM-aware dry loading & memory profiling with OOM protection & memory cleanup
6. Camera device probing & backend capability negotiation (DSHOW, MSMF, ANY)
7. Localhost port 8000 availability check
8. Structured readiness evaluation:
   - repo_ready
   - dependencies_ready
   - cuda_ready
   - checkpoint_ready
   - camera_device_present
   - camera_open_success
   - camera_frame_success
   - camera_mode_known
   - port_ready
   - evidence_dir_ready
   - models_loadable
   - ready_for_laptop_live_validation
   - ready_for_production (strictly NO)
9. Generates reports/ASUS_A17_PREFLIGHT.json & reports/ASUS_A17_CAMERA_CAPABILITIES.json

Outputs canonical console report:
LAPTOP_PREFLIGHT_COMPLETE
"""

import gc
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import psutil
import torch

# Suppress OpenCV C++ low-level warnings during camera probing
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("laptop_preflight")

CERTIFIED_HASHES = {
    "models/trained/yolo26m.pt": "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7",
    "models/trained/stage1_best.pt": "6d713808f0bc670e8bf06e01b006fac73d8d6e5db7130584cec1658b003ce98a",
    "models/trained/stage1_5_best.pt": "68690cf82715dc6dad2eb35a08711531170e935eb7dbe1c30fafabae341e2c2c",
    "models/trained/v4_posture_best.pt": "529a23f96ebec6051ad29279fba3cd5279e7aa8fe2e46292ab4bb9c9523ca180",
    "models/trained/v4_headpose_yaw_best.pt": "5d15eec5941cfc8d2468d09c27bff8eca2014e62bb12fc9a95c0c6239fbbca55",
    "models/fallback/posture_320/best_model.pt": "070a2e328a161b1c957576ec13647d65846a073f9dd35488c7c6edc847f0f4cf",
    "models/fallback/headpose_resnet18/best_model.pt": "bc31d46cfea007ebddfb6a8d5845641a32fd1cc018a831c4c8ae8b61e579a7d9",
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
    """Dynamically query manufacturer, model, CPU, RAM speed using PowerShell / CIM."""
    info = {
        "manufacturer": "Unknown",
        "model": "Unknown",
        "cpu_name": platform.processor(),
        "ram_speed_mts": "Unknown",
        "os_version": platform.platform(),
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


def query_pnp_cameras() -> List[Dict[str, str]]:
    """Query Windows PnP for physical camera devices."""
    cameras = []
    if os.name == "nt":
        try:
            ps_cmd = "Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -eq 'Camera' -or $_.Service -eq 'usbvideo' } | Select-Object Name, DeviceID, Status | ConvertTo-Json"
            res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout)
                if isinstance(data, dict):
                    data = [data]
                for item in data:
                    cameras.append({
                        "name": str(item.get("Name", "")).strip(),
                        "device_id": str(item.get("DeviceID", "")).strip(),
                        "status": str(item.get("Status", "")).strip(),
                    })
        except Exception as e:
            logger.debug(f"PnP camera query error: {e}")
    return cameras


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
    """Verify all 7 checkpoints with LFS pointer detection and SHA-256 integrity."""
    results = {}
    all_ok = True
    lfs_pointers = []

    for path, expected in CERTIFIED_HASHES.items():
        actual_path = path
        if not os.path.exists(actual_path):
            if path == "models/trained/yolo26m.pt" and os.path.exists("yolo26m.pt"):
                actual_path = "yolo26m.pt"
            elif "models/fallback/posture_320" in path and os.path.exists("runs/v4c/C1_mobilenet_v3_small_tight_person_crop_320/best_model.pt"):
                actual_path = "runs/v4c/C1_mobilenet_v3_small_tight_person_crop_320/best_model.pt"
            elif "models/fallback/headpose_resnet18" in path and os.path.exists("runs/v4c/headpose_resnet18_yaw/best_model.pt"):
                actual_path = "runs/v4c/headpose_resnet18_yaw/best_model.pt"

        if not os.path.exists(actual_path):
            results[path] = {"exists": False, "is_lfs_pointer": False, "match": False}
            all_ok = False
            continue

        if check_git_lfs_pointer(actual_path):
            lfs_pointers.append(path)
            results[path] = {"exists": True, "is_lfs_pointer": True, "match": False}
            all_ok = False
            continue

        with open(actual_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        ok = (h.lower() == expected.lower())
        results[path] = {"exists": True, "is_lfs_pointer": False, "actual_hash": h, "match": ok}
        if not ok:
            all_ok = False

    return results, all_ok, lfs_pointers


def probe_gpu_and_driver() -> Dict[str, Any]:
    """Query NVIDIA GPU, VRAM, and driver version dynamically."""
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
    VRAM usage on target 4 GB GPU. Sequential and resident loading with explicit cleanup.
    """
    if not torch.cuda.is_available():
        return {
            "status": "SKIP_NO_CUDA",
            "vram_allocated_peak_mb": 0.0,
            "oom_detected": False,
            "failing_model": None,
            "peak_vram_mb": 0.0,
        }

    torch.cuda.empty_cache()
    gc.collect()
    vram_initial = torch.cuda.memory_allocated(0) / (1024**2)
    step_metrics = {"vram_initial_mb": round(vram_initial, 1)}
    failing_model = None
    oom_detected = False

    loaded_models = []
    try:
        from src.detection.object_detector import YOLOObjectDetector
        from src.detection.behavior_detector import YOLOBehaviorDetector
        from src.models.posture.posture_classifier import load_posture_checkpoint
        from src.models.headpose.headpose_estimator import load_headpose_checkpoint

        # Step 1: General Object Detector (YOLO26m)
        yolo = YOLOObjectDetector(model_path="models/trained/yolo26m.pt", device="cuda:0")
        loaded_models.append(yolo)
        vram_after_yolo = torch.cuda.memory_allocated(0) / (1024**2)
        step_metrics["vram_after_yolo_mb"] = round(vram_after_yolo, 1)

        # Step 2: Posture Classifier (MobileNetV3)
        pos_model, _ = load_posture_checkpoint("models/trained/v4_posture_best.pt", device="cuda:0")
        loaded_models.append(pos_model)
        vram_after_posture = torch.cuda.memory_allocated(0) / (1024**2)
        step_metrics["vram_after_posture_mb"] = round(vram_after_posture, 1)

        # Step 3: Headpose Estimator (HopeNet)
        hp_model, _ = load_headpose_checkpoint("models/trained/v4_headpose_yaw_best.pt", device="cuda:0")
        loaded_models.append(hp_model)
        vram_after_headpose = torch.cuda.memory_allocated(0) / (1024**2)
        step_metrics["vram_after_headpose_mb"] = round(vram_after_headpose, 1)

        # Step 4: Macro Behavior Detector (Stage 1.5)
        macro = YOLOBehaviorDetector(model_path="models/trained/stage1_5_best.pt", device="cuda:0")
        loaded_models.append(macro)
        vram_after_macro = torch.cuda.memory_allocated(0) / (1024**2)
        step_metrics["vram_after_macro_mb"] = round(vram_after_macro, 1)

        peak_vram = torch.cuda.max_memory_allocated(0) / (1024**2)
        step_metrics["peak_vram_mb"] = round(peak_vram, 1)
        step_metrics["status"] = "PASS"

    except torch.cuda.OutOfMemoryError as oom_err:
        oom_detected = True
        failing_model = "CUDA_OOM"
        logger.error(f"CUDA Out of Memory during model dry load: {oom_err}")
        step_metrics["status"] = "FAIL_CUDA_OOM"
    except Exception as ex:
        failing_model = str(type(ex).__name__)
        logger.error(f"Exception during model dry load: {ex}")
        step_metrics["status"] = f"FAIL_{failing_model}"
    finally:
        # Explicit cleanup to ensure VRAM is released after dry load
        del loaded_models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "status": step_metrics.get("status", "FAIL"),
        "step_metrics": step_metrics,
        "oom_detected": oom_detected,
        "failing_model": failing_model,
        "peak_vram_mb": step_metrics.get("peak_vram_mb", 0.0),
    }


def probe_cameras() -> Tuple[List[Dict[str, Any]], bool, int, str, Optional[Tuple[int, int]], Optional[float], Dict[str, Any]]:
    """
    Probes camera indices (0..3) across backends.
    Distinguishes: configured, device_present, open_success, frame_success, connected, streaming.
    """
    pnp_devices = query_pnp_cameras()
    device_present = len(pnp_devices) > 0

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
    detailed_modes: Dict[str, Any] = {}

    for idx in range(4):
        for b_name, b_flag in backends:
            cam_entry = {
                "index": idx,
                "backend": b_name,
                "configured": True,
                "device_present": device_present,
                "open_success": False,
                "frame_success": False,
                "connected": False,
                "streaming": False,
                "reported_width": 0,
                "reported_height": 0,
                "reported_fps": 0.0,
                "actual_width": 0,
                "actual_height": 0,
            }
            try:
                cap = cv2.VideoCapture(idx, b_flag)
                if cap.isOpened():
                    cam_entry["open_success"] = True
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        h, w = frame.shape[:2]
                        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                        cam_entry["frame_success"] = True
                        cam_entry["connected"] = True
                        cam_entry["actual_width"] = w
                        cam_entry["actual_height"] = h
                        cam_entry["reported_width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or w)
                        cam_entry["reported_height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or h)
                        cam_entry["reported_fps"] = fps

                        discovered.append(cam_entry)
                        if first_idx == -1:
                            first_idx = idx
                            first_backend = b_name
                            actual_res = (w, h)
                            actual_fps = fps

                            # Test capability resolutions on the winning device
                            modes_tested = []
                            for req_w, req_h, req_fps in [
                                (1920, 1080, 30.0),
                                (1280, 720, 30.0),
                                (1280, 720, 25.0),
                                (640, 480, 30.0),
                            ]:
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
                                cap.set(cv2.CAP_PROP_FPS, req_fps)
                                time.sleep(0.05)
                                r_test, f_test = cap.read()
                                if r_test and f_test is not None:
                                    th, tw = f_test.shape[:2]
                                    tfps = cap.get(cv2.CAP_PROP_FPS)
                                    modes_tested.append({
                                        "requested": f"{req_w}x{req_h}@{req_fps}",
                                        "delivered": f"{tw}x{th}@{tfps:.1f}",
                                        "success": True,
                                    })
                                else:
                                    modes_tested.append({
                                        "requested": f"{req_w}x{req_h}@{req_fps}",
                                        "delivered": "NONE",
                                        "success": False,
                                    })
                            detailed_modes[f"{idx}:{b_name}"] = modes_tested

                        cap.release()
                        break
                    cap.release()
            except Exception as e:
                logger.debug(f"Camera probe exception on idx {idx} {b_name}: {e}")

    cam_discovered = len(discovered) > 0
    camera_semantics = {
        "pnp_devices": pnp_devices,
        "configured": True,
        "device_present": device_present or cam_discovered,
        "open_success": any(c["open_success"] for c in discovered),
        "frame_success": any(c["frame_success"] for c in discovered),
        "connected": any(c["connected"] for c in discovered),
        "streaming": False,  # Not actively streaming in preflight
        "detailed_modes": detailed_modes,
    }
    return discovered, cam_discovered, first_idx, first_backend, actual_res, actual_fps, camera_semantics


def check_port(host="127.0.0.1", port=8000) -> bool:
    """Check if port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except socket.error:
            return False


def check_evidence_dir(path: str = "evidence/laptop_preflight") -> bool:
    """Verify evidence directory can be created and written."""
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, ".probe_test")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_file)
        return True
    except Exception:
        return False


def compare_expectation(sys_info: Dict[str, Any], gpu_info: Dict[str, Any], mem_gb: float) -> str:
    """
    Compare actual hardware to expected reference:
    ASUS TUF Gaming A17 / FA707RC, Ryzen 7 6800H, 16 GB RAM, RTX 3050 Laptop, 4 GB VRAM.
    Returns: YES, PARTIAL, or NO.
    """
    matches = 0

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


def run_preflight() -> Dict[str, Any]:
    """Execute complete deterministic preflight audit and return structured report."""
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
    import torchvision
    import ultralytics
    torch_ver = torch.__version__
    torch_cuda = torch.version.cuda if torch.cuda.is_available() else "N/A"
    opencv_ver = cv2.__version__
    fastapi_ver = fastapi.__version__
    uvicorn_ver = uvicorn.__version__
    torchvision_ver = torchvision.__version__
    ultralytics_ver = ultralytics.__version__

    # 4. Checkpoints & Git LFS pointer check
    ckpt_results, ckpt_all_ok, lfs_ptrs = verify_checkpoints()

    # 5. Safe VRAM Profiling
    vram_profile = run_vram_aware_dry_load()
    models_loadable = (vram_profile["status"] == "PASS")

    # 6. Camera Discovery
    cams, cam_discovered, cam_idx, cam_backend, cam_res, cam_fps, cam_semantics = probe_cameras()

    # 7. Localhost Port 8000
    port_8000_ok = check_port("127.0.0.1", 8000)

    # 8. Evidence directory
    evidence_ok = check_evidence_dir()

    # 9. Expectation Comparison
    exp_match = compare_expectation(sys_info, gpu_info, ram_gb)

    # 10. Deterministic readiness computations
    repo_ready = True
    dependencies_ready = (
        torch_ver is not None
        and opencv_ver is not None
        and fastapi_ver is not None
        and uvicorn_ver is not None
    )
    cuda_ready = bool(gpu_info["cuda_available"])
    checkpoint_ready = ckpt_all_ok and (len(lfs_ptrs) == 0)
    camera_device_present = bool(cam_semantics["device_present"])
    camera_open_success = bool(cam_semantics["open_success"])
    camera_frame_success = bool(cam_semantics["frame_success"])
    camera_mode_known = (cam_res is not None)
    port_ready = port_8000_ok
    evidence_dir_ready = evidence_ok

    # Ready for live validation on laptop:
    # Requires software, CUDA, checkpoints, loadable models, available port, evidence dir,
    # and working physical camera delivering real frames!
    ready_for_laptop_live_validation = (
        repo_ready
        and dependencies_ready
        and cuda_ready
        and checkpoint_ready
        and models_loadable
        and camera_frame_success
        and port_ready
        and evidence_dir_ready
    )

    if not camera_device_present or not camera_frame_success:
        ready_for_laptop_live_str = "DEFERRED_NO_CAMERA_HARDWARE (Connect physical webcam or enable integrated webcam)"
    elif ready_for_laptop_live_validation:
        ready_for_laptop_live_str = "YES"
    else:
        ready_for_laptop_live_str = "NO"

    # Production readiness remains strictly NO
    ready_for_production_str = "NO (Prototype / Educational Demo Scope; not certified for unmonitored high-stakes testing)"

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware": {
            "manufacturer": sys_info["manufacturer"],
            "model": sys_info["model"],
            "cpu": sys_info["cpu_name"],
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_threads": psutil.cpu_count(logical=True),
            "ram_gb": ram_gb,
            "ram_speed": sys_info["ram_speed_mts"],
            "disk_free_gb": disk_free_gb,
            "os": sys_info["os_version"],
            "expected_hardware_match": exp_match,
        },
        "gpu": gpu_info,
        "software": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "torch": torch_ver,
            "torchvision": torchvision_ver,
            "cuda_build": torch_cuda,
            "opencv": opencv_ver,
            "fastapi": fastapi_ver,
            "uvicorn": uvicorn_ver,
            "ultralytics": ultralytics_ver,
        },
        "checkpoints": {
            "all_ok": ckpt_all_ok,
            "lfs_pointers_detected": lfs_ptrs,
            "details": ckpt_results,
        },
        "vram_profile": vram_profile,
        "camera": {
            "discovered": cam_discovered,
            "winning_index": cam_idx,
            "winning_backend": cam_backend,
            "actual_resolution": [cam_res[0], cam_res[1]] if cam_res else None,
            "actual_fps": cam_fps,
            "semantics": cam_semantics,
            "probed_candidates": cams,
        },
        "readiness": {
            "repo_ready": repo_ready,
            "dependencies_ready": dependencies_ready,
            "cuda_ready": cuda_ready,
            "checkpoint_ready": checkpoint_ready,
            "camera_device_present": camera_device_present,
            "camera_open_success": camera_open_success,
            "camera_frame_success": camera_frame_success,
            "camera_mode_known": camera_mode_known,
            "port_ready": port_ready,
            "evidence_dir_ready": evidence_dir_ready,
            "models_loadable": models_loadable,
            "READY_FOR_LAPTOP_LIVE_VALIDATION": ready_for_laptop_live_str,
            "READY_FOR_PRODUCTION": ready_for_production_str,
        },
    }

    # Save structured reports
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "ASUS_A17_PREFLIGHT.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(reports_dir / "ASUS_A17_CAMERA_CAPABILITIES.json", "w", encoding="utf-8") as f:
        json.dump({
            "pnp_devices": cam_semantics["pnp_devices"],
            "primary_camera_index": cam_idx,
            "primary_backend": cam_backend,
            "default_resolution": cam_res,
            "default_fps": cam_fps,
            "supported_modes": cam_semantics.get("detailed_modes", {}),
            "camera_semantics": {
                "configured": cam_semantics["configured"],
                "device_present": cam_semantics["device_present"],
                "open_success": cam_semantics["open_success"],
                "frame_success": cam_semantics["frame_success"],
                "connected": cam_semantics["connected"],
                "streaming": cam_semantics["streaming"],
            }
        }, f, indent=2)

    return report


def main():
    print("======================================================================")
    print("ASUS TUF GAMING A17 LIVE-DEMO TARGET PREFLIGHT")
    print("======================================================================")

    report = run_preflight()
    sys_info = report["hardware"]
    gpu_info = report["gpu"]
    vram_profile = report["vram_profile"]
    readiness = report["readiness"]
    cam = report["camera"]

    if report["checkpoints"]["lfs_pointers_detected"]:
        print("\n[CRITICAL WARNING: GIT LFS POINTERS DETECTED]")
        print("The following checkpoints are Git LFS pointer files (not physical model weights):")
        for lp in report["checkpoints"]["lfs_pointers_detected"]:
            print(f"  - {lp}")
        print("----------------------------------------------------------------------\n")

    if vram_profile["oom_detected"]:
        gpu_mem_status = "FAIL / REQUIRES_RUNTIME_PROFILE_ADJUSTMENT"
    elif vram_profile["status"] == "PASS":
        gpu_mem_status = f"PASS (Peak {vram_profile['peak_vram_mb']} MB / Headroom {round(gpu_info['vram_mb'] - vram_profile['peak_vram_mb'], 1)} MB)"
    else:
        gpu_mem_status = vram_profile["status"]

    print("LAPTOP_PREFLIGHT_COMPLETE\n")
    print(f"SYSTEM_MANUFACTURER = {sys_info['manufacturer']}")
    print(f"SYSTEM_MODEL = {sys_info['model']}\n")
    print(f"CPU = {sys_info['cpu']}")
    print(f"PHYSICAL_CORES = {sys_info['physical_cores']}")
    print(f"LOGICAL_THREADS = {sys_info['logical_threads']}\n")
    print(f"RAM_GB = {sys_info['ram_gb']} (Speed: {sys_info['ram_speed']})\n")
    print(f"NVIDIA_GPU = {gpu_info['gpu_name']}")
    print(f"GPU_VRAM_MB = {gpu_info['vram_mb']}")
    print(f"NVIDIA_DRIVER = {gpu_info['driver_version']} (smi_available: {gpu_info['nvidia_smi_available']})\n")
    print(f"CUDA_AVAILABLE = {'YES' if gpu_info['cuda_available'] else 'NO'}")
    print(f"TORCH_VERSION = {report['software']['torch']}")
    print(f"TORCH_CUDA_VERSION = {report['software']['cuda_build']}\n")
    print(f"CAMERA_DISCOVERED = {'YES' if cam['discovered'] else 'NO'}")
    print(f"CAMERA_INDEX = {cam['winning_index'] if cam['discovered'] else 'NONE'}")
    print(f"CAMERA_BACKEND = {cam['winning_backend'] if cam['discovered'] else 'NONE'}")
    res_str = f"{cam['actual_resolution'][0]}x{cam['actual_resolution'][1]}" if cam["actual_resolution"] else "NONE"
    print(f"CAMERA_ACTUAL_RESOLUTION = {res_str}")
    print(f"CAMERA_ACTUAL_FPS = {cam['actual_fps'] if cam['actual_fps'] is not None else 'NONE'}\n")
    print(f"CHECKPOINTS_VERIFIED = {'YES (7/7 Certified OK)' if report['checkpoints']['all_ok'] else 'FAIL'}")
    print(f"PORT_8000_AVAILABLE = {'YES' if readiness['port_ready'] else 'NO'}\n")
    print(f"EXPECTED_HARDWARE_MATCH = {sys_info['expected_hardware_match']}\n")
    print(f"GPU_MEMORY_PREFLIGHT = {gpu_mem_status}\n")
    print(f"READY_FOR_LAPTOP_LIVE_VALIDATION = {readiness['READY_FOR_LAPTOP_LIVE_VALIDATION']}\n")
    print(f"READY_FOR_PRODUCTION = {readiness['READY_FOR_PRODUCTION']}\n")
    print("======================================================================")


if __name__ == "__main__":
    main()
