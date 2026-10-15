"""Local capability discovery and session-bound Auto configuration (no behavior rules)."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import threading
from pathlib import Path

import psutil
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / ".runtime"
_write_lock = threading.RLock()


def command(args, timeout=12):
    p = subprocess.run(args, capture_output=True, text=True, errors="replace", timeout=timeout,
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    return p.stdout + p.stderr


def save_json(path, data):
    with _write_lock:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


def discover_cameras():
    """DirectShow enumeration order matches OpenCV DSHOW indices; monikers persist across boots."""
    if shutil.which("ffmpeg"):
        out = command(["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"])
        cameras = []
        for line in out.splitlines():
            m = re.search(r'"(.*)" \(video\)', line)
            if m:
                cameras.append({"index": len(cameras), "name": m[1], "id": ""})
            elif cameras and not cameras[-1]["id"]:
                m = re.search(r'Alternative name "(.*)"', line)
                if m:
                    cameras[-1]["id"] = m[1]
        if cameras:
            return cameras
    # Single-camera PnP fallback is unambiguous. Multiple PnP devices cannot safely map to indices.
    out = command(["powershell.exe", "-NoProfile", "-Command",
                   "Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -in @('Camera','Image') } | Select-Object Name,PNPDeviceID | ConvertTo-Json"])
    devices = json.loads(out) if out.strip() else []
    devices = [devices] if isinstance(devices, dict) else devices
    if len(devices) > 1:
        raise RuntimeError("Multiple cameras: install FFmpeg for stable DirectShow device selection.")
    return [{"index": 0, "name": d["Name"], "id": d["PNPDeviceID"]} for d in devices]


def choose_camera(cameras, saved_id=None, index=None):
    if not cameras:
        raise RuntimeError("BLOCKED: No physical camera. Connect it and check Windows camera permissions.")
    if saved_id and index is None:
        match = next((c for c in cameras if c["id"] == saved_id), None)
        if match is None:
            raise RuntimeError("BLOCKED: Saved camera is disconnected. Use Setup ExamGuard Vision to explicitly select a replacement.")
        return match
    if index is not None:
        match = next((c for c in cameras if c["index"] == index), None)
        if match is None:
            raise RuntimeError("Selected camera index is unavailable.")
        return match
    if len(cameras) == 1:
        return cameras[0]
    import tkinter as tk
    from tkinter import simpledialog
    root = tk.Tk()
    root.withdraw()
    answer = simpledialog.askinteger("ExamGuard Vision", "Choose camera once:\n" + "\n".join(f'{c["index"]}: {c["name"]}' for c in cameras), parent=root)
    root.destroy()
    if answer is None:
        raise RuntimeError("Camera selection cancelled. No monitoring session started.")
    return choose_camera(cameras, index=answer)


def camera_modes(camera):
    if not shutil.which("ffmpeg"):
        return []
    out = command(["ffmpeg", "-hide_banner", "-list_options", "true", "-f", "dshow", "-i", f'video={camera["id"]}'])
    found = set()
    for line in out.splitlines():
        m = re.search(r'(vcodec=mjpeg|pixel_format=\w+).*min s=(\d+)x(\d+) fps=([\d.]+)', line)
        if m:
            found.add((int(m[2]), int(m[3]), float(m[4]), "MJPG" if m[1] == "vcodec=mjpeg" else m[1].split("=")[1]))
    return [dict(width=w, height=h, fps=f, fourcc=c) for w, h, f, c in sorted(found)]


def discover_hardware():
    data = {"os": platform.platform(), "architecture": platform.machine(), "python": sys.version,
            "cpu": platform.processor(), "cpu_threads": psutil.cpu_count(),
            "ram_mb": round(psutil.virtual_memory().total / 2**20),
            "ram_available_mb": round(psutil.virtual_memory().available / 2**20),
            "versions": {}, "nvidia": [], "device": "cpu", "fallback_reason": None,
            "supported_backends": ["cpu"]}
    for name in ("torch", "torchvision", "opencv-python", "ultralytics", "numpy"):
        data["versions"][name] = importlib.metadata.version(name)
    data["cuda_build"] = torch.version.cuda
    if shutil.which("nvidia-smi"):
        rows = command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"])
        data["nvidia"] = [row.strip() for row in rows.splitlines() if "," in row]
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable; CPU inference selected.")
        # A real kernel, not just a driver discovery claim.
        x = torch.ones((32, 32), device="cuda:0")
        assert float((x @ x).mean().item()) == 32.0
        torch.cuda.synchronize()
        props = torch.cuda.get_device_properties(0)
        data.update(device="cuda:0", gpu=props.name, vram_mb=round(props.total_memory / 2**20),
                    compute_capability=list(torch.cuda.get_device_capability(0)), cuda_arches=torch.cuda.get_arch_list())
        data["supported_backends"].append("cuda:0")
    except Exception as exc:
        data["fallback_reason"] = str(exc)
    return data


def validate_checkpoints():
    from src.pilot.preflight import CERTIFIED_HASHES
    from src.orchestration.model_registry import compute_sha256
    cache_path = LOCAL / "checkpoint-integrity.json"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    results = {}
    for relative, expected in CERTIFIED_HASHES.items():
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size < 1024:
            raise RuntimeError(f"BLOCKED: Missing/LFS-pointer checkpoint {relative}. Run git lfs pull during setup.")
        stat = path.stat()
        signature = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, expected]
        old = cache.get(relative, {})
        digest = old.get("sha256") if old.get("signature") == signature else compute_sha256(path)
        if digest != expected:
            raise RuntimeError(f"BLOCKED: Corrupt checkpoint {relative}; SHA-256 mismatch.")
        results[relative] = {"signature": signature, "sha256": digest}
    save_json(cache_path, results)
    return results


def fingerprint(hardware, camera):
    stable = {k: hardware.get(k) for k in ("os", "architecture", "python", "cpu", "ram_mb", "versions", "nvidia", "device")}
    stable["camera"] = camera
    # Invalidate measured configuration on changes to runtime or frozen settings.
    for relative in ("configs/stage2_pipeline.yaml", "configs/v4d_fusion.yaml", "src/video/webcam.py", "src/orchestration/adaptive_runtime.py"):
        stable[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def select_detector_image_size(hardware):
    """Choose the validated detector scale from capability, never GPU name."""
    return 960 if (
        str(hardware.get("device", "cpu")).startswith("cuda")
        and hardware.get("vram_mb", 0) >= 8192
        and hardware.get("ram_available_mb", 0) >= 4096
    ) else 640


def prepare_auto(index=None, force=False):
    LOCAL.mkdir(exist_ok=True)
    probe = LOCAL / "write-check.tmp"
    probe.write_text("ok")
    probe.unlink()
    state_path = LOCAL / "auto-settings.json"
    try:
        saved = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    hardware = discover_hardware()
    source_path = LOCAL / "source-settings.json"
    source_settings = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {"type": "webcam"}
    source_type = source_settings["type"]
    if source_type == "webcam":
        saved_id = source_settings.get("camera", {}).get("id") or (None if force else saved.get("camera", {}).get("id"))
        camera = choose_camera(discover_cameras(), saved_id, index)
    else:
        camera = {"index": 0, "name": source_settings["name"], "id": source_settings["path"], "type": source_type}
    validate_checkpoints()
    key = fingerprint(hardware, camera)
    cached = not force and saved.get("fingerprint") == key and saved.get("verified")
    modes = saved.get("advertised_modes", []) if cached else (camera_modes(camera) if source_type == "webcam" else [])
    mode = saved["mode"] if cached else (negotiate_mode(camera, modes, hardware) if source_type == "webcam" else source_settings["mode"])
    with (ROOT / "configs/stage2_pipeline.yaml").open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["device"] = hardware["device"]
    for model in config["models"].values():
        model["device"] = hardware["device"]
    # Controlled RTX 5070 benchmark: 960 recovered materially stronger small-
    # phone predictions with a modest latency increase, while 1280 cost more
    # and was not consistently better. Keep 640 on CPU/low-VRAM systems,
    # including the 4 GB ASUS reference configuration.
    detector_image_size = select_detector_image_size(hardware)
    for detector_key in ("general_object_detector", "detector"):
        if detector_key in config["models"]:
            config["models"][detector_key]["image_size"] = detector_image_size
    config["api"]["host"] = "127.0.0.1"
    config["video_source"].update(default_type=source_type, default_path=camera["index"] if source_type == "webcam" else source_settings["path"], camera_id=f'{source_type}_{camera["index"]}',
                                  width=mode["width"], height=mode["height"], target_fps=mode["fps"],
                                  camera_backend="CAP_DSHOW", fourcc=mode.get("fourcc"), async_capture=True)
    # Fixed model resolutions/precision and evidence policy; only scheduling/buffering adapt.
    config["performance"].update(
        headpose_precision="fp32",
        posture_precision="fp32",
        preview_max_width=1280,
        detector_image_size=detector_image_size,
        detector_resolution_policy="960_ON_VALIDATED_HIGH_MEMORY_CUDA_ELSE_640",
    )
    if cached and "cadence" in saved:
        config["cadence"] = saved["cadence"]
    observed = saved.get("observed_runtime", {}) if cached else {}
    if observed:
        # Measured full-pipeline throughput is more representative than isolated model timings.
        target = max(1.0, min(mode["fps"], observed.get("ai_fps", 12.0)))
        if observed.get("latency_p95_ms", 0) > 120 or observed.get("queue_depth", 0) > 1:
            target *= 0.8
        if observed.get("peak_gpu_mb", 0) > hardware.get("vram_mb", 1) * 0.85 or hardware["ram_available_mb"] < 2048:
            target *= 0.8
        config["performance"]["auto_ai_target_fps"] = round(target, 2)
    if hardware["device"] == "cpu":
        config["batching"].update(max_posture_batch_size=4, max_headpose_batch_size=4)
    state = dict(fingerprint=key, hardware=hardware, camera=camera, mode=mode, advertised_modes=modes,
                 verified=False, reused_verified_settings=bool(cached), cadence=config["cadence"], observed_runtime=observed)
    path = LOCAL / "auto-runtime.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path, state


def negotiate_mode(camera, modes, hardware):
    from src.video.webcam import WebcamSource
    # Prefer 4K30 only when advertised, within host RAM budget, and physically measured sustainable.
    max_pixels = 3840 * 2160 if hardware["device"].startswith("cuda") and hardware["ram_available_mb"] > 4096 else 1920 * 1080
    candidates = sorted([m for m in modes if m["width"] * m["height"] <= max_pixels and m["fps"] >= 25 and m["fourcc"] == "MJPG"],
                        key=lambda m: m["width"] * m["height"], reverse=True)
    if not candidates:
        candidates = [dict(width=1280, height=720, fps=30.0, fourcc="MJPG"), dict(width=640, height=480, fps=30.0, fourcc=None)]
    for mode in candidates[:5]:
        source = WebcamSource(camera["index"], width=mode["width"], height=mode["height"], fps=mode["fps"], preferred_backend="CAP_DSHOW", fourcc=mode["fourcc"])
        try:
            if not source.open():
                continue
            for _ in range(5):
                source.read()
            start = time.perf_counter()
            count = 0
            while time.perf_counter() - start < 2.0:
                if source.read() is None:
                    break
                count += 1
            measured = count / (time.perf_counter() - start)
            if count and measured >= min(20, mode["fps"] * 0.75):
                return dict(width=source.width, height=source.height, fps=source.fps,
                            fourcc=source.actual_fourcc, measured_capture_fps=round(measured, 2), backend=source.backend_name)
        finally:
            source.release()
    raise RuntimeError("BLOCKED: No sustainable camera mode. Check camera permissions, USB connection or another camera application.")


def validate_model_execution(registry):
    """Every required model executes; failures block readiness. Timings use CUDA synchronization."""
    import numpy as np
    device = registry.device
    cuda = device.type == "cuda"
    frame = np.zeros((768, 768, 3), dtype=np.uint8)
    outputs = {}
    with torch.inference_mode():
        for name, wrapper in (("detector", registry.detector), ("macro", registry.macro_detector)):
            for _ in range(3):
                wrapper.detect(frame)
            if cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = wrapper.detect(frame)
            if cuda:
                torch.cuda.synchronize()
            # Ultralytics keeps an unfused CPU checkpoint model and executes a separate
            # fused AutoBackend model. Inspect the backend that actually performed predict().
            actual = str(next(wrapper._model.predictor.model.model.parameters()).device)
            if actual != str(device):
                raise RuntimeError(f"{name} device mismatch: {actual} != {device}")
            outputs[name] = dict(device=actual, forward_ms=round((time.perf_counter()-start)*1000, 2), detections=len(result))
        # Ultralytics select_device enables TF32 globally. Restore strict FP32 for classifier
        # score consistency; no threshold or checkpoint changes are used to mask numeric drift.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        for name, model in (("posture", registry.posture_model), ("headpose", registry.headpose_model)):
            x = torch.zeros((1, 3, 224, 224), device=device)
            model(x)
            if cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            y = model(x)
            if cuda:
                torch.cuda.synchronize()
            tensors = list(y) if isinstance(y, tuple) else [y]
            if not all(t.device == x.device and torch.isfinite(t).all().item() for t in tensors):
                raise RuntimeError(f"Invalid {name} model outputs")
            outputs[name] = dict(device=str(x.device), shapes=[list(t.shape) for t in tensors], forward_ms=round((time.perf_counter()-start)*1000, 2))
    return outputs


def select_cadence(model_results, hardware, measured_camera_fps):
    """Conservative measured bootstrap; apply at startup only, never inside an event window."""
    budget_ms = sum(result["forward_ms"] for result in model_results.values())
    rate = max(1.0, min(measured_camera_fps, 30.0, 700.0 / max(1.0, budget_ms)))
    # VRAM pressure changes batching/cadence, never model shape or scientific thresholds.
    if hardware.get("vram_mb", 0) and hardware["vram_mb"] < 2048:
        rate = min(rate, 8.0)
    return {"detector_hz": round(rate, 2), "posture_hz": min(10.0, rate),
            "headpose_hz": min(6.0, rate), "phone_hz": min(6.0, rate), "macro_hz": min(6.0, rate)}
