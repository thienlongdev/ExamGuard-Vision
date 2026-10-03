"""Preflight discovery and camera probe script."""
import hashlib
import json
import os
import platform
import socket
import sys
import time
import cv2
import psutil
import torch

CERTIFIED_HASHES = {
    "models/trained/yolo26m.pt": "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7",
    "models/trained/stage1_best.pt": "6d713808f0bc670e8bf06e01b006fac73d8d6e5db7130584cec1658b003ce98a",
    "models/trained/stage1_5_best.pt": "68690cf82715dc6dad2eb35a08711531170e935eb7dbe1c30fafabae341e2c2c",
    "models/trained/v4_posture_best.pt": "529a23f96ebec6051ad29279fba3cd5279e7aa8fe2e46292ab4bb9c9523ca180",
    "models/trained/v4_headpose_yaw_best.pt": "5d15eec5941cfc8d2468d09c27bff8eca2014e62bb12fc9a95c0c6239fbbca55",
    "models/fallback/posture_320/best_model.pt": "070a2e328a161b1c957576ec13647d65846a073f9dd35488c7c6edc847f0f4cf",
    "models/fallback/headpose_resnet18/best_model.pt": "bc31d46cfea007ebddfb6a8d5845641a32fd1cc018a831c4c8ae8b61e579a7d9",
}

def verify_checkpoints():
    results = {}
    all_ok = True
    for p, expected in CERTIFIED_HASHES.items():
        if not os.path.exists(p):
            results[p] = {"exists": False, "expected": expected, "actual": None, "match": False}
            all_ok = False
            continue
        with open(p, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        match = (actual == expected)
        results[p] = {"exists": True, "expected": expected, "actual": actual, "match": match}
        if not match:
            all_ok = False
    return results, all_ok

def probe_environment():
    import ultralytics
    import fastapi
    import uvicorn
    
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    vram_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2) if cuda_avail else 0.0
    mem = psutil.virtual_memory()
    
    env_info = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_avail,
        "cuda_version": torch.version.cuda if cuda_avail else None,
        "ultralytics_version": ultralytics.__version__,
        "opencv_version": cv2.__version__,
        "fastapi_version": fastapi.__version__,
        "uvicorn_version": uvicorn.__version__,
        "gpu_name": device_name,
        "gpu_vram_total_mb": round(vram_total_mb, 1),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_available_gb": round(mem.available / (1024**3), 2),
    }
    return env_info

def probe_cameras():
    cameras = []
    backends = [
        ("CAP_DSHOW", cv2.CAP_DSHOW) if os.name == "nt" else ("CAP_ANY", cv2.CAP_ANY),
        ("CAP_ANY", cv2.CAP_ANY),
    ]
    for idx in range(4):
        cam_info = {"index": idx, "open_success": False, "backend": None, "reported_width": 0, "reported_height": 0, "reported_fps": 0.0}
        for b_name, b_flag in backends:
            cap = cv2.VideoCapture(idx, b_flag)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cam_info["open_success"] = True
                    cam_info["backend"] = b_name
                    cam_info["reported_width"] = w
                    cam_info["reported_height"] = h
                    cam_info["reported_fps"] = fps
                    cam_info["test_frame_shape"] = list(frame.shape)
                    cap.release()
                    break
                cap.release()
        cameras.append(cam_info)
    return cameras

def negotiate_camera_capabilities(index, backend_str):
    b_flag = cv2.CAP_DSHOW if backend_str == "CAP_DSHOW" else cv2.CAP_ANY
    modes = [
        (1920, 1080, 30.0),
        (1280, 720, 30.0),
        (1280, 720, 25.0),
        (640, 480, 30.0),
    ]
    results = []
    selected_mode = None
    
    for req_w, req_h, req_fps in modes:
        cap = cv2.VideoCapture(index, b_flag)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
        cap.set(cv2.CAP_PROP_FPS, req_fps)
        time.sleep(0.1)
        
        # Read a few test frames to confirm stable grab
        grabbed = False
        actual_w = 0
        actual_h = 0
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                grabbed = True
                actual_h, actual_w = frame.shape[:2]
        
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        mode_res = {
            "requested": {"width": req_w, "height": req_h, "fps": req_fps},
            "actual": {"width": actual_w, "height": actual_h, "fps": actual_fps},
            "success": grabbed and actual_w > 0 and actual_h > 0
        }
        results.append(mode_res)
        if mode_res["success"] and selected_mode is None:
            selected_mode = mode_res
            
    return results, selected_mode

def check_port(host="127.0.0.1", port=8000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True, port
        except socket.error:
            # find next available port
            for alt_port in range(8001, 8020):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                    try:
                        s2.bind((host, alt_port))
                        return False, alt_port
                    except socket.error:
                        pass
            return False, -1

if __name__ == "__main__":
    os.makedirs("runs/local_live", exist_ok=True)
    
    print("--- 1. Checkpoint Integrity ---")
    hashes, hashes_ok = verify_checkpoints()
    print(f"7 Checkpoints Integrity OK: {hashes_ok}")
    with open("runs/local_live/checkpoint_hashes.json", "w") as f:
        json.dump({"verified_7_of_7": hashes_ok, "checkpoints": hashes}, f, indent=2)
    if not hashes_ok:
        print("CRITICAL HARD STOP: Checkpoint hash mismatch!")
        sys.exit(1)
        
    print("\n--- 2. Environment Preflight ---")
    env = probe_environment()
    print(json.dumps(env, indent=2))
    with open("runs/local_live/test_environment.json", "w") as f:
        json.dump(env, f, indent=2)
        
    print("\n--- 3. Camera Enumeration ---")
    cams = probe_cameras()
    print(json.dumps(cams, indent=2))
    with open("runs/local_live/camera_probe.json", "w") as f:
        json.dump(cams, f, indent=2)
        
    valid_cams = [c for c in cams if c["open_success"]]
    if not valid_cams:
        print("ERROR: No operational webcam discovered on indices 0..3!")
        sys.exit(2)
        
    # Prefer index 0 if valid
    chosen_cam = valid_cams[0]
    for c in valid_cams:
        if c["index"] == 0:
            chosen_cam = c
            break
            
    print(f"Selected Camera Index: {chosen_cam['index']} (Backend: {chosen_cam['backend']})")
    
    print("\n--- 4. Camera Capability Negotiation ---")
    neg_modes, selected = negotiate_camera_capabilities(chosen_cam["index"], chosen_cam["backend"])
    print(json.dumps({"negotiation": neg_modes, "selected": selected}, indent=2))
    with open("runs/local_live/camera_actual_mode.json", "w") as f:
        json.dump({"camera_index": chosen_cam["index"], "backend": chosen_cam["backend"], "negotiation": neg_modes, "selected_mode": selected}, f, indent=2)
        
    print("\n--- 5. Port Preflight ---")
    p_ok, p_val = check_port("127.0.0.1", 8000)
    print(f"Port 8000 available: {p_ok}, assigned port: {p_val}")
    
    print("\nPREFLIGHT CHECKS PASSED.")
