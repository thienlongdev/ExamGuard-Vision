import os
import sys
import hashlib
import json
import torch

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

def get_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def inspect_checkpoint(path):
    print(f"\n==========================================")
    print(f"Inspecting: {path}")
    if not os.path.exists(path):
        print("DOES NOT EXIST")
        return None
    
    sha256 = get_sha256(path)
    print(f"SHA-256: {sha256}")
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Size: {size_mb:.2f} MB")
    
    meta = {
        "checkpoint_path": path,
        "sha256": sha256,
        "size_mb": round(size_mb, 2)
    }
    
    # Inspect raw checkpoint dict
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        meta["is_dict"] = True
        meta["keys"] = list(raw.keys())
        print(f"Keys: {meta['keys']}")
        if "epoch" in raw:
            meta["epoch"] = raw["epoch"]
            print(f"Epoch: {raw['epoch']}")
        if "date" in raw:
            meta["date"] = str(raw["date"])
        if "version" in raw:
            meta["version"] = str(raw["version"])
            print(f"Version: {raw['version']}")
        if "train_args" in raw:
            targs = raw["train_args"]
            if isinstance(targs, dict):
                meta["train_args"] = {k: str(v) for k, v in targs.items() if k in ["model", "imgsz", "data", "batch", "epochs", "task", "device"]}
            else:
                meta["train_args"] = str(targs)
            print(f"Train args: {meta.get('train_args')}")
        if "args" in raw:
            args = raw["args"]
            if hasattr(args, "__dict__"):
                meta["args"] = {k: str(v) for k, v in args.__dict__.items() if k in ["model", "imgsz", "data", "batch", "epochs", "task", "device"]}
            elif isinstance(args, dict):
                meta["args"] = {k: str(v) for k, v in args.items() if k in ["model", "imgsz", "data", "batch", "epochs", "task", "device"]}
            print(f"Args: {meta.get('args')}")
        if "names" in raw:
            meta["names"] = raw["names"]
            meta["num_classes"] = len(raw["names"])
            print(f"Names ({len(raw['names'])} classes): {raw['names']}")
    else:
        meta["is_dict"] = False
        print(f"Raw object type: {type(raw)}")
    
    # Try Ultralytics YOLO loading
    if YOLO:
        try:
            yolo_model = YOLO(path)
            meta["ultralytics_task"] = getattr(yolo_model, "task", None)
            meta["model_class"] = type(yolo_model.model).__name__ if hasattr(yolo_model, "model") else None
            if hasattr(yolo_model, "names"):
                meta["model_names"] = yolo_model.names
            if hasattr(yolo_model, "model") and yolo_model.model is not None:
                m = yolo_model.model
                if hasattr(m, "stride"):
                    meta["stride"] = int(m.stride.max().item()) if hasattr(m.stride, "max") else str(m.stride)
                if hasattr(m, "yaml"):
                    meta["yaml"] = m.yaml
                param_count = sum(p.numel() for p in m.parameters())
                meta["parameter_count"] = param_count
                print(f"Ultralytics Model: task={meta['ultralytics_task']}, class={meta['model_class']}, params={param_count:,}, stride={meta.get('stride')}")
        except Exception as e:
            print(f"Ultralytics load exception: {e}")
            meta["ultralytics_error"] = str(e)
            
    return meta

if __name__ == "__main__":
    candidates = [
        "models/trained/stage1_best.pt",
        "models/trained/stage1_5_best.pt",
        "models/trained/v4_posture_best.pt",
        "models/trained/v4_headpose_yaw_best.pt",
        "models/fallback/posture_320/best_model.pt",
        "models/fallback/headpose_resnet18/best_model.pt",
        "models/trained/yolo26m.pt",
    ]
    results = {}
    for c in candidates:
        r = inspect_checkpoint(c)
        if r:
            results[c] = r
            
    os.makedirs("runs/stage2_integrity", exist_ok=True)
    with open("runs/stage2_integrity/checkpoint_inspection_raw.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved runs/stage2_integrity/checkpoint_inspection_raw.json")
