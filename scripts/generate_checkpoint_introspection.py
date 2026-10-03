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

def introspect_checkpoint(path, role_hint=None):
    if not os.path.exists(path):
        return None
    
    sha = get_sha256(path)
    size_mb = round(os.path.getsize(path) / (1024 * 1024), 2)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    
    entry = {
        "checkpoint_path": path.replace("\\", "/"),
        "sha256": sha,
        "size_mb": size_mb,
        "role_hint": role_hint,
    }
    
    if isinstance(raw, dict):
        entry["version"] = str(raw.get("version", "unknown"))
        entry["epoch"] = raw.get("epoch", None)
        
        # Train args
        train_args = raw.get("train_args", {})
        if isinstance(train_args, dict):
            entry["train_args"] = {
                "task": train_args.get("task"),
                "model": train_args.get("model"),
                "data": train_args.get("data"),
                "epochs": train_args.get("epochs"),
                "batch": train_args.get("batch"),
                "imgsz": train_args.get("imgsz"),
                "device": train_args.get("device"),
            }
            entry["training_imgsz"] = train_args.get("imgsz")
        else:
            entry["train_args"] = str(train_args)
            entry["training_imgsz"] = None
            
        # Names
        model_obj = raw.get("model")
        if hasattr(model_obj, "names"):
            names = model_obj.names
        else:
            names = raw.get("names", {})
            
        entry["names"] = names
        entry["num_classes"] = len(names) if names else 0
        
        # Model architecture & params
        if model_obj is not None:
            entry["model_class"] = type(model_obj).__name__
            if hasattr(model_obj, "yaml"):
                entry["yaml_descriptor"] = model_obj.yaml
                entry["architecture_yaml"] = model_obj.yaml.get("yaml_file", "unknown")
                entry["scale"] = model_obj.yaml.get("scale", "unknown")
            else:
                entry["yaml_descriptor"] = None
                entry["architecture_yaml"] = "ULTRALYTICS_YOLO_CUSTOM_CHECKPOINT"
                entry["scale"] = "unknown"
            
            if hasattr(model_obj, "stride"):
                entry["stride"] = int(model_obj.stride.max().item()) if hasattr(model_obj.stride, "max") else str(model_obj.stride)
            else:
                entry["stride"] = 32
                
            entry["parameter_count"] = sum(p.numel() for p in model_obj.parameters())
    else:
        entry["model_class"] = type(raw).__name__
        entry["names"] = getattr(raw, "names", {})
        entry["num_classes"] = len(entry["names"])
        entry["parameter_count"] = sum(p.numel() for p in raw.parameters()) if hasattr(raw, "parameters") else 0
        entry["architecture_yaml"] = "ULTRALYTICS_YOLO_CUSTOM_CHECKPOINT"
        entry["stride"] = 32
        entry["training_imgsz"] = None

    # Load with Ultralytics runtime to verify runtime task
    if YOLO:
        try:
            y = YOLO(path)
            entry["ultralytics_task"] = getattr(y, "task", "detect")
            entry["ultralytics_runtime_class"] = type(y.model).__name__ if hasattr(y, "model") else None
            if not entry.get("names") and hasattr(y, "names"):
                entry["names"] = y.names
                entry["num_classes"] = len(y.names)
        except Exception as e:
            entry["ultralytics_runtime_error"] = str(e)
            
    # Classify taxonomy facts
    names_dict = entry.get("names", {})
    has_person = any(v == "person" for v in names_dict.values()) if isinstance(names_dict, dict) else False
    has_cell_phone = any(v == "cell phone" or v == "phone" for v in names_dict.values()) if isinstance(names_dict, dict) else False
    has_custom_behavior = any(v in ["normal", "head_down", "turn_head", "discuss", "stand"] for v in names_dict.values()) if isinstance(names_dict, dict) else False
    
    entry["taxonomy_analysis"] = {
        "has_person_class": has_person,
        "person_class_id": [k for k, v in names_dict.items() if v == "person"][0] if has_person else None,
        "has_cell_phone_class": has_cell_phone,
        "cell_phone_class_id": [k for k, v in names_dict.items() if v in ["cell phone", "phone"]][0] if has_cell_phone else None,
        "has_custom_behavior_taxonomy": has_custom_behavior,
        "is_coco_compatible": has_person and has_cell_phone,
    }
    
    return entry

def main():
    checkpoints = [
        ("models/trained/stage1_best.pt", "Protected Baseline Stage 1"),
        ("models/trained/stage1_5_best.pt", "Macro Behavior Stage 1.5"),
        ("yolo26m.pt", "Local General Object Detector Candidate"),
        ("yolo11n.pt", "Local Lightweight Object Detector Candidate"),
    ]
    
    records = {}
    for p, hint in checkpoints:
        r = introspect_checkpoint(p, hint)
        if r:
            records[p] = r
            
    os.makedirs("runs/stage2_integrity", exist_ok=True)
    os.makedirs("reports/stage2_integrity", exist_ok=True)
    
    out_json = "runs/stage2_integrity/detector_checkpoint_introspection.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Generated {out_json}")

if __name__ == "__main__":
    main()
