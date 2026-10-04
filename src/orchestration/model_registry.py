"""
Stage 2 Model Registry & Singleton Lifecycle Manager
===================================================
Loads frozen perception checkpoints once, computes exact SHA-256 digests,
dynamically introspects physical model architecture and taxonomy from weights,
warms up GPU execution, and prevents checkpoint reloading during runtime.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import threading
import torch
import torch.nn as nn

from src.detection.object_detector import YOLOObjectDetector
from src.detection.behavior_detector import YOLOBehaviorDetector
from src.models.posture.posture_classifier import load_posture_checkpoint, PosturePredictor
from src.models.headpose.headpose_estimator import load_headpose_checkpoint, HeadPosePredictor

logger = logging.getLogger(__name__)


def compute_sha256(filepath: str | Path) -> str:
    """Compute exact SHA-256 hex digest of a file."""
    p = Path(filepath)
    if not p.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def introspect_yolo_metadata(
    checkpoint_path: str,
    runtime_imgsz: int,
    runtime_role: str,
    device: str,
    yolo_wrapper: Optional[Any] = None,
) -> Dict[str, Any]:
    """Dynamically derive self-describing metadata from physical YOLO checkpoint."""
    p = Path(checkpoint_path)
    sha256 = compute_sha256(p)
    meta: Dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path).replace("\\", "/"),
        "sha256": sha256,
        "runtime_role": runtime_role,
        "runtime_image_size": [runtime_imgsz, runtime_imgsz],
        "device": str(device),
        "precision_mode": "fp32",
    }

    if not p.exists():
        meta["status"] = "FILE_NOT_FOUND"
        return meta

    # Inspect physical checkpoint dictionary directly
    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict):
            meta["framework_version"] = str(raw.get("version", "unknown"))
            train_args = raw.get("train_args")
            if isinstance(train_args, dict):
                meta["training_imgsz"] = train_args.get("imgsz")
                meta["training_task"] = train_args.get("task")
                meta["training_model_origin"] = str(train_args.get("model"))
            else:
                meta["training_imgsz"] = None
                meta["training_task"] = None

            model_obj = raw.get("model")
            if model_obj is not None:
                meta["model_class"] = type(model_obj).__name__
                if hasattr(model_obj, "yaml") and isinstance(model_obj.yaml, dict):
                    yaml_f = model_obj.yaml.get("yaml_file")
                    scale = model_obj.yaml.get("scale")
                    meta["architecture_descriptor"] = f"{yaml_f} (scale={scale})" if yaml_f else "ULTRALYTICS_YOLO_CUSTOM_CHECKPOINT"
                else:
                    meta["architecture_descriptor"] = "ULTRALYTICS_YOLO_CUSTOM_CHECKPOINT"

                if hasattr(model_obj, "stride"):
                    meta["stride"] = int(model_obj.stride.max().item()) if hasattr(model_obj.stride, "max") else 32
                else:
                    meta["stride"] = 32

                meta["parameter_count"] = sum(param.numel() for param in model_obj.parameters())
                if hasattr(model_obj, "names") and isinstance(model_obj.names, dict):
                    meta["actual_names"] = model_obj.names
                    meta["num_classes"] = len(model_obj.names)
        else:
            meta["architecture_descriptor"] = "ULTRALYTICS_YOLO_CUSTOM_CHECKPOINT"
            meta["parameter_count"] = sum(param.numel() for param in raw.parameters()) if hasattr(raw, "parameters") else 0
            meta["stride"] = 32
    except Exception as e:
        logger.warning(f"Could not read raw checkpoint dictionary for '{checkpoint_path}': {e}")
        meta["architecture_descriptor"] = "ULTRALYTICS_YOLO_CUSTOM_CHECKPOINT"

    # Fill names from wrapper if missing
    if "actual_names" not in meta and yolo_wrapper is not None:
        if hasattr(yolo_wrapper, "model_names"):
            meta["actual_names"] = yolo_wrapper.model_names
            meta["num_classes"] = len(yolo_wrapper.model_names)
        elif hasattr(yolo_wrapper, "_model") and hasattr(yolo_wrapper._model, "names"):
            meta["actual_names"] = yolo_wrapper._model.names
            meta["num_classes"] = len(yolo_wrapper._model.names)

    # Resolution provenance check
    train_sz = meta.get("training_imgsz")
    if train_sz is not None and str(train_sz) == str(runtime_imgsz):
        meta["resolution_provenance"] = "CERTIFIED_TRAINING_RESOLUTION"
    elif train_sz is not None:
        meta["resolution_provenance"] = "RUNTIME_RESOLUTION_VARIANT_UNVALIDATED_FOR_ACCURACY"
    else:
        meta["resolution_provenance"] = "UNVALIDATED_RESOLUTION"

    return meta


class ModelRegistry:
    """Singleton model cache and provenance registry for Stage 2 orchestration."""

    _instance: Optional["ModelRegistry"] = None

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and self.config.get("device", "cuda:0").startswith("cuda") else "cpu"
        )
        self.models_cfg = self.config.get("models", {})
        self.inference_lock = threading.RLock()

        # Registry metadata dictionary
        self._registry_metadata: Dict[str, Dict[str, Any]] = {}

        # Cached model instances
        self.detector: Optional[YOLOObjectDetector] = None
        self.macro_detector: Optional[YOLOBehaviorDetector] = None
        self.posture_predictor: Optional[PosturePredictor] = None
        self.posture_model: Optional[nn.Module] = None
        self.headpose_predictor: Optional[HeadPosePredictor] = None
        self.headpose_model: Optional[nn.Module] = None

        self._initialized = False

    @classmethod
    def get_instance(cls, config: Optional[Dict[str, Any]] = None) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def initialize_models(self) -> None:
        """Load and warm up all configured perception models once."""
        if self._initialized:
            logger.info("ModelRegistry already initialized. Using cached instances.")
            return

        logger.info(f"Initializing ModelRegistry on target device: {self.device}")

        # 1. General Object Detector (for Person & Phone)
        # Check general_object_detector first, fallback to detector
        det_cfg = self.models_cfg.get("general_object_detector") or self.models_cfg.get("detector", {})
        det_path = det_cfg.get("model_path", "models/trained/yolo26m.pt")
        if not os.path.exists(det_path):
            candidate = os.path.join("models", "trained", os.path.basename(det_path))
            if os.path.exists(candidate):
                det_path = candidate
        # Ensure detector is a genuine person/phone capable checkpoint
        det_imgsz = int(det_cfg.get("image_size", 640))
        target_classes = det_cfg.get("target_classes", ["person", "cell phone"])

        logger.info(f"Loading General Object Detector: {det_path}")
        self.detector = YOLOObjectDetector(
            model_path=det_path,
            confidence=float(det_cfg.get("confidence", 0.35)),
            iou_threshold=float(det_cfg.get("iou_threshold", 0.45)),
            image_size=det_imgsz,
            device=str(self.device),
            target_classes=target_classes,
            person_confidence=float(det_cfg.get("person_confidence", det_cfg.get("confidence", 0.35))),
            phone_candidate_confidence=float(det_cfg.get("phone_candidate_confidence", 0.20)),
            phone_strong_confidence=float(det_cfg.get("phone_strong_confidence", 0.35)),
            min_phone_width_px=float(det_cfg.get("min_phone_width_px", 10.0)),
            min_phone_height_px=float(det_cfg.get("min_phone_height_px", 10.0)),
        )

        det_meta = introspect_yolo_metadata(
            checkpoint_path=det_path,
            runtime_imgsz=det_imgsz,
            runtime_role="GENERAL_OBJECT_DETECTOR",
            device=str(self.device),
            yolo_wrapper=self.detector,
        )
        det_meta["resolved_target_classes"] = self.detector.target_classes
        det_meta["person_class_id"] = self.detector.person_class_id
        det_meta["phone_class_id"] = self.detector.phone_class_id
        self._registry_metadata["detector"] = det_meta
        self._registry_metadata["general_object_detector"] = det_meta

        # 2. Macro behavior detector (Stage 1.5)
        macro_cfg = self.models_cfg.get("macro_behavior_detector", {})
        macro_path = macro_cfg.get("model_path", "models/trained/stage1_5_best.pt")
        macro_imgsz = int(macro_cfg.get("image_size", 768))

        logger.info(f"Loading Macro Behavior Detector: {macro_path}")
        self.macro_detector = YOLOBehaviorDetector(
            model_path=macro_path,
            confidence=float(macro_cfg.get("confidence", 0.40)),
            image_size=macro_imgsz,
            device=str(self.device),
            allow_fallback=bool(macro_cfg.get("allow_fallback", True)),
        )

        macro_meta = introspect_yolo_metadata(
            checkpoint_path=macro_path,
            runtime_imgsz=macro_imgsz,
            runtime_role="MACRO_BEHAVIOR_DETECTOR",
            device=str(self.device),
            yolo_wrapper=self.macro_detector,
        )
        self._registry_metadata["macro_behavior_detector"] = macro_meta

        # 3. Primary Posture Classifier (MobileNetV3-Small 224)
        pos_cfg = self.models_cfg.get("posture", {})
        pos_path = pos_cfg.get("primary_checkpoint", "models/trained/v4_posture_best.pt")
        pos_sha = compute_sha256(pos_path)
        logger.info(f"Loading Posture Classifier: {pos_path} (SHA256: {pos_sha[:16]}...)")
        pos_model, pos_ckpt = load_posture_checkpoint(pos_path, device=self.device)
        self.posture_model = pos_model
        self.posture_predictor = PosturePredictor(pos_model, device=self.device)
        pos_params = sum(p.numel() for p in pos_model.parameters()) if pos_model else 0
        pos_model_name = pos_ckpt.get("model_name", "mobilenet_v3_small")
        self._registry_metadata["posture"] = {
            "checkpoint_path": str(pos_path).replace("\\", "/"),
            "sha256": pos_sha,
            "model_name": pos_model_name,
            "actual loaded class": type(pos_model).__name__,
            "actual_loaded_class": type(pos_model).__name__,
            "architecture": pos_model_name,
            "num_classes": pos_ckpt.get("num_classes", len(pos_ckpt.get("class_names", []))),
            "class_names": pos_ckpt.get("class_names", []),
            "input_resolution": [224, 224],
            "device": str(self.device),
            "precision": pos_cfg.get("precision_mode", "fp32"),
            "precision_mode": pos_cfg.get("precision_mode", "fp32"),
            "parameter_count": pos_params,
            "runtime_role": "SPECIALIZED_POSTURE_CLASSIFIER",
        }

        # 4. Primary Head-Pose Estimator (HopeNet-Yaw)
        hp_cfg = self.models_cfg.get("headpose", {})
        hp_path = hp_cfg.get("primary_checkpoint", "models/trained/v4_headpose_yaw_best.pt")
        hp_sha = compute_sha256(hp_path)
        logger.info(f"Loading Head-Pose Estimator: {hp_path} (SHA256: {hp_sha[:16]}...)")
        hp_model, hp_ckpt = load_headpose_checkpoint(hp_path, device=self.device)
        self.headpose_model = hp_model
        self.headpose_predictor = HeadPosePredictor(hp_model, device=self.device)
        hp_params = sum(p.numel() for p in hp_model.parameters()) if hp_model else 0
        hp_model_name = hp_ckpt.get("model_name", "hopenet_yaw")
        hp_min = hp_model.min_angle if hasattr(hp_model, "min_angle") else -99.0
        hp_max = hp_model.max_angle if hasattr(hp_model, "max_angle") else 99.0
        self._registry_metadata["headpose"] = {
            "checkpoint_path": str(hp_path).replace("\\", "/"),
            "sha256": hp_sha,
            "model_name": hp_model_name,
            "actual loaded class": type(hp_model).__name__,
            "actual_loaded_class": type(hp_model).__name__,
            "architecture": hp_model_name,
            "task": hp_ckpt.get("task", "YAW_REGRESSION"),
            "native_support": [hp_min, hp_max],
            "primary_range_deg": [hp_min, hp_max],
            "input_resolution": [224, 224],
            "device": str(self.device),
            "precision": hp_cfg.get("precision_mode", "fp32"),
            "precision_mode": hp_cfg.get("precision_mode", "fp32"),
            "parameter_count": hp_params,
            "runtime_role": "SPECIALIZED_HEADPOSE_ESTIMATOR",
        }

        # Warm up models on GPU if applicable
        self._warmup()
        self._initialized = True
        logger.info("ModelRegistry fully initialized and warmed up.")

    def _warmup(self) -> None:
        """Execute single dummy inference pass to warm up CUDA kernels."""
        if not torch.cuda.is_available() or "cuda" not in str(self.device):
            return
        logger.info("Warming up perception models on CUDA...")
        try:
            with torch.inference_mode():
                # Warm up posture
                dummy_posture = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=self.device)
                self.posture_model(dummy_posture)

                # Warm up headpose
                dummy_headpose = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=self.device)
                self.headpose_model(dummy_headpose)

                torch.cuda.synchronize()
            logger.info("CUDA warm-up complete.")
        except Exception as e:
            logger.warning(f"CUDA warm-up encountered non-critical error: {e}")

    def get_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Return provenance registry metadata for all active models."""
        return self._registry_metadata
