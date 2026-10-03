"""
Unified Head-Pose Estimator Factory and Inference Interface
Provides model construction, checkpoint serialization/loading, and inference output
conforming strictly to configs/v4_fusion_contract.yaml.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.headpose.hopenet import HopeNetYaw, ResNet18Yaw


@dataclass
class HeadPoseInferenceOutput:
    """Output structure conforming to configs/v4_fusion_contract.yaml Module B."""
    student_id: Optional[int]
    timestamp: Optional[float]
    frame_index: Optional[int]
    yaw_deg: float
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    pose_confidence: float = 1.0
    face_quality: str = "EXCELLENT"
    capability_flags: Tuple[str, ...] = ("HEAD_POSE_ELIGIBLE",)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def canonicalize_yaw(angle: Union[float, int, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Canonicalizes Euler yaw angle to [-180, 180) degrees via continuous periodic wrapping.
    Leaves values already in [-180, 180) unchanged (exact identity).
    Never clamps or truncates angles.
    Raises ValueError on NaN or Inf.
    """
    import math
    if isinstance(angle, (int, float, np.floating, np.integer)):
        val = float(angle)
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Invalid angle value (NaN or Inf): {val}")
        return ((val + 180.0) % 360.0) - 180.0
    elif isinstance(angle, np.ndarray):
        if not np.isfinite(angle).all():
            raise ValueError("Input array contains NaN or Inf values")
        orig_dtype = angle.dtype
        res = ((angle.astype(np.float64) + 180.0) % 360.0) - 180.0
        return res.astype(orig_dtype)
    elif isinstance(angle, torch.Tensor):
        if not torch.isfinite(angle).all():
            raise ValueError("Input tensor contains NaN or Inf values")
        orig_dtype = angle.dtype
        res = torch.remainder(angle.double() + 180.0, 360.0) - 180.0
        return res.to(orig_dtype)
    else:
        raise TypeError(f"Unsupported type for angle canonicalization: {type(angle)}")



def shortest_angular_difference(
    prediction_deg: Union[float, int, np.ndarray, torch.Tensor],
    target_deg: Union[float, int, np.ndarray, torch.Tensor],
    signed: bool = False,
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Computes canonical shortest angular difference between prediction and target on S^1:
        delta_deg = ((prediction_deg - target_deg + 180.0) % 360.0) - 180.0
        angular_error_deg = abs(delta_deg)

    Guarantees shortest distance on circle in [0, 180] degrees (or [-180, 180) if signed=True).
    Seamlessly resolves +/-180 boundary discontinuities:
        e.g. target = +179°, pred = -179° -> delta = +2°, error = 2°
    Strictly validates input: raises ValueError on NaN or Inf.
    """
    import math

    if isinstance(prediction_deg, (int, float, np.floating, np.integer)) and isinstance(target_deg, (int, float, np.floating, np.integer)):
        p = float(prediction_deg)
        t = float(target_deg)
        if math.isnan(p) or math.isinf(p) or math.isnan(t) or math.isinf(t):
            raise ValueError(f"Invalid input: NaN or Inf encountered (pred={p}, target={t})")
        delta = ((p - t + 180.0) % 360.0) - 180.0
        return delta if signed else abs(delta)

    elif isinstance(prediction_deg, np.ndarray) or isinstance(target_deg, np.ndarray):
        p = np.asarray(prediction_deg, dtype=np.float64)
        t = np.asarray(target_deg, dtype=np.float64)
        if not np.isfinite(p).all() or not np.isfinite(t).all():
            raise ValueError("Input array contains NaN or Inf values")
        delta = ((p - t + 180.0) % 360.0) - 180.0
        return delta if signed else np.abs(delta)

    elif isinstance(prediction_deg, torch.Tensor) or isinstance(target_deg, torch.Tensor):
        p = torch.as_tensor(prediction_deg, dtype=torch.float32)
        t = torch.as_tensor(target_deg, dtype=torch.float32)
        if not torch.isfinite(p).all() or not torch.isfinite(t).all():
            raise ValueError("Input tensor contains NaN or Inf values")
        delta = torch.remainder(p - t + 180.0, 360.0) - 180.0
        return delta if signed else torch.abs(delta)

    else:
        raise TypeError(f"Unsupported types: pred={type(prediction_deg)}, target={type(target_deg)}")


def circular_mae(
    y_true: Union[np.ndarray, torch.Tensor, List[float]],
    y_pred: Union[np.ndarray, torch.Tensor, List[float]],
) -> float:
    """Computes Mean Absolute Angular Error across S^1 using shortest angular difference."""
    errors = shortest_angular_difference(y_pred, y_true, signed=False)
    if isinstance(errors, torch.Tensor):
        return float(torch.mean(errors.float()).item())
    elif isinstance(errors, np.ndarray):
        return float(np.mean(errors))
    elif isinstance(errors, (int, float, np.floating, np.integer)):
        return float(errors)
    else:
        return float(np.mean(errors))


def encode_yaw_sin_cos(
    yaw_deg: Union[float, int, np.ndarray, torch.Tensor]
) -> Union[Tuple[float, float], np.ndarray, torch.Tensor]:
    """
    Encodes Euler yaw degrees to unit-circle 2D representation: (sin(yaw), cos(yaw)).
    Continuous mapping with zero boundary discontinuity on S^1.
    """
    import math

    if isinstance(yaw_deg, (int, float, np.floating, np.integer)):
        y = float(yaw_deg)
        if math.isnan(y) or math.isinf(y):
            raise ValueError(f"Invalid yaw value: {y}")
        rad = math.radians(y)
        return (math.sin(rad), math.cos(rad))

    elif isinstance(yaw_deg, np.ndarray):
        y = np.asarray(yaw_deg, dtype=np.float64)
        if not np.isfinite(y).all():
            raise ValueError("Input array contains NaN or Inf values")
        rad = np.radians(y)
        return np.stack([np.sin(rad), np.cos(rad)], axis=-1)

    elif isinstance(yaw_deg, torch.Tensor):
        y = torch.as_tensor(yaw_deg, dtype=torch.float32)
        if not torch.isfinite(y).all():
            raise ValueError("Input tensor contains NaN or Inf values")
        rad = y * (torch.pi / 180.0)
        return torch.stack([torch.sin(rad), torch.cos(rad)], dim=-1)

    else:
        raise TypeError(f"Unsupported type: {type(yaw_deg)}")


def decode_yaw_sin_cos(
    sin_val: Union[float, int, np.ndarray, torch.Tensor],
    cos_val: Union[float, int, np.ndarray, torch.Tensor],
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Decodes (sin, cos) unit-circle representation back to canonical Euler yaw degrees in [-180, 180).
    Uses atan2(sin, cos) * 180 / pi followed by canonicalization.
    """
    import math

    if isinstance(sin_val, (int, float, np.floating, np.integer)) and isinstance(cos_val, (int, float, np.floating, np.integer)):
        s = float(sin_val)
        c = float(cos_val)
        if math.isnan(s) or math.isinf(s) or math.isnan(c) or math.isinf(c):
            raise ValueError(f"Invalid input: NaN or Inf encountered (sin={s}, cos={c})")
        deg = math.degrees(math.atan2(s, c))
        return canonicalize_yaw(deg)

    elif isinstance(sin_val, np.ndarray) or isinstance(cos_val, np.ndarray):
        s = np.asarray(sin_val, dtype=np.float64)
        c = np.asarray(cos_val, dtype=np.float64)
        if not np.isfinite(s).all() or not np.isfinite(c).all():
            raise ValueError("Input array contains NaN or Inf values")
        deg = np.degrees(np.arctan2(s, c))
        return canonicalize_yaw(deg)

    elif isinstance(sin_val, torch.Tensor) or isinstance(cos_val, torch.Tensor):
        s = torch.as_tensor(sin_val, dtype=torch.float32)
        c = torch.as_tensor(cos_val, dtype=torch.float32)
        if not torch.isfinite(s).all() or not torch.isfinite(c).all():
            raise ValueError("Input tensor contains NaN or Inf values")
        deg = torch.atan2(s, c) * (180.0 / torch.pi)
        return canonicalize_yaw(deg)

    else:
        raise TypeError(f"Unsupported types: sin={type(sin_val)}, cos={type(cos_val)}")



def create_headpose_model(
    model_name: str = "hopenet_yaw",
    pretrained: bool = True,
) -> nn.Module:
    """
    Factory function for head-pose estimation models.
    Supported model names:
    - 'hopenet_yaw': ResNet50 backbone with 66-bin classification + continuous expectation
    - 'resnet18_yaw': Lightweight continuous regression baseline
    """
    norm_name = model_name.lower().replace("-", "_")
    if norm_name in ["hopenet", "hopenet_yaw", "hopenet_resnet50"]:
        return HopeNetYaw(num_bins=66, min_angle=-99.0, max_angle=99.0, pretrained=pretrained)
    elif norm_name in ["resnet18_yaw", "resnet18"]:
        return ResNet18Yaw(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported head-pose architecture: '{model_name}'. Supported: 'hopenet_yaw', 'resnet18_yaw'")


def save_headpose_checkpoint(
    model: nn.Module,
    filepath: Union[str, Path],
    model_name: str,
    epoch: int,
    metrics: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Saves head-pose model weights with full training provenance metadata."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_name": model_name,
        "task": "YAW_REGRESSION",
        "epoch": epoch,
        "metrics": metrics,
        "state_dict": model.state_dict(),
        "extra_metadata": extra_metadata or {},
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state

    torch.save(payload, str(filepath))


def load_headpose_checkpoint(
    filepath: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> tuple[nn.Module, Dict[str, Any]]:
    """Loads saved head-pose model from disk."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Head-pose checkpoint not found: {filepath}")

    checkpoint = torch.load(str(filepath), map_location=device, weights_only=False)
    model_name = checkpoint.get("model_name", "hopenet_yaw")

    model = create_headpose_model(model_name=model_name, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


class HeadPosePredictor:
    """Inference wrapper for head pose conforming to fusion contract."""

    def __init__(self, model: nn.Module, device: Union[str, torch.device] = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in str(device) else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_tensor(
        self,
        tensor_batch: torch.Tensor,
        student_ids: Optional[List[Optional[int]]] = None,
        timestamps: Optional[List[Optional[float]]] = None,
        frame_indices: Optional[List[Optional[int]]] = None,
        face_qualities: Optional[List[str]] = None,
    ) -> List[HeadPoseInferenceOutput]:
        """
        Runs inference on preprocessed face/head batch (B, 3, 224, 224).
        """
        tensor_batch = tensor_batch.to(self.device)
        logits, expected_yaw = self.model(tensor_batch)

        yaws = expected_yaw.cpu().numpy()
        batch_size = len(yaws)

        # Compute confidence from logits if available
        if logits is not None:
            probs = F.softmax(logits, dim=1)
            confs = probs.max(dim=1).values.cpu().numpy()
        else:
            confs = [1.0] * batch_size

        outputs = []
        for i in range(batch_size):
            s_id = student_ids[i] if student_ids and i < len(student_ids) else None
            t_stamp = timestamps[i] if timestamps and i < len(timestamps) else None
            f_idx = frame_indices[i] if frame_indices and i < len(frame_indices) else None
            fq = face_qualities[i] if face_qualities and i < len(face_qualities) else "EXCELLENT"

            out = HeadPoseInferenceOutput(
                student_id=s_id,
                timestamp=t_stamp,
                frame_index=f_idx,
                yaw_deg=float(yaws[i]),
                pitch_deg=None,
                roll_deg=None,
                pose_confidence=float(confs[i]),
                face_quality=fq,
                capability_flags=("HEAD_POSE_ELIGIBLE",),
            )
            outputs.append(out)

        return outputs
