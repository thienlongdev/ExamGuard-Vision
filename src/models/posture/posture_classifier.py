"""
Unified Posture Classifier Factory and Inference Interface
Provides model construction, checkpoint serialization/loading, and inference output
conforming strictly to configs/v4_fusion_contract.yaml.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.posture.resnet_cbam import resnet18_cbam, resnet50_cbam
from src.models.posture.mobilenet_posture import mobilenet_posture

# Canonical 4-class ontology mapping
POSTURE_CLASSES = [
    "NORMAL_UPRIGHT",     # index 0
    "NORMAL_READ_WRITE",   # index 1
    "HEAD_REST_SLEEP",     # index 2 (minority)
    "TURN_HEAD_CLEAR",     # index 3
]

POSTURE_CLASS_TO_IDX = {name: idx for idx, name in enumerate(POSTURE_CLASSES)}
POSTURE_IDX_TO_CLASS = {idx: name for idx, name in enumerate(POSTURE_CLASSES)}


@dataclass
class PostureInferenceOutput:
    """Output structure conforming to configs/v4_fusion_contract.yaml"""
    student_id: Optional[int]
    timestamp: Optional[float]
    normal_upright_score: float
    normal_read_write_score: float
    head_rest_sleep_score: float
    turn_head_clear_score: float
    predicted_class: str
    confidence: float
    crop_quality: str = "GOOD"
    crop_type: str = "TIGHT_PERSON_CROP"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_posture_model(
    model_name: str,
    num_classes: int = 4,
    pretrained: bool = True,
) -> nn.Module:
    """
    Factory function for candidate posture classification models.

    Supported model names:
    - 'resnet18_cbam': ResNet-18 with CBAM attention
    - 'resnet50_cbam': ResNet-50 with CBAM attention
    - 'mobilenet_v3_small' / 'mobilenetv4_small': MobileNet lightweight edge baseline
    """
    normalized_name = model_name.lower().replace("-", "_")

    if normalized_name in ["resnet18_cbam", "resnet18"]:
        return resnet18_cbam(num_classes=num_classes, pretrained=pretrained)
    elif normalized_name in ["resnet50_cbam", "resnet50"]:
        return resnet50_cbam(num_classes=num_classes, pretrained=pretrained)
    elif normalized_name in ["mobilenet_v3_small", "mobilenetv4_small", "mobilenet_small", "mobilenet"]:
        return mobilenet_posture(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(
            f"Unsupported posture model architecture: '{model_name}'. "
            f"Supported: 'resnet18_cbam', 'resnet50_cbam', 'mobilenet_v3_small'"
        )


def save_posture_checkpoint(
    model: nn.Module,
    filepath: Union[str, Path],
    model_name: str,
    epoch: int,
    metrics: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Saves model state dict and full provenance metadata."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_name": model_name,
        "num_classes": 4,
        "class_names": POSTURE_CLASSES,
        "epoch": epoch,
        "metrics": metrics,
        "state_dict": model.state_dict(),
        "extra_metadata": extra_metadata or {},
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state

    torch.save(payload, str(filepath))


def load_posture_checkpoint(
    filepath: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> tuple[nn.Module, Dict[str, Any]]:
    """Loads posture model from a saved checkpoint dictionary."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(str(filepath), map_location=device, weights_only=False)
    model_name = checkpoint.get("model_name", "resnet18_cbam")
    num_classes = checkpoint.get("num_classes", 4)

    model = create_posture_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


class PosturePredictor:
    """Inference wrapper implementing the contract schema."""

    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device] = "cuda:0",
        precision_mode: str = "fp32",
    ):
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in str(device) else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.precision_mode = str(precision_mode).lower()
        self.use_amp = (self.precision_mode == "fp16" and self.device.type == "cuda")

    def predict_tensor(
        self,
        tensor_batch: torch.Tensor,
        student_ids: Optional[List[Optional[int]]] = None,
        timestamps: Optional[List[Optional[float]]] = None,
        crop_qualities: Optional[List[str]] = None,
        crop_types: Optional[List[str]] = None,
    ) -> List[PostureInferenceOutput]:
        """
        Runs inference on a preprocessed batch tensor (B, 3, H, W) normalized to ImageNet mean/std.
        """
        tensor_batch = tensor_batch.to(self.device, non_blocking=True)
        with torch.inference_mode():
            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = self.model(tensor_batch)
            else:
                logits = self.model(tensor_batch)
            probs = F.softmax(logits.float(), dim=1).cpu().numpy()

        batch_size = probs.shape[0]
        outputs = []
        for i in range(batch_size):
            p = probs[i]
            pred_idx = int(p.argmax())
            pred_class = POSTURE_CLASSES[pred_idx]
            conf = float(p[pred_idx])

            s_id = student_ids[i] if student_ids and i < len(student_ids) else None
            t_stamp = timestamps[i] if timestamps and i < len(timestamps) else None
            c_qual = crop_qualities[i] if crop_qualities and i < len(crop_qualities) else "GOOD"
            c_type = crop_types[i] if crop_types and i < len(crop_types) else "TIGHT_PERSON_CROP"

            out = PostureInferenceOutput(
                student_id=s_id,
                timestamp=t_stamp,
                normal_upright_score=float(p[0]),
                normal_read_write_score=float(p[1]),
                head_rest_sleep_score=float(p[2]),
                turn_head_clear_score=float(p[3]),
                predicted_class=pred_class,
                confidence=conf,
                crop_quality=c_qual,
                crop_type=c_type,
            )
            outputs.append(out)

        return outputs
