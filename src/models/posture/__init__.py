"""
Posture classifier package
"""

from src.models.posture.cbam import CBAM, ChannelAttention, SpatialAttention
from src.models.posture.resnet_cbam import (
    BasicBlockCBAM,
    BottleneckCBAM,
    ResNetCBAM,
    resnet18_cbam,
    resnet50_cbam,
)
from src.models.posture.mobilenet_posture import mobilenet_posture
from src.models.posture.posture_classifier import (
    POSTURE_CLASSES,
    POSTURE_CLASS_TO_IDX,
    POSTURE_IDX_TO_CLASS,
    PostureInferenceOutput,
    create_posture_model,
    save_posture_checkpoint,
    load_posture_checkpoint,
    PosturePredictor,
)

__all__ = [
    "CBAM",
    "ChannelAttention",
    "SpatialAttention",
    "BasicBlockCBAM",
    "BottleneckCBAM",
    "ResNetCBAM",
    "resnet18_cbam",
    "resnet50_cbam",
    "mobilenet_posture",
    "POSTURE_CLASSES",
    "POSTURE_CLASS_TO_IDX",
    "POSTURE_IDX_TO_CLASS",
    "PostureInferenceOutput",
    "create_posture_model",
    "save_posture_checkpoint",
    "load_posture_checkpoint",
    "PosturePredictor",
]
