"""
Head-pose models package
"""

from src.models.headpose.hopenet import HopeNetYaw, ResNet18Yaw
from src.models.headpose.headpose_estimator import (
    HeadPoseInferenceOutput,
    create_headpose_model,
    save_headpose_checkpoint,
    load_headpose_checkpoint,
    HeadPosePredictor,
    canonicalize_yaw,
    shortest_angular_difference,
    circular_mae,
    encode_yaw_sin_cos,
    decode_yaw_sin_cos,
)

__all__ = [
    "HopeNetYaw",
    "ResNet18Yaw",
    "HeadPoseInferenceOutput",
    "create_headpose_model",
    "save_headpose_checkpoint",
    "load_headpose_checkpoint",
    "HeadPosePredictor",
    "canonicalize_yaw",
    "shortest_angular_difference",
    "circular_mae",
    "encode_yaw_sin_cos",
    "decode_yaw_sin_cos",
]

