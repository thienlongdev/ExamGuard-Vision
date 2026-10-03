"""Detection module."""

from src.detection.types import BBox, Detection, BehaviorDetection
from src.detection.object_detector import ObjectDetector, YOLOObjectDetector
from src.detection.behavior_detector import BehaviorDetector, YOLOBehaviorDetector

__all__ = [
    "BBox",
    "Detection",
    "BehaviorDetection",
    "ObjectDetector",
    "YOLOObjectDetector",
    "BehaviorDetector",
    "YOLOBehaviorDetector",
]
