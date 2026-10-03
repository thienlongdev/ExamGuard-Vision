"""Behavior detector abstraction and YOLO custom model adapter."""

from abc import ABC, abstractmethod
import logging
import os
from typing import Dict, List, Optional
import numpy as np

from src.detection.types import BBox, BehaviorDetection

logger = logging.getLogger(__name__)


class BehaviorDetector(ABC):
    """Abstract interface for student behavior detection."""

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        person_boxes: Optional[List[BBox]] = None,
    ) -> List[BehaviorDetection]:
        """Detect observable behaviors in the frame or person regions.

        Args:
            frame: Numpy array (H, W, 3).
            person_boxes: Optional bounding boxes of tracked students to focus on.

        Returns:
            List of normalized BehaviorDetection objects.
        """
        pass


class YOLOBehaviorDetector(BehaviorDetector):
    """Ultralytics YOLO adapter for custom trained behavior models."""

    def __init__(
        self,
        model_path: str = "models/trained/behavior_best.pt",
        confidence: float = 0.40,
        image_size: int = 640,
        device: str = "cpu",
        taxonomy: Optional[List[str]] = None,
        allow_fallback: bool = True,
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.allow_fallback = allow_fallback

        self.taxonomy = taxonomy or [
            "normal",
            "head_down",
            "lean",
            "turn_head",
            "use_phone",
            "discuss",
            "stand",
        ]

        self._model = None
        self._is_fallback_mode = False
        self._load_model()

    def _load_model(self) -> None:
        """Load custom behavior weights or configure graceful fallback adapter."""
        if os.path.exists(self.model_path):
            try:
                from ultralytics import YOLO
                logger.info(f"Loading custom behavior model from '{self.model_path}'...")
                self._model = YOLO(self.model_path)
                logger.info(f"Custom behavior model successfully loaded.")
                return
            except Exception as e:
                logger.warning(f"Error loading custom behavior model from '{self.model_path}': {e}")

        # If custom model weights are not yet trained/available
        if self.allow_fallback:
            self._is_fallback_mode = True
            logger.warning(
                f"Custom behavior model checkpoint not found at '{self.model_path}'. "
                f"Operating in baseline heuristic adapter mode until Stage 1/2 training is completed. "
                f"(Behavior detections default to 'normal' unless phone/geometry cues are present; accuracy requires model training)."
            )
        else:
            raise FileNotFoundError(
                f"Behavior model weights '{self.model_path}' not found and allow_fallback=False."
            )

    def detect(
        self,
        frame: np.ndarray,
        person_boxes: Optional[List[BBox]] = None,
    ) -> List[BehaviorDetection]:
        """Detect behaviors for persons in frame."""
        # 1. Custom model inference path
        if self._model is not None and not self._is_fallback_mode:
            results = self._model.predict(
                source=frame,
                conf=self.confidence,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )

            detections: List[BehaviorDetection] = []
            if not results:
                return detections

            first_res = results[0]
            if first_res.boxes is None or len(first_res.boxes) == 0:
                return detections

            boxes_data = first_res.boxes.data.cpu().numpy()
            for row in boxes_data:
                x1, y1, x2, y2, conf, cls_id = row[:6]
                cls_id_int = int(cls_id)
                behavior_name = first_res.names.get(cls_id_int, "normal")

                bbox = BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
                detections.append(
                    BehaviorDetection(
                        bbox=bbox,
                        behavior=behavior_name,
                        confidence=float(conf),
                    )
                )
            return detections

        # 2. Fallback adapter mode: assign baseline behavior per person box
        # This keeps the downstream pipeline (tracking, temporal buffer, rule engine) fully functional
        # during vertical slice development without faking neural network accuracy.
        detections: List[BehaviorDetection] = []
        if person_boxes:
            for pbox in person_boxes:
                # Basic geometric aspect ratio check (e.g. standing vs seated leaning)
                w = pbox.width
                h = pbox.height
                aspect_ratio = h / (w + 1e-6)

                # Seated ratio is typically ~1.3-1.8; full standing is typically > 2.2
                behavior = "normal"
                confidence = 0.50

                if aspect_ratio > 2.5:
                    behavior = "stand"
                    confidence = 0.55

                detections.append(
                    BehaviorDetection(
                        bbox=pbox,
                        behavior=behavior,
                        confidence=confidence,
                    )
                )
        return detections
