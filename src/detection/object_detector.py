"""Object detector abstraction and YOLO implementation for V1 targets (person, cell phone)."""

from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Optional
import numpy as np

from src.detection.types import BBox, Detection

logger = logging.getLogger(__name__)


class ObjectDetector(ABC):
    """Abstract interface for general object detection."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run object detection on an RGB/BGR image frame.

        Args:
            frame: Numpy array (H, W, 3).

        Returns:
            List of normalized Detection objects.
        """
        pass


class YOLOObjectDetector(ObjectDetector):
    """Ultralytics YOLO adapter for V1 objects (person, cell phone)."""

    def __init__(
        self,
        model_path: str = "yolo26m.pt",
        confidence: float = 0.35,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        device: str = "cpu",
        target_classes: Optional[Dict[int, str] | List[str]] = None,
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device
        self._raw_target_classes = target_classes

        self._model = None
        self.target_classes: Dict[int, str] = {}
        self.person_class_id: Optional[int] = None
        self.phone_class_id: Optional[int] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the YOLO model and dynamically verify/resolve target class indices."""
        try:
            from ultralytics import YOLO
            logger.info(f"Loading object detector model from '{self.model_path}' on device '{self.device}'...")
            self._model = YOLO(self.model_path)
            logger.info(f"YOLO object detector successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model from '{self.model_path}': {e}")
            raise

        actual_names = getattr(self._model, "names", {})
        if not isinstance(actual_names, dict) and hasattr(self._model, "model") and hasattr(self._model.model, "names"):
            actual_names = self._model.model.names

        # Name-to-ID lookup from actual physical checkpoint
        name_to_id = {v.lower().strip(): k for k, v in actual_names.items()}

        resolved_classes: Dict[int, str] = {}

        if self._raw_target_classes is None:
            # Dynamically discover 'person' and 'cell phone' / 'phone' from checkpoint
            for candidate in ["person", "cell phone", "phone"]:
                if candidate in name_to_id:
                    cid = name_to_id[candidate]
                    resolved_classes[cid] = actual_names[cid]
        elif isinstance(self._raw_target_classes, list):
            for name in self._raw_target_classes:
                normalized = name.lower().strip()
                if normalized in name_to_id:
                    cid = name_to_id[normalized]
                    resolved_classes[cid] = actual_names[cid]
                else:
                    logger.warning(
                        f"Target class '{name}' requested but NOT present in checkpoint '{self.model_path}' "
                        f"taxonomy: {actual_names}"
                    )
        elif isinstance(self._raw_target_classes, dict):
            for cid, name in self._raw_target_classes.items():
                normalized = str(name).lower().strip()
                # Verify if cid matches physical name
                if cid in actual_names and actual_names[cid].lower().strip() == normalized:
                    resolved_classes[cid] = actual_names[cid]
                elif normalized in name_to_id:
                    # Relocate to true physical class id
                    actual_cid = name_to_id[normalized]
                    logger.info(
                        f"Re-mapping target class '{name}' from assumed ID {cid} to actual physical ID {actual_cid}"
                    )
                    resolved_classes[actual_cid] = actual_names[actual_cid]
                else:
                    logger.warning(
                        f"Requested class {cid}:'{name}' does not match checkpoint taxonomy: {actual_names}. "
                        f"Class is OMITTED from detector targets."
                    )

        self.target_classes = resolved_classes
        # Find person and phone IDs
        for cid, name in self.target_classes.items():
            if name.lower().strip() == "person":
                self.person_class_id = cid
            elif name.lower().strip() in ["cell phone", "phone"]:
                self.phone_class_id = cid

        logger.info(
            f"Resolved detector target classes from physical checkpoint: {self.target_classes} "
            f"(person_id={self.person_class_id}, phone_id={self.phone_class_id})"
        )

    @property
    def is_person_available(self) -> bool:
        return self.person_class_id is not None

    @property
    def is_phone_available(self) -> bool:
        return self.phone_class_id is not None

    @property
    def model_names(self) -> Dict[int, str]:
        if self._model and hasattr(self._model, "names"):
            return self._model.names
        return {}

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect persons and cell phones in the frame."""
        if self._model is None:
            return []

        classes_to_filter = list(self.target_classes.keys())

        # Run inference
        results = self._model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            classes=classes_to_filter,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        first_res = results[0]
        if first_res.boxes is None or len(first_res.boxes) == 0:
            return detections

        boxes_data = first_res.boxes.data.cpu().numpy()
        for row in boxes_data:
            # Format: [x1, y1, x2, y2, conf, cls]
            x1, y1, x2, y2, conf, cls_id = row[:6]
            cls_id_int = int(cls_id)
            class_name = self.target_classes.get(cls_id_int, first_res.names.get(cls_id_int, f"class_{cls_id_int}"))

            bbox = BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
            detections.append(
                Detection(
                    bbox=bbox,
                    class_id=cls_id_int,
                    class_name=class_name,
                    confidence=float(conf),
                )
            )

        return detections
