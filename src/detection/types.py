"""Normalized detection types and geometry structures.

Ensures the rest of the system is completely independent of specific detector libraries.
"""

from dataclasses import dataclass
from typing import Tuple, Union


@dataclass(frozen=True)
class BBox:
    """Bounding box defined by (x1, y1, x2, y2) in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_int_tuple(self) -> Tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2)))

    def __iter__(self):
        return iter((self.x1, self.y1, self.x2, self.y2))

    def __getitem__(self, idx: int) -> float:
        return (self.x1, self.y1, self.x2, self.y2)[idx]

    def __len__(self) -> int:
        return 4

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point (x, y) is inside the bounding box."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def expand(self, ratio_x: float, ratio_y: float) -> "BBox":
        """Expand bounding box proportionally around its center."""
        w = self.width
        h = self.height
        pad_x = w * ratio_x
        pad_y = h * ratio_y
        return BBox(
            x1=self.x1 - pad_x,
            y1=self.y1 - pad_y,
            x2=self.x2 + pad_x,
            y2=self.y2 + pad_y,
        )

    def iou(self, other: "BBox") -> float:
        """Calculate Intersection over Union (IoU) with another bounding box."""
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area

    def overlap_ratio(self, other: "BBox") -> float:
        """Calculate intersection area divided by self.area."""
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        if self.area <= 0.0:
            return 0.0
        return inter_area / self.area


@dataclass
class Detection:
    """Normalized object detector output."""
    bbox: BBox
    class_id: int
    class_name: str
    confidence: float


@dataclass
class BehaviorDetection:
    """Normalized behavior detector output."""
    bbox: BBox
    behavior: str
    confidence: float
