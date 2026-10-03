"""Tests for spatial object association and BBox geometry."""

import pytest
from src.analysis.object_association import ObjectAssociator
from src.detection.types import BBox, Detection
from src.tracking.tracker import Track


def test_bbox_geometry():
    box1 = BBox(x1=0, y1=0, x2=10, y2=10)
    box2 = BBox(x1=5, y1=0, x2=15, y2=10)

    assert box1.width == 10
    assert box1.height == 10
    assert box1.area == 100
    assert box1.center == (5.0, 5.0)

    # IoU: intersection is [5,0] to [10,10] = 5*10 = 50. Union = 100+100-50 = 150. IoU = 50/150 = 0.333
    assert abs(box1.iou(box2) - (50.0 / 150.0)) < 1e-4

    # Containment
    assert box1.contains_point(5, 5) is True
    assert box1.contains_point(12, 5) is False

    # Expansion
    expanded = box1.expand(0.1, 0.1)
    assert expanded.x1 == -1.0
    assert expanded.x2 == 11.0


def test_object_associator_single_student():
    associator = ObjectAssociator(expand_student_bbox_ratio=0.2)

    student_track = Track(
        track_id=1,
        bbox=BBox(100, 100, 300, 500),
        confidence=0.95,
        timestamp=10.0,
    )

    # Phone inside student's lap/desk area
    phone_det = Detection(
        bbox=BBox(150, 400, 200, 480),
        class_id=67,
        class_name="cell phone",
        confidence=0.88,
    )

    results = associator.associate(tracks=[student_track], detections=[phone_det])
    assert 1 in results
    assert results[1].has_phone is True
    assert len(results[1].phone_detections) == 1
    assert results[1].highest_phone_confidence == 0.88


def test_object_associator_multiple_students():
    associator = ObjectAssociator(expand_student_bbox_ratio=0.2)

    # Student 1 on the left
    s1 = Track(track_id=1, bbox=BBox(50, 100, 200, 400), confidence=0.9, timestamp=1.0)
    # Student 2 on the right
    s2 = Track(track_id=2, bbox=BBox(600, 100, 750, 400), confidence=0.9, timestamp=1.0)

    # Phone near Student 2
    phone_s2 = Detection(
        bbox=BBox(620, 250, 680, 320),
        class_id=67,
        class_name="cell phone",
        confidence=0.92,
    )

    results = associator.associate(tracks=[s1, s2], detections=[phone_s2])

    assert results[1].has_phone is False
    assert results[2].has_phone is True
    assert results[2].highest_phone_confidence == 0.92
