"""
Tests for V4D Capability & Scale Gating Engine
=============================================
Covers:
- Posture scale gating (height thresholds)
- Camera profile constraints (e.g. HIGH_ANGLE_CCTV disables facial head-pose)
- Head-pose resolution eligibility
- HopeNet primary support range [-99, +99)
- ResNet18 Circular fallback outside native range
- Severe occlusion / unresolvable face producing explicit UNAVAILABLE (no fabrication of yaw = 0)
"""

import pytest
from src.fusion.capability import CapabilityGate
from src.fusion.types import ObservationStatus, HeadPoseSupportStatus


@pytest.fixture
def gate():
    config = {
        "camera_profiles": {
            "HIGH_ANGLE_CCTV": {
                "posture_supported": True,
                "headpose_supported": False,
                "min_person_height_px": 120.0,
            },
            "FRONT_OBLIQUE_CCTV": {
                "posture_supported": True,
                "headpose_supported": True,
                "min_person_height_px": 120.0,
                "min_head_dimension_px": 25.0,
            }
        },
        "scale_gating": {
            "posture": {
                "min_person_height_px": 120.0,
                "min_person_width_px": 40.0,
                "high_res_fallback_available": True,
                "default_resolution": 224,
                "use_320_fallback": False,
            },
            "head_pose": {
                "min_head_width_px": 25.0,
                "min_head_height_px": 25.0,
                "primary_yaw_range_deg": [-99.0, 99.0],
                "enable_circular_fallback": True,
            }
        }
    }
    return CapabilityGate(config)


def test_posture_scale_gating(gate):
    # Valid box: H=150, W=50
    valid_box = (10.0, 10.0, 60.0, 160.0)
    assert gate.is_posture_eligible(valid_box) is True

    # Too short: H=100 < 120
    small_box = (10.0, 10.0, 60.0, 110.0)
    assert gate.is_posture_eligible(small_box) is False

    # Too narrow: W=30 < 40
    narrow_box = (10.0, 10.0, 40.0, 160.0)
    assert gate.is_posture_eligible(narrow_box) is False


def test_camera_profile_headpose_restriction(gate):
    person_box = (10.0, 10.0, 100.0, 200.0)
    head_box = (30.0, 10.0, 80.0, 60.0)

    # In FRONT_OBLIQUE_CCTV: eligible
    assert gate.is_head_pose_eligible(person_box, head_box, camera_profile="FRONT_OBLIQUE_CCTV") is True

    # In HIGH_ANGLE_CCTV: head-pose is unsupported by profile
    assert gate.is_head_pose_eligible(person_box, head_box, camera_profile="HIGH_ANGLE_CCTV") is False


def test_headpose_dispatch_primary_hopenet(gate):
    # Within [-99, +99): should select HopeNet-Yaw
    cue = gate.dispatch_head_pose(raw_yaw_candidate=35.0, is_eligible=True)
    assert cue.status == ObservationStatus.AVAILABLE
    assert cue.source_model == "HopeNet-Yaw"
    assert cue.support_status == HeadPoseSupportStatus.WITHIN_PRIMARY_SUPPORT
    assert cue.yaw_deg == 35.0


def test_headpose_dispatch_circular_fallback(gate):
    # Extreme angle 115.0 deg (outside [-99, +99)): should fall back to ResNet18-Circular
    cue = gate.dispatch_head_pose(raw_yaw_candidate=115.0, is_eligible=True)
    assert cue.status == ObservationStatus.AVAILABLE
    assert cue.source_model == "ResNet18-Circular"
    assert cue.support_status == HeadPoseSupportStatus.FALLBACK_CIRCULAR
    assert cue.yaw_deg == 115.0


def test_headpose_dispatch_unresolvable_face(gate):
    # When ineligible (e.g. tiny face or occluded), MUST produce UNAVAILABLE (no fabrication of yaw = 0!)
    cue = gate.dispatch_head_pose(raw_yaw_candidate=15.0, is_eligible=False)
    assert cue.status == ObservationStatus.UNAVAILABLE
    assert cue.yaw_deg is None
    assert cue.support_status == HeadPoseSupportStatus.FACE_UNRESOLVABLE
