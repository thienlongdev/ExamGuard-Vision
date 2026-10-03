"""
Pilot Camera Profile & Capability Schema
=========================================
Implements Part A requirements:
- Complete CameraProfile schema
- ViewpointProfile enumeration
- Seat / Desk Zone representations
- CameraCapabilitySummary with quantitative coverage metrics
- Provenance tracking (DEFAULT, CALIBRATED_FROM_VISIBILITY, ENGINEERING_HEURISTIC, PHYSICALLY_VALIDATED)
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class ViewpointProfile(str, Enum):
    CEILING_HIGH = "CEILING_HIGH"
    FRONT_OBLIQUE = "FRONT_OBLIQUE"
    REAR_OBLIQUE = "REAR_OBLIQUE"
    SIDE_OBLIQUE = "SIDE_OBLIQUE"
    DESK_LEVEL = "DESK_LEVEL"
    UNKNOWN = "UNKNOWN"


class CapabilityLevel(str, Enum):
    FULL = "FULL"
    LIMITED = "LIMITED"
    UNAVAILABLE = "UNAVAILABLE"


class TrackingCapability(str, Enum):
    SUPPORTED = "SUPPORTED"
    DEGRADED = "DEGRADED"


class ProvenanceLevel(str, Enum):
    DEFAULT = "DEFAULT"
    CALIBRATED_FROM_VISIBILITY = "CALIBRATED_FROM_VISIBILITY"
    ENGINEERING_HEURISTIC = "ENGINEERING_HEURISTIC"
    PHYSICALLY_VALIDATED = "PHYSICALLY_VALIDATED"


@dataclass
class Resolution:
    width: int
    height: int

    def to_dict(self) -> Dict[str, int]:
        return {"width": self.width, "height": self.height}


@dataclass
class SeatZone:
    zone_id: str
    desk_polygon: Optional[List[List[float]]] = None
    seat_polygon: Optional[List[List[float]]] = None

    def contains_point(self, pt: Tuple[float, float]) -> bool:
        """Check if point (x, y) is inside seat or desk polygon."""
        import cv2
        for poly in [self.seat_polygon, self.desk_polygon]:
            if poly and len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32)
                res = cv2.pointPolygonTest(pts, pt, False)
                if res >= 0:
                    return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PrivacySettings:
    anonymize_tracks: bool = True
    enable_face_recognition: bool = False # STRICTLY PROHIBITED
    store_full_room_stream: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CameraProfile:
    camera_id: str
    name: str
    source_type: str = "video_file" # "video_file", "webcam", "rtsp"
    source_uri: str = "samples/sample_exam.mp4"
    resolution: Resolution = field(default_factory=lambda: Resolution(1920, 1080))
    nominal_fps: float = 25.0
    expected_student_count: int = 15
    room_id: str = "hall_101"

    # Viewpoint & Mounting Geometry (optional, never fabricate)
    viewpoint_profile: ViewpointProfile = ViewpointProfile.UNKNOWN
    camera_height_m: Optional[float] = None
    camera_pitch_deg: Optional[float] = None
    camera_roll_deg: Optional[float] = None
    horizontal_fov_deg: Optional[float] = None

    # Examination ROI & Seat Zones
    roi_polygon: Optional[List[List[float]]] = None
    seat_zones: List[SeatZone] = field(default_factory=list)

    # Runtime Detector Image Sizes
    general_detector_imgsz: int = 640
    macro_detector_imgsz: int = 768

    # Branch activation
    posture_enabled: bool = True
    headpose_enabled: bool = True
    phone_enabled: bool = True
    evidence_enabled: bool = True

    # Privacy
    privacy: PrivacySettings = field(default_factory=PrivacySettings)

    # Provenance
    provenance_level: ProvenanceLevel = ProvenanceLevel.DEFAULT
    calibrated_at: Optional[str] = None
    calibration_session_id: Optional[str] = None

    def filter_roi(self, bbox: Tuple[float, float, float, float]) -> bool:
        """Check if center of bbox is inside examination ROI polygon."""
        if not self.roi_polygon or len(self.roi_polygon) < 3:
            return True
        import cv2
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        pts = np.array(self.roi_polygon, dtype=np.int32)
        res = cv2.pointPolygonTest(pts, (cx, cy), False)
        return res >= 0

    def find_seat_zone(self, bbox: Tuple[float, float, float, float]) -> Optional[str]:
        """Find matching seat zone for bbox center if defined."""
        if not self.seat_zones:
            return None
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        for sz in self.seat_zones:
            if sz.contains_point((cx, cy)):
                return sz.zone_id
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "source_type": self.source_type,
            "source_uri": self.source_uri,
            "resolution": self.resolution.to_dict(),
            "nominal_fps": self.nominal_fps,
            "expected_student_count": self.expected_student_count,
            "room_id": self.room_id,
            "viewpoint_profile": self.viewpoint_profile.value,
            "camera_height_m": self.camera_height_m,
            "camera_pitch_deg": self.camera_pitch_deg,
            "camera_roll_deg": self.camera_roll_deg,
            "horizontal_fov_deg": self.horizontal_fov_deg,
            "roi_polygon": self.roi_polygon,
            "seat_zones": [sz.to_dict() for sz in self.seat_zones],
            "general_detector_imgsz": self.general_detector_imgsz,
            "macro_detector_imgsz": self.macro_detector_imgsz,
            "posture_enabled": self.posture_enabled,
            "headpose_enabled": self.headpose_enabled,
            "phone_enabled": self.phone_enabled,
            "evidence_enabled": self.evidence_enabled,
            "privacy": self.privacy.to_dict(),
            "provenance_level": self.provenance_level.value,
            "calibrated_at": self.calibrated_at,
            "calibration_session_id": self.calibration_session_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraProfile":
        res_data = data.get("resolution", {"width": 1920, "height": 1080})
        res = Resolution(width=int(res_data.get("width", 1920)), height=int(res_data.get("height", 1080)))

        priv_data = data.get("privacy", {})
        priv = PrivacySettings(
            anonymize_tracks=bool(priv_data.get("anonymize_tracks", True)),
            enable_face_recognition=False, # STRICTLY ENFORCED FALSE
            store_full_room_stream=bool(priv_data.get("store_full_room_stream", False)),
        )

        sz_list = []
        for sz_data in data.get("seat_zones", []):
            if isinstance(sz_data, dict):
                sz_list.append(SeatZone(
                    zone_id=str(sz_data.get("zone_id", "zone")),
                    desk_polygon=sz_data.get("desk_polygon"),
                    seat_polygon=sz_data.get("seat_polygon"),
                ))

        vp_str = str(data.get("viewpoint_profile", "UNKNOWN")).upper()
        vp = ViewpointProfile(vp_str) if vp_str in ViewpointProfile.__members__ else ViewpointProfile.UNKNOWN

        prov_str = str(data.get("provenance_level", data.get("provenance", {}).get("source_level", "DEFAULT"))).upper()
        prov = ProvenanceLevel(prov_str) if prov_str in ProvenanceLevel.__members__ else ProvenanceLevel.DEFAULT

        return cls(
            camera_id=str(data.get("camera_id", "cam_0")),
            name=str(data.get("name", "Exam Camera")),
            source_type=str(data.get("source_type", "video_file")),
            source_uri=str(data.get("source_uri", data.get("default_path", "samples/sample_exam.mp4"))),
            resolution=res,
            nominal_fps=float(data.get("nominal_fps", 25.0)),
            expected_student_count=int(data.get("expected_student_count", 15)),
            room_id=str(data.get("room_id", "room_default")),
            viewpoint_profile=vp,
            camera_height_m=float(data["camera_height_m"]) if data.get("camera_height_m") is not None else None,
            camera_pitch_deg=float(data["camera_pitch_deg"]) if data.get("camera_pitch_deg") is not None else None,
            camera_roll_deg=float(data["camera_roll_deg"]) if data.get("camera_roll_deg") is not None else None,
            horizontal_fov_deg=float(data["horizontal_fov_deg"]) if data.get("horizontal_fov_deg") is not None else None,
            roi_polygon=data.get("roi_polygon"),
            seat_zones=sz_list,
            general_detector_imgsz=int(data.get("general_detector_imgsz", 640)),
            macro_detector_imgsz=int(data.get("macro_detector_imgsz", 768)),
            posture_enabled=bool(data.get("posture_enabled", True)),
            headpose_enabled=bool(data.get("headpose_enabled", True)),
            phone_enabled=bool(data.get("phone_enabled", True)),
            evidence_enabled=bool(data.get("evidence_enabled", True)),
            privacy=priv,
            provenance_level=prov,
            calibrated_at=data.get("calibrated_at"),
            calibration_session_id=data.get("calibration_session_id"),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "CameraProfile":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False, indent=2)


@dataclass
class CameraCapabilitySummary:
    """Runtime camera capability summary and quantitative coverage metrics."""
    camera_id: str
    posture_capability: CapabilityLevel
    headpose_capability: CapabilityLevel
    phone_capability: CapabilityLevel
    tracking_capability: TrackingCapability

    # Quantitative coverage percentages (0.0 - 100.0)
    pct_height_gte_120: float
    pct_height_60_119: float
    pct_height_lt_60: float
    pct_head_crop_gte_25: float
    phone_median_apparent_size_px: Optional[Tuple[float, float]] = None

    # Geometry & Operational Diagnostic Flags
    diagnostic_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "POSTURE_CAPABILITY": self.posture_capability.value,
            "HEADPOSE_CAPABILITY": self.headpose_capability.value,
            "PHONE_CAPABILITY": self.phone_capability.value,
            "TRACKING_CAPABILITY": self.tracking_capability.value,
            "quantitative_coverage": {
                "pct_person_height_gte_120px": round(self.pct_height_gte_120, 2),
                "pct_person_height_60_119px": round(self.pct_height_60_119, 2),
                "pct_person_height_lt_60px": round(self.pct_height_lt_60, 2),
                "pct_head_crop_gte_25x25px": round(self.pct_head_crop_gte_25, 2),
                "phone_median_apparent_size_px": (
                    [round(self.phone_median_apparent_size_px[0], 1), round(self.phone_median_apparent_size_px[1], 1)]
                    if self.phone_median_apparent_size_px else None
                ),
            },
            "diagnostic_flags": self.diagnostic_flags,
        }


def evaluate_camera_capability(
    profile: CameraProfile,
    person_heights_px: List[float],
    head_dimensions_px: List[float],
    phone_boxes_wh: List[Tuple[float, float]],
    fps_observed: float,
    drop_pct: float,
) -> CameraCapabilitySummary:
    """
    Computes camera capability levels and quantitative coverage from observed distributions.
    Does NOT fabricate accuracy claims; represents physical visibility and processing capacity.
    """
    total_persons = len(person_heights_px)
    if total_persons == 0:
        pct_gte_120 = 0.0
        pct_60_119 = 0.0
        pct_lt_60 = 0.0
    else:
        pct_gte_120 = (sum(1 for h in person_heights_px if h >= 120.0) / total_persons) * 100.0
        pct_60_119 = (sum(1 for h in person_heights_px if 60.0 <= h < 120.0) / total_persons) * 100.0
        pct_lt_60 = (sum(1 for h in person_heights_px if h < 60.0) / total_persons) * 100.0

    total_heads = len(head_dimensions_px)
    if total_heads == 0:
        pct_head_gte_25 = 0.0
    else:
        pct_head_gte_25 = (sum(1 for d in head_dimensions_px if d >= 25.0) / total_heads) * 100.0

    # Phone median size
    if phone_boxes_wh:
        med_w = float(np.median([wh[0] for wh in phone_boxes_wh]))
        med_h = float(np.median([wh[1] for wh in phone_boxes_wh]))
        phone_median = (med_w, med_h)
    else:
        phone_median = None

    # Determine Posture Capability
    if not profile.posture_enabled or pct_gte_120 + pct_60_119 < 20.0:
        posture_cap = CapabilityLevel.UNAVAILABLE
    elif pct_gte_120 >= 70.0:
        posture_cap = CapabilityLevel.FULL
    else:
        posture_cap = CapabilityLevel.LIMITED

    # Determine Head-Pose Capability
    # Ceiling high angles or rear view disable facial yaw
    if profile.viewpoint_profile in [ViewpointProfile.CEILING_HIGH, ViewpointProfile.REAR_OBLIQUE]:
        headpose_cap = CapabilityLevel.UNAVAILABLE
    elif not profile.headpose_enabled or pct_head_gte_25 < 30.0:
        headpose_cap = CapabilityLevel.UNAVAILABLE
    elif pct_head_gte_25 >= 75.0:
        headpose_cap = CapabilityLevel.FULL
    else:
        headpose_cap = CapabilityLevel.LIMITED

    # Determine Phone Capability
    if not profile.phone_enabled:
        phone_cap = CapabilityLevel.UNAVAILABLE
    elif phone_median is not None and phone_median[0] >= 25.0 and phone_median[1] >= 25.0:
        phone_cap = CapabilityLevel.FULL
    elif phone_median is not None and phone_median[0] >= 15.0:
        phone_cap = CapabilityLevel.LIMITED
    else:
        phone_cap = CapabilityLevel.UNAVAILABLE

    # Tracking capability
    if drop_pct > 15.0 or fps_observed < 12.0:
        tracking_cap = TrackingCapability.DEGRADED
    else:
        tracking_cap = TrackingCapability.SUPPORTED

    # Diagnostic flags
    diag_flags = []
    if profile.viewpoint_profile == ViewpointProfile.CEILING_HIGH or (profile.camera_pitch_deg is not None and profile.camera_pitch_deg < -45.0):
        diag_flags.append("HIGH_ANGLE_WARNING")
    if pct_lt_60 > 40.0:
        diag_flags.append("LOW_PERSON_PIXEL_COVERAGE")
    if headpose_cap == CapabilityLevel.UNAVAILABLE:
        diag_flags.append("LOW_HEADPOSE_COVERAGE")
    if pct_head_gte_25 < 25.0 and profile.viewpoint_profile != ViewpointProfile.CEILING_HIGH:
        diag_flags.append("HEADPOSE_UNAVAILABLE")
    if phone_cap != CapabilityLevel.FULL:
        diag_flags.append("PHONE_VISIBILITY_LIMITED")
    if drop_pct > 10.0:
        diag_flags.append("HIGH_FRAME_DROP_RATE")

    return CameraCapabilitySummary(
        camera_id=profile.camera_id,
        posture_capability=posture_cap,
        headpose_capability=headpose_cap,
        phone_capability=phone_cap,
        tracking_capability=tracking_cap,
        pct_height_gte_120=pct_gte_120,
        pct_height_60_119=pct_60_119,
        pct_height_lt_60=pct_lt_60,
        pct_head_crop_gte_25=pct_head_gte_25,
        phone_median_apparent_size_px=phone_median,
        diagnostic_flags=diag_flags,
    )
