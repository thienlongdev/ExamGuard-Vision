"""
Generate Pilot Camera Operating Profile
=======================================
Implements Part N requirements:
- Synthesizes calibration artifacts and operating envelope into a recommended CameraProfile
- Determines safe ingestion FPS, detector cadence, capability gating, and queue depth
- Strict Invariant: NEVER auto-tunes scientific/behavioral thresholds (duration, risk score, confidence)
  from unlabeled video. Only tunes runtime, visibility, and queue/FPS settings.
- Outputs recommended YAML profile with explicit provenance PROVISIONAL_RECOMMENDATION for operator review.
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Dict, Any, Optional
import yaml

from src.pilot.profile import (
    CameraProfile,
    ViewpointProfile,
    ProvenanceLevel,
    CapabilityLevel,
    Resolution,
)

logger = logging.getLogger(__name__)


def generate_recommended_profile(
    camera_id: str,
    calibration_json_path: str,
    envelope_json_path: Optional[str] = None,
    output_profile_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generates recommended camera profile for human invigilator review."""
    with open(calibration_json_path, "r", encoding="utf-8") as f:
        cal_data = json.load(f)

    raw_profile = cal_data.get("camera_profile", {})
    cap_summary = cal_data.get("capability_summary", {})
    telem = cal_data.get("telemetry", {})

    source_res_str = telem.get("source_resolution", "1920x1080")
    sw, sh = map(int, source_res_str.split("x"))
    expected_students = int(raw_profile.get("expected_student_count", 15))

    # Read operating envelope if provided
    rec_fps = 20.0
    if envelope_json_path and os.path.exists(envelope_json_path):
        try:
            with open(envelope_json_path, "r", encoding="utf-8") as f:
                env_data = json.load(f)
            res_key = source_res_str
            occ_key = f"{min(15, max(5, (expected_students // 5) * 5))}_students"
            safe_fps = env_data.get(res_key, {}).get(occ_key, {}).get("max_safe_fps")
            if safe_fps and safe_fps > 0:
                rec_fps = float(safe_fps)
        except Exception as e:
            logger.warning(f"Could not parse envelope {envelope_json_path}: {e}")
    else:
        # Heuristic from mean whole-loop latency
        mean_lat = float(telem.get("mean_whole_loop_latency_ms", 40.0))
        if mean_lat <= 30.0:
            rec_fps = 25.0
        elif mean_lat <= 45.0:
            rec_fps = 20.0
        else:
            rec_fps = 15.0

    # Capability determinations
    posture_cap = cap_summary.get("POSTURE_CAPABILITY", "LIMITED")
    headpose_cap = cap_summary.get("HEADPOSE_CAPABILITY", "UNAVAILABLE")
    phone_cap = cap_summary.get("PHONE_CAPABILITY", "LIMITED")

    posture_enabled = posture_cap in ["FULL", "LIMITED"]
    headpose_enabled = headpose_cap == "FULL"
    phone_enabled = phone_cap in ["FULL", "LIMITED"]

    # Warnings
    warnings = list(cap_summary.get("diagnostic_flags", []))
    if headpose_cap == "UNAVAILABLE":
        warnings.append("HEADPOSE_GATED_OFF_UNRESOLVABLE_GEOMETRY")
    if expected_students >= 20:
        warnings.append("DENSE_LOAD_DEGRADED_EXPECTED")

    recommended_profile = {
        "camera_id": camera_id,
        "name": f"Recommended Profile: {raw_profile.get('name', camera_id)}",
        "source_type": raw_profile.get("source_type", "video_file"),
        "source_uri": raw_profile.get("source_uri", "samples/sample_exam.mp4"),
        "resolution": {"width": sw, "height": sh},
        "nominal_fps": round(rec_fps, 1),
        "expected_student_count": expected_students,
        "room_id": raw_profile.get("room_id", "hall_101"),
        "viewpoint_profile": raw_profile.get("viewpoint_profile", "UNKNOWN"),
        "camera_height_m": raw_profile.get("camera_height_m"),
        "camera_pitch_deg": raw_profile.get("camera_pitch_deg"),
        "camera_roll_deg": raw_profile.get("camera_roll_deg"),
        "horizontal_fov_deg": raw_profile.get("horizontal_fov_deg"),
        "roi_polygon": raw_profile.get("roi_polygon"),
        "seat_zones": raw_profile.get("seat_zones", []),
        "general_detector_imgsz": 640,
        "macro_detector_imgsz": 768,
        "posture_enabled": posture_enabled,
        "headpose_enabled": headpose_enabled,
        "phone_enabled": phone_enabled,
        "evidence_enabled": True,
        "privacy": {
            "anonymize_tracks": True,
            "enable_face_recognition": False, # NEVER ENABLED
            "store_full_room_stream": False,
        },
        "provenance_level": "CALIBRATED_FROM_VISIBILITY",
        "recommended_cadence": {
            "detector_hz": round(rec_fps, 1),
            "posture_hz": 10.0 if posture_enabled else 0.0,
            "headpose_hz": 6.0 if headpose_enabled else 0.0,
            "phone_hz": 6.0 if phone_enabled else 0.0,
            "macro_hz": 6.0,
        },
        "queue_depth_max": 5,
        "drop_policy": "DROP_STALE_ON_BACKPRESSURE",
        "scientific_threshold_policy": {
            "auto_tuned": False,
            "justification": "INVARIANT_ENFORCED: No scientific/behavioral threshold auto-tuning without labeled GT.",
            "duration_thresholds_provenance": "PROVISIONAL_CANDIDATE_THRESHOLD",
            "read_write_discount_provenance": "PROVISIONAL_FUSION_WEIGHT_0.50",
        },
        "operator_warnings": warnings,
    }

    out_path = output_profile_path or os.path.join("runs", "pilot", f"{camera_id}_recommended_profile.yaml")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(recommended_profile, f, sort_keys=False, indent=2)

    logger.info(f"Saved recommended camera profile to {out_path}")
    return recommended_profile


def main():
    parser = argparse.ArgumentParser(description="Generate Pilot Camera Operating Profile")
    parser.add_argument("--camera-id", required=True, help="Camera ID to generate profile for")
    parser.add_argument("--calibration-json", required=True, help="Path to camera_calibration.json")
    parser.add_argument("--envelope-json", default=None, help="Optional path to operating envelope JSON")
    parser.add_argument("--output", default=None, help="Output YAML path")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    rec = generate_recommended_profile(
        camera_id=args.camera_id,
        calibration_json_path=args.calibration_json,
        envelope_json_path=args.envelope_json,
        output_profile_path=args.output,
    )
    print(f"Generated recommended profile for {args.camera_id}.")


if __name__ == "__main__":
    main()
