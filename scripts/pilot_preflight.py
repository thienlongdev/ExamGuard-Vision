"""
Pilot Preflight Command Line Tool
=================================
Implements Part K requirements:
Usage:
  python scripts/pilot_preflight.py --profile configs/pilot/camera_profile_template.yaml
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pilot.profile import CameraProfile, ViewpointProfile
from src.pilot.preflight import PilotPreflightChecker

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Pilot Preflight Verification")
    parser.add_argument("--profile", default="configs/pilot/camera_profile_template.yaml", help="Path to CameraProfile YAML")
    parser.add_argument("--camera-id", default=None, help="Override camera ID")
    parser.add_argument("--source", default=None, help="Override source URI")
    parser.add_argument("--output-dir", default=os.path.join("runs", "pilot", "preflight"), help="Output directory")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if os.path.exists(args.profile):
        profile = CameraProfile.from_yaml(args.profile)
    else:
        profile = CameraProfile(
            camera_id=args.camera_id or "cam_default",
            name="Default Camera",
            source_uri=args.source or "samples/sample_exam.mp4",
        )

    if args.camera_id:
        profile.camera_id = args.camera_id
    if args.source:
        profile.source_uri = args.source

    checker = PilotPreflightChecker(profile=profile)
    report = checker.run_preflight(run_source_check=True)

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, f"{profile.camera_id}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    print("\n" + "=" * 65)
    print(f"PILOT PREFLIGHT REPORT: {profile.camera_id}")
    print(f"FINAL VERDICT: {report.final_verdict}")
    print("-" * 65)
    print("CRITICAL GATES:")
    for k, v in report.critical_gates.items():
        print(f"  [{v['status']}] {k}: {v['details']}")
    print("PERFORMANCE GATES:")
    for k, v in report.performance_gates.items():
        print(f"  [{v['status']}] {k}: {v['details']}")
    print("CAPABILITY WARNINGS:")
    for k, v in report.capability_warnings.items():
        print(f"  [{v['status']}] {k}: {v['details']}")
    print("OPERATIONAL WARNINGS:")
    for k, v in report.operational_warnings.items():
        print(f"  [{v['status']}] {k}: {v['details']}")
    print(f"\nArtifact saved to: {out_json}")
    print("=" * 65 + "\n")

    if report.final_verdict == "PILOT_PREFLIGHT_FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
