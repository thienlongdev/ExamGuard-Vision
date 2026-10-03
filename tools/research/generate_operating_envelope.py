"""
Deterministic Operating Envelope Generator
===========================================
Mechanically derives operating_envelope.json from resolution_occupancy_matrix.json.
Enforces Section 21-24 requirements:
- Only scenarios passing the strict Occupancy Validity Gate may contribute.
- Evaluates line-rate classification rule: STABLE_LINE_RATE, STABLE_WITH_BOUNDED_DROPS, UNSTABLE.
- Mechanically selects max_safe_fps as the highest tested FPS satisfying stable criteria.
- Persists selection reasoning and provenance.
- Zero hardcoded recommendation tables allowed.
"""

import json
import logging
import os
import sys
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_operating_envelope")


def derive_operating_envelope(
    matrix_path: str = "runs/pilot/resolution_occupancy_matrix.json",
    output_path: str = "runs/pilot/operating_envelope.json",
    backup_path: str = "runs/pilot/PILOT_OPERATING_ENVELOPE.json",
) -> Dict[str, Any]:
    """
    Reads resolution_occupancy_matrix.json and algorithmically determines
    safe FPS for each resolution and student occupancy level.
    """
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Matrix artifact not found at {matrix_path}")

    with open(matrix_path, "r", encoding="utf-8") as f:
        matrix_data = json.load(f)

    # If matrix_data is dict with standard_results, unpack it
    if isinstance(matrix_data, dict) and "standard_results" in matrix_data:
        scenarios = matrix_data["standard_results"]
    elif isinstance(matrix_data, list):
        scenarios = matrix_data
    else:
        raise ValueError("Invalid matrix data format")

    logger.info(f"Loaded {len(scenarios)} scenarios from {matrix_path}")

    # Group by resolution and occupancy
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for sc in scenarios:
        res = sc["source_resolution"]
        occ = f"{sc['target_scene_occupancy']}_students"
        if res not in grouped:
            grouped[res] = {}
        if occ not in grouped[res]:
            grouped[res][occ] = []
        grouped[res][occ].append(sc)

    envelope: Dict[str, Any] = {
        "metadata": {
            "source_matrix": matrix_path,
            "generator": "scripts/generate_operating_envelope.py",
            "methodology": "STRICT_ARTIFACT_DERIVED_MECHANICAL_SELECTION",
            "occupancy_gate_enforced": True,
            "provenance": "Zero manual numeric overrides. Derived from raw telemetry.",
        },
        "resolutions": {},
    }

    for res, occ_dict in sorted(grouped.items()):
        envelope["resolutions"][res] = {}

        for occ_key, sc_list in sorted(occ_dict.items(), key=lambda x: int(x[0].split("_")[0])):
            # Sort by requested source FPS ascending
            sc_list = sorted(sc_list, key=lambda s: s["requested_source_fps"])

            status_by_fps = {}
            valid_candidates = []
            selection_reasoning = []

            for s in sc_list:
                fps_val = s["requested_source_fps"]
                fps_label = f"{int(fps_val)}_fps"
                gate_passed = s.get("occupancy_validity_gate", {}).get("passed", False)
                stability = s.get("stability_classification", "UNSTABLE")
                c2r_p95 = s.get("capture_to_result_latency_ms", {}).get("p95", 999.0)
                drop_pct = s.get("drops", {}).get("percentage", 100.0)
                eff_fps = s.get("effective_processed_fps", 0.0)

                status_by_fps[fps_label] = {
                    "stability": stability,
                    "occupancy_gate_passed": gate_passed,
                    "effective_fps": eff_fps,
                    "drop_percentage": drop_pct,
                    "capture_to_result_p95_ms": c2r_p95,
                    "actual_mean_active_tracks": s.get("actual_active_tracks", {}).get("mean", 0.0),
                }

                # Section 22: ONLY VALID OCCUPANCY SCENARIOS MAY CONTRIBUTE
                if not gate_passed:
                    selection_reasoning.append(f"{fps_label}: EXCLUDED (Occupancy validity gate failed)")
                    continue

                if stability == "STABLE_LINE_RATE":
                    valid_candidates.append((fps_val, "STABLE_LINE_RATE"))
                    selection_reasoning.append(f"{fps_label}: STABLE_LINE_RATE (Throughput maintained, drops <= 1%, latency P95 <= 150ms)")
                elif stability == "STABLE_WITH_BOUNDED_DROPS":
                    valid_candidates.append((fps_val, "STABLE_WITH_BOUNDED_DROPS"))
                    selection_reasoning.append(f"{fps_label}: STABLE_WITH_BOUNDED_DROPS (Drop rate {drop_pct:.1f}% <= 15%, bounded queue)")
                else:
                    selection_reasoning.append(f"{fps_label}: UNSTABLE ({s.get('limiting_factor', 'UNSTABLE')})")

            # Section 24: Max safe FPS is highest tested FPS satisfying stable criteria
            if valid_candidates:
                # Prefer highest STABLE_LINE_RATE
                line_rate_candidates = [fps for fps, stab in valid_candidates if stab == "STABLE_LINE_RATE"]
                if line_rate_candidates:
                    max_safe_fps = max(line_rate_candidates)
                    confidence = "HIGH_CONFIDENCE_LINE_RATE"
                else:
                    max_safe_fps = max(fps for fps, stab in valid_candidates)
                    confidence = "BOUNDED_DROPS_OPERATIONAL"
            else:
                max_safe_fps = 0.0
                confidence = "UNSUPPORTED_WORKLOAD"

            envelope["resolutions"][res][occ_key] = {
                "max_safe_fps": max_safe_fps,
                "confidence_tier": confidence,
                "status_by_fps": status_by_fps,
                "selection_reasoning": selection_reasoning,
            }

            logger.info(f"Derived: {res} | {occ_key} -> max_safe_fps = {max_safe_fps} ({confidence})")

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2)

    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2)

    logger.info(f"Operating envelope written to {output_path} and {backup_path}")
    return envelope


if __name__ == "__main__":
    matrix_file = sys.argv[1] if len(sys.argv) > 1 else "runs/pilot/resolution_occupancy_matrix.json"
    derive_operating_envelope(matrix_file)
