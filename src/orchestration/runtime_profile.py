"""
Runtime Profile Framework for Portable Live Demonstration
===========================================================
Defines execution profiles (AUTO, FULL, BALANCED, LOW_POWER) for target hardware.

CRITICAL GOVERNANCE RULES:
1. Scientific behavior thresholds (V4D event triggers, angle gates, min durations)
   MUST REMAIN IDENTICAL across all profiles. Only runtime cadence, batching,
   and resolution may adapt.
2. Selection for AUTO profile MUST BE BENCHMARK-DRIVEN, computed from measured
   telemetry (capture FPS, processed FPS, latency P95/P99, drop rate, queue depth,
   VRAM, RAM, thermal stability). NO profile selection based solely on GPU model string.
3. BOOTSTRAP_PROFILE_CANDIDATE is a safe initial starting candidate for the upcoming
   ASUS TUF A17 benchmark, NOT the finalized profile.
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, Tuple, Optional


class RuntimeProfileName(str, Enum):
    AUTO = "AUTO"
    FULL = "FULL"
    BALANCED = "BALANCED"
    LOW_POWER = "LOW_POWER"


@dataclass(frozen=True)
class RuntimeProfileConfig:
    profile_name: RuntimeProfileName
    description: str
    camera_resolution: Tuple[int, int]
    camera_requested_fps: float
    detector_imgsz: int
    detector_cadence_hz: float
    bytetrack_every_frame: bool
    posture_resolution: Tuple[int, int]
    posture_cadence_hz: float
    headpose_resolution: Tuple[int, int]
    headpose_cadence_hz: float
    macro_imgsz: int
    macro_cadence_hz: float
    queue_depth: int
    evidence_mode: str
    behavior_thresholds_immutable: bool = True  # Scientific thresholds MUST NOT mutate!

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["profile_name"] = self.profile_name.value
        return d


# 1. Non-canonical initial candidate for the future live benchmark on ASUS A17
BOOTSTRAP_PROFILE_CANDIDATE = RuntimeProfileConfig(
    profile_name=RuntimeProfileName.BALANCED,
    description="Safe bootstrap starting candidate for ASUS TUF A17 RTX 3050 live benchmark",
    camera_resolution=(1280, 720),
    camera_requested_fps=30.0,
    detector_imgsz=640,
    detector_cadence_hz=12.0,       # candidate 10-15 Hz
    bytetrack_every_frame=True,     # ByteTrack runs every captured frame
    posture_resolution=(224, 224),
    posture_cadence_hz=10.0,        # candidate 8-10 Hz
    headpose_resolution=(224, 224),
    headpose_cadence_hz=5.0,        # candidate 4-6 Hz
    macro_imgsz=768,
    macro_cadence_hz=4.0,           # candidate 3-5 Hz
    queue_depth=5,
    evidence_mode="EVENT_SNAPSHOT_ONLY",
    behavior_thresholds_immutable=True,
)

# 2. Defined Profiles Catalog
PROFILES_CATALOG: Dict[RuntimeProfileName, RuntimeProfileConfig] = {
    RuntimeProfileName.FULL: RuntimeProfileConfig(
        profile_name=RuntimeProfileName.FULL,
        description="High-cadence profile for powerful workstations (e.g. RTX 5070 / Desktop)",
        camera_resolution=(1280, 720),
        camera_requested_fps=30.0,
        detector_imgsz=640,
        detector_cadence_hz=30.0,
        bytetrack_every_frame=True,
        posture_resolution=(224, 224),
        posture_cadence_hz=15.0,
        headpose_resolution=(224, 224),
        headpose_cadence_hz=10.0,
        macro_imgsz=768,
        macro_cadence_hz=6.0,
        queue_depth=5,
        evidence_mode="FULL_SNAPSHOT_AND_CLIP",
        behavior_thresholds_immutable=True,
    ),
    RuntimeProfileName.BALANCED: BOOTSTRAP_PROFILE_CANDIDATE,
    RuntimeProfileName.LOW_POWER: RuntimeProfileConfig(
        profile_name=RuntimeProfileName.LOW_POWER,
        description="Reduced-cadence profile for thermal/battery conservation or resource contention",
        camera_resolution=(1280, 720),
        camera_requested_fps=30.0,
        detector_imgsz=640,
        detector_cadence_hz=8.0,
        bytetrack_every_frame=True,
        posture_resolution=(224, 224),
        posture_cadence_hz=6.0,
        headpose_resolution=(224, 224),
        headpose_cadence_hz=3.0,
        macro_imgsz=768,
        macro_cadence_hz=2.0,
        queue_depth=3,
        evidence_mode="EVENT_SNAPSHOT_ONLY",
        behavior_thresholds_immutable=True,
    ),
}


def evaluate_runtime_profile_from_telemetry(
    metrics: Dict[str, Any]
) -> Tuple[RuntimeProfileName, Dict[str, Any]]:
    """
    Evaluates measured runtime telemetry to select the optimal runtime profile.

    STRICT GOVERNANCE:
    - Profile selection MUST be derived from measured throughput, latency, queue, VRAM, and drops.
    - Profile selection MUST NOT be based solely on GPU model string or vendor name.
    """
    processed_fps = metrics.get("processed_fps_mean") or 0.0
    latency_p95 = metrics.get("latency_p95_ms") or 999.0
    drop_pct = metrics.get("drop_percent") or 0.0
    queue_wait_ms = metrics.get("queue_wait_mean_ms") or 999.0
    vram_used_mb = metrics.get("vram_used_mb") or 0.0
    vram_total_mb = metrics.get("vram_total_mb") or 4096.0

    eval_rationale = {
        "processed_fps": processed_fps,
        "latency_p95_ms": latency_p95,
        "drop_percent": drop_pct,
        "queue_wait_mean_ms": queue_wait_ms,
        "vram_headroom_mb": max(0.0, vram_total_mb - vram_used_mb),
    }

    # If severe backpressure, dropped frames > 10% or latency > 120ms: LOW_POWER
    if drop_pct > 10.0 or latency_p95 > 120.0 or queue_wait_ms > 50.0:
        recommended = RuntimeProfileName.LOW_POWER
        eval_rationale["selection_reason"] = "Backpressure/latency threshold exceeded; selecting LOW_POWER"
    # If comfortable processing >= 20 FPS, low latency <= 60ms, negligible drops <= 2%: FULL
    elif processed_fps >= 20.0 and latency_p95 <= 60.0 and drop_pct <= 2.0:
        recommended = RuntimeProfileName.FULL
        eval_rationale["selection_reason"] = "High throughput and low latency verified; selecting FULL"
    # Otherwise: BALANCED
    else:
        recommended = RuntimeProfileName.BALANCED
        eval_rationale["selection_reason"] = "Optimal balanced operational envelope; selecting BALANCED"

    return recommended, eval_rationale
