"""
Pilot Camera Calibration Toolkit
==================================
Implements Part B requirements:
- Offline calibration on local video or stream
- Measurement of frame timing, decode latency, tracking lifespan, detection distributions
- Crop size distributions (person height, head dimensions, phone size)
- Empirical capability coverage calculation without fabricating accuracy
- Geometry diagnostic flags (HIGH_ANGLE_WARNING, LOW_PERSON_PIXEL_COVERAGE, etc.)
- Generation of camera_calibration.json, camera_calibration.md, sample_overlays, and histograms
"""

from dataclasses import dataclass, field
import json
import logging
import os
import time
from typing import Dict, List, Optional, Any, Tuple
import cv2
import numpy as np

from src.video.base import VideoFrame, VideoSource
from src.video.factory import create_video_source
from src.orchestration.stage2_pipeline import Stage2Pipeline
from src.pilot.profile import (
    CameraProfile,
    CameraCapabilitySummary,
    evaluate_camera_capability,
    ViewpointProfile,
    CapabilityLevel,
    TrackingCapability,
)

logger = logging.getLogger(__name__)


@dataclass
class CalibrationTelemetry:
    """Raw telemetry collected during camera calibration."""
    camera_id: str
    source_uri: str
    source_resolution: Tuple[int, int]
    nominal_fps: float
    frames_collected: int
    duration_sec: float

    # Timing
    received_fps: float
    interframe_intervals_ms: List[float]
    decode_latencies_ms: List[float]
    whole_loop_latencies_ms: List[float]

    # Detections & Tracking
    person_counts_per_frame: List[int]
    track_counts_per_frame: List[int]
    person_bbox_heights_px: List[float]
    person_bbox_widths_px: List[float]
    head_dimensions_px: List[float]
    phone_sizes_wh_px: List[Tuple[float, float]]

    # Lifespan & Continuity
    track_lifespans_frames: List[int]
    occlusion_overlap_count: int
    dropped_frames_count: int
    max_queue_depth: int

    # Resource metrics
    gpu_allocated_vram_mb: float
    gpu_reserved_vram_mb: float
    host_rss_mb: float


class CameraCalibrator:
    """Calibrates an examination hall camera offline or live."""

    def __init__(
        self,
        profile: CameraProfile,
        pipeline: Optional[Stage2Pipeline] = None,
        output_dir: Optional[str] = None,
    ):
        self.profile = profile
        self.output_dir = output_dir or os.path.join("runs", "pilot", "calibration", profile.camera_id)
        self.overlays_dir = os.path.join(self.output_dir, "sample_overlays")
        self.histograms_dir = os.path.join(self.output_dir, "capability_histograms")

        os.makedirs(self.overlays_dir, exist_ok=True)
        os.makedirs(self.histograms_dir, exist_ok=True)

        self.pipeline = pipeline or Stage2Pipeline(enable_debug_overlay=True)

    def calibrate(
        self,
        video_source: Optional[VideoSource] = None,
        max_frames: int = 150,
        max_duration_sec: float = 60.0,
    ) -> Tuple[CameraCapabilitySummary, CalibrationTelemetry]:
        """Runs the calibration collection window."""
        source = video_source or create_video_source(
            source_type=self.profile.source_type,
            source=self.profile.source_uri,
            source_id=self.profile.camera_id,
            fps=self.profile.nominal_fps,
        )

        if not source.is_opened():
            source.open()

        interframe_times = []
        decode_latencies = []
        loop_latencies = []
        person_counts = []
        track_counts = []
        person_heights = []
        person_widths = []
        head_dimensions = []
        phone_sizes = []
        saved_overlays = 0
        last_ts = None
        t_start = time.time()
        frames_done = 0

        logger.info(f"Starting camera calibration for {self.profile.camera_id}...")

        while frames_done < max_frames and (time.time() - t_start) < max_duration_sec:
            t0 = time.perf_counter()
            vf = source.read()
            t_decode = (time.perf_counter() - t0) * 1000.0

            if vf is None:
                break

            decode_latencies.append(t_decode)
            if last_ts is not None:
                interframe_times.append((vf.timestamp - last_ts) * 1000.0)
            last_ts = vf.timestamp

            # Process through pipeline
            res = self.pipeline.process_frame(vf)
            loop_latencies.append(res.metrics.whole_loop_end_to_end_ms)

            # Extract student tracks & ROI
            visible_tracks = [t for t in res.tracks if self.profile.filter_roi(t.bbox)]
            track_counts.append(len(visible_tracks))
            person_counts.append(len(res.tracks))

            for t in visible_tracks:
                x1, y1, x2, y2 = t.bbox
                h = max(0.0, y2 - y1)
                w = max(0.0, x2 - x1)
                person_heights.append(h)
                person_widths.append(w)
                # Head bbox proxy: upper 25% of tight person crop
                head_dimensions.append(max(w * 0.5, h * 0.25))

            # Phone detections
            for u in res.unified_updates:
                if u.phone and u.phone.detected and u.phone.phone_bbox:
                    px1, py1, px2, py2 = u.phone.phone_bbox
                    phone_sizes.append((max(0.0, px2 - px1), max(0.0, py2 - py1)))

            # Save sample overlays (first 3 frames)
            if saved_overlays < 3 and res.annotated_frame is not None:
                overlay_path = os.path.join(self.overlays_dir, f"sample_overlay_{frames_done:04d}.jpg")
                self._draw_calibration_overlay(res.annotated_frame, visible_tracks, overlay_path)
                saved_overlays += 1

            frames_done += 1

        duration = max(0.001, time.time() - t_start)
        rec_fps = frames_done / duration

        # Host and GPU telemetry
        import psutil
        rss_mb = psutil.Process().memory_info().rss / (1024.0 ** 2)
        import torch
        if torch.cuda.is_available():
            vram_alloc = torch.cuda.memory_allocated(0) / (1024.0 ** 2)
            vram_res = torch.cuda.memory_reserved(0) / (1024.0 ** 2)
        else:
            vram_alloc = 0.0
            vram_res = 0.0

        # Lifespans from metadata
        lifespans = [m.get("track_age_frames", 1) for m in self.pipeline._track_metadata.values()]
        if not lifespans:
            lifespans = [1]

        drop_pct = (self.pipeline.dropped_frames_count / max(1, frames_done + self.pipeline.dropped_frames_count)) * 100.0

        telemetry = CalibrationTelemetry(
            camera_id=self.profile.camera_id,
            source_uri=self.profile.source_uri,
            source_resolution=(source.width, source.height),
            nominal_fps=self.profile.nominal_fps,
            frames_collected=frames_done,
            duration_sec=duration,
            received_fps=rec_fps,
            interframe_intervals_ms=interframe_times,
            decode_latencies_ms=decode_latencies,
            whole_loop_latencies_ms=loop_latencies,
            person_counts_per_frame=person_counts,
            track_counts_per_frame=track_counts,
            person_bbox_heights_px=person_heights,
            person_bbox_widths_px=person_widths,
            head_dimensions_px=head_dimensions,
            phone_sizes_wh_px=phone_sizes,
            track_lifespans_frames=lifespans,
            occlusion_overlap_count=0,
            dropped_frames_count=self.pipeline.dropped_frames_count,
            max_queue_depth=self.pipeline.ingestion_queue.qsize,
            gpu_allocated_vram_mb=vram_alloc,
            gpu_reserved_vram_mb=vram_res,
            host_rss_mb=rss_mb,
        )

        capability = evaluate_camera_capability(
            profile=self.profile,
            person_heights_px=person_heights,
            head_dimensions_px=head_dimensions,
            phone_boxes_wh=phone_sizes,
            fps_observed=rec_fps,
            drop_pct=drop_pct,
        )

        # Generate reports and histograms
        self._generate_histogram_plot(telemetry)
        self._write_calibration_artifacts(capability, telemetry)

        return capability, telemetry

    def _draw_calibration_overlay(self, frame: np.ndarray, tracks: List[Any], out_path: str) -> None:
        """Draws ROI polygon, seat zones, and tracks on sample overlay."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Draw ROI polygon
        if self.profile.roi_polygon and len(self.profile.roi_polygon) >= 3:
            pts = np.array(self.profile.roi_polygon, dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(vis, "EXAM ROI", (pts[0][0] + 5, pts[0][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw Seat Zones
        for sz in self.profile.seat_zones:
            if sz.desk_polygon and len(sz.desk_polygon) >= 3:
                pts = np.array(sz.desk_polygon, dtype=np.int32)
                cv2.polylines(vis, [pts], isClosed=True, color=(255, 128, 0), thickness=2)
                cv2.putText(vis, f"Seat: {sz.zone_id}", (pts[0][0], pts[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)

        # Header overlay
        header = f"CALIBRATION OVERLAY | Cam: {self.profile.camera_id} | Viewpoint: {self.profile.viewpoint_profile.value}"
        cv2.rectangle(vis, (0, 0), (w, 35), (20, 20, 20), -1)
        cv2.putText(vis, header, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imwrite(out_path, vis)

    def _generate_histogram_plot(self, tel: CalibrationTelemetry) -> None:
        """Saves capability histogram images using matplotlib or numpy/cv2 fallback."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # 1. Person Height Distribution
            ax = axes[0, 0]
            if tel.person_bbox_heights_px:
                ax.hist(tel.person_bbox_heights_px, bins=15, color="#3b82f6", edgecolor="black")
                ax.axvline(120.0, color="green", linestyle="--", label="Posture Full (120px)")
                ax.axvline(60.0, color="orange", linestyle="--", label="Posture Min (60px)")
                ax.legend()
            ax.set_title("Person BBox Height Distribution (px)")
            ax.set_xlabel("Height (px)")
            ax.set_ylabel("Count")

            # 2. Head Dimension Distribution
            ax = axes[0, 1]
            if tel.head_dimensions_px:
                ax.hist(tel.head_dimensions_px, bins=15, color="#8b5cf6", edgecolor="black")
                ax.axvline(25.0, color="green", linestyle="--", label="Yaw Eligible (25px)")
                ax.legend()
            ax.set_title("Head Region Dimension Distribution (px)")
            ax.set_xlabel("Dimension (px)")

            # 3. Whole-loop Latency Distribution
            ax = axes[1, 0]
            if tel.whole_loop_latencies_ms:
                ax.hist(tel.whole_loop_latencies_ms, bins=15, color="#10b981", edgecolor="black")
                ax.axvline(33.3, color="red", linestyle="--", label="30 FPS (33.3ms)")
                ax.axvline(40.0, color="amber" if hasattr(plt, "amber") else "orange", linestyle="--", label="25 FPS (40ms)")
                ax.legend()
            ax.set_title("Whole Loop Latency Distribution (ms)")
            ax.set_xlabel("Latency (ms)")

            # 4. Active Tracks Count
            ax = axes[1, 1]
            if tel.track_counts_per_frame:
                ax.plot(tel.track_counts_per_frame, color="#f59e0b", lw=2)
            ax.set_title("Active Student Tracks over Calibration")
            ax.set_xlabel("Frame Index")
            ax.set_ylabel("Tracks")

            plt.tight_layout()
            out_img = os.path.join(self.histograms_dir, "calibration_histograms.png")
            plt.savefig(out_img, dpi=120)
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate matplotlib histogram: {e}")

    def _write_calibration_artifacts(
        self,
        cap: CameraCapabilitySummary,
        tel: CalibrationTelemetry,
    ) -> None:
        """Writes camera_calibration.json and camera_calibration.md."""
        json_path = os.path.join(self.output_dir, "camera_calibration.json")
        md_path = os.path.join(self.output_dir, "camera_calibration.md")

        cal_dict = {
            "camera_profile": self.profile.to_dict(),
            "capability_summary": cap.to_dict(),
            "telemetry": {
                "source_resolution": f"{tel.source_resolution[0]}x{tel.source_resolution[1]}",
                "nominal_fps": tel.nominal_fps,
                "received_fps": round(tel.received_fps, 2),
                "frames_collected": tel.frames_collected,
                "duration_sec": round(tel.duration_sec, 2),
                "mean_decode_latency_ms": round(float(np.mean(tel.decode_latencies_ms)), 2) if tel.decode_latencies_ms else 0.0,
                "mean_whole_loop_latency_ms": round(float(np.mean(tel.whole_loop_latencies_ms)), 2) if tel.whole_loop_latencies_ms else 0.0,
                "p95_whole_loop_latency_ms": round(float(np.percentile(tel.whole_loop_latencies_ms, 95)), 2) if tel.whole_loop_latencies_ms else 0.0,
                "mean_person_detections_per_frame": round(float(np.mean(tel.person_counts_per_frame)), 2) if tel.person_counts_per_frame else 0.0,
                "mean_active_tracks_per_frame": round(float(np.mean(tel.track_counts_per_frame)), 2) if tel.track_counts_per_frame else 0.0,
                "mean_track_lifespan_frames": round(float(np.mean(tel.track_lifespans_frames)), 1),
                "dropped_frames_count": tel.dropped_frames_count,
                "max_queue_depth": tel.max_queue_depth,
                "gpu_allocated_vram_mb": round(tel.gpu_allocated_vram_mb, 1),
                "host_rss_mb": round(tel.host_rss_mb, 1),
            },
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(cal_dict, f, indent=2)

        # Markdown Report
        md_content = f"""# Camera Calibration Report: {self.profile.camera_id}

## 1. Camera Profile & Mounting
- **Camera ID**: `{self.profile.camera_id}`
- **Camera Name**: {self.profile.name}
- **Viewpoint Profile**: `{self.profile.viewpoint_profile.value}`
- **Source Resolution**: {tel.source_resolution[0]}x{tel.source_resolution[1]}
- **Nominal FPS**: {self.profile.nominal_fps:.1f} (Received: {tel.received_fps:.1f} FPS)
- **Room ID**: `{self.profile.room_id}`
- **Mounting Details**: Height {self.profile.camera_height_m or 'UNKNOWN'}m, Pitch {self.profile.camera_pitch_deg or 'UNKNOWN'}°

---

## 2. Capability Evaluation
| Capability Branch | Level | Quantitative Physical Visibility Coverage |
| :--- | :--- | :--- |
| **Posture Capability** | **{cap.posture_capability.value}** | ≥120px: {cap.pct_height_gte_120:.1f}%, 60-119px: {cap.pct_height_60_119:.1f}%, <60px: {cap.pct_height_lt_60:.1f}% |
| **Head-Pose Capability** | **{cap.headpose_capability.value}** | Head crop ≥25x25px: {cap.pct_head_crop_gte_25:.1f}% |
| **Phone Capability** | **{cap.phone_capability.value}** | Median apparent size: {cap.phone_median_apparent_size_px or 'NOT_OBSERVED'} |
| **Tracking Capability** | **{cap.tracking_capability.value}** | Mean track lifespan: {np.mean(tel.track_lifespans_frames):.1f} frames |

---

## 3. Camera Geometry Diagnostics
{chr(10).join(f"- `{flag}`" for flag in cap.diagnostic_flags) if cap.diagnostic_flags else "- No diagnostic warnings (Nominal coverage)"}

---

## 4. Physical Runtime Telemetry
- **Frames Processed**: {tel.frames_collected}
- **Mean Whole-Loop Latency**: {np.mean(tel.whole_loop_latencies_ms):.2f} ms (P95: {np.percentile(tel.whole_loop_latencies_ms, 95):.2f} ms)
- **Decode Latency**: {np.mean(tel.decode_latencies_ms):.2f} ms
- **Frame Drop Count**: {tel.dropped_frames_count}
- **Max Ingestion Queue Depth**: {tel.max_queue_depth}
- **GPU VRAM Allocated**: {tel.gpu_allocated_vram_mb:.1f} MB
- **Host RSS Memory**: {tel.host_rss_mb:.1f} MB

*Notice: All capability indicators represent physical visibility and execution feasibility. Zero behavioral accuracy claims are made without labeled ground truth.*
"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
