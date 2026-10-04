"""
Stage 2 Full End-to-End Pipeline Orchestrator
=============================================
Orchestrates:
VideoSource -> Frame Decode / Monotonic Timestamp -> Bounded Ingestion Queue ->
General Object Detector (Person/Phone) -> ByteTrack ->
Crop Scheduler & Batching (Posture / Head-Pose) ->
Phone Spatial Association -> Macro Cues (Cadence Gated) ->
UnifiedTrackUpdate -> V4D MultiCueFusionEngine ->
Event Lifecycle State Machine -> Risk Aggregator -> Evidence Manager ->
FastAPI / WebSocket / Visualizer.

Strict Governance:
- Observable evidence only (NEVER declares CHEATING, CHEATER, GUILTY, FRAUD).
- Frozen 4-class posture taxonomy.
- Decoupled phone association branch.
- Deterministic per-track cadence & batching.
- True physical component timing instrumentation.
- Real bounded queue with DROP_STALE_ON_BACKPRESSURE.
- Real multi-cue risk aggregation with correlated cue clustering.
- Real track continuity and age metadata.
"""

from dataclasses import dataclass, field, asdict
import json
import logging
import os
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
import cv2
import numpy as np
import yaml

from src.video.base import VideoFrame, VideoSource
from src.video.factory import create_video_source
from src.detection.types import BBox, Detection
from src.detection.object_detector import YOLOObjectDetector
from src.detection.behavior_detector import YOLOBehaviorDetector
from src.tracking.tracker import Track
from src.tracking.bytetrack import ByteTrackTracker

from src.fusion.types import (
    UnifiedTrackUpdate,
    TrackingState,
    MacroBehaviorCue,
    ObservationStatus,
    FusedEvent,
    RiskLevel,
    EventFamily,
)
from src.fusion.cue_state import PerTrackCueState
from src.fusion.fusion_engine import MultiCueFusionEngine
from src.fusion.event_engine import EventEngine
from src.fusion.risk_aggregator import RiskAggregator
from src.fusion.temporal_buffer import TemporalBuffer

from src.orchestration.model_registry import ModelRegistry
from src.orchestration.crop_scheduler import CropScheduler
from src.orchestration.phone_associator import PhoneAssociator
from src.orchestration.evidence_manager import IntegratedEvidenceManager

logger = logging.getLogger(__name__)


@dataclass
class Stage2FrameMetrics:
    """Detailed physical wall-clock execution latency instrumentation for a single frame."""
    frame_idx: int
    timestamp_sec: float
    source_read_ms: float = 0.0
    decode_ms: float = 0.0                     # Alias for source_read_ms
    general_detector_ms: float = 0.0
    detector_ms: float = 0.0                   # Alias for general_detector_ms
    macro_behavior_ms: float = 0.0
    tracker_ms: float = 0.0
    crop_extraction_ms: float = 0.0
    posture_preprocess_ms: float = 0.0
    posture_inference_ms: float = 0.0
    crop_preprocess_ms: float = 0.0            # Backward compat: crop_extract + preproc
    posture_ms: float = 0.0                    # Backward compat: posture_inference_ms
    headpose_preprocess_ms: float = 0.0
    headpose_inference_ms: float = 0.0
    headpose_ms: float = 0.0                   # Backward compat: headpose_inference_ms
    phone_association_ms: float = 0.0
    phone_ms: float = 0.0                      # Backward compat: phone_association_ms
    fusion_ms: float = 0.0                     # Physical fusion engine update time
    event_engine_ms: float = 0.0               # Physical event state machine time
    event_ms: float = 0.0                      # Backward compat alias
    risk_aggregation_ms: float = 0.0           # Physical risk assessment time
    evidence_manager_ms: float = 0.0           # Physical evidence lifecycle time
    evidence_ms: float = 0.0                   # Backward compat alias
    serialization_ms: float = 0.0              # Real physical JSON serialization time
    debug_overlay_ms: float = 0.0              # Physical rendering time if enabled
    post_decode_pipeline_ms: float = 0.0       # Independent wall-clock post-decode
    total_pipeline_ms: float = 0.0             # Backward compat alias for post_decode_pipeline_ms
    whole_loop_end_to_end_ms: float = 0.0      # Canonical wall clock: read() -> publication
    queue_wait_ms: float = 0.0
    processing_ms: float = 0.0
    capture_to_result_ms: float = 0.0
    ai_frame_age_ms: float = 0.0               # Real-time latency: result_wall_time - capture_timestamp
    active_tracks_count: int = 0
    active_events_count: int = 0
    queue_depth: int = 0
    dropped_frames_count: int = 0
    stale_skipped_count: int = 0               # Intentional skips for real-time freshness

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Stage2FrameResult:
    """Complete result container for an orchestrated frame."""
    frame_idx: int
    timestamp_sec: float
    tracks: List[Track]
    unified_updates: List[UnifiedTrackUpdate]
    lifecycle_events: List[Tuple[FusedEvent, str]]  # (event, "OPEN"|"UPDATE"|"CLOSE")
    active_events: List[FusedEvent]
    metrics: Stage2FrameMetrics
    annotated_frame: Optional[np.ndarray] = None


class BoundedFrameQueue:
    """
    Thread-safe bounded queue for frame ingestion with latest-frame / drop-stale semantics.
    Enforces low latency by buffering only the freshest frames (default capacity 1-2).
    Tracks captured, AI-selected, stale-skipped, and driver-dropped frames honestly.
    """

    def __init__(self, maxsize: int = 2, drop_policy: str = "DROP_STALE_ON_BACKPRESSURE"):
        self.maxsize = max(1, maxsize)
        self.drop_policy = drop_policy
        self._queue: queue.Queue = queue.Queue(maxsize=self.maxsize)
        self._lock = threading.Lock()
        self.captured_count = 0
        self.ai_selected_count = 0
        self.stale_skipped_count = 0
        self.dropped_frames_count = 0
        self.dropped_frames_log: List[Dict[str, Any]] = []

    def push(self, frame: VideoFrame) -> Optional[VideoFrame]:
        """Push frame into bounded queue. If full and drop_policy is DROP_STALE, drop oldest."""
        with self._lock:
            self.captured_count += 1
            dropped = None
            if self._queue.full():
                if self.drop_policy == "DROP_STALE_ON_BACKPRESSURE":
                    try:
                        item = self._queue.get_nowait()
                        dropped = item[0] if isinstance(item, tuple) else item
                        self.stale_skipped_count += 1
                        self.dropped_frames_count += 1
                        self.dropped_frames_log.append({
                            "dropped_frame_idx": dropped.frame_idx,
                            "timestamp_sec": dropped.timestamp,
                            "drop_wall_time": time.time(),
                            "queue_depth_at_drop": self._queue.qsize(),
                            "reason": "STALE_SKIPPED_FOR_FRESHNESS",
                        })
                    except queue.Empty:
                        pass
                else:
                    self.dropped_frames_count += 1
                    return frame

            enter_t = time.time()
            self._queue.put((frame, enter_t))
            return dropped

    def pop(self, timeout: float = 0.5, drain_stale: bool = False) -> Optional[VideoFrame]:
        """Pop next frame. If drain_stale is True, discards intermediate older frames."""
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

        # Drain any older intermediate frames if drain_stale requested
        if drain_stale and not self._queue.empty():
            with self._lock:
                while not self._queue.empty():
                    try:
                        stale_item = self._queue.get_nowait()
                        stale_f = stale_item[0] if isinstance(stale_item, tuple) else stale_item
                        self.stale_skipped_count += 1
                        self.dropped_frames_count += 1
                        item = stale_item
                    except queue.Empty:
                        break

        with self._lock:
            self.ai_selected_count += 1

        if isinstance(item, tuple):
            frame, enter_t = item
            exit_t = time.time()
            frame._queue_enter_time = enter_t
            frame._queue_exit_time = exit_t
            frame._queue_wait_ms = (exit_t - enter_t) * 1000.0
            if hasattr(frame, "timestamp") and frame.timestamp and frame.timestamp > 0:
                frame._ai_frame_age_ms = (exit_t - frame.timestamp) * 1000.0
            else:
                frame._ai_frame_age_ms = frame._queue_wait_ms
            return frame
        return item

    def pop_latest(self, timeout: float = 0.5) -> Optional[VideoFrame]:
        return self.pop(timeout=timeout, drain_stale=True)

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def is_empty(self) -> bool:
        return self._queue.empty()


class Stage2Pipeline:
    """Full End-to-End Orchestrator for Exam Monitoring."""

    def __init__(
        self,
        config_path: str = "configs/stage2_pipeline.yaml",
        video_source: Optional[VideoSource] = None,
        enable_debug_overlay: bool = True,
        camera_id: Optional[str] = None,
        model_registry: Optional[ModelRegistry] = None,
    ):
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.pipeline_version = str(self.config.get("version", "2.1.0-integrity"))
        self.enable_debug_overlay = enable_debug_overlay

        # 1. Model Registry (Singleton, cached, self-describing, warmed up)
        self.registry = model_registry or ModelRegistry.get_instance(self.config)
        self.registry.initialize_models()

        # 2. Tracking (ByteTrack)
        track_cfg = self.config.get("tracking", {})
        self.tracker = ByteTrackTracker(
            track_high_thresh=float(track_cfg.get("track_high_thresh", 0.45)),
            track_low_thresh=float(track_cfg.get("track_low_thresh", 0.10)),
            new_track_thresh=float(track_cfg.get("new_track_thresh", 0.50)),
            track_buffer=int(track_cfg.get("track_buffer", 30)),
            match_thresh=float(track_cfg.get("match_thresh", 0.80)),
            frame_rate=int(track_cfg.get("frame_rate", 30)),
        )

        # 3. Crop Scheduler & Batched GPU Inference with Scale Gating
        self.crop_scheduler = CropScheduler(
            model_registry=self.registry,
            config=self.config,
        )

        # 4. Phone Associator
        phone_cfg = self.config.get("phone_association", {})
        self.phone_associator = PhoneAssociator(
            expand_ratio=float(phone_cfg.get("expand_student_bbox_ratio", 0.20)),
            max_distance_ratio=float(phone_cfg.get("max_center_distance_ratio", 0.65)),
            min_iou_overlap=float(phone_cfg.get("min_iou_overlap", 0.04)),
            ambiguity_margin=float(phone_cfg.get("ambiguity_margin", 0.15)),
        )

        # 5. Macro Cadence Control
        cadence_cfg = self.config.get("cadence", {})
        self.macro_hz = float(cadence_cfg.get("macro_hz", 6.0))
        self.macro_interval = 1.0 / max(0.1, self.macro_hz)
        self._last_macro_timestamp = -1.0
        self._cached_macro_dets: List[Any] = []

        # 6. Persistent Track State Metadata (real track age, continuity, missing duration)
        self._track_metadata: Dict[int, Dict[str, Any]] = {}

        # 7. V4D Temporal Multi-Cue Fusion & Event Engines
        fusion_cfg_path = self.config.get("fusion", {}).get("config_path", "configs/v4d_fusion.yaml")
        with open(fusion_cfg_path, "r", encoding="utf-8") as f:
            self.fusion_config = yaml.safe_load(f)

        self.temporal_buffer = TemporalBuffer(
            time_horizon_seconds=float(self.fusion_config.get("temporal_buffer", {}).get("time_horizon_seconds", 30.0)),
            max_samples_per_track=int(self.fusion_config.get("temporal_buffer", {}).get("max_samples_per_track", 300)),
            eviction_inactive_seconds=float(self.fusion_config.get("temporal_buffer", {}).get("eviction_inactive_seconds", 15.0)),
            track_continuity_tolerance_sec=float(self.fusion_config.get("temporal_buffer", {}).get("track_continuity_tolerance_sec", 2.0)),
        )
        self.fusion_engine = MultiCueFusionEngine(
            config=self.fusion_config,
            temporal_buffer=self.temporal_buffer,
        )
        self.event_engine = EventEngine(config=self.fusion_config)
        self.risk_aggregator = RiskAggregator(config=self.fusion_config)

        # 8. Evidence Manager
        self.evidence_listeners: List[Callable[[str, str, str], None]] = []
        ev_cfg = self.config.get("evidence", {})
        self.evidence_manager = IntegratedEvidenceManager(
            output_dir=str(ev_cfg.get("output_dir", "storage/evidence")),
            max_snapshots_per_event=int(ev_cfg.get("max_snapshots_per_event", 3)),
            pre_event_seconds=float(ev_cfg.get("clip_pre_event_sec", 3.0)),
            post_event_seconds=float(ev_cfg.get("clip_post_event_sec", 3.0)),
            enabled=bool(ev_cfg.get("enabled", True)),
            on_evidence_ready=self._on_evidence_ready,
        )

        # 9. Video Source
        vs_cfg = self.config.get("video_source", {})
        cid = camera_id or vs_cfg.get("camera_id", "cam_0")
        self.camera_id = cid

        if video_source is not None:
            self.source = video_source
        else:
            stype = vs_cfg.get("default_type", "video_file")
            spath = vs_cfg.get("default_path", "samples/sample_exam.mp4")
            fps = float(vs_cfg.get("target_fps", 30.0))
            self.source = create_video_source(source_type=stype, source=spath, source_id=cid, fps=fps)

        # Determine semantic source origin from video source (never falsely elevated)
        if hasattr(self.source, "source_origin") and self.source.source_origin:
            self.source_origin = str(self.source.source_origin)
        elif hasattr(self.source, "source_type"):
            stype = str(self.source.source_type).lower()
            if stype == "webcam":
                self.source_origin = "PHYSICAL_LIVE_CAMERA"
            elif stype in ["video_file", "file"]:
                self.source_origin = "VIDEO_FILE"
            elif stype == "rtsp":
                self.source_origin = "RTSP_STREAM"
            elif stype == "software_fixture":
                self.source_origin = "SOFTWARE_VALIDATION_FIXTURE"
            else:
                self.source_origin = "UNKNOWN"
        else:
            self.source_origin = "UNKNOWN"

        # 10. Real Bounded Ingestion Queue & Backpressure
        bp_cfg = self.config.get("backpressure", {})
        self.max_decode_queue = int(bp_cfg.get("max_decode_queue_depth", 5))
        self.drop_policy = str(bp_cfg.get("drop_policy", "DROP_STALE_ON_BACKPRESSURE"))
        self.ingestion_queue = BoundedFrameQueue(maxsize=self.max_decode_queue, drop_policy=self.drop_policy)
        self.processed_frames_count = 0

        # Event listeners (e.g. for WebSocket broadcasting)
        self.event_listeners: List[Callable[[FusedEvent, str], None]] = []
        self._active_events_map: Dict[str, FusedEvent] = {}

        # Downstream display frame & track buffer for browser streaming (MJPEG)
        self._latest_jpeg_frame: Optional[bytes] = None
        self._latest_tracks_summary: List[Dict[str, Any]] = []
        self._frame_lock = threading.Lock()

        import collections
        self._processed_timestamps: collections.deque = collections.deque(maxlen=30)
        self._captured_timestamps: collections.deque = collections.deque(maxlen=30)

    @property
    def observed_capture_fps(self) -> Optional[float]:
        if len(self._captured_timestamps) >= 5:
            dt = self._captured_timestamps[-1] - self._captured_timestamps[0]
            if dt > 0.01:
                return round((len(self._captured_timestamps) - 1) / dt, 1)
        return None

    @property
    def observed_processed_fps(self) -> Optional[float]:
        if len(self._processed_timestamps) >= 5:
            dt = self._processed_timestamps[-1] - self._processed_timestamps[0]
            if dt > 0.01:
                return round((len(self._processed_timestamps) - 1) / dt, 1)
        return None

    @property
    def observed_inference_fps(self) -> Optional[float]:
        return self.observed_processed_fps

    @property
    def dropped_frames_count(self) -> int:
        return self.ingestion_queue.dropped_frames_count

    def reset(self) -> None:
        """Reset transient state when switching or starting a new monitoring session (Workstream 10)."""
        with self.registry.inference_lock:
            if hasattr(self.tracker, "reset"):
                try:
                    self.tracker.reset()
                except Exception:
                    pass
            self.temporal_buffer.clear()
            self.event_engine.reset()
            self.risk_aggregator.reset()
            if hasattr(self.phone_associator, "reset"):
                try:
                    self.phone_associator.reset()
                except Exception:
                    pass
            self.crop_scheduler.reset()
            self.evidence_manager.reset()
            self._active_events_map.clear()
            self._track_metadata.clear()
            with self._frame_lock:
                self._latest_tracks_summary.clear()
            logger.info("Stage2Pipeline transient state reset for new monitoring session.")

    def add_event_listener(self, listener: Callable[[FusedEvent, str], None]) -> None:
        """Register a callback for lifecycle events."""
        self.event_listeners.append(listener)

    def _broadcast_event(self, event: FusedEvent, action: str) -> None:
        """Broadcast lifecycle transition to registered listeners."""
        for cb in self.event_listeners:
            try:
                cb(event, action)
            except Exception as e:
                logger.error(f"Error in event listener callback: {e}")

    def add_evidence_listener(self, listener: Callable[[str, str, str], None]) -> None:
        """Register a callback for evidence ready events (event_id, evidence_type, file_path)."""
        self.evidence_listeners.append(listener)

    def _on_evidence_ready(self, event_id: str, evidence_type: str, file_path: str) -> None:
        for cb in self.evidence_listeners:
            try:
                cb(event_id, evidence_type, file_path)
            except Exception as e:
                logger.debug(f"Error in evidence listener callback: {e}")

    def _compute_multi_cue_risk_inputs(
        self,
        ev: FusedEvent,
        cue_state: PerTrackCueState,
    ) -> Tuple[int, int, float]:
        """
        Derives active supporting cues, independent cue clusters, and mean supporting reliability.
        Protects against correlated double-counting (e.g. TURN_HEAD_CLEAR posture + yaw).

        Clusters:
        - ORIENTATION_CLUSTER: posture turn + continuous yaw (correlated, 1 cluster)
        - PHONE_CLUSTER: phone detected & associated (independent)
        - DISCUSSION_CLUSTER: macro discuss / peer pairing (independent)
        - STANDING_CLUSTER: macro stand / posture stand (independent)
        - POSTURE_REST_CLUSTER: posture sleep / head rest (independent)
        """
        supporting_reliabilities: List[float] = []
        active_clusters: Set[str] = set()
        active_cues_count = 0
        ev_type = ev.event_type

        # 1. Orientation Cluster (TURN_HEAD_CLEAR posture and/or headpose yaw)
        turn_posture_active = (
            cue_state.posture_status == ObservationStatus.AVAILABLE and
            cue_state.posture_probs.get("TURN_HEAD_CLEAR", 0.0) >= 0.40
        )
        yaw_active = (
            cue_state.headpose_status == ObservationStatus.AVAILABLE and
            cue_state.smoothed_yaw_deg is not None and
            abs(cue_state.smoothed_yaw_deg) >= 25.0
        )

        if ev_type in [
            EventFamily.SUSTAINED_LATERAL_HEAD_ORIENTATION.value,
            EventFamily.MULTI_CUE_ATTENTION_SHIFT.value,
        ]:
            if turn_posture_active:
                active_cues_count += 1
                supporting_reliabilities.append(cue_state.posture_reliability)
                active_clusters.add("ORIENTATION_CLUSTER")
            if yaw_active:
                active_cues_count += 1
                supporting_reliabilities.append(cue_state.headpose_reliability)
                active_clusters.add("ORIENTATION_CLUSTER")

        # 2. Phone Cluster
        phone_active = (
            cue_state.phone_status == ObservationStatus.AVAILABLE and
            cue_state.phone_detected and
            cue_state.phone_association_status in ["CLEAR_ASSOCIATION", "AMBIGUOUS_ASSOCIATION"]
        )
        if phone_active:
            active_cues_count += 1
            supporting_reliabilities.append(cue_state.phone_reliability)
            active_clusters.add("PHONE_CLUSTER")

        # 3. Posture Rest Cluster
        rest_active = (
            cue_state.posture_status == ObservationStatus.AVAILABLE and
            cue_state.posture_probs.get("HEAD_REST_SLEEP", 0.0) >= 0.40
        )
        if ev_type == EventFamily.SUSTAINED_HEAD_REST.value and rest_active:
            active_cues_count += 1
            supporting_reliabilities.append(cue_state.posture_reliability)
            active_clusters.add("POSTURE_REST_CLUSTER")

        # 4. Discussion Cluster
        discuss_active = (
            cue_state.macro_status == ObservationStatus.AVAILABLE and
            cue_state.discuss_score >= 0.40
        )
        if ev_type == EventFamily.DISCUSSION_CANDIDATE.value and discuss_active:
            active_cues_count += 1
            supporting_reliabilities.append(1.0)
            active_clusters.add("DISCUSSION_CLUSTER")

        # 5. Standing Cluster
        stand_active = (
            cue_state.macro_status == ObservationStatus.AVAILABLE and
            cue_state.stand_score >= 0.40
        )
        if ev_type == EventFamily.STANDING.value and stand_active:
            active_cues_count += 1
            supporting_reliabilities.append(1.0)
            active_clusters.add("STANDING_CLUSTER")

        # Default fallback if no specific cues met primary threshold
        if not supporting_reliabilities:
            active_cues_count = max(1, active_cues_count)
            independent_cues_count = 1
            mean_rel = cue_state.posture_reliability if cue_state.posture_reliability > 0.0 else 1.0
        else:
            independent_cues_count = max(1, len(active_clusters))
            mean_rel = sum(supporting_reliabilities) / len(supporting_reliabilities)

        # Competing negative evidence discount (NORMAL_READ_WRITE suppresses SLEEP)
        if cue_state.read_write_suppression_active:
            mean_rel *= 0.50

        return active_cues_count, independent_cues_count, mean_rel

    def process_frame(
        self,
        video_frame: VideoFrame,
        source_read_ms: float = 0.0,
        t_loop_start: Optional[float] = None,
        injected_tracks: Optional[List[Track]] = None,
    ) -> Stage2FrameResult:
        """
        Process a single frame through the complete Stage 2 orchestration.
        All module boundaries are physically instrumented.
        """
        if video_frame is None:
            return None

        t_pipeline_start = time.perf_counter()
        frame = video_frame.frame
        ts = video_frame.timestamp
        frame_idx = video_frame.frame_idx
        h, w = frame.shape[:2]
        cam_id = video_frame.source_id

        # Update rolling FPS tracking
        self._processed_timestamps.append(time.time())
        if ts is not None and ts > 0:
            self._captured_timestamps.append(ts)

        # Feed rolling clip buffer immediately (if not already pushed by capture worker)
        if not getattr(video_frame, "_evidence_pushed", False):
            self.evidence_manager.push_frame(frame, ts)

        # 1. Full-frame General Object Detection (Person & Phone) or Injected Tracks for C2 load scaling
        if injected_tracks is not None:
            tracks = injected_tracks
            detections = []
            general_det_ms = 0.0
            track_ms = 0.0
        else:
            t_det_start = time.perf_counter()
            with self.registry.inference_lock:
                detections = self.registry.detector.detect(frame)
            t_det_end = time.perf_counter()
            general_det_ms = (t_det_end - t_det_start) * 1000.0

            # 2. Multi-Object Tracking (ByteTrack)
            t_track_start = time.perf_counter()
            tracks = self.tracker.update(detections, ts, (h, w))
            t_track_end = time.perf_counter()
            track_ms = (t_track_end - t_track_start) * 1000.0

        # Update persistent track state metadata & handle expired tracks
        active_track_ids = [t.track_id for t in tracks]
        self.crop_scheduler.cleanup_expired_tracks(active_track_ids)

        for track in tracks:
            tid = track.track_id
            if tid not in self._track_metadata:
                self._track_metadata[tid] = {
                    "first_seen_timestamp": ts,
                    "last_seen_timestamp": ts,
                    "track_age_frames": 1,
                    "time_since_seen_sec": 0.0,
                    "missing_duration": 0.0,
                    "continuity_status": "NEW",
                    "baseline_y1_samples": [],
                    "baseline_height_samples": [],
                    "baseline_aspect_samples": [],
                    "baseline_y1": None,
                    "baseline_height": None,
                    "baseline_aspect": None,
                }
            else:
                m_entry = self._track_metadata[tid]
                m_entry["track_age_frames"] += 1
                dt = max(0.0, ts - m_entry["last_seen_timestamp"])
                m_entry["time_since_seen_sec"] = dt
                m_entry["missing_duration"] = dt if track.lost else 0.0
                m_entry["last_seen_timestamp"] = ts
                m_entry["continuity_status"] = "LOST" if track.lost else "TRACKED"

        if frame_idx % 10 == 0 or frame_idx == 1:
            expired_ids = self.temporal_buffer.evict_inactive_tracks(ts, camera_id=cam_id)
            if expired_ids:
                closed_events = self.event_engine.handle_track_expiration(expired_ids, ts, camera_id=cam_id)
                for ev in closed_events:
                    active_ev = self._active_events_map.pop(ev.event_id, None)
                    ev_to_close = active_ev or ev
                    ev_to_close.status = "closed"
                    ev_to_close.lifecycle_status = "closed"
                    ev_to_close.end_timestamp = ts
                    ev_to_close.duration = max(0.0, ts - getattr(ev_to_close, "start_timestamp", ts))
                    self.evidence_manager.handle_event_lifecycle(ev_to_close, "CLOSE", frame, ts, fps=getattr(self.source, "fps", 30.0))
                    self._broadcast_event(ev_to_close, "CLOSE")
                for eid in expired_ids:
                    self._track_metadata.pop(eid, None)

        # 3. Phone spatial association with ambiguity checks and temporal accumulation
        t_phone_start = time.perf_counter()
        phone_associations = self.phone_associator.associate(tracks, detections, timestamp_sec=ts)
        t_phone_end = time.perf_counter()
        phone_assoc_ms = (t_phone_end - t_phone_start) * 1000.0

        # Build attention tracks set for adaptive scheduler (active events & phone candidates)
        attention_track_ids = set()
        for ev in self._active_events_map.values():
            tid = getattr(ev, "track_id", getattr(ev, "student_id", None))
            if tid is not None:
                attention_track_ids.add(tid)
        for tid, assoc in phone_associations.items():
            if assoc.detected or assoc.status == "AMBIGUOUS":
                attention_track_ids.add(tid)

        # 4. Cadence-scheduled batched GPU crop inference (Posture & Head-Pose)
        with self.registry.inference_lock:
            posture_cues, headpose_cues, crop_timings = self.crop_scheduler.schedule_and_infer(
                frame, tracks, ts, attention_track_ids=attention_track_ids
            )

        # 5. Macro behavior detection with cadence gating
        time_since_macro = ts - self._last_macro_timestamp
        if self._last_macro_timestamp < 0.0 or time_since_macro >= (self.macro_interval - 1e-4):
            t_macro_start = time.perf_counter()
            person_boxes = [t.bbox for t in tracks]
            with self.registry.inference_lock:
                macro_dets = self.registry.macro_detector.detect(frame, person_boxes)
            t_macro_end = time.perf_counter()
            macro_ms = (t_macro_end - t_macro_start) * 1000.0
            self._last_macro_timestamp = ts
            self._cached_macro_dets = macro_dets
            macro_eval_status = ObservationStatus.AVAILABLE
        else:
            macro_dets = self._cached_macro_dets
            macro_ms = 0.0
            macro_eval_status = ObservationStatus.AVAILABLE if self._cached_macro_dets else ObservationStatus.NOT_EVALUATED

        # Match tracks to macro detections via IoU / spatial containment with exclusive matching for stand
        macro_track_map = {}
        if macro_dets and tracks:
            matches: List[Tuple[float, int, Any]] = []
            for track in tracks:
                for m_det in macro_dets:
                    iou = track.bbox.iou(m_det.bbox)
                    is_contained = track.bbox.contains_point(*m_det.bbox.center)
                    if iou >= 0.15 or is_contained:
                        score = iou + (0.5 if is_contained else 0.0)
                        matches.append((score, track.track_id, m_det))

            matches.sort(key=lambda x: x[0], reverse=True)
            assigned_dets = set()
            for score, tid, m_det in matches:
                if tid in macro_track_map:
                    continue
                det_id = id(m_det)
                if m_det.behavior != "discuss" and det_id in assigned_dets:
                    continue
                macro_track_map[tid] = m_det
                assigned_dets.add(det_id)

        # 6. Physical Fusion Engine & Event Engine Updates
        t_fusion_total = 0.0
        t_event_total = 0.0
        t_risk_total = 0.0
        t_evidence_total = 0.0

        unified_updates: List[UnifiedTrackUpdate] = []
        lifecycle_events: List[Tuple[FusedEvent, str]] = []

        for track in tracks:
            t_id = track.track_id
            pos_cue = posture_cues.get(t_id)
            hp_cue = headpose_cues.get(t_id)
            phone_assoc = phone_associations.get(t_id)
            phone_cue = phone_assoc.to_phone_cue() if phone_assoc else None

            # Track state metadata & Seated baseline update
            t_meta = self._track_metadata.setdefault(t_id, {})
            cur_y1 = float(track.bbox.y1)
            cur_h = float(track.bbox.height)
            cur_w = float(track.bbox.width)
            cur_ar = cur_h / max(1.0, cur_w)

            # Update seated baseline during initial stabilization or normal/read-write posture
            is_seated_cue = False
            if t_meta.get("track_age_frames", 1) <= 15:
                is_seated_cue = True
            elif pos_cue and pos_cue.status == ObservationStatus.AVAILABLE:
                if pos_cue.predicted_class in ("NORMAL_UPRIGHT", "NORMAL_READ_WRITE"):
                    is_seated_cue = True

            if is_seated_cue:
                y1_list = t_meta.setdefault("baseline_y1_samples", [])
                h_list = t_meta.setdefault("baseline_height_samples", [])
                ar_list = t_meta.setdefault("baseline_aspect_samples", [])
                y1_list.append(cur_y1)
                h_list.append(cur_h)
                ar_list.append(cur_ar)
                if len(y1_list) > 60:
                    y1_list.pop(0)
                    h_list.pop(0)
                    ar_list.pop(0)
                t_meta["baseline_y1"] = float(np.median(y1_list))
                t_meta["baseline_height"] = float(np.median(h_list))
                t_meta["baseline_aspect"] = float(np.median(ar_list))

            # Macro cue evaluation with conservative Standing validation gate
            macro_det = macro_track_map.get(t_id)
            macro_cue = MacroBehaviorCue(status=macro_eval_status)
            if macro_det:
                if macro_det.behavior == "stand":
                    conf = float(macro_det.confidence)
                    is_stand_valid = (conf >= 0.55)

                    # Posture veto: reading/writing, upright seated, or head on desk is not standing
                    if pos_cue and pos_cue.status == ObservationStatus.AVAILABLE:
                        rw_score = pos_cue.probabilities.get("NORMAL_READ_WRITE", 0.0)
                        sleep_score = pos_cue.probabilities.get("HEAD_REST_SLEEP", 0.0)
                        if rw_score >= 0.30 or sleep_score >= 0.30:
                            is_stand_valid = False

                    # Geometry gate: Leaning forward towards camera lowers head; actual standing extends upward!
                    base_y1 = t_meta.get("baseline_y1")
                    base_h = t_meta.get("baseline_height")
                    base_ar = t_meta.get("baseline_aspect")
                    samples_cnt = len(t_meta.get("baseline_y1_samples", []))
                    if base_y1 is not None and base_h is not None and samples_cnt >= 4:
                        head_rose = (cur_y1 < base_y1 - 0.12 * base_h)
                        aspect_tall = (base_ar is not None and cur_ar > base_ar * 1.30)
                        if not (head_rose or aspect_tall):
                            is_stand_valid = False
                    elif pos_cue and pos_cue.status == ObservationStatus.AVAILABLE:
                        if pos_cue.predicted_class in ("NORMAL_READ_WRITE", "NORMAL_UPRIGHT", "HEAD_REST_SLEEP"):
                            is_stand_valid = False

                    macro_cue.stand_score = conf if is_stand_valid else 0.0
                elif macro_det.behavior == "discuss":
                    macro_cue.discuss_score = macro_det.confidence

            # Track state metadata (real age and real time since seen)
            t_meta = self._track_metadata.get(t_id, {})
            track_age = t_meta.get("track_age_frames", 1)
            time_since_seen = t_meta.get("time_since_seen_sec", 0.0)

            tracking_state = TrackingState(
                track_id=t_id,
                timestamp_sec=ts,
                bbox=track.bbox.as_tuple(),
                track_age_frames=track_age,
                time_since_seen_sec=time_since_seen,
                visibility_score=track.confidence,
                occluded=track.lost,
            )

            update = UnifiedTrackUpdate(
                track_id=t_id,
                timestamp_sec=ts,
                camera_id=cam_id,
                source_origin=self.source_origin,
                tracking=tracking_state,
                posture=pos_cue or PostureCue(),
                headpose=hp_cue or HeadPoseCue(),
                phone=phone_cue or None,
                macro_behavior=macro_cue,
            )
            unified_updates.append(update)

            # Ingest into V4D MultiCueFusionEngine
            t_f0 = time.perf_counter()
            cue_state = self.fusion_engine.update_track(update)
            t_fusion_total += (time.perf_counter() - t_f0)

            # 7. Process event state machines
            if cue_state is not None:
                t_e0 = time.perf_counter()
                emitted = self.event_engine.process_cue_state(cue_state, camera_id=cam_id)
                t_event_total += (time.perf_counter() - t_e0)

                for ev, action in emitted:
                    # Risk Aggregator scoring with real multi-cue derivation
                    t_r0 = time.perf_counter()
                    cues_cnt, indep_cnt, mean_rel = self._compute_multi_cue_risk_inputs(ev, cue_state)
                    scored_ev = self.risk_aggregator.assess_event_risk(
                        ev,
                        active_cues_count=cues_cnt,
                        independent_cues_count=indep_cnt,
                        mean_reliability=mean_rel,
                    )
                    t_risk_total += (time.perf_counter() - t_r0)

                    # Manage active events map
                    if action == "OPEN":
                        self._active_events_map[scored_ev.event_id] = scored_ev
                    elif action == "CLOSE":
                        self._active_events_map.pop(scored_ev.event_id, None)

                    # Evidence Manager transition handling
                    t_ev0 = time.perf_counter()
                    self.evidence_manager.handle_event_lifecycle(scored_ev, action, frame, ts, fps=self.source.fps)
                    t_evidence_total += (time.perf_counter() - t_ev0)

                    # Broadcast lifecycle transition
                    self._broadcast_event(scored_ev, action)
                    lifecycle_events.append((scored_ev, action))

        # 7b. Process unassociated phone observations (Room-level cue persistence)
        unassoc_phones = self.phone_associator.get_unassociated_phones()
        unassoc_machine_key = (cam_id, -1, EventFamily.PHONE_VISIBLE_UNASSOCIATED)
        if unassoc_phones or unassoc_machine_key in self.event_engine._machines:
            unassoc_emitted = self.event_engine.process_unassociated_phones(unassoc_phones, ts, camera_id=cam_id)
            for ev, action in unassoc_emitted:
                scored_ev = self.risk_aggregator.assess_event_risk(
                    ev,
                    active_cues_count=1,
                    independent_cues_count=1,
                    mean_reliability=0.85,
                )
                if action == "OPEN":
                    self._active_events_map[scored_ev.event_id] = scored_ev
                elif action == "CLOSE":
                    self._active_events_map.pop(scored_ev.event_id, None)
                self.evidence_manager.handle_event_lifecycle(scored_ev, action, frame, ts, fps=self.source.fps)
                self._broadcast_event(scored_ev, action)
                lifecycle_events.append((scored_ev, action))

        # 8. Real Serialization Path Measurement
        t_ser_start = time.perf_counter()
        if lifecycle_events:
            for ev, act in lifecycle_events:
                _ = json.dumps({"action": act, "event": ev.to_dict()})
        else:
            # Measure actual serialization of active frame summary payload
            _ = json.dumps({
                "frame_idx": frame_idx,
                "timestamp": ts,
                "active_tracks": len(tracks),
                "active_events": len(self._active_events_map),
            })
        t_ser_end = time.perf_counter()
        ser_ms = (t_ser_end - t_ser_start) * 1000.0

        # 9. Render optional debug visualization (NO CHEATER / CHEATING LABELS)
        annotated_frame = None
        t_dbg_start = time.perf_counter()
        if self.enable_debug_overlay:
            annotated_frame = self._render_debug_overlay(
                frame.copy(), tracks, posture_cues, headpose_cues, phone_associations, self._active_events_map
            )
        debug_overlay_ms = (time.perf_counter() - t_dbg_start) * 1000.0 if self.enable_debug_overlay else 0.0

        t_pipeline_end = time.perf_counter()
        post_decode_ms = (t_pipeline_end - t_pipeline_start) * 1000.0

        queue_wait_ms = getattr(video_frame, "_queue_wait_ms", 0.0)
        processing_ms = post_decode_ms
        capture_to_result_ms = queue_wait_ms + processing_ms

        # Real-time latency metric (result timestamp - original capture timestamp)
        now_wall = time.time()
        if ts is not None and ts > 0:
            ai_frame_age_ms = max(0.0, (now_wall - ts) * 1000.0)
        else:
            ai_frame_age_ms = getattr(video_frame, "_ai_frame_age_ms", capture_to_result_ms)

        # Independent canonical whole-loop end-to-end measurement
        if t_loop_start is not None:
            whole_loop_ms = (t_pipeline_end - t_loop_start) * 1000.0
        else:
            whole_loop_ms = post_decode_ms + source_read_ms

        # Timing metrics breakdown (all physical, zero fabricated percentages)
        metrics = Stage2FrameMetrics(
            frame_idx=frame_idx,
            timestamp_sec=ts,
            source_read_ms=round(source_read_ms, 3),
            decode_ms=round(source_read_ms, 3),
            general_detector_ms=round(general_det_ms, 3),
            detector_ms=round(general_det_ms, 3),
            macro_behavior_ms=round(macro_ms, 3),
            tracker_ms=round(track_ms, 3),
            crop_extraction_ms=round(crop_timings.get("crop_extraction_ms", 0.0), 3),
            posture_preprocess_ms=round(crop_timings.get("posture_preprocess_ms", 0.0), 3),
            posture_inference_ms=round(crop_timings.get("posture_inference_ms", 0.0), 3),
            crop_preprocess_ms=round(crop_timings.get("crop_preprocess_ms", 0.0), 3),
            posture_ms=round(crop_timings.get("posture_ms", 0.0), 3),
            headpose_preprocess_ms=round(crop_timings.get("headpose_preprocess_ms", 0.0), 3),
            headpose_inference_ms=round(crop_timings.get("headpose_inference_ms", 0.0), 3),
            headpose_ms=round(crop_timings.get("headpose_ms", 0.0), 3),
            phone_association_ms=round(phone_assoc_ms, 3),
            phone_ms=round(phone_assoc_ms, 3),
            fusion_ms=round(t_fusion_total * 1000.0, 3),
            event_engine_ms=round(t_event_total * 1000.0, 3),
            event_ms=round(t_event_total * 1000.0, 3),
            risk_aggregation_ms=round(t_risk_total * 1000.0, 3),
            evidence_manager_ms=round(t_evidence_total * 1000.0, 3),
            evidence_ms=round(t_evidence_total * 1000.0, 3),
            serialization_ms=round(ser_ms, 3),
            debug_overlay_ms=round(debug_overlay_ms, 3),
            post_decode_pipeline_ms=round(post_decode_ms, 3),
            total_pipeline_ms=round(post_decode_ms, 3),
            whole_loop_end_to_end_ms=round(whole_loop_ms, 3),
            queue_wait_ms=round(queue_wait_ms, 3),
            processing_ms=round(processing_ms, 3),
            capture_to_result_ms=round(capture_to_result_ms, 3),
            ai_frame_age_ms=round(ai_frame_age_ms, 3),
            active_tracks_count=len(tracks),
            active_events_count=len(self._active_events_map),
            queue_depth=self.ingestion_queue.qsize,
            dropped_frames_count=self.ingestion_queue.dropped_frames_count,
            stale_skipped_count=self.ingestion_queue.stale_skipped_count,
        )

        self.processed_frames_count += 1

        # Update thread-safe latest frame buffer for downstream dashboard streaming
        # Product web stream receives CLEAN physical frame; web frontend renders product overlays
        self._update_latest_frame_buffer(
            frame,
            tracks,
            posture_cues,
            headpose_cues,
            phone_associations,
        )

        return Stage2FrameResult(
            frame_idx=frame_idx,
            timestamp_sec=ts,
            tracks=tracks,
            unified_updates=unified_updates,
            lifecycle_events=lifecycle_events,
            active_events=list(self._active_events_map.values()),
            metrics=metrics,
            annotated_frame=annotated_frame,
        )

    def _update_latest_frame_buffer(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        posture_cues: Dict[int, Any],
        headpose_cues: Dict[int, Any],
        phone_associations: Dict[int, Any],
    ) -> None:
        """Thread-safe frame buffer update for downstream HTTP/MJPEG streaming."""
        try:
            ret, jpeg_buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                tracks_summary = []
                for t in tracks:
                    tid = t.track_id
                    bx = t.bbox.as_int_tuple() if hasattr(t.bbox, "as_int_tuple") else list(t.bbox)
                    pos = posture_cues.get(tid)
                    hp = headpose_cues.get(tid)
                    ph = phone_associations.get(tid)
                    track_events = [e for e in self._active_events_map.values() if e.track_id == tid]
                    max_risk = "LOW"
                    primary_ev = None
                    for e in track_events:
                        if e.risk_level == "HIGH":
                            max_risk = "HIGH"
                            primary_ev = e
                            break
                        elif e.risk_level == "MEDIUM" and max_risk != "HIGH":
                            max_risk = "MEDIUM"
                            primary_ev = e

                    tracks_summary.append({
                        "track_id": tid,
                        "bbox": bx,
                        "posture": pos.predicted_class if (pos and hasattr(pos, "predicted_class") and pos.predicted_class) else "N/A",
                        "posture_conf": round(float(pos.confidence), 2) if (pos and hasattr(pos, "confidence") and pos.confidence is not None) else None,
                        "yaw_deg": round(float(hp.yaw_deg), 1) if (hp and hasattr(hp, "yaw_deg") and hp.yaw_deg is not None) else None,
                        "phone_status": ph.status if ph and hasattr(ph, "status") else "NONE",
                        "active_events": [e.event_type for e in track_events],
                        "primary_event_type": primary_ev.event_type if primary_ev else None,
                        "risk_level": max_risk,
                    })
                with self._frame_lock:
                    self._latest_jpeg_frame = jpeg_buf.tobytes()
                    self._latest_tracks_summary = tracks_summary
        except Exception as e:
            logger.debug(f"Frame buffer update failed: {e}")

    def set_external_display_frame(self, frame: np.ndarray) -> None:
        """Allow orchestrator to feed HUD-rendered frame to browser stream."""
        try:
            ret, jpeg_buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                with self._frame_lock:
                    self._latest_jpeg_frame = jpeg_buf.tobytes()
        except Exception:
            pass

    def _render_debug_overlay(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        posture_cues: Dict[int, Any],
        headpose_cues: Dict[int, Any],
        phone_associations: Dict[int, Any],
        active_events: Dict[str, FusedEvent],
    ) -> np.ndarray:
        """Render informative observable debug annotations (strictly no cheating words)."""
        events_by_track: Dict[int, List[FusedEvent]] = {}
        for ev in active_events.values():
            events_by_track.setdefault(ev.track_id, []).append(ev)

        for track in tracks:
            t_id = track.track_id
            x1, y1, x2, y2 = track.bbox.as_int_tuple()

            # Determine color from active event risk level
            track_evs = events_by_track.get(t_id, [])
            if any(e.risk_level == "HIGH" for e in track_evs):
                color = (0, 0, 240)    # Red for HIGH risk evidence
                thick = 3
            elif any(e.risk_level == "MEDIUM" for e in track_evs):
                color = (0, 165, 255)  # Orange for MEDIUM
                thick = 2
            elif track_evs:
                color = (0, 215, 255)  # Yellow for LOW
                thick = 2
            else:
                color = (0, 220, 0)    # Green for normal observable baseline
                thick = 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

            # Build label lines
            pos = posture_cues.get(t_id)
            hp = headpose_cues.get(t_id)
            phone = phone_associations.get(t_id)

            pos_str = f"Posture: {pos.predicted_class} ({pos.confidence:.2f})" if (pos and pos.predicted_class) else "Posture: N/A"
            hp_str = f"Yaw: {hp.yaw_deg:.1f} deg" if (hp and hp.yaw_deg is not None) else "Yaw: N/A"
            phone_str = f"Phone: {phone.status}" if phone else "Phone: NONE"

            lines = [f"ID {t_id}", pos_str, hp_str, phone_str]
            if track_evs:
                ev_names = [e.event_type for e in track_evs]
                lines.append(f"Events: {', '.join(ev_names)}")

            # Draw background box for text
            start_y = max(20, y1 - 10 - len(lines) * 16)
            for j, line in enumerate(lines):
                ly = start_y + j * 16
                cv2.putText(frame, line, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2)
                cv2.putText(frame, line, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        return frame

    def run_stream(
        self,
        max_frames: Optional[int] = None,
        on_frame_callback: Optional[Callable[[Stage2FrameResult], None]] = None,
    ) -> List[Stage2FrameMetrics]:
        """
        Run integrated pipeline loop on the configured VideoSource using real
        bounded ingestion queue with DROP_STALE_ON_BACKPRESSURE backpressure.
        """
        if not self.source.is_opened():
            if not self.source.open():
                raise RuntimeError(f"Could not open VideoSource: {self.source.source_id}")

        all_metrics: List[Stage2FrameMetrics] = []
        stop_event = threading.Event()
        read_done_event = threading.Event()

        # Producer thread function
        def producer_worker():
            f_count = 0
            while not stop_event.is_set():
                t0_read = time.perf_counter()
                vframe = self.source.read()
                t1_read = time.perf_counter()
                read_duration_ms = (t1_read - t0_read) * 1000.0

                if vframe is None:
                    break

                # Attach source read latency and wall start time
                vframe.extra_metadata = {
                    "source_read_ms": read_duration_ms,
                    "t0_read": t0_read,
                }

                self.ingestion_queue.push(vframe)
                f_count += 1
                if max_frames is not None and f_count >= max_frames:
                    break

            read_done_event.set()

        producer_thread = threading.Thread(target=producer_worker, daemon=True)
        producer_thread.start()

        logger.info(f"Starting Stage 2 pipeline loop on source '{self.source.source_id}' with backpressure...")

        try:
            while True:
                # Consumer pop from bounded queue
                vframe = self.ingestion_queue.pop(timeout=0.2)
                if vframe is None:
                    if read_done_event.is_set() and self.ingestion_queue.is_empty:
                        break
                    continue

                source_read_ms = vframe.extra_metadata.get("source_read_ms", 0.0) if hasattr(vframe, "extra_metadata") and vframe.extra_metadata else 0.0
                t_loop_start = vframe.extra_metadata.get("t0_read", None) if hasattr(vframe, "extra_metadata") and vframe.extra_metadata else None

                res = self.process_frame(vframe, source_read_ms=source_read_ms, t_loop_start=t_loop_start)
                all_metrics.append(res.metrics)

                if on_frame_callback is not None:
                    on_frame_callback(res)

                if max_frames is not None and self.processed_frames_count >= max_frames:
                    stop_event.set()
                    break

        finally:
            stop_event.set()
            producer_thread.join(timeout=1.0)
            self.source.release()
            self.evidence_manager.shutdown()

        logger.info(
            f"Pipeline loop finished. Processed {self.processed_frames_count} frames, "
            f"dropped {self.dropped_frames_count} frames on backpressure."
        )
        return all_metrics

    def reset_runtime_state(self) -> None:
        """Clear all transient runtime state across tracking, buffering, event, and evidence systems.
        
        Models remain loaded and cached. Model weights/checkpoints are NOT reloaded or mutated.
        """
        self.tracker.reset()
        self.temporal_buffer.clear()
        self.crop_scheduler.reset()
        self.event_engine.reset()
        self._track_metadata.clear()
        self._active_events_map.clear()
        self._last_macro_timestamp = -1.0
        self._cached_macro_dets = []
        self.ingestion_queue = BoundedFrameQueue(maxsize=self.max_decode_queue, drop_policy=self.drop_policy)
        self.processed_frames_count = 0
        self._processed_timestamps.clear()
        self._captured_timestamps.clear()
        if self.evidence_manager is not None:
            self.evidence_manager.reset()
        logger.info("Stage2Pipeline runtime state successfully reset.")

    def close_camera_session(self, camera_id: str = "cam_0") -> None:
        """Explicitly close a camera session, evicting its tracks and events."""
        expired = self.temporal_buffer.evict_inactive_tracks(float("inf"), camera_id=camera_id)
        if expired:
            closed = self.event_engine.handle_track_expiration(expired, time.time(), camera_id=camera_id)
            for ev in closed:
                active_ev = self._active_events_map.pop(ev.event_id, None)
                ev_to_close = active_ev or ev
                ev_to_close.status = "closed"
                ev_to_close.lifecycle_status = "closed"
                self._broadcast_event(ev_to_close, "CLOSE")
            for eid in expired:
                self._track_metadata.pop(eid, None)
        self.temporal_buffer.clear(camera_id=camera_id)
        self.event_engine.reset(camera_id=camera_id)
        logger.info(f"Camera session '{camera_id}' closed and cleaned up.")

