"""
Multi-Camera Single-Node Orchestration Engine for ExamGuard Vision.
Coordinates concurrent camera sources on a single edge node with:
- Shared ModelRegistry (single GPU perception model set to prevent RTX 3050 VRAM exhaustion)
- Strictly isolated per-camera tracking, temporal buffers, cue states, and evidence buffers
- Canonical scoped identity (camera_id, track_id) with ZERO cross-camera Re-ID
- Independent failure isolation and background auto-reconnect
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import queue
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
import cv2
import numpy as np
import yaml

from src.video.base import VideoFrame, VideoSource
from src.video.factory import create_video_source
from src.orchestration.model_registry import ModelRegistry
from src.orchestration.stage2_pipeline import Stage2Pipeline, BoundedFrameQueue
from src.persistence.models import CameraConfig, SessionCamera
from src.persistence.service import PersistenceService
from src.security.redactor import redact_sensitive_text

logger = logging.getLogger(__name__)

DEFAULT_CAMERAS_CONFIG_PATH = "configs/runtime/cameras.yaml"


@dataclass
class CameraTelemetry:
    camera_id: str
    name: str
    source_type: str
    configured: bool = True
    device_present: bool = True
    connected: bool = False
    streaming: bool = False
    status_display: str = "ĐANG KHỞI ĐỘNG"  # TRỰC TIẾP, MẤT KẾT NỐI, ĐANG KẾT NỐI LẠI
    capture_fps: float = 0.0
    processing_fps: float = 0.0
    queue_depth: int = 0
    drop_rate: float = 0.0
    stale_skipped: int = 0
    ai_frame_age_ms: float = 0.0
    backend_name: str = "AUTO"
    active_tracks: int = 0
    reconnect_attempts: int = 0
    last_frame_at: Optional[float] = None
    room: Optional[str] = "Phòng thi chính"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "source_type": self.source_type,
            "configured": self.configured,
            "device_present": self.device_present,
            "connected": self.connected,
            "streaming": self.streaming,
            "status": "STREAMING" if self.streaming else ("CONNECTED" if self.connected else "DISCONNECTED"),
            "status_display": self.status_display,
            "capture_fps": round(self.capture_fps, 1),
            "processing_fps": round(self.processing_fps, 1),
            "fps": round(self.processing_fps or self.capture_fps, 1),
            "queue_depth": self.queue_depth,
            "drop_rate": round(self.drop_rate, 3),
            "stale_skipped": self.stale_skipped,
            "ai_frame_age_ms": round(self.ai_frame_age_ms, 1),
            "backend_name": self.backend_name,
            "active_tracks": self.active_tracks,
            "reconnect_attempts": self.reconnect_attempts,
            "last_frame_at": self.last_frame_at,
            "room": self.room,
        }


class CameraPipelineWorker:
    """
    Independent worker managing a single camera ingest, inference, and reconnect lifecycle.
    Shares ModelRegistry with other workers to conserve GPU VRAM.
    """

    def __init__(
        self,
        config: CameraConfig,
        shared_registry: ModelRegistry,
        stage2_config_path: str = "configs/stage2_pipeline.yaml",
        on_event_callback: Optional[Callable[[Any, str], None]] = None,
        persistence_service: Optional[PersistenceService] = None,
        on_evidence_callback: Optional[Callable[[str, str, str], None]] = None,
    ):
        self.config = config
        self.camera_id = config.camera_id
        self.name = config.name
        self.shared_registry = shared_registry
        self.stage2_config_path = stage2_config_path
        self.on_event_callback = on_event_callback
        self.persistence_service = persistence_service
        self.on_evidence_callback = on_evidence_callback

        self.telemetry = CameraTelemetry(
            camera_id=self.camera_id,
            name=self.name,
            source_type=self.config.source_type,
            room=self.config.room,
        )

        self._stop_event = threading.Event()
        self._ingest_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        self._reconnect_thread: Optional[threading.Thread] = None

        self.video_source: Optional[VideoSource] = None
        self.pipeline: Optional[Stage2Pipeline] = None

        # Thread-safe preview buffer
        self._latest_jpeg: Optional[bytes] = None
        self._latest_tracks: List[Dict[str, Any]] = []
        self._preview_lock = threading.Lock()

        # Telemetry measurement windows
        import collections
        self._capture_timestamps = collections.deque(maxlen=30)
        self._process_timestamps = collections.deque(maxlen=30)
        self._dropped_count = 0
        self._total_captured = 0

    def _resolve_source_target(self) -> Any:
        """Resolve device index, video path, or secret RTSP URL from environment."""
        stype = self.config.source_type.lower()
        if stype in ["webcam", "camera"]:
            return self.config.device_index if self.config.device_index is not None else 0
        elif stype in ["video_file", "video", "file"]:
            return self.config.source_uri_ref or "tests/fixtures/sample_exam.mp4"
        elif stype in ["rtsp", "cctv"]:
            ref = self.config.source_uri_ref or ""
            # If reference matches an environment variable (e.g. EXAMGUARD_RTSP_CAM01)
            if ref.startswith("env:") or ref.startswith("EXAMGUARD_"):
                env_name = ref.split("env:")[-1]
                url = os.environ.get(env_name, "")
                if not url:
                    logger.warning(f"RTSP environment variable '{env_name}' not found for camera {self.camera_id}")
                return url
            return ref
        return 0

    def initialize_pipeline(self) -> bool:
        """Instantiate isolated Stage2Pipeline instance sharing the ModelRegistry."""
        try:
            target = self._resolve_source_target()
            redacted_target = redact_sensitive_text(str(target))
            logger.info(f"Initializing video source for [{self.camera_id}]: {self.config.source_type} -> {redacted_target}")

            self.video_source = create_video_source(
                source_type=self.config.source_type,
                source=target,
                source_id=self.camera_id,
                width=self.config.resolution_width or 1280,
                height=self.config.resolution_height or 720,
                fps=self.config.target_capture_fps or 30.0,
            )

            # Create dedicated Stage2Pipeline for this camera sharing the singleton ModelRegistry
            self.pipeline = Stage2Pipeline(
                config_path=self.stage2_config_path,
                video_source=self.video_source,
                enable_debug_overlay=True,
                camera_id=self.camera_id,
                model_registry=self.shared_registry,
            )

            if self.on_event_callback:
                self.pipeline.add_event_listener(self.on_event_callback)
            if self.on_evidence_callback and hasattr(self.pipeline, "add_evidence_listener"):
                self.pipeline.add_evidence_listener(self.on_evidence_callback)

            # Check if source opened successfully
            if hasattr(self.video_source, "is_opened") and self.video_source.is_opened():
                self.telemetry.connected = True
                self.telemetry.streaming = True
                self.telemetry.status_display = "TRỰC TIẾP"
                if hasattr(self.video_source, "backend_name"):
                    self.telemetry.backend_name = self.video_source.backend_name
            else:
                self.telemetry.connected = False
                self.telemetry.streaming = False
                self.telemetry.status_display = "MẤT KẾT NỐI"

            return True
        except Exception as e:
            logger.error(f"Failed to initialize pipeline for camera {self.camera_id}: {e}")
            self.telemetry.connected = False
            self.telemetry.streaming = False
            self.telemetry.status_display = "LỖI NGUỒN"
            return False

    def start(self) -> None:
        """Start ingest and processing worker threads."""
        if not self.pipeline:
            if not self.initialize_pipeline():
                self._trigger_reconnect()
                return

        self._stop_event.clear()
        self._ingest_thread = threading.Thread(
            target=self._ingest_loop,
            name=f"Ingest-{self.camera_id}",
            daemon=True,
        )
        self._process_thread = threading.Thread(
            target=self._process_loop,
            name=f"Process-{self.camera_id}",
            daemon=True,
        )

        self._ingest_thread.start()
        self._process_thread.start()
        logger.info(f"Started camera worker threads for [{self.camera_id}].")

    def stop(self) -> None:
        """Stop worker threads and safely release camera handles."""
        self._stop_event.set()

        if self._ingest_thread and self._ingest_thread.is_alive():
            self._ingest_thread.join(timeout=1.5)
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=1.5)
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            self._reconnect_thread.join(timeout=0.5)

        if self.video_source and hasattr(self.video_source, "release"):
            try:
                self.video_source.release()
            except Exception:
                pass

        self.telemetry.streaming = False
        self.telemetry.connected = False
        self.telemetry.status_display = "ĐÃ DỪNG"
        logger.info(f"Stopped camera worker [{self.camera_id}].")

    def _ingest_loop(self) -> None:
        """Ingest loop: read frames from source into bounded queue and monitor connectivity."""
        last_frame_time = time.time()
        while not self._stop_event.is_set():
            if self.video_source is None:
                time.sleep(0.1)
                continue

            try:
                frame = self.video_source.read()
            except Exception as e:
                logger.warning(f"Read exception on camera [{self.camera_id}]: {e}")
                frame = None

            now = time.time()
            if frame is not None:
                self._total_captured += 1
                self._capture_timestamps.append(now)
                self.telemetry.last_frame_at = now
                self.telemetry.connected = True
                self.telemetry.streaming = True
                self.telemetry.status_display = "TRỰC TIẾP"
                last_frame_time = now

                if len(self._capture_timestamps) >= 2:
                    dt = self._capture_timestamps[-1] - self._capture_timestamps[0]
                    self.telemetry.capture_fps = (len(self._capture_timestamps) - 1) / max(0.001, dt)

                # Section 7: Feed rolling evidence buffer immediately from camera capture!
                if self.pipeline and hasattr(self.pipeline, "evidence_manager"):
                    frame._evidence_pushed = True
                    self.pipeline.evidence_manager.push_frame(frame.frame, frame.timestamp)

                dropped = self.pipeline.ingestion_queue.push(frame)
                if dropped is not None:
                    self._dropped_count += 1
                self.telemetry.stale_skipped = getattr(self.pipeline.ingestion_queue, "stale_skipped_count", 0)
                if self._total_captured > 0:
                    self.telemetry.drop_rate = self._dropped_count / self._total_captured
                self.telemetry.queue_depth = self.pipeline.ingestion_queue.qsize
            else:
                # No frame returned: check if source disconnected
                if now - last_frame_time > 3.0:
                    if self.telemetry.streaming:
                        logger.warning(f"Camera [{self.camera_id}] lost frame signal. Transitioning to MẤT KẾT NỐI.")
                        self.telemetry.streaming = False
                        self.telemetry.connected = False
                        self.telemetry.status_display = "MẤT KẾT NỐI"
                        self._trigger_reconnect()
                time.sleep(0.03)

    def _process_loop(self) -> None:
        """Inference & tracking loop: process frames through pipeline and generate preview."""
        last_preview_time = 0.0
        while not self._stop_event.is_set():
            if not self.pipeline:
                time.sleep(0.05)
                continue

            frame = self.pipeline.ingestion_queue.pop(timeout=0.1, drain_stale=True)
            if frame is None:
                continue

            now = time.time()
            self._process_timestamps.append(now)
            if len(self._process_timestamps) >= 2:
                dt = self._process_timestamps[-1] - self._process_timestamps[0]
                self.telemetry.processing_fps = (len(self._process_timestamps) - 1) / max(0.001, dt)

            # Record AI frame age
            if hasattr(frame, "timestamp") and frame.timestamp and frame.timestamp > 0:
                self.telemetry.ai_frame_age_ms = max(0.0, (now - frame.timestamp) * 1000.0)
            elif hasattr(frame, "_ai_frame_age_ms"):
                self.telemetry.ai_frame_age_ms = frame._ai_frame_age_ms

            # Process through pipeline
            try:
                result = self.pipeline.process_frame(frame)
                self.telemetry.active_tracks = len(result.tracks) if result else 0

                # Throttled preview generation (max ~15 FPS for browser efficiency)
                if now - last_preview_time >= 0.065:
                    last_preview_time = now
                    pipe_tracks = []
                    if self.pipeline and hasattr(self.pipeline, "_latest_tracks_summary"):
                        with self.pipeline._frame_lock:
                            pipe_tracks = list(self.pipeline._latest_tracks_summary)
                    elif result and result.tracks:
                        pipe_tracks = [
                            {
                                "camera_id": self.camera_id,
                                "track_id": t.track_id,
                                "bbox": list(t.bbox.as_int_tuple() if hasattr(t.bbox, "as_int_tuple") else t.bbox),
                                "risk_level": "SAFE",
                                "review_severity": "GREEN",
                            }
                            for t in result.tracks
                        ]

                    with self._preview_lock:
                        self._latest_tracks = pipe_tracks
                        if result.annotated_frame is not None:
                            annotated = result.annotated_frame.copy()
                            cv2.putText(
                                annotated,
                                f"CAM: {self.name}",
                                (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 200),
                                2,
                                cv2.LINE_AA,
                            )
                            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                            self._latest_jpeg = buf.tobytes()
            except Exception as e:
                logger.error(f"Error processing frame on camera [{self.camera_id}]: {e}")

    def _trigger_reconnect(self) -> None:
        """Launch background auto-reconnect worker if not already running."""
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            return
        self._reconnect_thread = threading.Thread(
            target=self._reconnect_worker,
            name=f"Reconnect-{self.camera_id}",
            daemon=True,
        )
        self._reconnect_thread.start()

    def _reconnect_worker(self) -> None:
        """Exponential backoff reconnect loop (never stalls other camera feeds)."""
        backoff = 1.0
        max_backoff = 10.0
        while not self._stop_event.is_set() and not self.telemetry.streaming:
            self.telemetry.reconnect_attempts += 1
            self.telemetry.status_display = "ĐANG KẾT NỐI LẠI"
            logger.info(f"Attempting to reconnect [{self.camera_id}] (attempt #{self.telemetry.reconnect_attempts})...")

            if self.video_source and hasattr(self.video_source, "release"):
                try:
                    self.video_source.release()
                except Exception:
                    pass

            time.sleep(backoff)
            if self._stop_event.is_set():
                break

            try:
                target = self._resolve_source_target()
                self.video_source = create_video_source(
                    source_type=self.config.source_type,
                    source=target,
                    source_id=self.camera_id,
                    width=self.config.resolution_width or 1280,
                    height=self.config.resolution_height or 720,
                    fps=self.config.target_capture_fps or 30.0,
                )
                if hasattr(self.video_source, "is_opened") and self.video_source.is_opened():
                    logger.info(f"Camera [{self.camera_id}] successfully reconnected!")
                    if self.pipeline:
                        self.pipeline.source = self.video_source
                        # Reset tracker on reconnect to avoid track ID collision with stale state
                        self.pipeline.tracker.reset()
                    self.telemetry.connected = True
                    self.telemetry.streaming = True
                    self.telemetry.status_display = "TRỰC TIẾP"
                    break
            except Exception as e:
                logger.debug(f"Reconnect attempt failed for [{self.camera_id}]: {e}")

            backoff = min(max_backoff, backoff * 1.5)

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Return latest encoded preview JPEG frame."""
        with self._preview_lock:
            if self._latest_jpeg is not None:
                return self._latest_jpeg
        if self.pipeline and hasattr(self.pipeline, "_latest_jpeg_frame"):
            with self.pipeline._frame_lock:
                return self.pipeline._latest_jpeg_frame
        return None

    def get_latest_tracks(self) -> List[Dict[str, Any]]:
        """Return latest active tracks for this camera."""
        if self.pipeline and hasattr(self.pipeline, "_latest_tracks_summary"):
            with self.pipeline._frame_lock:
                if self.pipeline._latest_tracks_summary:
                    return list(self.pipeline._latest_tracks_summary)
        with self._preview_lock:
            return list(self._latest_tracks)


class CameraManager:
    """
    Central orchestrator managing multi-camera ingest, shared perception models,
    hero camera promotion, and failure resilience.
    """

    _instance: Optional["CameraManager"] = None

    @classmethod
    def get_instance(
        cls,
        config_path: str = DEFAULT_CAMERAS_CONFIG_PATH,
        persistence_service: Optional[PersistenceService] = None,
    ) -> "CameraManager":
        if cls._instance is None:
            cls._instance = cls(config_path=config_path, persistence_service=persistence_service)
        return cls._instance

    def __init__(
        self,
        config_path: str = DEFAULT_CAMERAS_CONFIG_PATH,
        persistence_service: Optional[PersistenceService] = None,
    ):
        self.config_path = config_path
        self.ps = persistence_service or PersistenceService.get_instance()
        self.repo_root = Path(__file__).resolve().parent.parent.parent

        # 1. Shared ModelRegistry (Singleton across all cameras)
        self.shared_registry = ModelRegistry.get_instance()
        self.shared_registry.initialize_models()

        self.workers: Dict[str, CameraPipelineWorker] = {}
        self.hero_camera_id: str = "cam01"
        self._lock = threading.RLock()
        self._event_listeners: List[Callable[[Any, str], None]] = []
        self._evidence_listeners: List[Callable[[str, str, str], None]] = []

        self._load_and_sync_configuration()

    def _load_and_sync_configuration(self) -> None:
        """Load cameras from YAML config or database, syncing them into cameras table."""
        full_cfg_path = self.repo_root / self.config_path
        yaml_cams: List[Dict[str, Any]] = []

        if full_cfg_path.is_file():
            try:
                with open(full_cfg_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    yaml_cams = data.get("cameras", [])
                    if data.get("default_camera_id"):
                        self.hero_camera_id = data["default_camera_id"]
            except Exception as e:
                logger.warning(f"Could not load cameras YAML from {full_cfg_path}: {e}")

        # If no cameras defined in YAML, provide default webcam
        if not yaml_cams:
            yaml_cams = [{
                "camera_id": "cam01",
                "name": "Camera phòng thi chính",
                "source_type": "webcam",
                "device_index": 0,
                "enabled": True,
                "resolution_width": 1280,
                "resolution_height": 720,
                "target_capture_fps": 30.0,
                "room": "Phòng thi chính",
            }]

        # Sync to DB
        for c in yaml_cams:
            cam_obj = CameraConfig(
                camera_id=c.get("camera_id", "cam01"),
                name=c.get("name", "Camera"),
                source_type=c.get("source_type", c.get("type", "webcam")),
                device_index=c.get("device_index", 0),
                source_uri_ref=c.get("source_uri_ref", c.get("uri_env")),
                enabled=1 if c.get("enabled", True) else 0,
                resolution_width=c.get("resolution_width", 1280),
                resolution_height=c.get("resolution_height", 720),
                target_capture_fps=float(c.get("target_capture_fps", c.get("target_fps", 30.0))),
                room=c.get("room", "Phòng thi chính"),
            )
            self.ps.cameras.add_or_update_camera(cam_obj)

    def register_event_listener(self, listener: Callable[[Any, str], None]) -> None:
        """Register global listener for lifecycle events across all cameras."""
        self._event_listeners.append(listener)
        for worker in self.workers.values():
            if worker.pipeline:
                worker.pipeline.add_event_listener(listener)

    def register_evidence_listener(self, listener: Callable[[str, str, str], None]) -> None:
        """Register global listener for evidence readiness across all cameras."""
        self._evidence_listeners.append(listener)
        for worker in self.workers.values():
            if worker.pipeline and hasattr(worker.pipeline, "add_evidence_listener"):
                worker.pipeline.add_evidence_listener(listener)

    def add_evidence_listener(self, listener: Callable[[str, str, str], None]) -> None:
        self.register_evidence_listener(listener)

    def _on_worker_event(self, event: Any, action: str) -> None:
        """Relay event from any camera worker to all registered listeners."""
        for cb in self._event_listeners:
            try:
                cb(event, action)
            except Exception as e:
                logger.error(f"Error in camera manager event listener: {e}")

    def _on_worker_evidence(self, event_id: str, evidence_type: str, file_path: str) -> None:
        """Relay evidence readiness from any camera worker to all registered listeners."""
        for cb in self._evidence_listeners:
            try:
                cb(event_id, evidence_type, file_path)
            except Exception as e:
                logger.error(f"Error in camera manager evidence listener: {e}")

    def reset(self) -> None:
        """Reset pipelines on all camera workers for new monitoring session (Workstream 10)."""
        with self._lock:
            for worker in self._workers.values():
                if worker.pipeline and hasattr(worker.pipeline, "reset"):
                    try:
                        worker.pipeline.reset()
                    except Exception as e:
                        logger.debug(f"Error resetting worker pipeline {worker.camera_id}: {e}")
            logger.info("CameraManager workers reset for new monitoring session.")

    def finalize_all_active(self) -> None:
        """Finalize all pending evidence recordings across workers (e.g. before session close)."""
        with self._lock:
            for worker in self._workers.values():
                if worker.pipeline and hasattr(worker.pipeline, "evidence_manager") and worker.pipeline.evidence_manager:
                    try:
                        if worker.pipeline.evidence_manager.clip_recorder:
                            worker.pipeline.evidence_manager.clip_recorder.finalize_all_active()
                    except Exception as e:
                        logger.debug(f"Error finalizing recordings for {worker.camera_id}: {e}")

    def start_all_enabled_cameras(self) -> int:
        """Start workers for all enabled cameras in database."""
        with self._lock:
            enabled_cams = self.ps.cameras.list_enabled_cameras()
            logger.info(f"Starting {len(enabled_cams)} enabled cameras on single node...")

            started = 0
            for cam_cfg in enabled_cams:
                cid = cam_cfg.camera_id
                if cid in self.workers:
                    continue

                worker = CameraPipelineWorker(
                    config=cam_cfg,
                    shared_registry=self.shared_registry,
                    on_event_callback=self._on_worker_event,
                    persistence_service=self.ps,
                    on_evidence_callback=self._on_worker_evidence,
                )
                self.workers[cid] = worker
                worker.start()
                started += 1

                # Link to active session if exists
                if self.ps.active_session:
                    self.ps.cameras.attach_camera_to_session(self.ps.active_session.session_id, cid)

            if enabled_cams and (self.hero_camera_id not in self.workers):
                self.hero_camera_id = enabled_cams[0].camera_id

            return started

    def stop_all_cameras(self) -> None:
        """Stop all camera workers and detach from active session."""
        with self._lock:
            sid = self.ps.active_session.session_id if self.ps.active_session else None
            for cid, worker in self.workers.items():
                worker.stop()
                if sid:
                    self.ps.cameras.detach_camera_from_session(sid, cid, status="ENDED")
            self.workers.clear()
            logger.info("All camera workers stopped.")

    def restart_camera(self, camera_id: str) -> bool:
        """Restart a single camera worker without disturbing other cameras."""
        with self._lock:
            worker = self.workers.get(camera_id)
            if worker:
                worker.stop()
                cam_cfg = self.ps.cameras.get_camera(camera_id)
                if cam_cfg and cam_cfg.enabled:
                    worker.config = cam_cfg
                    worker.start()
                    return True
                else:
                    self.workers.pop(camera_id, None)
                    return True
            else:
                cam_cfg = self.ps.cameras.get_camera(camera_id)
                if cam_cfg and cam_cfg.enabled:
                    new_worker = CameraPipelineWorker(
                        config=cam_cfg,
                        shared_registry=self.shared_registry,
                        on_event_callback=self._on_worker_event,
                        persistence_service=self.ps,
                        on_evidence_callback=self._on_worker_evidence,
                    )
                    self.workers[camera_id] = new_worker
                    new_worker.start()
                    return True
            return False

    def set_hero_camera(self, camera_id: str) -> bool:
        """Set active focused hero camera."""
        if camera_id in self.workers or self.ps.cameras.get_camera(camera_id):
            self.hero_camera_id = camera_id
            return True
        return False

    def get_telemetry_list(self) -> List[Dict[str, Any]]:
        """Return status telemetry for all configured cameras."""
        with self._lock:
            all_cfgs = self.ps.cameras.list_cameras()
            results = []
            for cfg in all_cfgs:
                worker = self.workers.get(cfg.camera_id)
                if worker:
                    t_dict = worker.telemetry.to_dict()
                    t_dict["enabled"] = bool(cfg.enabled)
                    t_dict["is_hero"] = (cfg.camera_id == self.hero_camera_id)
                    results.append(t_dict)
                else:
                    results.append({
                        "camera_id": cfg.camera_id,
                        "name": cfg.name,
                        "source_type": cfg.source_type,
                        "configured": True,
                        "device_present": False,
                        "connected": False,
                        "streaming": False,
                        "status": "DISABLED" if not cfg.enabled else "OFFLINE",
                        "status_display": "CHƯA KÍCH HOẠT" if not cfg.enabled else "NGOẠI TUYẾN",
                        "capture_fps": 0.0,
                        "processing_fps": 0.0,
                        "fps": 0.0,
                        "queue_depth": 0,
                        "drop_rate": 0.0,
                        "active_tracks": 0,
                        "reconnect_attempts": 0,
                        "room": cfg.room,
                        "enabled": bool(cfg.enabled),
                        "is_hero": (cfg.camera_id == self.hero_camera_id),
                    })
            return results

    def resolve_canonical_camera_id(self, camera_id: Optional[str] = None) -> str:
        """Resolve any user/alias camera ID (e.g. CAM 01, webcam_0, camera_0) to canonical worker ID."""
        if not camera_id:
            return self.hero_camera_id
        if camera_id in self.workers:
            return camera_id
        norm = camera_id.lower().replace(" ", "").replace("_", "").replace("-", "")
        for cid in self.workers:
            if cid.lower().replace(" ", "").replace("_", "").replace("-", "") == norm:
                return cid
        for cfg in self.ps.cameras.list_cameras():
            cid = cfg.camera_id
            if cid.lower().replace(" ", "").replace("_", "").replace("-", "") == norm:
                return cid
        return self.hero_camera_id

    def get_all_statuses(self) -> List[Dict[str, Any]]:
        """Return status telemetry for all configured cameras (endpoint compatibility)."""
        return self.get_telemetry_list()

    def get_preview_jpeg(self, camera_id: Optional[str] = None) -> Optional[bytes]:
        """Return latest preview frame for requested camera (or hero camera) with fallback."""
        cid = self.resolve_canonical_camera_id(camera_id)
        worker = self.workers.get(cid)
        if worker:
            frame = worker.get_latest_jpeg()
            if frame is not None:
                return frame
        # Fallback to any active worker that has a frame
        for w in self.workers.values():
            buf = w.get_latest_jpeg()
            if buf is not None:
                return buf
        return None

    def get_camera_frame(self, camera_id: Optional[str] = None) -> Optional[bytes]:
        """Alias for get_preview_jpeg matching main.py endpoint calls."""
        return self.get_preview_jpeg(camera_id)

    def get_tracks_summary(self, camera_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return latest tracks summary for requested camera."""
        cid = self.resolve_canonical_camera_id(camera_id)
        worker = self.workers.get(cid)
        if worker:
            return worker.get_latest_tracks()
        for w in self.workers.values():
            t = w.get_latest_tracks()
            if t:
                return t
        return []

    def get_camera_tracks(self, camera_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Alias for get_tracks_summary matching main.py endpoint calls."""
        return self.get_tracks_summary(camera_id)
