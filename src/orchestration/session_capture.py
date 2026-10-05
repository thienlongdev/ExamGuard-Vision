"""
Session-bound physical capture controller.

Binds webcam ownership and AI frame processing to an ACTIVE MonitoringSession:

    IDLE ──start(session_id)──> ACTIVE ──stop()──> IDLE

While IDLE the physical camera handle is released, the live loop processes no
frames, and the latest preview JPEG / track summary are empty. Models stay
loaded in the pipeline (warm) — only capture and inference on real frames stop.
"""

import logging
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

STATE_IDLE = "IDLE"
STATE_ACTIVE = "ACTIVE"
STATE_STOPPING = "STOPPING"


class SessionCaptureController:
    def __init__(self, source: Any, pipeline: Any, camera_id: str):
        self.source = source
        self.pipeline = pipeline
        self.camera_id = camera_id

        self.state = STATE_IDLE
        self.session_id: Optional[str] = None
        self.open_count = 0
        self.release_count = 0

        # Serializes start/stop transitions
        self._transition_lock = threading.RLock()
        # Held by the live loop for one read+process iteration; stop() takes it to drain in-flight work
        self._frame_lock = threading.Lock()
        self._active_evt = threading.Event()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        return self._active_evt.is_set()

    def is_bound_to(self, session_id: Optional[str]) -> bool:
        """True while the controller owns capture for this session (including during orderly stop)."""
        return bool(session_id) and self.session_id == session_id and self.state != STATE_IDLE

    def wait_until_active(self, timeout: float) -> bool:
        return self._active_evt.wait(timeout)

    def status(self) -> Dict[str, Any]:
        opened = False
        try:
            opened = bool(self.source.is_opened())
        except Exception:
            pass
        return {
            "state": self.state,
            "session_id": self.session_id,
            "camera_id": self.camera_id,
            "source_opened": opened,
            "open_count": self.open_count,
            "release_count": self.release_count,
        }

    @contextmanager
    def frame_slot(self) -> Iterator[bool]:
        """Live-loop guard: yields True only if a frame may be captured and processed now."""
        with self._frame_lock:
            yield self._active_evt.is_set()

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def start(self, session_id: str) -> bool:
        """Open the camera for this session. Idempotent for the session that already owns capture."""
        with self._transition_lock:
            if self.state == STATE_ACTIVE and self.session_id == session_id:
                return True
            if self.state != STATE_IDLE:
                self.stop()

            # Fresh session-scoped runtime state (tracker IDs, temporal buffers, event FSMs)
            self._clear_runtime_state()

            if not self.source.open():
                logger.error(f"Session '{session_id}': could not open camera '{self.camera_id}'.")
                try:
                    self.source.release()
                except Exception:
                    pass
                return False

            self.open_count += 1
            self.session_id = session_id
            self.state = STATE_ACTIVE
            self._active_evt.set()
            logger.info(f"Session '{session_id}': camera '{self.camera_id}' opened, capture + AI active.")
            return True

    def stop(self) -> None:
        """Orderly shutdown: close live events, flush evidence, release camera, clear live state."""
        with self._transition_lock:
            if self.state == STATE_IDLE:
                return
            sid = self.session_id
            self.state = STATE_STOPPING
            # 1. The live loop stops picking up new frames after the current one
            self._active_evt.clear()

            with self._frame_lock:
                # 2. Close active event lifecycles while the session is still ACTIVE so CLOSE persists
                try:
                    if hasattr(self.pipeline, "close_camera_session"):
                        self.pipeline.close_camera_session(self.camera_id)
                except Exception as e:
                    logger.warning(f"Session '{sid}': error closing live events: {e}")

                # 3. Finalize pending evidence clips
                try:
                    em = getattr(self.pipeline, "evidence_manager", None)
                    if em is not None and getattr(em, "clip_recorder", None) is not None:
                        em.clip_recorder.finalize_all_active()
                except Exception as e:
                    logger.warning(f"Session '{sid}': error finalizing evidence clips: {e}")

                # 4. Release the physical camera handle
                try:
                    self.source.release()
                except Exception as e:
                    logger.warning(f"Session '{sid}': error releasing camera: {e}")
                self.release_count += 1

                # 5. Clear preview frame, track summary and session-scoped runtime state
                self._clear_runtime_state()

            self.session_id = None
            self.state = STATE_IDLE
            logger.info(f"Session '{sid}': camera '{self.camera_id}' released, capture + AI stopped.")

    def _clear_runtime_state(self) -> None:
        p = self.pipeline
        try:
            if hasattr(p, "reset_runtime_state"):
                p.reset_runtime_state()
        except Exception as e:
            logger.warning(f"Could not reset pipeline runtime state: {e}")
        lock = getattr(p, "_frame_lock", None)
        try:
            if lock is not None:
                with lock:
                    p._latest_jpeg_frame = None
                    p._latest_tracks_summary = []
            else:
                p._latest_jpeg_frame = None
                p._latest_tracks_summary = []
        except Exception:
            pass
