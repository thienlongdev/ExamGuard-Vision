"""
Rolling buffer and asynchronous clip recorder for incident video evidence.
Enforces:
- Memory-bounded buffer (stores compressed JPEG bytes with timestamps, max MB limit)
- Non-blocking asynchronous MP4 clip encoding
- Atomic file write (.tmp -> final)
- SHA-256 hash calculation and completion callback
- Graceful error isolation (encoding failure never crashes perception)
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import inspect
import logging
import os
import threading
import time
from typing import Deque, List, Optional, Callable, Dict, Any, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BufferedFrame:
    jpeg_bytes: bytes
    timestamp: float
    width: int
    height: int


@dataclass
class CodecCapability:
    backend_api: int
    fourcc_str: str
    extension: str
    mime_type: str
    container: str
    is_valid: bool = False


_PROBED_CODEC: Optional[CodecCapability] = None

# A clip below these bounds is not reviewable evidence and must never be marked READY.
MIN_CLIP_FRAMES = 8
MIN_CLIP_DURATION_SEC = 1.0


def select_frames_for_fps(frames: List["BufferedFrame"], fps: float) -> List["BufferedFrame"]:
    """Subsample frames so consecutive frames are at least 1/fps apart (keeps real-time playback speed)."""
    if fps <= 0 or len(frames) < 2:
        return list(frames)
    min_gap = (1.0 / fps) * 0.9
    selected = [frames[0]]
    for bf in frames[1:]:
        if bf.timestamp - selected[-1].timestamp >= min_gap:
            selected.append(bf)
    return selected


def probe_video_codec() -> CodecCapability:
    """
    Startup capability probe.
    Tests candidate encoders with a short synthetic clip to ensure browser-compatible
    video output (duration > 0, frame count > 0, non-empty, parseable container).
    Caches result.
    """
    global _PROBED_CODEC
    if _PROBED_CODEC is not None:
        return _PROBED_CODEC

    import tempfile
    candidates: List[Tuple[int, str, str, str, str]] = []
    if os.name == "nt":
        # Windows: cv2.CAP_MSMF with H264 produces AVC1 in MP4 that Chromium plays natively
        candidates.append((cv2.CAP_MSMF, "H264", ".mp4", "video/mp4", "MP4"))
        candidates.append((cv2.CAP_FFMPEG, "avc1", ".mp4", "video/mp4", "MP4"))
        candidates.append((cv2.CAP_FFMPEG, "H264", ".mp4", "video/mp4", "MP4"))
        candidates.append((cv2.CAP_FFMPEG, "VP80", ".webm", "video/webm", "WEBM"))
        candidates.append((0, "mp4v", ".mp4", "video/mp4", "MP4"))
    else:
        # Linux/macOS
        candidates.append((cv2.CAP_FFMPEG, "avc1", ".mp4", "video/mp4", "MP4"))
        candidates.append((cv2.CAP_FFMPEG, "H264", ".mp4", "video/mp4", "MP4"))
        candidates.append((cv2.CAP_FFMPEG, "VP80", ".webm", "video/webm", "WEBM"))
        candidates.append((0, "mp4v", ".mp4", "video/mp4", "MP4"))

    for backend, fourcc_str, ext, mime, container in candidates:
        temp_dir = tempfile.gettempdir()
        test_file = os.path.join(temp_dir, f"eg_probe_{fourcc_str}_{int(time.time())}{ext}")
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            w, h = 320, 240
            if backend != 0:
                writer = cv2.VideoWriter(test_file, backend, fourcc, 15.0, (w, h))
            else:
                writer = cv2.VideoWriter(test_file, fourcc, 15.0, (w, h))

            if not writer.isOpened():
                continue

            dummy = np.zeros((h, w, 3), dtype=np.uint8)
            for _ in range(5):
                writer.write(dummy)
            writer.release()

            if not os.path.exists(test_file) or os.path.getsize(test_file) < 256:
                if os.path.exists(test_file):
                    os.remove(test_file)
                continue

            cap = cv2.VideoCapture(test_file)
            valid = False
            if cap.isOpened():
                cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if cnt > 0:
                    valid = True
                cap.release()

            if os.path.exists(test_file):
                os.remove(test_file)

            if valid:
                logger.info(f"Evidence video codec probed successfully: backend={backend}, fourcc={fourcc_str}, ext={ext}, mime={mime}")
                _PROBED_CODEC = CodecCapability(
                    backend_api=backend,
                    fourcc_str=fourcc_str,
                    extension=ext,
                    mime_type=mime,
                    container=container,
                    is_valid=True,
                )
                return _PROBED_CODEC
        except Exception as e:
            if os.path.exists(test_file):
                try:
                    os.remove(test_file)
                except Exception:
                    pass
            logger.debug(f"Codec probe failed for {fourcc_str}: {e}")

    logger.warning("All candidate video encoders failed probe. Using fallback MP4V.")
    _PROBED_CODEC = CodecCapability(
        backend_api=0,
        fourcc_str="mp4v",
        extension=".mp4",
        mime_type="video/mp4",
        container="MP4",
        is_valid=False,
    )
    return _PROBED_CODEC


class RollingClipRecorder:
    """Maintains a memory-bounded circular pre-event buffer and writes verified video clips."""

    def __init__(
        self,
        output_dir: str = "storage/evidence/clips",
        pre_event_seconds: float = 5.0,
        post_event_seconds: float = 5.0,
        max_fps: float = 15.0,
        jpeg_quality: int = 80,
        max_buffer_mb: float = 64.0,
        on_clip_ready: Optional[Callable[..., None]] = None,
        on_clip_failed: Optional[Callable[[str, str, Exception], None]] = None,
    ):
        self.output_dir = output_dir
        self.pre_event_seconds = pre_event_seconds
        self.post_event_seconds = post_event_seconds
        self.max_fps = max_fps
        self.jpeg_quality = jpeg_quality
        self.max_buffer_bytes = int(max_buffer_mb * 1024 * 1024)
        self.on_clip_ready = on_clip_ready
        self.on_clip_failed = on_clip_failed

        # Maximum frames in rolling buffer (e.g. 5s * 30fps = 150 frames + margin)
        self.buffer_maxlen = int(pre_event_seconds * 30.0) + 30
        self._rolling_buffer: Deque[BufferedFrame] = deque(maxlen=self.buffer_maxlen)
        self._current_buffer_bytes = 0

        # Active recording tasks awaiting post-event frames:
        self._active_recordings: List[Dict[str, Any]] = []
        # Capture thread pushes frames while the pipeline thread triggers clips.
        self._lock = threading.Lock()

        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="clip_worker")
        os.makedirs(self.output_dir, exist_ok=True)
        # Warmup codec probe
        probe_video_codec()

    def push_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """Compress frame to JPEG bytes, add to rolling buffer, and feed active clip jobs."""
        h, w = frame.shape[:2]
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not success:
            return

        jpeg_data = encoded.tobytes()
        buf_frame = BufferedFrame(
            jpeg_bytes=jpeg_data,
            timestamp=timestamp,
            width=w,
            height=h,
        )

        completed_jobs = []
        with self._lock:
            # Enforce memory safety bound
            while self._current_buffer_bytes + len(jpeg_data) > self.max_buffer_bytes and self._rolling_buffer:
                evicted = self._rolling_buffer.popleft()
                self._current_buffer_bytes -= len(evicted.jpeg_bytes)

            if len(self._rolling_buffer) == self._rolling_buffer.maxlen and self._rolling_buffer:
                self._current_buffer_bytes -= len(self._rolling_buffer[0].jpeg_bytes)
            self._rolling_buffer.append(buf_frame)
            self._current_buffer_bytes += len(jpeg_data)

            # Append to active clip jobs
            for job in self._active_recordings:
                job["frames"].append(buf_frame)
                if timestamp >= job["end_timestamp"]:
                    completed_jobs.append(job)
            for job in completed_jobs:
                self._active_recordings.remove(job)

        # Dispatch completed jobs to background encoding
        for job in completed_jobs:
            self._dispatch_encode(
                job["event_id"],
                job["filepath"],
                job["frames"],
                job["fps"],
            )

    def trigger_clip(
        self,
        event_id: str,
        timestamp: float,
        fps: float = 15.0,
        custom_filepath: Optional[str] = None,
    ) -> str:
        """Trigger evidence clip capture centered around current event."""
        codec_info = probe_video_codec()
        if custom_filepath:
            filepath = custom_filepath
        else:
            filename = f"{event_id}_evidence{codec_info.extension}"
            filepath = os.path.join(self.output_dir, filename)

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        clip_fps = min(fps, self.max_fps)
        end_time = timestamp + self.post_event_seconds
        cutoff = timestamp - self.pre_event_seconds

        with self._lock:
            # One event = one clip: an already-recording event is never re-triggered
            for existing in self._active_recordings:
                if existing["event_id"] == event_id:
                    return existing["filepath"]

            # Collect pre-event frames from rolling buffer
            pre_frames = [bf for bf in self._rolling_buffer if bf.timestamp >= cutoff]
            job = {
                "event_id": event_id,
                "filepath": filepath,
                "frames": pre_frames,
                "end_timestamp": end_time,
                "fps": clip_fps,
            }
            self._active_recordings.append(job)
        return filepath

    def finalize_all_active(self) -> None:
        """Immediately dispatch any remaining active recordings (e.g. on stream end or shutdown)."""
        with self._lock:
            pending = list(self._active_recordings)
            self._active_recordings.clear()
        for job in pending:
            if job["frames"]:
                self._dispatch_encode(
                    job["event_id"],
                    job["filepath"],
                    job["frames"],
                    job["fps"],
                )

    def _dispatch_encode(
        self,
        event_id: str,
        filepath: str,
        frames: List[BufferedFrame],
        fps: float,
    ) -> None:
        """Encode frames into a verified browser-compatible video file in a background worker."""
        if not frames:
            if self.on_clip_failed:
                self.on_clip_failed(event_id, filepath, RuntimeError("No frames available for video clip"))
            return

        frames = select_frames_for_fps(frames, fps)

        def _encode():
            codec_info = probe_video_codec()
            tmp_path = f"{filepath}.tmp{codec_info.extension}"
            try:
                w, h = frames[0].width, frames[0].height
                writer = None
                fourcc = cv2.VideoWriter_fourcc(*codec_info.fourcc_str)

                if codec_info.backend_api != 0:
                    writer = cv2.VideoWriter(tmp_path, codec_info.backend_api, fourcc, fps, (w, h))

                if writer is None or not writer.isOpened():
                    # Fallback chain
                    if os.name == "nt":
                        writer = cv2.VideoWriter(tmp_path, cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"H264"), fps, (w, h))
                    if not writer or not writer.isOpened():
                        writer = cv2.VideoWriter(tmp_path, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
                    if not writer or not writer.isOpened():
                        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

                if not writer or not writer.isOpened():
                    raise RuntimeError(f"Cannot initialize VideoWriter for {tmp_path}")

                written_count = 0
                for bf in frames:
                    arr = np.frombuffer(bf.jpeg_bytes, dtype=np.uint8)
                    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if decoded is not None:
                        writer.write(decoded)
                        written_count += 1
                writer.release()

                # Strict post-encode validation (Workstreams 17, 21)
                if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 512:
                    raise RuntimeError(f"Encoded clip {tmp_path} is missing or under 512 bytes")

                cap = cv2.VideoCapture(tmp_path)
                if not cap.isOpened():
                    raise RuntimeError(f"Encoded clip {tmp_path} cannot be opened by VideoCapture")

                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_fps = cap.get(cv2.CAP_PROP_FPS) or fps
                decodable = 0
                while decodable < MIN_CLIP_FRAMES and cap.read()[0]:
                    decodable += 1
                cap.release()

                # READY requires a finalized container with a real, playable duration.
                if n_frames < MIN_CLIP_FRAMES or written_count < MIN_CLIP_FRAMES or decodable < MIN_CLIP_FRAMES:
                    raise RuntimeError(
                        f"Encoded clip {tmp_path} too short: {n_frames} container frames, "
                        f"{written_count} written (min {MIN_CLIP_FRAMES})"
                    )

                # Playable duration as the browser will see it (frames / container fps)
                duration_sec = round(n_frames / max(1.0, cap_fps), 2)
                if duration_sec < MIN_CLIP_DURATION_SEC:
                    raise RuntimeError(
                        f"Encoded clip {tmp_path} duration {duration_sec}s below minimum {MIN_CLIP_DURATION_SEC}s"
                    )

                # Atomic rename
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass
                os.replace(tmp_path, filepath)

                # Compute SHA-256
                hasher = hashlib.sha256()
                with open(filepath, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
                clip_hash = hasher.hexdigest()
                clip_size = os.path.getsize(filepath)

                logger.info(
                    f"Saved verified video evidence clip: {filepath} ({written_count} frames, "
                    f"{duration_sec}s, {clip_size} bytes, codec={codec_info.fourcc_str})"
                )
                if self.on_clip_ready:
                    sig = inspect.signature(self.on_clip_ready)
                    if len(sig.parameters) >= 8:
                        self.on_clip_ready(
                            event_id,
                            filepath,
                            clip_hash,
                            clip_size,
                            duration_sec,
                            written_count,
                            codec_info.fourcc_str,
                            codec_info.mime_type,
                        )
                    else:
                        try:
                            self.on_clip_ready(
                                event_id,
                                filepath,
                                clip_hash,
                                clip_size,
                                duration_sec,
                                written_count,
                                codec_info.fourcc_str,
                                codec_info.mime_type,
                            )
                        except TypeError:
                            self.on_clip_ready(event_id, filepath, clip_hash, clip_size)

            except Exception as e:
                logger.error(f"Error encoding video clip {filepath}: {e}")
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                if self.on_clip_failed:
                    self.on_clip_failed(event_id, filepath, e)

        self._executor.submit(_encode)

    def shutdown(self) -> None:
        """Gracefully finalize pending recordings and wait for writes to complete."""
        self.finalize_all_active()
        self._executor.shutdown(wait=True)
