"""Factory functions to instantiate VideoSource instances from config or CLI args."""

import logging
from typing import Any, Dict, Optional, Union

from src.video.base import VideoSource
from src.video.webcam import WebcamSource
from src.video.video_file import VideoFileSource
from src.video.rtsp import RTSPSource

logger = logging.getLogger(__name__)


def create_video_source(
    source_type: str,
    source: Union[int, str],
    source_id: Optional[str] = None,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    extra_config: Optional[Dict[str, Any]] = None,
) -> VideoSource:
    """Create a VideoSource based on source type.

    Args:
        source_type: One of 'webcam', 'video', 'video_file', or 'rtsp'.
        source: Device index (e.g. 0), file path, or RTSP URL.
        source_id: Logical camera identifier.
        width: Desired frame width.
        height: Desired frame height.
        fps: Desired frame rate.
        extra_config: Additional parameters (e.g. loop, reconnect attempts).

    Returns:
        Instantiated VideoSource.
    """
    st = source_type.lower().strip()
    sid = source_id or f"{st}-{source}"
    extra = extra_config or {}

    if st in ("webcam", "camera"):
        dev_idx = int(source) if isinstance(source, (int, str)) and str(source).isdigit() else 0
        return WebcamSource(
            source=dev_idx,
            source_id=sid,
            width=width,
            height=height,
            fps=fps,
        )

    elif st in ("video", "video_file", "file"):
        loop = bool(extra.get("loop", False))
        realtime_pace = bool(extra.get("realtime_pace", True))
        return VideoFileSource(
            file_path=str(source),
            source_id=sid,
            loop=loop,
            realtime_pace=realtime_pace,
        )

    elif st in ("rtsp", "cctv", "ip_camera"):
        reconnect_attempts = int(extra.get("reconnect_attempts", 10))
        reconnect_delay = float(extra.get("reconnect_delay_seconds", 1.5))
        return RTSPSource(
            rtsp_url=str(source),
            source_id=sid,
            max_reconnect_attempts=reconnect_attempts,
            initial_reconnect_delay=reconnect_delay,
            expected_fps=fps,
        )

    else:
        raise ValueError(f"Unsupported video source type: '{source_type}'. Choose 'webcam', 'video', or 'rtsp'.")
