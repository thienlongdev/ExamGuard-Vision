"""Tests for VideoSource abstraction and factory."""

import pytest
from src.video.factory import create_video_source
from src.video.webcam import WebcamSource
from src.video.video_file import VideoFileSource
from src.video.rtsp import RTSPSource


def test_video_source_factory_webcam():
    src = create_video_source("webcam", 0, "test-webcam", width=1280, height=720, fps=30.0)
    assert isinstance(src, WebcamSource)
    assert src.source_id == "test-webcam"
    assert src.source_type == "webcam"


def test_video_source_factory_video_file():
    src = create_video_source("video", "samples/exam.mp4", "test-video", extra_config={"loop": True})
    assert isinstance(src, VideoFileSource)
    assert src.file_path == "samples/exam.mp4"
    assert src.loop is True


def test_video_source_factory_rtsp():
    src = create_video_source("rtsp", "rtsp://192.168.1.100:554/live", "cctv-1")
    assert isinstance(src, RTSPSource)
    assert src.rtsp_url == "rtsp://192.168.1.100:554/live"


def test_video_source_factory_invalid():
    with pytest.raises(ValueError):
        create_video_source("unsupported_source_type", "input")
