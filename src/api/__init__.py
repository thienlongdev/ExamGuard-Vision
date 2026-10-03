"""API module."""

from src.api.schemas import (
    HealthResponse,
    CameraInfo,
    EventResponse,
    EventStatusUpdateRequest,
    SystemStatusResponse,
)
from src.api.websocket import ConnectionManager
from src.api.main import create_app, app

__all__ = [
    "HealthResponse",
    "CameraInfo",
    "EventResponse",
    "EventStatusUpdateRequest",
    "SystemStatusResponse",
    "ConnectionManager",
    "create_app",
    "app",
]
