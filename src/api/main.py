"""FastAPI application for Exam Suspicious Behavior Detection."""

import asyncio
import logging
import time
from typing import List, Optional, Any
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.api.schemas import (
    CameraInfo,
    CameraCounts,
    RuntimeStreamStatus,
    ConfiguredRates,
    ObservedRates,
    EventResponse,
    EventStatusUpdateRequest,
    HealthResponse,
    SystemStatusResponse,
)
from src.api.websocket import ConnectionManager
from src.api.static_ui import DASHBOARD_HTML
from src.behavior.event_manager import EventManager, SuspiciousEvent

logger = logging.getLogger(__name__)


def create_app(
    event_manager: Optional[EventManager] = None,
    connection_manager: Optional[ConnectionManager] = None,
    camera_id: str = "laptop_webcam_0",
    camera_type: str = "webcam",
    stage2_pipeline: Optional[Any] = None,
    camera_connected: bool = False,
    camera_streaming: bool = False,
    device_present: bool = False,
) -> FastAPI:
    # State stores
    ev_manager = event_manager or EventManager(camera_id=camera_id)
    ws_manager = connection_manager or ConnectionManager()

    # Wire event manager listener to WebSocket broadcaster
    loop: Optional[asyncio.AbstractEventLoop] = None

    def _on_new_event(event: SuspiciousEvent):
        nonlocal loop
        try:
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    ws_manager.broadcast({"type": "NEW_EVENT", "event": event.to_dict()}),
                    loop,
                )
        except Exception as e:
            logger.debug(f"Could not broadcast event to WS: {e}")

    ev_manager.add_listener(_on_new_event)

    # Wire Stage 2 FusedEvent lifecycle (OPEN, UPDATE, CLOSE) to WebSocket and ev_manager
    if stage2_pipeline is not None:
        def _on_stage2_lifecycle(event: Any, action: str):
            nonlocal loop
            # 1. Sync event to ev_manager for HTTP API endpoints
            try:
                if hasattr(event, "event_id"):
                    eid = event.event_id
                    existing = ev_manager.get_event(eid)
                    cur_status = existing.status if existing else "new"
                    cur_notes = existing.reviewer_notes if existing else None
                    snap_path = getattr(event, "evidence_summary", {}).get("snapshot_path") if hasattr(event, "evidence_summary") and isinstance(event.evidence_summary, dict) else None
                    clip_path = getattr(event, "evidence_summary", {}).get("clip_path") if hasattr(event, "evidence_summary") and isinstance(event.evidence_summary, dict) else None
                    ts = getattr(event, "last_update_timestamp", time.time())
                    st = getattr(event, "start_timestamp", ts)
                    et = getattr(event, "end_timestamp", ts) or ts
                    
                    susp_ev = SuspiciousEvent(
                        event_id=eid,
                        track_id=event.track_id,
                        camera_id=event.camera_id,
                        timestamp=ts,
                        start_time=st,
                        end_time=et,
                        event_type=event.event_type,
                        risk_level=getattr(event, "risk_level", "LOW"),
                        score=float(getattr(event, "risk_score", 0.0)),
                        evidence=getattr(event, "evidence_summary", {}),
                        snapshot_path=snap_path,
                        clip_path=clip_path,
                        status=cur_status,
                        reviewer_notes=cur_notes,
                        event_origin=getattr(event, "event_origin", "LIVE_OBSERVATION"),
                    )
                    ev_manager._events[eid] = susp_ev
            except Exception as e:
                logger.debug(f"Could not sync Stage 2 event to ev_manager: {e}")

            # 2. Broadcast lifecycle to WebSocket clients
            try:
                if loop is None or not loop.is_running():
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        pass

                payload = {
                    "type": f"EVENT_{action}",
                    "action": action,
                    "event": event.to_dict() if hasattr(event, "to_dict") else str(event),
                }

                if loop and loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        ws_manager.broadcast(payload),
                        loop,
                    )
            except Exception as e:
                logger.debug(f"Could not broadcast Stage 2 event to WS: {e}")

        stage2_pipeline.add_event_listener(_on_stage2_lifecycle)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal loop
        loop = asyncio.get_running_loop()
        logger.info("FastAPI backend started.")
        yield

    app = FastAPI(
        title="Exam Suspicious Behavior Detection API",
        description="Production-oriented suspicious behavior detection and invigilator review backend",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def get_dashboard():
        """Serve embedded invigilator review dashboard."""
        return DASHBOARD_HTML

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(
            status="ok",
            version="0.1.0",
            timestamp=time.time(),
        )

    @app.get("/api/cameras", response_model=List[CameraInfo])
    async def list_cameras():
        is_stream = camera_streaming
        if stage2_pipeline is not None and hasattr(stage2_pipeline, "source"):
            if hasattr(stage2_pipeline.source, "is_opened") and stage2_pipeline.source.is_opened():
                is_stream = True
        is_conn = camera_connected or is_stream
        dev_pres = device_present or is_conn
        status_str = "STREAMING" if is_stream else ("CONNECTED" if is_conn else "NO_PHYSICAL_CAMERA")

        return [
            CameraInfo(
                camera_id=camera_id,
                name=f"Exam Room Camera ({camera_type.upper()})",
                source_type=camera_type,
                configured=True,
                device_present=dev_pres,
                connected=is_conn,
                streaming=is_stream,
                status=status_str,
                configured_capture_fps=30.0,
                configured_resolution="1280x720",
                observed_capture_fps=30.0 if is_stream else None,
                observed_resolution="1280x720" if is_stream else None,
                is_active=is_stream,
                fps=30.0 if is_stream else None,
                resolution="1280x720",
            )
        ]

    @app.get("/api/events", response_model=List[EventResponse])
    async def get_events(
        risk_level: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ):
        events = ev_manager.list_events(risk_level=risk_level, status=status, limit=limit)
        return [EventResponse(**e.to_dict()) for e in events]

    @app.get("/api/events/{event_id}", response_model=EventResponse)
    async def get_event(event_id: str):
        event = ev_manager.get_event(event_id)
        if not event:
            raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found.")
        return EventResponse(**event.to_dict())

    @app.patch("/api/events/{event_id}", response_model=EventResponse)
    async def update_event_status(event_id: str, req: EventStatusUpdateRequest):
        """Update review status (confirmed / dismissed) by human invigilator."""
        success = ev_manager.update_event_status(
            event_id=event_id,
            new_status=req.status,
            reviewer_notes=req.reviewer_notes,
        )
        if not success:
            raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found.")

        updated_event = ev_manager.get_event(event_id)
        # Notify WebSocket clients about status update
        await ws_manager.broadcast({
            "type": "EVENT_STATUS_UPDATED",
            "event_id": event_id,
            "status": req.status,
        })
        return EventResponse(**updated_event.to_dict())

    @app.get("/api/system/status", response_model=SystemStatusResponse)
    async def get_system_status():
        all_events = ev_manager.list_events(limit=1000)
        models_meta = None
        try:
            from src.orchestration.model_registry import ModelRegistry
            reg = ModelRegistry.get_instance()
            if not reg._initialized:
                reg.initialize_models()
            models_meta = reg.get_metadata()
        except Exception:
            pass

        # Pipeline runtime telemetry if stage2_pipeline is wired
        drop_pct = None
        q_depth = 0
        active_st = 0
        is_stream = camera_streaming

        if stage2_pipeline is not None:
            if hasattr(stage2_pipeline, "source") and hasattr(stage2_pipeline.source, "is_opened"):
                if stage2_pipeline.source.is_opened():
                    is_stream = True
            tot = stage2_pipeline.processed_frames_count + stage2_pipeline.dropped_frames_count
            if tot > 0:
                drop_pct = round((stage2_pipeline.dropped_frames_count / tot) * 100.0, 1)
            q_depth = stage2_pipeline.ingestion_queue.qsize
            active_st = len(stage2_pipeline._track_metadata)

        is_conn = camera_connected or is_stream
        dev_pres = device_present or is_conn
        active_streams = 1 if is_stream else 0

        camera_counts = CameraCounts(
            registered=1,
            configured=1,
            connected=1 if is_conn else 0,
            streaming=active_streams,
        )

        runtime_stream = RuntimeStreamStatus(
            active=is_stream,
            source_type=camera_type.upper(),
            device_present=dev_pres,
        )

        configured_rates = ConfiguredRates(
            capture_fps=30.0,
            inference_fps=12.0,
        )

        observed_rates = ObservedRates(
            capture_fps=30.0 if is_stream else None,
            processed_fps=(getattr(stage2_pipeline, "processed_fps", None) if is_stream else None),
            inference_fps=12.0 if is_stream else None,
        )

        import torch
        vram = torch.cuda.memory_allocated(0) / (1024.0 ** 2) if torch.cuda.is_available() else 0.0

        return SystemStatusResponse(
            camera_counts=camera_counts,
            runtime_stream=runtime_stream,
            configured_rates=configured_rates,
            observed_rates=observed_rates,
            active_cameras=active_streams,
            active_students=active_st,
            total_events=len(all_events),
            new_events=sum(1 for e in all_events if e.status == "new"),
            confirmed_events=sum(1 for e in all_events if e.status in ["confirmed", "confirmed_event"]),
            dismissed_events=sum(1 for e in all_events if e.status == "dismissed"),
            effective_fps=30.0 if active_streams > 0 else None,
            inference_fps=12.0 if active_streams > 0 else None,
            head_pose_enabled=True if models_meta and "headpose" in models_meta else False,
            pipeline_version="2.0.0-orchestration",
            model_registry=models_meta,
            drop_percentage=drop_pct,
            queue_depth=q_depth,
            gpu_vram_allocated_mb=round(vram, 1),
            evidence_storage_status="HEALTHY",
            operator_warnings=[],
        )

    @app.get("/api/system/models")
    async def get_system_models():
        """Return provenance registry metadata including SHA-256 for all active models."""
        try:
            from src.orchestration.model_registry import ModelRegistry
            reg = ModelRegistry.get_instance()
            if not reg._initialized:
                reg.initialize_models()
            return reg.get_metadata()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model registry unavailable: {e}")

    @app.websocket("/ws/events")
    async def websocket_events_endpoint(websocket: WebSocket):
        await ws_manager.connect(websocket)
        try:
            while True:
                # Keep alive and receive any client messages/pings
                await websocket.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
        except Exception:
            ws_manager.disconnect(websocket)

    return app


# Default module-level application instance for uvicorn run
app = create_app()
