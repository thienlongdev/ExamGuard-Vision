"""FastAPI application for Exam Suspicious Behavior Detection."""

import asyncio
import logging
from pathlib import Path
import re
import time
from typing import List, Optional, Any
import urllib.parse
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, Response
from starlette.staticfiles import StaticFiles

from datetime import datetime
import json

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
    SessionResponse,
    SessionSummary,
    SessionUpdateRequest,
    EvidenceVerifyResponse,
    BackupResponse,
)
from src.api.websocket import ConnectionManager
from src.api.static_ui import load_dashboard_html, DASHBOARD_HTML
from src.behavior.event_manager import EventManager, SuspiciousEvent
from src.persistence.service import PersistenceService
from src.persistence.backup import (
    create_local_backup,
    verify_backup_directory,
    preview_retention_candidates,
)

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
    persistence_service: Optional[PersistenceService] = None,
) -> FastAPI:
    ps = persistence_service or PersistenceService.get_instance()
    if ps.active_session is None:
        try:
            ps.initialize_runtime_session()
        except Exception as e:
            logger.warning(f"Could not auto-initialize runtime session: {e}")

    def _sanitize_evidence_path(raw_path: Optional[str]) -> Optional[str]:
        if not raw_path:
            return None
        raw_str = str(raw_path).replace("\\", "/")
        if raw_str.startswith("/api/evidence/"):
            return raw_str
        for marker in ["storage/sessions/", "storage/evidence/", "runs/local_live/", "evidence/"]:
            if marker in raw_str:
                sub = raw_str.split(marker)[-1].lstrip("/")
                if marker == "storage/sessions/":
                    return f"/api/evidence/sessions/{sub}"
                return f"/api/evidence/{sub}"
        if "sessions/" in raw_str:
            sub = "sessions/" + raw_str.split("sessions/")[-1]
            return f"/api/evidence/{sub}"
        # If path already contains session subfolder structure (e.g. asus_a17_demo/snapshots/...)
        parts = [p for p in raw_str.strip("/").split("/") if p]
        if len(parts) >= 2 and any(p in ["snapshots", "clips", "metadata"] for p in parts):
            return f"/api/evidence/{'/'.join(parts)}"
        if "snapshots/" in raw_str:
            sub = "snapshots/" + raw_str.split("snapshots/")[-1]
            return f"/api/evidence/{sub}"
        if "clips/" in raw_str:
            sub = "clips/" + raw_str.split("clips/")[-1]
            return f"/api/evidence/{sub}"
        filename = Path(raw_str).name
        return f"/api/evidence/snapshots/{filename}"

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
                    if not snap_path and hasattr(event, "evidence_summary") and isinstance(event.evidence_summary, dict):
                        snap_path = event.evidence_summary.get("open_snapshot_path")
                    if snap_path:
                        snap_path = _sanitize_evidence_path(str(snap_path))
                    clip_path = getattr(event, "evidence_summary", {}).get("clip_path") if hasattr(event, "evidence_summary") and isinstance(event.evidence_summary, dict) else None
                    if clip_path:
                        clip_path = _sanitize_evidence_path(str(clip_path))
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
                        lifecycle_status=getattr(event, "lifecycle_status", "active"),
                        review_status=cur_status if cur_status in ["confirmed", "dismissed"] else getattr(event, "review_status", "awaiting"),
                        observation_snapshot=getattr(event, "observation_snapshot", {}),
                        reviewer_notes=cur_notes,
                        event_origin=getattr(event, "event_origin", "UNKNOWN"),
                    )
                    ev_manager._events[eid] = susp_ev
            except Exception as e:
                logger.debug(f"Could not sync Stage 2 event to ev_manager: {e}")

            # 2. Persist to SQLite via PersistenceService
            try:
                ps.persist_stage2_event(event, action)
            except Exception as e:
                logger.debug(f"Could not persist stage2 event to DB: {e}")

            # 3. Broadcast lifecycle to WebSocket clients
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

        if hasattr(stage2_pipeline, "add_event_listener"):
            stage2_pipeline.add_event_listener(_on_stage2_lifecycle)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal loop
        loop = asyncio.get_running_loop()
        logger.info("FastAPI backend started.")

        async def _heartbeat_loop():
            while True:
                try:
                    await asyncio.sleep(5.0)
                    ps.heartbeat()
                except asyncio.CancelledError:
                    break
                except Exception:
                    pass

        hb_task = asyncio.create_task(_heartbeat_loop())
        try:
            yield
        finally:
            hb_task.cancel()
            try:
                ps.close_active_session(reason="GRACEFUL_STOP")
            except Exception as e:
                logger.debug(f"Error closing session on shutdown: {e}")

    app = FastAPI(
        title="Exam Suspicious Behavior Detection API",
        description="Production-oriented suspicious behavior detection and invigilator review backend",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.event_manager = ev_manager

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def get_dashboard():
        """Serve embedded invigilator review dashboard."""
        return load_dashboard_html()

    @app.get("/dashboard", response_class=HTMLResponse)
    async def get_dashboard_alias():
        """Serve embedded invigilator review dashboard (alias route)."""
        return load_dashboard_html()

    @app.get("/api/cameras/stream")
    @app.get("/api/cameras/{cam_id}/stream")
    async def get_camera_stream(cam_id: Optional[str] = None):
        """MJPEG video stream downstream of active perception pipeline."""
        if stage2_pipeline is None:
            raise HTTPException(status_code=503, detail="Camera pipeline is not initialized.")

        async def frame_generator():
            while True:
                if hasattr(stage2_pipeline, "source") and hasattr(stage2_pipeline.source, "is_opened"):
                    if not stage2_pipeline.source.is_opened():
                        break

                jpeg_data = getattr(stage2_pipeline, "_latest_jpeg_frame", None)
                if jpeg_data is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg_data)).encode() + b"\r\n\r\n" +
                        jpeg_data + b"\r\n"
                    )
                await asyncio.sleep(0.04)

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/cameras/frame")
    @app.get("/api/cameras/{cam_id}/frame")
    async def get_camera_frame(cam_id: Optional[str] = None):
        """Single latest JPEG frame from active pipeline."""
        if stage2_pipeline is None or getattr(stage2_pipeline, "_latest_jpeg_frame", None) is None:
            raise HTTPException(status_code=404, detail="No active camera frame available.")
        return Response(content=stage2_pipeline._latest_jpeg_frame, media_type="image/jpeg")

    @app.get("/api/cameras/tracks")
    @app.get("/api/cameras/{cam_id}/tracks")
    async def get_camera_tracks(cam_id: Optional[str] = None):
        """Latest active track bounding boxes and telemetry for web overlay."""
        if stage2_pipeline is None:
            return []
        return getattr(stage2_pipeline, "_latest_tracks_summary", [])

    @app.get("/api/evidence/verify/{evidence_id}", response_model=EvidenceVerifyResponse)
    async def verify_evidence_endpoint(evidence_id: str):
        """Verify evidence file integrity on disk against database SHA-256."""
        res = ps.verify_evidence_integrity(evidence_id)
        return EvidenceVerifyResponse(**res)

    @app.get("/api/evidence/{file_path:path}")
    async def get_evidence_file(file_path: str):
        """Safely serve snapshot evidence images and video clips with strict path traversal prevention."""
        allowed_roots = [
            Path("storage/sessions").resolve(),
            Path("storage/evidence").resolve(),
            Path("storage").resolve(),
            Path("evidence").resolve(),
            Path("runs/local_live").resolve(),
        ]

        # Multi-pass unquote to defeat single and double URL encodings (e.g. %252e)
        unquoted = file_path
        for _ in range(3):
            unquoted = urllib.parse.unquote(unquoted)

        # Strict path traversal prevention: check for parent directory references
        if ".." in file_path or ".." in unquoted or "%2e" in file_path.lower() or "%2e" in unquoted.lower():
            raise HTTPException(status_code=403, detail="Path traversal forbidden.")

        # Check for Windows drive letters (e.g. C:, D:), UNC paths (\\\\ or //), or absolute root paths
        if re.search(r"^[a-zA-Z]:", file_path) or re.search(r"^[a-zA-Z]:", unquoted):
            raise HTTPException(status_code=403, detail="Drive letters and absolute paths forbidden.")
        if file_path.startswith(("\\\\", "//")) or unquoted.startswith(("\\\\", "//")):
            raise HTTPException(status_code=403, detail="UNC paths forbidden.")

        clean_path = unquoted.lstrip("/\\")
        if not clean_path:
            raise HTTPException(status_code=404, detail="Empty evidence path.")

        target_path: Optional[Path] = None

        # 1. Direct canonical relative path check under allowed roots
        for root in allowed_roots:
            try:
                candidate = (root / clean_path).resolve()
                candidate.relative_to(root)
                if candidate.is_file():
                    target_path = candidate
                    break
            except (ValueError, RuntimeError):
                continue

        # 2. Legacy fallback search by filename (with strict ambiguity detection)
        if target_path is None:
            filename = Path(clean_path).name
            if filename:
                matches: List[Path] = []
                for root in allowed_roots:
                    if root.exists():
                        try:
                            for match in root.glob(f"**/{filename}"):
                                if match.is_file():
                                    try:
                                        resolved = match.resolve()
                                        resolved.relative_to(root)
                                        matches.append(resolved)
                                    except (ValueError, RuntimeError):
                                        continue
                        except Exception as e:
                            logger.warning(f"Error globbing for evidence {filename} under {root}: {e}")

                # Deduplicate identical canonical paths
                unique_matches = list(dict.fromkeys(matches))

                if len(unique_matches) == 1:
                    target_path = unique_matches[0]
                elif len(unique_matches) > 1:
                    logger.warning(
                        f"Ambiguous evidence lookup for filename '{filename}': {len(unique_matches)} matches found across sessions."
                    )
                    raise HTTPException(
                        status_code=409,
                        detail=f"Ambiguous evidence reference '{filename}'. Multiple conflicting evidence records exist.",
                    )

        if target_path is None or not target_path.is_file():
            raise HTTPException(status_code=404, detail="Evidence artifact not found.")

        # Ensure final resolved target_path is strictly within an allowed root
        is_safe = False
        for root in allowed_roots:
            try:
                target_path.resolve().relative_to(root)
                is_safe = True
                break
            except ValueError:
                continue
        if not is_safe:
            raise HTTPException(status_code=403, detail="Evidence artifact outside allowed root.")

        suffix = target_path.suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".mp4": "video/mp4",
            ".json": "application/json",
        }
        if suffix not in media_types:
            raise HTTPException(status_code=403, detail="Forbidden evidence file type.")

        return FileResponse(
            path=str(target_path),
            media_type=media_types[suffix],
            headers={"Cache-Control": "public, max-age=3600"},
        )

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

        obs_cap_fps = getattr(stage2_pipeline, "observed_capture_fps", None) if is_stream else None
        obs_res = None
        if is_stream and stage2_pipeline is not None and hasattr(stage2_pipeline, "source"):
            if hasattr(stage2_pipeline.source, "width") and hasattr(stage2_pipeline.source, "height"):
                obs_res = f"{stage2_pipeline.source.width}x{stage2_pipeline.source.height}"

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
                observed_capture_fps=obs_cap_fps,
                observed_resolution=obs_res,
                is_active=is_stream,
                fps=obs_cap_fps,
                resolution="1280x720",
            )
        ]

    def _sanitize_event_dict(d: dict) -> dict:
        sp = d.get("snapshot_path")
        if sp:
            d["snapshot_path"] = _sanitize_evidence_path(sp)
        cp = d.get("clip_path")
        if cp:
            d["clip_path"] = _sanitize_evidence_path(cp)
        st = str(d.get("status", "")).lower()
        if st in ["confirmed", "confirmed_event", "reviewed"]:
            d["review_status"] = "confirmed"
        elif st in ["dismissed"]:
            d["review_status"] = "dismissed"
        elif "review_status" not in d or not d["review_status"]:
            d["review_status"] = "awaiting"
        if "lifecycle_status" not in d or not d["lifecycle_status"]:
            d["lifecycle_status"] = "closed" if d.get("status") == "closed" else "active"
        return d

    @app.get("/api/events", response_model=List[EventResponse])
    async def get_events(
        risk_level: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ):
        events = ev_manager.list_events(risk_level=risk_level, status=status, limit=limit)
        return [EventResponse(**_sanitize_event_dict(e.to_dict())) for e in events]

    @app.get("/api/events/{event_id}", response_model=EventResponse)
    async def get_event(event_id: str):
        event = ev_manager.get_event(event_id)
        if not event:
            raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found.")
        return EventResponse(**_sanitize_event_dict(event.to_dict()))

    @app.patch("/api/events/{event_id}", response_model=EventResponse)
    async def update_event_status(event_id: str, req: EventStatusUpdateRequest):
        """Update review status (confirmed / dismissed) by human invigilator."""
        norm_decision = (
            "CONFIRMED"
            if req.status.lower() in ["confirmed", "confirmed_event", "reviewed"]
            else "DISMISSED"
        )
        try:
            ps.record_review_decision(
                event_id=event_id,
                decision=norm_decision,
                note=req.reviewer_notes,
            )
        except Exception as e:
            logger.error(f"Failed to persist review decision for {event_id}: {e}")

        success = ev_manager.update_event_status(
            event_id=event_id,
            new_status=req.status,
            reviewer_notes=req.reviewer_notes,
        )

        db_ev = ps.events.get_event(event_id)
        if not success and not db_ev:
            raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found.")

        updated_event = ev_manager.get_event(event_id)
        # Notify WebSocket clients about status update
        await ws_manager.broadcast({
            "type": "EVENT_STATUS_UPDATED",
            "event_id": event_id,
            "status": req.status,
            "reviewer_notes": req.reviewer_notes,
        })
        if updated_event:
            return EventResponse(**_sanitize_event_dict(updated_event.to_dict()))

        if db_ev:
            st_ts = datetime.fromisoformat(db_ev.opened_at).timestamp() if db_ev.opened_at else time.time()
            et_ts = datetime.fromisoformat(db_ev.closed_at).timestamp() if db_ev.closed_at else st_ts
            d = {
                "event_id": db_ev.event_id,
                "track_id": db_ev.track_id or 0,
                "camera_id": db_ev.camera_id,
                "timestamp": st_ts,
                "start_time": st_ts,
                "end_time": et_ts,
                "event_type": db_ev.event_type,
                "risk_level": db_ev.severity,
                "score": db_ev.score,
                "evidence": json.loads(db_ev.evidence_summary_json or "{}"),
                "snapshot_path": _sanitize_evidence_path(json.loads(db_ev.evidence_summary_json or "{}").get("snapshot_path")),
                "clip_path": _sanitize_evidence_path(json.loads(db_ev.evidence_summary_json or "{}").get("clip_path")),
                "status": db_ev.review_status,
                "lifecycle_status": db_ev.lifecycle_status,
                "review_status": db_ev.review_status,
                "observation_snapshot": json.loads(db_ev.observation_snapshot_json or "{}"),
                "reviewer_notes": req.reviewer_notes,
                "event_origin": db_ev.source_origin,
            }
            return EventResponse(**_sanitize_event_dict(d))
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found.")

    # --- SESSION MANAGEMENT & HISTORY ENDPOINTS ---

    @app.get("/api/sessions", response_model=List[SessionResponse])
    async def list_sessions(
        status: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """List historical and active exam monitoring sessions."""
        sessions = ps.sessions.list_sessions(limit=limit, offset=offset, status=status, search=search)
        resp = []
        for s in sessions:
            summary_dict = ps.sessions.get_session_summary(s.session_id)
            resp.append(
                SessionResponse(
                    session_id=s.session_id,
                    name=s.name,
                    room=s.room,
                    invigilator_name=s.invigilator_name,
                    started_at=s.started_at,
                    ended_at=s.ended_at,
                    status=s.status,
                    camera_count=s.camera_count,
                    created_at=s.created_at,
                    updated_at=s.updated_at,
                    last_heartbeat_at=s.last_heartbeat_at,
                    close_reason=s.close_reason,
                    summary=SessionSummary(**summary_dict),
                )
            )
        return resp

    @app.get("/api/sessions/current", response_model=SessionResponse)
    async def get_current_session():
        """Retrieve the currently active monitoring session."""
        active = ps.active_session or ps.sessions.get_active_session()
        if not active:
            raise HTTPException(status_code=404, detail="No active session found.")
        summary_dict = ps.sessions.get_session_summary(active.session_id)
        return SessionResponse(
            session_id=active.session_id,
            name=active.name,
            room=active.room,
            invigilator_name=active.invigilator_name,
            started_at=active.started_at,
            ended_at=active.ended_at,
            status=active.status,
            camera_count=active.camera_count,
            created_at=active.created_at,
            updated_at=active.updated_at,
            last_heartbeat_at=active.last_heartbeat_at,
            close_reason=active.close_reason,
            summary=SessionSummary(**summary_dict),
        )

    @app.get("/api/sessions/{session_id}", response_model=SessionResponse)
    async def get_session_detail(session_id: str):
        """Retrieve details and KPI summary for a specific session."""
        sess = ps.sessions.get_session(session_id)
        if not sess:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        summary_dict = ps.sessions.get_session_summary(session_id)
        return SessionResponse(
            session_id=sess.session_id,
            name=sess.name,
            room=sess.room,
            invigilator_name=sess.invigilator_name,
            started_at=sess.started_at,
            ended_at=sess.ended_at,
            status=sess.status,
            camera_count=sess.camera_count,
            created_at=sess.created_at,
            updated_at=sess.updated_at,
            last_heartbeat_at=sess.last_heartbeat_at,
            close_reason=sess.close_reason,
            summary=SessionSummary(**summary_dict),
        )

    @app.patch("/api/sessions/{session_id}", response_model=SessionResponse)
    async def update_session_meta(session_id: str, req: SessionUpdateRequest):
        """Update session metadata (name, room, invigilator_name)."""
        success = ps.update_session_metadata(
            session_id=session_id,
            name=req.name,
            room=req.room,
            invigilator_name=req.invigilator_name,
        )
        if not success:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        sess = ps.sessions.get_session(session_id)
        summary_dict = ps.sessions.get_session_summary(session_id)
        return SessionResponse(
            session_id=sess.session_id,
            name=sess.name,
            room=sess.room,
            invigilator_name=sess.invigilator_name,
            started_at=sess.started_at,
            ended_at=sess.ended_at,
            status=sess.status,
            camera_count=sess.camera_count,
            created_at=sess.created_at,
            updated_at=sess.updated_at,
            last_heartbeat_at=sess.last_heartbeat_at,
            close_reason=sess.close_reason,
            summary=SessionSummary(**summary_dict),
        )

    @app.get("/api/sessions/{session_id}/events", response_model=List[EventResponse])
    async def get_session_events(
        session_id: str,
        severity: Optional[str] = None,
        review_status: Optional[str] = None,
        limit: int = 500,
    ):
        """Retrieve all events belonging to a historical or active session."""
        persisted_events = ps.events.list_events_by_session(
            session_id=session_id,
            severity=severity,
            review_status=review_status,
            limit=limit,
        )
        results = []
        for pe in persisted_events:
            obs = {}
            if pe.observation_snapshot_json:
                try:
                    obs = json.loads(pe.observation_snapshot_json)
                except Exception:
                    pass
            ev_sum = {}
            if pe.evidence_summary_json:
                try:
                    ev_sum = json.loads(pe.evidence_summary_json)
                except Exception:
                    pass

            latest_rev = ps.reviews.get_latest_review(pe.event_id)
            reviewer_note = latest_rev.note if latest_rev else None

            snap_path = ev_sum.get("snapshot_path") or ev_sum.get("open_snapshot_path")
            clip_path = ev_sum.get("clip_path")
            if not snap_path or not clip_path:
                for evd in ps.evidence.list_evidence_for_event(pe.event_id):
                    if evd.evidence_type == "SNAPSHOT" and not snap_path:
                        snap_path = evd.relative_path
                    elif evd.evidence_type == "VIDEO_CLIP" and not clip_path:
                        clip_path = evd.relative_path

            st_ts = datetime.fromisoformat(pe.opened_at).timestamp() if pe.opened_at else time.time()
            et_ts = datetime.fromisoformat(pe.closed_at).timestamp() if pe.closed_at else st_ts

            d = {
                "event_id": pe.event_id,
                "track_id": pe.track_id or 0,
                "camera_id": pe.camera_id,
                "timestamp": st_ts,
                "start_time": st_ts,
                "end_time": et_ts,
                "event_type": pe.event_type,
                "risk_level": pe.severity,
                "score": pe.score,
                "evidence": ev_sum,
                "snapshot_path": _sanitize_evidence_path(snap_path),
                "clip_path": _sanitize_evidence_path(clip_path),
                "status": pe.review_status,
                "lifecycle_status": pe.lifecycle_status,
                "review_status": pe.review_status,
                "observation_snapshot": obs,
                "reviewer_notes": reviewer_note,
                "event_origin": pe.source_origin,
            }
            results.append(EventResponse(**d))
        return results

    @app.get("/api/sessions/{session_id}/audit")
    async def get_session_audit_logs(session_id: str):
        """Retrieve audit log history for a session."""
        logs = ps.audit.list_logs_for_session(session_id)
        return [l.to_dict() for l in logs]

    @app.post("/api/backup", response_model=BackupResponse)
    async def trigger_backup():
        """Create a complete local backup of ExamGuard database and evidence files."""
        res = create_local_backup(ps)
        return BackupResponse(**res)

    @app.get("/api/backup/status")
    async def get_backup_status():
        """Retrieve the latest backup logs and status."""
        logs = ps.audit.list_all_logs(limit=200)
        backup_logs = [l.to_dict() for l in logs if "BACKUP" in l.action]
        return {
            "last_backup": backup_logs[-1] if backup_logs else None,
            "total_backups": len(backup_logs),
        }

    @app.get("/api/retention/preview")
    async def preview_retention_endpoint():
        """Preview retention cleanup candidates without deletion."""
        return preview_retention_candidates(ps)

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

        obs_proc_fps = getattr(stage2_pipeline, "observed_processed_fps", None) if is_stream else None
        obs_cap_fps = getattr(stage2_pipeline, "observed_capture_fps", None) if is_stream else None
        obs_inf_fps = getattr(stage2_pipeline, "observed_inference_fps", None) if is_stream else None

        observed_rates = ObservedRates(
            capture_fps=obs_cap_fps,
            processed_fps=obs_proc_fps,
            inference_fps=obs_inf_fps,
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
            effective_fps=obs_proc_fps,
            inference_fps=obs_inf_fps,
            head_pose_enabled=True if models_meta and "headpose" in models_meta else False,
            pipeline_version="2.0.0-orchestration",
            model_registry=models_meta,
            drop_percentage=drop_pct,
            queue_depth=q_depth,
            gpu_vram_allocated_mb=round(vram, 1),
            evidence_storage_status="HEALTHY" if ps.db.check_health() else "ERROR",
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
