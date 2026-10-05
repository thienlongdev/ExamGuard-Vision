"""FastAPI application for Exam Suspicious Behavior Detection."""

import asyncio
from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import List, Optional, Any, Tuple
import urllib.parse

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, RedirectResponse
from starlette.staticfiles import StaticFiles

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
    SessionPageResponse,
    SessionSummary,
    SessionStartRequest,
    SessionEndRequest,
    SessionUpdateRequest,
    EvidenceVerifyResponse,
    BackupResponse,
    LoginRequest,
    SetupRequest,
    UserResponse,
    AuthMeResponse,
    UserCreateRequest,
    UserUpdateRequest,
    UserResetPasswordRequest,
    CameraConfigItem,
    CameraConfigUpdate,
    BackupCreateRequest,
    BackupVerifyRequest,
    SecurityStatusResponse,
)
from src.api.websocket import ConnectionManager
from src.api.static_ui import (
    load_dashboard_html,
    load_login_html,
    load_setup_html,
    load_recovery_html,
    LOGIN_HTML,
    SETUP_HTML,
    RECOVERY_HTML,
)
from src.behavior.event_manager import EventManager, SuspiciousEvent
from src.orchestration.camera_manager import CameraManager
from src.persistence.service import PersistenceService, AmbiguousEvidenceError
from src.persistence.repositories.session_repository import classify_session_kind
from src.evidence.clip_recorder import MIN_CLIP_DURATION_SEC, MIN_CLIP_FRAMES
from src.persistence.backup import (
    create_local_backup,
    create_encrypted_backup,
    verify_backup_directory,
    preview_retention_candidates,
)
from src.security.models import Role, Permission, User, AuthSession
from src.security.permissions import has_permission
from src.security.csrf import generate_csrf_token, verify_csrf_token
from src.security.password import validate_password_policy
from src.security.key_provider import get_key_provider

logger = logging.getLogger(__name__)


def _iso_to_epoch(iso: Optional[str], default: float) -> float:
    """
    ISO timestamp -> epoch seconds. Windows raises OSError for local times near/before
    the epoch (e.g. 1970-01-01T07:00 in UTC+7); one such row must not break a whole listing.
    """
    if not iso:
        return default
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return default
    except (OSError, OverflowError):
        dt = datetime.fromisoformat(iso)
        return (dt - datetime(1970, 1, 1)).total_seconds() + time.timezone


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
    camera_manager: Optional[CameraManager] = None,
    enforce_auth: bool = True,
    capture_controller: Optional[Any] = None,
    fresh_boot: bool = False,
) -> FastAPI:
    ps = persistence_service or PersistenceService.get_instance()
    # Reconnect to active session in database if one exists, otherwise leave idle awaiting explicit start (Workstream 3)
    if ps.active_session is None:
        try:
            ps.get_current_monitoring_session()
        except Exception as e:
            logger.warning(f"Could not load active monitoring session: {e}")

    def _sanitize_evidence_path(raw_path: Optional[str]) -> Optional[str]:
        if not raw_path:
            return None
        raw_str = str(raw_path).replace("\\", "/")
        if raw_str.endswith(".enc"):
            raw_str = raw_str[:-4]
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
        # If path already contains session subfolder structure
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

    # Track evidence view audits to prevent flooding on range chunks
    _evidence_view_audit_cache: dict = {}

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

    def _event_admissible(event: Any) -> bool:
        """Hard gate: only events from the camera owned by the ACTIVE monitoring session are accepted."""
        try:
            if not ps.is_event_admissible(getattr(event, "camera_id", None)):
                return False
            if capture_controller is not None:
                active = ps.active_session
                return bool(active) and capture_controller.is_bound_to(active.session_id)
            return True
        except Exception as e:
            logger.debug(f"Event admission check failed: {e}")
            return False

    # Wire Stage 2 FusedEvent lifecycle (OPEN, UPDATE, CLOSE) to WebSocket and ev_manager
    if stage2_pipeline is not None:
        def _on_stage2_lifecycle(event: Any, action: str):
            nonlocal loop
            if not _event_admissible(event):
                return
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

        def _on_stage2_evidence_ready(event_id: str, evidence_type: str, file_path: str):
            nonlocal loop
            sanitized = _sanitize_evidence_path(file_path)
            try:
                existing = ev_manager.get_event(event_id)
                if existing:
                    if evidence_type == "SNAPSHOT":
                        existing.snapshot_path = sanitized
                    elif evidence_type == "VIDEO_CLIP":
                        existing.clip_path = sanitized
                    ev_manager._events[event_id] = existing
            except Exception as e:
                logger.debug(f"Could not update ev_manager for evidence ready: {e}")

            try:
                if loop is None or not loop.is_running():
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        pass
                payload = {
                    "type": "EVIDENCE_READY",
                    "event_id": event_id,
                    "evidence_type": evidence_type,
                    "file_path": sanitized,
                }
                if loop and loop.is_running():
                    asyncio.run_coroutine_threadsafe(ws_manager.broadcast(payload), loop)
            except Exception as e:
                logger.debug(f"Could not broadcast evidence ready to WS: {e}")

        if hasattr(stage2_pipeline, "add_evidence_listener"):
            stage2_pipeline.add_evidence_listener(_on_stage2_evidence_ready)

    # Wire CameraManager multi-camera events if present
    if camera_manager is not None:
        def _on_multi_camera_event(event: Any, action: str):
            nonlocal loop
            if not _event_admissible(event):
                return
            try:
                if hasattr(event, "event_id"):
                    eid = event.event_id
                    existing = ev_manager.get_event(eid)
                    cur_status = existing.status if existing else "new"
                    cur_notes = existing.reviewer_notes if existing else None
                    snap_path = getattr(event, "evidence_summary", {}).get("snapshot_path") if hasattr(event, "evidence_summary") and isinstance(event.evidence_summary, dict) else None
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
                logger.debug(f"Could not sync multi-camera event: {e}")

            try:
                ps.persist_stage2_event(event, action)
            except Exception as e:
                logger.debug(f"Could not persist multi-camera event to DB: {e}")

            try:
                if loop and loop.is_running():
                    payload = {
                        "type": f"EVENT_{action}",
                        "action": action,
                        "event": event.to_dict() if hasattr(event, "to_dict") else str(event),
                    }
                    asyncio.run_coroutine_threadsafe(ws_manager.broadcast(payload), loop)
            except Exception as e:
                logger.debug(f"Could not broadcast multi-camera event to WS: {e}")

        camera_manager.add_event_listener(_on_multi_camera_event)
        if hasattr(camera_manager, "add_evidence_listener"):
            camera_manager.add_evidence_listener(_on_stage2_evidence_ready)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal loop
        loop = asyncio.get_running_loop()
        if fresh_boot:
            # A real server start invalidates every previous web login and interrupts any monitoring
            # session the previous process left ACTIVE: a restart always requires a fresh login.
            ps.on_application_boot()
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
            if capture_controller is not None:
                try:
                    capture_controller.stop()
                except Exception as e:
                    logger.debug(f"Error stopping session capture on shutdown: {e}")
            try:
                # Only an explicit End Session closes a session; an application stop interrupts it
                ps.interrupt_active_session(reason="APPLICATION_STOPPED")
            except Exception as e:
                logger.debug(f"Error interrupting session on shutdown: {e}")
            if camera_manager is not None:
                try:
                    camera_manager.stop_all()
                except Exception as e:
                    logger.debug(f"Error stopping CameraManager: {e}")

    app = FastAPI(
        title="Exam Suspicious Behavior Detection API",
        description="Production-oriented suspicious behavior detection and invigilator review backend",
        version="0.2.0-production-foundation",
        lifespan=lifespan,
    )
    app.state.event_manager = ev_manager
    app.state.camera_manager = camera_manager
    app.state.persistence_service = ps

    # 1. TIGHTENED CORS: Same-origin & localhost only (no wildcard '*')
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:8000",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # 2. SECURITY HEADERS MIDDLEWARE
    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "img-src 'self' data: blob:; "
            "media-src 'self' blob:; "
            "style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; "
            "connect-src 'self' ws: wss:; "
            "font-src 'self' data:;"
        )
        return response

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # --- AUTHENTICATION & CSRF HELPERS ---

    def _get_auth_context(request: Request) -> Tuple[Optional[User], Optional[AuthSession]]:
        """Extract user and session from eg_session cookie."""
        if not enforce_auth:
            # Synthetic default admin for test compatibility
            dummy = User(
                user_id="usr_admin_default",
                username="admin",
                password_hash="",
                display_name="Quản trị viên",
                role=Role.ADMIN,
                is_active=True,
            )
            return dummy, None

        token = request.cookies.get("eg_session")
        if not token:
            return None, None
        return ps.validate_session_token(token)

    def _require_authenticated_user(request: Request) -> Tuple[User, Optional[AuthSession]]:
        user, session = _get_auth_context(request)
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="Yêu cầu đăng nhập để truy cập.")
        return user, session

    def _require_permission(request: Request, perm: Permission) -> Tuple[User, Optional[AuthSession]]:
        user, session = _require_authenticated_user(request)
        if not has_permission(user.role, perm):
            raise HTTPException(status_code=403, detail="Bạn không có quyền thực hiện thao tác này.")
        return user, session

    def _verify_csrf_if_applicable(request: Request, session: Optional[AuthSession]):
        if not enforce_auth or session is None:
            return
        if request.method in ["POST", "PATCH", "PUT", "DELETE"]:
            token = request.headers.get("X-CSRF-Token")
            if not token or not verify_csrf_token(token, session.token_hash):
                raise HTTPException(status_code=403, detail="CSRF token không hợp lệ hoặc đã hết hạn.")

    # --- PAGE ROUTING: DASHBOARD, LOGIN, FIRST-RUN SETUP ---

    @app.get("/", response_class=HTMLResponse)
    @app.get("/dashboard", response_class=HTMLResponse)
    async def get_dashboard(request: Request):
        """Serve dashboard if authenticated. If no users exist, redirect to /setup. Else redirect to /login."""
        if enforce_auth:
            if not ps.has_users():
                return RedirectResponse(url="/setup", status_code=302)
            user, _ = _get_auth_context(request)
            if not user or not user.is_active:
                return RedirectResponse(url="/login", status_code=302)
        # Never cached: Back after logout or a restart must hit the server and its auth check again
        return HTMLResponse(
            content=load_dashboard_html(),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
        )

    @app.get("/login", response_class=HTMLResponse)
    async def get_login_page(request: Request):
        """Serve login page. If no users exist, redirect to /setup. If already authenticated, redirect to /."""
        if enforce_auth:
            if not ps.has_users():
                return RedirectResponse(url="/setup", status_code=302)
            user, _ = _get_auth_context(request)
            if user and user.is_active:
                return RedirectResponse(url="/", status_code=302)
        return load_login_html()

    @app.get("/forgot-password", response_class=HTMLResponse)
    async def get_forgot_password_page():
        """Serve offline-first account recovery and password reset guide."""
        return load_recovery_html()

    @app.get("/setup", response_class=HTMLResponse)
    async def get_setup_page():
        """Serve first-run initial admin setup. If admin exists, redirect to /login."""
        if ps.has_users():
            return RedirectResponse(url="/login", status_code=302)
        return SETUP_HTML

    # --- AUTHENTICATION & SETUP API ENDPOINTS ---

    @app.post("/api/setup")
    async def initial_setup_endpoint(req: SetupRequest, request: Request):
        """First-run initial admin creation. Only allowed when 0 users exist."""
        if ps.has_users():
            raise HTTPException(status_code=403, detail="Thiết lập quản trị viên đã hoàn tất.")

        u_strip = req.username.strip()
        d_strip = req.display_name.strip()
        if not u_strip or not d_strip:
            raise HTTPException(status_code=400, detail="Vui lòng cung cấp đầy đủ tên đăng nhập và tên hiển thị.")

        valid_pw, pw_err = validate_password_policy(req.password)
        if not valid_pw:
            raise HTTPException(status_code=400, detail=pw_err or "Mật khẩu không đạt chính sách bảo mật.")

        try:
            admin_user = ps.create_initial_admin(
                username=u_strip,
                password=req.password,
                display_name=d_strip,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Create session and set cookie
        ua = request.headers.get("user-agent", "")
        session, raw_token = ps.create_user_session(admin_user.user_id, user_agent=ua)

        resp = JSONResponse(content={"success": True, "redirect": "/"})
        resp.set_cookie(
            key="eg_session",
            value=raw_token,
            httponly=True,
            samesite="lax",
            path="/",
            max_age=8 * 3600,
        )
        return resp

    @app.post("/api/auth/login", response_model=AuthMeResponse)
    async def login_endpoint(req: LoginRequest, request: Request):
        """Authenticate user credentials with Argon2id and rate-limiting lockout."""
        user, err_code = ps.authenticate_user(req.username.strip(), req.password)
        if err_code and err_code.startswith("USER_LOCKED"):
            parts = err_code.split(":")
            mins_str = f" sau {parts[1]} phút" if len(parts) > 1 and parts[1].isdigit() else " sau"
            raise HTTPException(
                status_code=423,
                detail=f"Tài khoản tạm thời bị khóa do nhập sai nhiều lần. Bạn có thể thử lại{mins_str} hoặc sử dụng 'Quên mật khẩu?' để được hỗ trợ.",
            )
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Tên đăng nhập hoặc mật khẩu không đúng.",
            )

        ua = request.headers.get("user-agent", "")
        session, raw_token = ps.create_user_session(user.user_id, user_agent=ua)
        csrf = generate_csrf_token(session.token_hash)

        resp_data = AuthMeResponse(
            user=UserResponse(
                user_id=user.user_id,
                username=user.username,
                display_name=user.display_name,
                role=user.role_value,
                role_display=user.role_display,
                is_active=bool(user.is_active),
                last_login_at=user.last_login_at,
                created_at=user.created_at,
                locked_until=user.locked_until,
                failed_login_count=user.failed_login_count,
            ),
            csrf_token=csrf,
            role_display=user.role_display,
        )

        resp = JSONResponse(content=resp_data.model_dump())
        resp.set_cookie(
            key="eg_session",
            value=raw_token,
            httponly=True,
            samesite="lax",
            path="/",
            max_age=8 * 3600,
        )
        return resp

    @app.post("/api/auth/logout")
    async def logout_endpoint(request: Request):
        """Revoke server-side session and clear cookie."""
        token = request.cookies.get("eg_session")
        if token:
            ps.revoke_session_token(token)
        resp = JSONResponse(content={"success": True})
        resp.delete_cookie(key="eg_session", path="/")
        return resp

    @app.get("/api/auth/me", response_model=AuthMeResponse)
    async def get_current_user_endpoint(request: Request):
        """Retrieve current authenticated user profile, role display, and CSRF token."""
        user, session = _require_authenticated_user(request)
        csrf = generate_csrf_token(session.token_hash) if session else "test_csrf_token"
        return AuthMeResponse(
            user=UserResponse(
                user_id=user.user_id,
                username=user.username,
                display_name=user.display_name,
                role=user.role_value,
                role_display=user.role_display,
                is_active=bool(user.is_active),
                last_login_at=user.last_login_at,
                created_at=user.created_at,
                locked_until=user.locked_until,
                failed_login_count=user.failed_login_count,
            ),
            csrf_token=csrf,
            role_display=user.role_display,
        )

    # --- USER MANAGEMENT ENDPOINTS (ADMIN ONLY) ---

    @app.get("/api/users", response_model=List[UserResponse])
    async def list_users_endpoint(request: Request):
        """List all application users (ADMIN only)."""
        _require_permission(request, Permission.USER_MANAGE)
        users = ps.users.list_users()
        return [
            UserResponse(
                user_id=u.user_id,
                username=u.username,
                display_name=u.display_name,
                role=u.role_value,
                role_display=u.role_display,
                is_active=bool(u.is_active),
                last_login_at=u.last_login_at,
                created_at=u.created_at,
                locked_until=u.locked_until,
                failed_login_count=u.failed_login_count,
            )
            for u in users
        ]

    @app.post("/api/users", response_model=UserResponse)
    async def create_user_endpoint(req: UserCreateRequest, request: Request):
        """Create new user with role (ADMIN only)."""
        admin, session = _require_permission(request, Permission.USER_MANAGE)
        _verify_csrf_if_applicable(request, session)

        u_strip = req.username.strip()
        d_strip = req.display_name.strip()
        if not u_strip or not d_strip:
            raise HTTPException(status_code=400, detail="Vui lòng cung cấp đầy đủ tên đăng nhập và tên hiển thị.")

        valid_pw, pw_err = validate_password_policy(req.password)
        if not valid_pw:
            raise HTTPException(status_code=400, detail=pw_err or "Mật khẩu không đạt chính sách bảo mật.")

        try:
            role = Role(req.role.upper())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Vai trò không hợp lệ: {req.role}")

        try:
            created = ps.create_user(
                username=u_strip,
                password=req.password,
                display_name=d_strip,
                role=role,
                created_by=admin.user_id,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return UserResponse(
            user_id=created.user_id,
            username=created.username,
            display_name=created.display_name,
            role=created.role_value,
            role_display=created.role_display,
            is_active=bool(created.is_active),
            last_login_at=created.last_login_at,
            created_at=created.created_at,
            locked_until=created.locked_until,
            failed_login_count=created.failed_login_count,
        )

    @app.patch("/api/users/{user_id}", response_model=UserResponse)
    async def update_user_endpoint(user_id: str, req: UserUpdateRequest, request: Request):
        """Update user role, active status, or display name (ADMIN only)."""
        admin, session = _require_permission(request, Permission.USER_MANAGE)
        _verify_csrf_if_applicable(request, session)

        role_enum = Role(req.role.upper()) if req.role else None
        updated = ps.update_user(
            user_id=user_id,
            role=role_enum,
            is_active=req.is_active,
            display_name=req.display_name,
            admin_user_id=admin.user_id,
        )
        if not updated:
            raise HTTPException(status_code=404, detail="Không tìm thấy người dùng.")

        return UserResponse(
            user_id=updated.user_id,
            username=updated.username,
            display_name=updated.display_name,
            role=updated.role_value,
            role_display=updated.role_display,
            is_active=bool(updated.is_active),
            last_login_at=updated.last_login_at,
            created_at=updated.created_at,
            locked_until=updated.locked_until,
            failed_login_count=updated.failed_login_count,
        )

    @app.post("/api/users/{user_id}/reset-password")
    async def reset_password_endpoint(user_id: str, req: UserResetPasswordRequest, request: Request):
        """Reset user password (ADMIN only)."""
        admin, session = _require_permission(request, Permission.USER_MANAGE)
        _verify_csrf_if_applicable(request, session)

        valid_pw, pw_err = validate_password_policy(req.new_password)
        if not valid_pw:
            raise HTTPException(status_code=400, detail=pw_err or "Mật khẩu không đạt chính sách bảo mật.")

        success = ps.reset_password(
            user_id,
            req.new_password,
            admin_user_id=admin.user_id,
            must_change_password=bool(req.must_change_password),
        )
        if not success:
            raise HTTPException(status_code=404, detail="Không tìm thấy người dùng.")
        return {"success": True, "message": "Đặt lại mật khẩu thành công."}

    @app.post("/api/users/{user_id}/unlock")
    async def unlock_user_endpoint(user_id: str, request: Request):
        """Unlock locked user account without changing password (ADMIN only)."""
        admin, session = _require_permission(request, Permission.USER_MANAGE)
        _verify_csrf_if_applicable(request, session)

        success = ps.unlock_user(user_id, admin_user_id=admin.user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Không tìm thấy người dùng.")
        return {"success": True, "message": "Mở khóa tài khoản thành công."}

    # --- EVIDENCE SERVING WITH RANGE SUPPORT & IN-MEMORY DECRYPTION ---

    @app.get("/api/evidence/verify/{evidence_id}", response_model=EvidenceVerifyResponse)
    async def verify_evidence_endpoint(evidence_id: str, request: Request):
        """Verify evidence file integrity on disk against database SHA-256."""
        if enforce_auth:
            _require_permission(request, Permission.EVIDENCE_VIEW)
        res = ps.verify_evidence_integrity(evidence_id)
        return EvidenceVerifyResponse(**res)

    @app.get("/api/evidence/{file_path:path}")
    async def get_evidence_file(file_path: str, request: Request):
        """
        Safely serve snapshot images and video clips with strict path traversal prevention,
        authorization checks, in-memory AES-256-GCM decryption, and HTTP Range support.
        """
        user = None
        if enforce_auth:
            user, _ = _require_permission(request, Permission.EVIDENCE_VIEW)

        # Multi-pass unquote to defeat single and double URL encodings
        unquoted = file_path
        for _ in range(3):
            unquoted = urllib.parse.unquote(unquoted)

        # Strict path traversal prevention
        if ".." in file_path or ".." in unquoted or "%2e" in file_path.lower() or "%2e" in unquoted.lower():
            raise HTTPException(status_code=403, detail="Path traversal forbidden.")

        # Check for Windows drive letters, UNC paths, or absolute paths
        if re.search(r"^[a-zA-Z]:", file_path) or re.search(r"^[a-zA-Z]:", unquoted):
            raise HTTPException(status_code=403, detail="Drive letters and absolute paths forbidden.")
        if file_path.startswith(("\\\\", "//")) or unquoted.startswith(("\\\\", "//")):
            raise HTTPException(status_code=403, detail="UNC paths forbidden.")

        clean_path = unquoted.lstrip("/\\")
        if not clean_path:
            raise HTTPException(status_code=404, detail="Empty evidence path.")

        # Strict prohibition on credential or security directory access
        lowered_path = clean_path.lower()
        if "security" in lowered_path or lowered_path.endswith(".dpapi") or "master-key" in lowered_path:
            raise HTTPException(status_code=403, detail="Access to security credentials forbidden.")

        # Attempt to load and decrypt evidence in memory via PersistenceService
        try:
            data, mime_type = ps.load_and_decrypt_evidence(clean_path)
        except AmbiguousEvidenceError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Tập tin bằng chứng không tồn tại.")
        except Exception as e:
            logger.error(f"Error loading evidence '{clean_path}': {e}")
            raise HTTPException(status_code=500, detail="Không thể giải mã hoặc đọc tập tin bằng chứng.")

        # Bounded audit logging of evidence access (at most once every 60s per user/file)
        if user is not None:
            now_ts = time.time()
            cache_key = f"{user.user_id}:{clean_path}"
            last_logged = _evidence_view_audit_cache.get(cache_key, 0.0)
            if now_ts - last_logged > 60.0:
                _evidence_view_audit_cache[cache_key] = now_ts
                try:
                    ps.audit.log_action(
                        audit_id=f"aud_view_{int(now_ts * 1000) % 1000000}",
                        action="EVIDENCE_VIEWED",
                        actor_type="USER",
                        actor_id=user.user_id,
                        actor_display_name=user.display_name,
                        details={"file": clean_path, "mime": mime_type},
                    )
                except Exception:
                    pass

        total_bytes = len(data)

        # HTTP Range header handling for HTML5 video seeking
        range_header = request.headers.get("range")
        if range_header and range_header.startswith("bytes="):
            try:
                range_val = range_header.replace("bytes=", "").strip()
                parts = range_val.split(",")[0].split("-")
                if parts[0]:
                    start = int(parts[0])
                    end = int(parts[1]) if len(parts) > 1 and parts[1] else total_bytes - 1
                else:
                    # Suffix range: last N bytes
                    start = max(0, total_bytes - int(parts[1]))
                    end = total_bytes - 1
                end = min(end, total_bytes - 1)
                if start >= total_bytes or start > end:
                    return Response(
                        status_code=416,
                        headers={"Content-Range": f"bytes */{total_bytes}"},
                    )
                chunk = data[start : end + 1]
                return Response(
                    content=chunk,
                    status_code=206,
                    media_type=mime_type,
                    headers={
                        "Content-Range": f"bytes {start}-{end}/{total_bytes}",
                        "Accept-Ranges": "bytes",
                        "Content-Length": str(len(chunk)),
                        "Cache-Control": "private, no-cache",
                    },
                )
            except Exception as e:
                logger.debug(f"Range parsing error: {e}")

        return Response(
            content=data,
            status_code=200,
            media_type=mime_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(total_bytes),
                "Cache-Control": "private, no-cache",
            },
        )

    # --- CANONICAL CAMERA REGISTRY (one source for gate dropdown, header and System > Camera) ---

    def _runtime_camera_descriptor() -> dict:
        """The camera this runtime captures from: its persisted config if present, else auto-discovered."""
        try:
            persisted = ps.cameras.get_camera(camera_id)
        except Exception:
            persisted = None
        if persisted is not None:
            return {
                "camera_id": persisted.camera_id,
                "name": persisted.name,
                "source_type": persisted.source_type,
                "device_index": persisted.device_index,
                "enabled": bool(persisted.enabled),
                "resolution_width": persisted.resolution_width,
                "resolution_height": persisted.resolution_height,
                "target_capture_fps": persisted.target_capture_fps,
                "room": persisted.room,
                "auto_discovered": False,
            }
        tail = camera_id.rsplit("_", 1)[-1]
        idx = int(tail) if tail.isdigit() else None
        return {
            "camera_id": camera_id,
            "name": f"CAM {(idx or 0) + 1:02d}",
            "source_type": camera_type,
            "device_index": idx,
            "enabled": True,
            "resolution_width": 1280,
            "resolution_height": 720,
            "target_capture_fps": 30.0,
            "room": None,
            "auto_discovered": True,
        }

    def _runtime_capture_state() -> Tuple[str, bool]:
        """(capture_state, streaming). Without a session controller the legacy always-on semantics apply."""
        if capture_controller is not None:
            return ("ACTIVE", True) if capture_controller.is_active() else ("IDLE", False)
        is_stream = camera_streaming
        if stage2_pipeline is not None and hasattr(stage2_pipeline, "source"):
            if hasattr(stage2_pipeline.source, "is_opened") and stage2_pipeline.source.is_opened():
                is_stream = True
        if is_stream:
            return "ACTIVE", True
        return ("IDLE" if (camera_connected or device_present) else "OFFLINE"), False

    # --- HEALTH & CAMERA STATUS ENDPOINTS ---

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Unauthenticated health check for launcher and local diagnostics."""
        return HealthResponse(
            status="ok",
            version="0.2.0-production-foundation",
            timestamp=time.time(),
        )

    @app.get("/api/cameras", response_model=List[CameraInfo])
    async def list_cameras(request: Request):
        """List all configured and active cameras."""
        if enforce_auth:
            _require_permission(request, Permission.MONITOR_VIEW)

        if camera_manager is not None:
            statuses = camera_manager.get_all_statuses()
            res = []
            for s in statuses:
                obs_fps = s.get("processing_fps") if s.get("streaming") else None
                res.append(
                    CameraInfo(
                        camera_id=s.get("camera_id", "cam"),
                        name=s.get("name", "Camera"),
                        source_type=s.get("source_type", "webcam"),
                        configured=s.get("configured", True),
                        device_present=s.get("connected", False),
                        connected=s.get("connected", False),
                        streaming=s.get("streaming", False),
                        status="STREAMING" if s.get("streaming") else ("CONNECTED" if s.get("connected") else "DISCONNECTED"),
                        configured_capture_fps=float(s.get("target_capture_fps", 30.0)),
                        configured_resolution=f"{s.get('resolution_width', 1280)}x{s.get('resolution_height', 720)}",
                        observed_capture_fps=s.get("capture_fps"),
                        observed_resolution=f"{s.get('resolution_width', 1280)}x{s.get('resolution_height', 720)}" if s.get("streaming") else None,
                        is_active=s.get("streaming", False),
                        fps=obs_fps,
                        resolution=f"{s.get('resolution_width', 1280)}x{s.get('resolution_height', 720)}",
                    )
                )
            return res

        # Single camera fallback
        desc = _runtime_camera_descriptor()
        capture_state, is_stream = _runtime_capture_state()
        dev_pres = device_present or camera_connected or is_stream
        # With session-bound capture, "connected" means the device handle is open; an idle camera is only present
        is_conn = is_stream if capture_controller is not None else (camera_connected or is_stream)
        if is_stream:
            status_str = "STREAMING"
        elif capture_controller is not None and dev_pres:
            status_str = "READY"
        else:
            status_str = "CONNECTED" if is_conn else "NO_PHYSICAL_CAMERA"
        active_sess = ps.active_session if is_stream else None

        obs_cap_fps = getattr(stage2_pipeline, "observed_capture_fps", None) if is_stream else None
        obs_res = None
        if is_stream and stage2_pipeline is not None and hasattr(stage2_pipeline, "source"):
            if hasattr(stage2_pipeline.source, "width") and hasattr(stage2_pipeline.source, "height"):
                obs_res = f"{stage2_pipeline.source.width}x{stage2_pipeline.source.height}"

        cfg_res = f"{desc['resolution_width']}x{desc['resolution_height']}"
        return [
            CameraInfo(
                camera_id=desc["camera_id"],
                name=desc["name"],
                source_type=desc["source_type"],
                configured=True,
                device_present=dev_pres,
                connected=is_conn,
                streaming=is_stream,
                status=status_str,
                configured_capture_fps=float(desc["target_capture_fps"]),
                configured_resolution=cfg_res,
                observed_capture_fps=obs_cap_fps,
                observed_resolution=obs_res,
                is_active=is_stream,
                fps=obs_cap_fps,
                resolution=cfg_res,
                capture_state=capture_state,
                device_index=desc["device_index"],
                auto_discovered=desc["auto_discovered"],
                monitoring_session_id=active_sess.session_id if active_sess else None,
            )
        ]

    @app.get("/api/cameras/stream")
    @app.get("/api/cameras/{cam_id}/stream")
    async def get_camera_stream(request: Request, cam_id: Optional[str] = None, max_frames: int = 0):
        """MJPEG video stream downstream of active perception pipeline."""
        if enforce_auth:
            _require_permission(request, Permission.MONITOR_VIEW)

        stream_headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        }

        if camera_manager is not None:
            target_cam = camera_manager.resolve_canonical_camera_id(cam_id)
            async def multi_frame_gen():
                yielded = 0
                try:
                    while True:
                        if await request.is_disconnected():
                            break
                        frame_bytes = camera_manager.get_camera_frame(target_cam)
                        if frame_bytes is not None:
                            yield (
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n"
                                b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n" +
                                frame_bytes + b"\r\n"
                            )
                            yielded += 1
                            if max_frames > 0 and yielded >= max_frames:
                                break
                        await asyncio.sleep(0.04)
                except (asyncio.CancelledError, GeneratorExit):
                    pass

            return StreamingResponse(
                multi_frame_gen(),
                media_type="multipart/x-mixed-replace; boundary=frame",
                headers=stream_headers,
            )

        if stage2_pipeline is None:
            raise HTTPException(status_code=503, detail="Camera pipeline is not initialized.")

        async def frame_generator():
            yielded = 0
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    jpeg_data = getattr(stage2_pipeline, "_latest_jpeg_frame", None)
                    if jpeg_data is not None:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(jpeg_data)).encode() + b"\r\n\r\n" +
                            jpeg_data + b"\r\n"
                        )
                        yielded += 1
                        if max_frames > 0 and yielded >= max_frames:
                            break
                    await asyncio.sleep(0.04)
            except (asyncio.CancelledError, GeneratorExit):
                pass

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers=stream_headers,
        )

    @app.get("/api/cameras/frame")
    @app.get("/api/cameras/{cam_id}/frame")
    async def get_camera_frame(request: Request, cam_id: Optional[str] = None):
        """Single latest JPEG frame from active pipeline."""
        if enforce_auth:
            _require_permission(request, Permission.MONITOR_VIEW)

        frame_headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }

        if camera_manager is not None:
            target_cam = camera_manager.resolve_canonical_camera_id(cam_id)
            frame_bytes = camera_manager.get_camera_frame(target_cam)
            if frame_bytes is None:
                raise HTTPException(status_code=404, detail=f"No active frame for camera '{target_cam}'.")
            return Response(content=frame_bytes, media_type="image/jpeg", headers=frame_headers)

        if stage2_pipeline is None or getattr(stage2_pipeline, "_latest_jpeg_frame", None) is None:
            raise HTTPException(status_code=404, detail="No active camera frame available.")
        return Response(content=stage2_pipeline._latest_jpeg_frame, media_type="image/jpeg", headers=frame_headers)

    @app.get("/api/cameras/tracks")
    @app.get("/api/cameras/{cam_id}/tracks")
    async def get_camera_tracks(request: Request, cam_id: Optional[str] = None):
        """Latest active track bounding boxes and telemetry for web overlay."""
        if enforce_auth:
            _require_permission(request, Permission.MONITOR_VIEW)

        if cam_id == "tracks":
            cam_id = None

        tracks = []
        if camera_manager is not None:
            target_cam = camera_manager.resolve_canonical_camera_id(cam_id)
            tracks = camera_manager.get_camera_tracks(target_cam)
        elif stage2_pipeline is not None:
            tracks = getattr(stage2_pipeline, "_latest_tracks_summary", [])

        return JSONResponse(
            content=tracks,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    @app.post("/api/cameras/hero")
    async def set_hero_camera(payload: dict, request: Request):
        """Select active hero camera for primary viewport."""
        if enforce_auth:
            _require_permission(request, Permission.MONITOR_VIEW)
        cid = payload.get("camera_id")
        if not cid:
            raise HTTPException(status_code=400, detail="Missing camera_id")
        if camera_manager is not None:
            camera_manager.set_hero_camera(cid)
        return {"hero_camera_id": cid}

    @app.get("/api/cameras/config", response_model=List[CameraConfigItem])
    async def get_cameras_config_endpoint(request: Request):
        """List camera configuration records (ADMIN only)."""
        _require_permission(request, Permission.CAMERA_CONFIGURE)
        configs = ps.cameras.list_cameras()
        items = [
            CameraConfigItem(
                camera_id=c.camera_id,
                name=c.name,
                source_type=c.source_type,
                device_index=c.device_index,
                enabled=c.enabled,
                resolution_width=c.resolution_width,
                resolution_height=c.resolution_height,
                target_capture_fps=c.target_capture_fps,
                room=c.room,
            )
            for c in configs
        ]
        if camera_manager is None:
            # The runtime camera is always represented, whether persisted or auto-discovered
            desc = _runtime_camera_descriptor()
            capture_state, _ = _runtime_capture_state()
            existing = next((it for it in items if it.camera_id == desc["camera_id"]), None)
            if existing is not None:
                existing.capture_state = capture_state
            else:
                items.insert(0, CameraConfigItem(**desc, capture_state=capture_state))
        return items

    @app.patch("/api/cameras/config/{cam_id}")
    async def update_camera_config_endpoint(cam_id: str, req: CameraConfigUpdate, request: Request):
        """Update camera configuration (ADMIN only)."""
        admin, session = _require_permission(request, Permission.CAMERA_CONFIGURE)
        _verify_csrf_if_applicable(request, session)

        existing = ps.cameras.get_camera(cam_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Không tìm thấy cấu hình camera.")

        if req.name is not None:
            existing.name = req.name
        if req.enabled is not None:
            existing.enabled = req.enabled
        if req.device_index is not None:
            existing.device_index = req.device_index
        if req.resolution_width is not None:
            existing.resolution_width = req.resolution_width
        if req.resolution_height is not None:
            existing.resolution_height = req.resolution_height
        if req.target_capture_fps is not None:
            existing.target_capture_fps = req.target_capture_fps
        if req.room is not None:
            existing.room = req.room

        ps.cameras.add_or_update_camera(existing)

        # Audit camera change
        ps.audit.log_action(
            audit_id=f"aud_cam_{int(time.time() * 1000) % 1000000}",
            action="CAMERA_CONFIG_CHANGED",
            actor_type="USER",
            actor_id=admin.user_id,
            actor_display_name=admin.display_name,
            details={"camera_id": cam_id, "name": existing.name, "enabled": existing.enabled},
        )
        return {"success": True, "camera_id": cam_id}

    # --- EVENTS & REVIEW ENDPOINTS ---

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
        ev_sum = d.get("evidence") if isinstance(d.get("evidence"), dict) else {}
        if not d.get("clip_status"):
            live_clip = str(ev_sum.get("clip_status") or "").upper()
            if live_clip in ("READY", "FAILED"):
                d["clip_status"] = live_clip
            elif live_clip or d.get("clip_path"):
                d["clip_status"] = "PENDING"
        if not d.get("snapshot_status"):
            d["snapshot_status"] = "READY" if d.get("snapshot_path") else (ev_sum.get("snapshot_status") or "PENDING")
        return d

    _session_active_cache: dict = {}

    def _is_session_active(session_id: Optional[str]) -> bool:
        if not session_id:
            return False
        if ps.active_session and ps.active_session.session_id == session_id:
            return True
        cached = _session_active_cache.get(session_id)
        if cached is not None and time.time() - cached[1] < 5.0:
            return cached[0]
        sess = ps.sessions.get_session(session_id)
        active = bool(sess and sess.status == "ACTIVE")
        _session_active_cache[session_id] = (active, time.time())
        return active

    def _clip_status_for(pe: Any, ev_sum: dict, rows: List[Any]) -> Tuple[Optional[str], str]:
        """
        Truthful video state from the stored artifact row: READY only for a verified,
        reviewable clip; known-short legacy clips are LEGACY_INVALID, never READY.
        """
        clip_rows = [r for r in rows if r.evidence_type == "VIDEO_CLIP"]
        if clip_rows:
            row = clip_rows[-1]
            state = (row.artifact_state or "").upper()
            if state == "READY":
                frames_ok = row.frame_count is None or row.frame_count >= MIN_CLIP_FRAMES
                duration_ok = row.duration_sec is not None and row.duration_sec >= MIN_CLIP_DURATION_SEC
                if frames_ok and duration_ok:
                    return row.relative_path, "READY"
                return row.relative_path, "LEGACY_INVALID"
            if state == "FAILED":
                return None, "FAILED"
            return None, "PENDING"
        if str(ev_sum.get("clip_state") or ev_sum.get("clip_status") or "").upper() == "FAILED":
            return None, "FAILED"
        if not _is_session_active(pe.session_id):
            return None, "LEGACY_INVALID"
        if pe.closed_at:
            try:
                if time.time() - _iso_to_epoch(pe.closed_at, time.time()) > 60.0:
                    return None, "FAILED"
            except ValueError:
                pass
        return None, "PENDING"

    def _persisted_event_to_dict(pe: Any) -> dict:
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

        evidence_rows = ps.evidence.list_evidence_for_event(pe.event_id)
        snap_path = None
        for evd in evidence_rows:
            if evd.evidence_type == "SNAPSHOT" and (evd.artifact_state or "READY").upper() == "READY":
                snap_path = evd.relative_path
        if not snap_path:
            snap_path = ev_sum.get("snapshot_path") or ev_sum.get("open_snapshot_path")
        clip_path, clip_status = _clip_status_for(pe, ev_sum, evidence_rows)

        st_ts = _iso_to_epoch(pe.opened_at, time.time())
        et_ts = _iso_to_epoch(pe.closed_at, st_ts)

        d = {
            "clip_status": clip_status,
            "snapshot_status": "READY" if snap_path else "PENDING",
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
            "review_status": pe.review_status,
            "lifecycle_status": pe.lifecycle_status or "active",
            "observation_snapshot": obs,
            "created_at": pe.created_at,
            "reviewer_notes": reviewer_note,
            "event_origin": pe.source_origin,
        }
        return _sanitize_event_dict(d)

    @app.get("/api/events", response_model=List[EventResponse])
    async def get_events(
        request: Request,
        risk_level: Optional[str] = None,
        status: Optional[str] = None,
        session_id: Optional[str] = None,
        all_sessions: bool = False,
        limit: int = 100,
    ):
        if enforce_auth:
            _require_permission(request, Permission.MONITOR_VIEW)

        target_sid = session_id
        if not target_sid and not all_sessions and ps.active_session:
            target_sid = ps.active_session.session_id

        if target_sid:
            persisted = ps.events.list_events_by_session(
                session_id=target_sid,
                severity=risk_level,
                review_status=status,
                limit=limit,
            )
            if persisted:
                return [EventResponse(**_persisted_event_to_dict(pe)) for pe in persisted]

        events = ev_manager.list_events(risk_level=risk_level, status=status, limit=limit)
        return [EventResponse(**_sanitize_event_dict(e.to_dict())) for e in events]

    @app.get("/api/events/{event_id}", response_model=EventResponse)
    async def get_event(event_id: str, request: Request):
        if enforce_auth:
            _require_permission(request, Permission.MONITOR_VIEW)
        event = ev_manager.get_event(event_id)
        if event:
            return EventResponse(**_sanitize_event_dict(event.to_dict()))
        db_ev = ps.events.get_event(event_id)
        if db_ev:
            return EventResponse(**_persisted_event_to_dict(db_ev))
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found.")

    @app.patch("/api/events/{event_id}", response_model=EventResponse)
    async def update_event_status(event_id: str, req: EventStatusUpdateRequest, request: Request):
        """Update review status (confirmed / dismissed) with authentic human reviewer attribution."""
        reviewer_id = None
        reviewer_name = None
        if enforce_auth:
            user, session = _require_authenticated_user(request)
            _verify_csrf_if_applicable(request, session)
            norm_chk = req.status.lower()
            if norm_chk in ["confirmed", "confirmed_event", "reviewed"]:
                if not has_permission(user.role, Permission.EVENT_CONFIRM):
                    raise HTTPException(status_code=403, detail="Bạn không có quyền xác nhận sự kiện.")
            else:
                if not has_permission(user.role, Permission.EVENT_DISMISS):
                    raise HTTPException(status_code=403, detail="Bạn không có quyền bỏ qua sự kiện.")
            reviewer_id = user.user_id
            reviewer_name = user.display_name

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
                reviewer_id=reviewer_id,
                reviewer_name=reviewer_name,
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
            "reviewer_name": reviewer_name,
        })
        if updated_event:
            return EventResponse(**_sanitize_event_dict(updated_event.to_dict()))

        if db_ev:
            st_ts = _iso_to_epoch(db_ev.opened_at, time.time())
            et_ts = _iso_to_epoch(db_ev.closed_at, st_ts)
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

    def _build_session_response(s: Any) -> SessionResponse:
        summary_dict = ps.sessions.get_session_summary(s.session_id)
        return SessionResponse(
            session_id=s.session_id,
            name=s.name,
            room=s.room,
            class_name=getattr(s, "class_name", None),
            subject_code=getattr(s, "subject_code", None),
            invigilator_name=s.invigilator_name,
            notes=getattr(s, "notes", None),
            started_at=s.started_at,
            ended_at=s.ended_at,
            status=s.status,
            camera_count=s.camera_count,
            created_at=s.created_at,
            updated_at=s.updated_at,
            last_heartbeat_at=s.last_heartbeat_at,
            close_reason=s.close_reason,
            evidence_failure_count=getattr(s, "evidence_failure_count", 0) or 0,
            summary=SessionSummary(**summary_dict),
        )

    @app.get("/api/sessions", response_model=List[SessionResponse])
    async def list_sessions(
        request: Request,
        status: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """List historical and active exam monitoring sessions."""
        if enforce_auth:
            _require_permission(request, Permission.HISTORY_VIEW)

        sessions = ps.sessions.list_sessions(limit=limit, offset=offset, status=status, search=search)
        return [_build_session_response(s) for s in sessions]

    _DATE_RANGE_DAYS = {"today": 0, "7d": 7, "30d": 30}

    @app.get("/api/sessions/page", response_model=SessionPageResponse)
    async def list_sessions_page(
        request: Request,
        page: int = 1,
        page_size: int = 10,
        status: Optional[str] = None,
        search: Optional[str] = None,
        date_range: Optional[str] = None,
        include_test: bool = False,
    ):
        """One row per session, filtered and paginated server-side for the History view."""
        if enforce_auth:
            _require_permission(request, Permission.HISTORY_VIEW)

        page_size = max(1, min(page_size, 50))
        started_after = None
        if date_range in _DATE_RANGE_DAYS:
            day_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            started_after = (day_start - timedelta(days=_DATE_RANGE_DAYS[date_range])).isoformat()

        candidates = ps.sessions.list_sessions(
            limit=100000, offset=0, status=status, search=search, started_after=started_after
        )
        kinds = {s.session_id: classify_session_kind(s.name) for s in candidates}
        visible = [s for s in candidates if include_test or kinds[s.session_id] == "OPERATIONAL" or s.status == "ACTIVE"]
        hidden = len(candidates) - len(visible)

        total = len(visible)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))
        page_items = visible[(page - 1) * page_size : page * page_size]

        items = []
        for s in page_items:
            resp = _build_session_response(s)
            resp.session_kind = kinds[s.session_id]
            items.append(resp)
        return SessionPageResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            hidden_test_count=hidden,
        )

    @app.get("/api/sessions/current", response_model=SessionResponse)
    async def get_current_session(request: Request):
        """Retrieve the currently active monitoring session."""
        if enforce_auth:
            _require_permission(request, Permission.MONITOR_VIEW)

        active = ps.get_current_monitoring_session()
        if not active:
            raise HTTPException(status_code=404, detail="No active monitoring session found.")
        return _build_session_response(active)

    @app.post("/api/sessions/start", response_model=SessionResponse)
    async def start_session(req: SessionStartRequest, request: Request):
        """
        Explicitly begin a new exam monitoring session (Workstream 4 & 7).
        Resets live tracking state, buffer, queue, and KPIs.
        """
        actor_id = None
        actor_name = None
        if enforce_auth:
            user, session = _require_permission(request, Permission.SESSION_EDIT)
            _verify_csrf_if_applicable(request, session)
            actor_id = user.user_id
            actor_name = user.display_name
            if not req.invigilator_name:
                req.invigilator_name = user.display_name

        if capture_controller is not None:
            return await _start_session_with_capture(req, actor_id)

        # Flush any active recordings from previous session
        if stage2_pipeline and hasattr(stage2_pipeline, "evidence_manager") and stage2_pipeline.evidence_manager:
            try:
                stage2_pipeline.evidence_manager.clip_recorder.finalize_all_active()
            except Exception:
                pass
        if camera_manager and hasattr(camera_manager, "finalize_all_active"):
            try:
                camera_manager.finalize_all_active()
            except Exception:
                pass

        new_sess = ps.start_monitoring_session(
            name=req.get_name(),
            room=req.get_room(),
            class_name=req.class_name,
            subject_code=req.subject_code,
            invigilator_name=req.invigilator_name,
            notes=req.notes,
            camera_ids=req.camera_ids,
            actor_id=actor_id,
        )

        # Workstream 10: Reset transient runtime state
        if stage2_pipeline and hasattr(stage2_pipeline, "reset"):
            try:
                stage2_pipeline.reset()
            except Exception:
                pass
        if camera_manager and hasattr(camera_manager, "reset"):
            try:
                camera_manager.reset()
            except Exception:
                pass

        ev_manager.clear()

        # Broadcast session start event to WebSockets
        payload = {
            "type": "MONITORING_SESSION_STARTED",
            "session": new_sess.to_dict(),
        }
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(ws_manager.broadcast(payload), loop)

        return _build_session_response(new_sess)

    async def _start_session_with_capture(req: SessionStartRequest, actor_id: Optional[str]) -> SessionResponse:
        """
        Session-bound start transaction: persist ACTIVE session bound to the canonical camera,
        then open the camera and start capture + AI. If the camera cannot open, the session is
        recorded as START_FAILED (never left as a hidden ACTIVE zombie) and the request fails.
        """
        # A running session is stopped in order (events closed while it is still ACTIVE) before switching
        previous = ps.get_current_monitoring_session()
        if previous is not None and capture_controller.is_bound_to(previous.session_id):
            await asyncio.to_thread(capture_controller.stop)

        cam_id = capture_controller.camera_id
        requested = [c for c in (req.camera_ids or []) if c]
        if requested and any(c != cam_id for c in requested):
            raise HTTPException(status_code=400, detail=f"Camera không hợp lệ. Camera khả dụng: {cam_id}.")

        new_sess = ps.start_monitoring_session(
            name=req.get_name(),
            room=req.get_room(),
            class_name=req.class_name,
            subject_code=req.subject_code,
            invigilator_name=req.invigilator_name,
            notes=req.notes,
            camera_ids=[cam_id],
            actor_id=actor_id,
        )
        ev_manager.clear()

        opened = await asyncio.to_thread(capture_controller.start, new_sess.session_id)
        if not opened:
            ps.mark_session_start_failed(new_sess.session_id, reason="CAMERA_OPEN_FAILED", actor_id=actor_id)
            raise HTTPException(
                status_code=503,
                detail="Không thể mở camera. Vui lòng kiểm tra kết nối camera và thử lại.",
            )

        payload = {"type": "MONITORING_SESSION_STARTED", "session": new_sess.to_dict()}
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(ws_manager.broadcast(payload), loop)
        return _build_session_response(new_sess)

    @app.post("/api/sessions/{session_id}/end", response_model=SessionResponse)
    async def end_session(session_id: str, req: SessionEndRequest, request: Request):
        """
        Explicitly close an active monitoring session with bounded finalization (Workstream 6).
        """
        actor_id = None
        if enforce_auth:
            user, session = _require_permission(request, Permission.SESSION_EDIT)
            _verify_csrf_if_applicable(request, session)
            actor_id = user.user_id

        existing = ps.sessions.get_session(session_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

        if capture_controller is not None and capture_controller.is_bound_to(session_id):
            # Stop new frames, close live events while still ACTIVE, flush evidence, release camera, clear live state
            await asyncio.to_thread(capture_controller.stop)

        # Finalize any pending clips across workers
        if stage2_pipeline and hasattr(stage2_pipeline, "evidence_manager") and stage2_pipeline.evidence_manager:
            try:
                stage2_pipeline.evidence_manager.clip_recorder.finalize_all_active()
            except Exception:
                pass
        if camera_manager and hasattr(camera_manager, "finalize_all_active"):
            try:
                camera_manager.finalize_all_active()
            except Exception:
                pass

        summary_dict = ps.sessions.get_session_summary(session_id)
        closed_sess = ps.end_monitoring_session(
            session_id=session_id,
            reason=req.reason,
            summary=summary_dict,
            actor_id=actor_id,
        )

        res_sess = closed_sess or existing
        # The closed session's events stay in SQLite/History; drop them from the live view
        ev_manager.clear()
        payload = {
            "type": "MONITORING_SESSION_ENDED",
            "session": res_sess.to_dict(),
        }
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(ws_manager.broadcast(payload), loop)

        return _build_session_response(res_sess)

    @app.get("/api/sessions/{session_id}", response_model=SessionResponse)
    async def get_session_detail(session_id: str, request: Request):
        """Retrieve details of a single session."""
        if enforce_auth:
            _require_permission(request, Permission.HISTORY_VIEW)

        sess = ps.sessions.get_session(session_id)
        if not sess:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        return _build_session_response(sess)

    @app.patch("/api/sessions/{session_id}", response_model=SessionResponse)
    async def update_session_metadata(session_id: str, req: SessionUpdateRequest, request: Request):
        """Update mutable metadata fields for a session."""
        actor_id = None
        actor_name = None
        if enforce_auth:
            user, session = _require_permission(request, Permission.SESSION_EDIT)
            _verify_csrf_if_applicable(request, session)
            actor_id = user.user_id
            actor_name = user.display_name

        existing = ps.sessions.get_session(session_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

        ps.update_session_metadata(
            session_id=session_id,
            name=req.name,
            room=req.room,
            invigilator_name=req.invigilator_name,
            actor_id=actor_id,
        )
        existing = ps.sessions.get_session(session_id)
        return _build_session_response(existing)

    @app.get("/api/sessions/{session_id}/events", response_model=List[EventResponse])
    async def get_session_events(
        session_id: str,
        request: Request,
        severity: Optional[str] = None,
        review_status: Optional[str] = None,
        limit: int = 500,
    ):
        """Retrieve all events belonging to a historical or active session."""
        if enforce_auth:
            _require_permission(request, Permission.HISTORY_VIEW)

        persisted_events = ps.events.list_events_by_session(
            session_id=session_id,
            severity=severity,
            review_status=review_status,
            limit=limit,
        )
        # Exact session_id isolation is enforced by the SQL filter above
        return [EventResponse(**_persisted_event_to_dict(pe)) for pe in persisted_events]

    @app.get("/api/sessions/{session_id}/audit")
    async def get_session_audit_logs(session_id: str, request: Request):
        """Retrieve audit log history for a session."""
        if enforce_auth:
            _require_permission(request, Permission.AUDIT_VIEW)

        logs = ps.audit.list_logs_for_session(session_id)
        return [l.to_dict() for l in logs]

    # --- BACKUP ENDPOINTS (ADMIN ONLY) ---

    @app.post("/api/backup", response_model=BackupResponse)
    async def trigger_backup(request: Request, body: Optional[BackupCreateRequest] = None):
        """Create a complete local or portable encrypted backup."""
        if enforce_auth:
            admin, session = _require_permission(request, Permission.BACKUP_CREATE)
            _verify_csrf_if_applicable(request, session)

        recovery_pw = body.recovery_passphrase if body else None
        if recovery_pw:
            res = create_encrypted_backup(ps, recovery_passphrase=recovery_pw)
        else:
            res = create_local_backup(ps)
        return BackupResponse(**res)

    @app.post("/api/backup/verify")
    async def verify_backup_endpoint(req: BackupVerifyRequest, request: Request):
        """Verify encrypted backup structure, manifest, wrapping key, and hashes."""
        if enforce_auth:
            admin, session = _require_permission(request, Permission.BACKUP_CREATE)
            _verify_csrf_if_applicable(request, session)

        res = verify_backup_directory(req.backup_id_or_path, recovery_passphrase=req.recovery_passphrase)
        return res

    @app.get("/api/backup/status")
    async def get_backup_status(request: Request):
        """Retrieve the latest backup logs and status."""
        if enforce_auth:
            _require_permission(request, Permission.BACKUP_CREATE)

        logs = ps.audit.list_all_logs(limit=200)
        backup_logs = [l.to_dict() for l in logs if "BACKUP" in l.action]
        return {
            "last_backup": backup_logs[-1] if backup_logs else None,
            "total_backups": len(backup_logs),
        }

    @app.get("/api/retention/preview")
    async def preview_retention_endpoint(request: Request):
        """Preview retention cleanup candidates without deletion."""
        if enforce_auth:
            _require_permission(request, Permission.BACKUP_CREATE)
        return preview_retention_candidates(ps)

    # --- SYSTEM & DIAGNOSTICS ENDPOINTS ---

    @app.get("/api/system/status", response_model=SystemStatusResponse)
    async def get_system_status(request: Request):
        if enforce_auth:
            _require_permission(request, Permission.MONITOR_VIEW)

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

        drop_pct = None
        q_depth = 0
        active_st = 0
        is_stream = camera_streaming

        if camera_manager is not None:
            statuses = camera_manager.get_all_statuses()
            is_stream = any(s.get("streaming", False) for s in statuses)
            is_conn = any(s.get("connected", False) for s in statuses)
            active_st = sum(s.get("active_tracks", 0) for s in statuses)
            camera_counts = CameraCounts(
                registered=len(statuses),
                configured=len(statuses),
                connected=sum(1 for s in statuses if s.get("connected")),
                streaming=sum(1 for s in statuses if s.get("streaming")),
            )
            obs_proc_fps = max([s.get("processing_fps", 0.0) for s in statuses], default=None)
            obs_cap_fps = max([s.get("capture_fps", 0.0) for s in statuses], default=None)
            obs_inf_fps = obs_proc_fps
        else:
            if stage2_pipeline is not None:
                if hasattr(stage2_pipeline, "source") and hasattr(stage2_pipeline.source, "is_opened"):
                    if stage2_pipeline.source.is_opened():
                        is_stream = True
                tot = stage2_pipeline.processed_frames_count + stage2_pipeline.dropped_frames_count
                if tot > 0:
                    drop_pct = round((stage2_pipeline.dropped_frames_count / tot) * 100.0, 1)
                q_depth = stage2_pipeline.ingestion_queue.qsize
                active_st = len(stage2_pipeline._track_metadata)

            if capture_controller is not None:
                # Camera ready != camera active: only an ACTIVE session's capture counts as streaming
                is_stream = capture_controller.is_active()
                is_conn = is_stream
                if not is_stream:
                    drop_pct = None
                    q_depth = 0
                    active_st = 0
            else:
                is_conn = camera_connected or is_stream
            camera_counts = CameraCounts(
                registered=1,
                configured=1,
                connected=1 if is_conn else 0,
                streaming=1 if is_stream else 0,
            )
            obs_proc_fps = getattr(stage2_pipeline, "observed_processed_fps", None) if is_stream else None
            obs_cap_fps = getattr(stage2_pipeline, "observed_capture_fps", None) if is_stream else None
            obs_inf_fps = getattr(stage2_pipeline, "observed_inference_fps", None) if is_stream else None

        dev_pres = device_present or is_conn
        active_sess = ps.active_session
        runtime_stream = RuntimeStreamStatus(
            active=is_stream,
            source_type=camera_type.upper(),
            device_present=dev_pres,
            capture_state=("ACTIVE" if is_stream else ("IDLE" if dev_pres else "OFFLINE")),
            monitoring_active=bool(active_sess and active_sess.status == "ACTIVE" and is_stream),
        )

        configured_rates = ConfiguredRates(
            capture_fps=30.0,
            inference_fps=12.0,
        )

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
            active_cameras=camera_counts.streaming,
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

    @app.get("/api/system/security", response_model=SecurityStatusResponse)
    async def get_system_security_status(request: Request):
        """Retrieve security health metrics (ADMIN only)."""
        _require_permission(request, Permission.SECURITY_ADMIN)

        kp = get_key_provider()
        key_storage_str = "Windows DPAPI" if "WindowsDPAPI" in type(kp).__name__ else ("InMemoryTest" if "InMemory" in type(kp).__name__ else "Environment")
        audit_valid, _ = ps.audit.verify_audit_chain()

        return SecurityStatusResponse(
            auth_status="Bật",
            evidence_encryption="AES-256-GCM",
            key_storage=key_storage_str,
            audit_chain_valid=audit_valid,
            total_users=len(ps.users.list_users()),
            active_sessions=ps.users.count_active_sessions(),
        )

    @app.get("/api/system/models")
    async def get_system_models(request: Request):
        """Return provenance registry metadata including SHA-256 for all active models."""
        if enforce_auth:
            _require_permission(request, Permission.MONITOR_VIEW)

        try:
            from src.orchestration.model_registry import ModelRegistry
            reg = ModelRegistry.get_instance()
            if not reg._initialized:
                reg.initialize_models()
            return reg.get_metadata()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model registry unavailable: {e}")

    # --- WEBSOCKET ENDPOINT WITH SESSION HANDSHAKE AUTHENTICATION ---

    @app.websocket("/ws/events")
    async def websocket_events_endpoint(websocket: WebSocket):
        """Real-time event notification stream. Rejects unauthenticated connections."""
        if enforce_auth:
            token = websocket.cookies.get("eg_session")
            user = None
            if token:
                user, _ = ps.validate_session_token(token)
            if not user or not user.is_active:
                await websocket.close(code=4401, reason="Unauthorized")
                return

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


# Default module-level application instance for uvicorn run (boot invalidation runs only when served)
app = create_app(fresh_boot=True)
