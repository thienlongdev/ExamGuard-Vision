"""FastAPI application for Exam Suspicious Behavior Detection."""

import asyncio
from datetime import datetime
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
    SessionSummary,
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
from src.api.static_ui import load_dashboard_html, LOGIN_HTML, SETUP_HTML
from src.behavior.event_manager import EventManager, SuspiciousEvent
from src.orchestration.camera_manager import CameraManager
from src.persistence.service import PersistenceService, AmbiguousEvidenceError
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

    # Wire CameraManager multi-camera events if present
    if camera_manager is not None:
        def _on_multi_camera_event(event: Any, action: str):
            nonlocal loop
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
        return load_dashboard_html()

    @app.get("/login", response_class=HTMLResponse)
    async def get_login_page(request: Request):
        """Serve login page. If no users exist, redirect to /setup. If already authenticated, redirect to /."""
        if enforce_auth:
            if not ps.has_users():
                return RedirectResponse(url="/setup", status_code=302)
            user, _ = _get_auth_context(request)
            if user and user.is_active:
                return RedirectResponse(url="/", status_code=302)
        return LOGIN_HTML

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
        if err_code == "USER_LOCKED":
            raise HTTPException(
                status_code=423,
                detail="Tài khoản tạm thời bị khóa do nhập sai nhiều lần. Vui lòng thử lại sau.",
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
                is_active=user.is_active,
                last_login_at=user.last_login_at,
                created_at=user.created_at,
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
                is_active=user.is_active,
                last_login_at=user.last_login_at,
                created_at=user.created_at,
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
                is_active=u.is_active,
                last_login_at=u.last_login_at,
                created_at=u.created_at,
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
            is_active=created.is_active,
            last_login_at=created.last_login_at,
            created_at=created.created_at,
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
            is_active=updated.is_active,
            last_login_at=updated.last_login_at,
            created_at=updated.created_at,
        )

    @app.post("/api/users/{user_id}/reset-password")
    async def reset_password_endpoint(user_id: str, req: UserResetPasswordRequest, request: Request):
        """Reset user password (ADMIN only)."""
        admin, session = _require_permission(request, Permission.USER_MANAGE)
        _verify_csrf_if_applicable(request, session)

        valid_pw, pw_err = validate_password_policy(req.new_password)
        if not valid_pw:
            raise HTTPException(status_code=400, detail=pw_err or "Mật khẩu không đạt chính sách bảo mật.")

        success = ps.reset_password(user_id, req.new_password, admin_user_id=admin.user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Không tìm thấy người dùng.")
        return {"success": True, "message": "Đặt lại mật khẩu thành công."}

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
                parts = range_val.split("-")
                start = int(parts[0]) if parts[0] else 0
                end = int(parts[1]) if len(parts) > 1 and parts[1] else total_bytes - 1
                if start >= total_bytes or end >= total_bytes or start > end:
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

    @app.get("/api/cameras/stream")
    @app.get("/api/cameras/{cam_id}/stream")
    async def get_camera_stream(request: Request, cam_id: Optional[str] = None):
        """MJPEG video stream downstream of active perception pipeline."""
        if enforce_auth:
            _require_permission(request, Permission.MONITOR_VIEW)

        if camera_manager is not None:
            target_cam = cam_id or camera_manager.hero_camera_id
            async def multi_frame_gen():
                while True:
                    frame_bytes = camera_manager.get_camera_frame(target_cam)
                    if frame_bytes is not None:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n" +
                            frame_bytes + b"\r\n"
                        )
                    await asyncio.sleep(0.04)

            return StreamingResponse(
                multi_frame_gen(),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

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
    async def get_camera_frame(request: Request, cam_id: Optional[str] = None):
        """Single latest JPEG frame from active pipeline."""
        if enforce_auth:
            _require_permission(request, Permission.MONITOR_VIEW)

        if camera_manager is not None:
            target_cam = cam_id or camera_manager.hero_camera_id
            frame_bytes = camera_manager.get_camera_frame(target_cam)
            if frame_bytes is None:
                raise HTTPException(status_code=404, detail=f"No active frame for camera '{target_cam}'.")
            return Response(content=frame_bytes, media_type="image/jpeg")

        if stage2_pipeline is None or getattr(stage2_pipeline, "_latest_jpeg_frame", None) is None:
            raise HTTPException(status_code=404, detail="No active camera frame available.")
        return Response(content=stage2_pipeline._latest_jpeg_frame, media_type="image/jpeg")

    @app.get("/api/cameras/tracks")
    @app.get("/api/cameras/{cam_id}/tracks")
    async def get_camera_tracks(request: Request, cam_id: Optional[str] = None):
        """Latest active track bounding boxes and telemetry for web overlay."""
        if enforce_auth:
            _require_permission(request, Permission.MONITOR_VIEW)

        if camera_manager is not None:
            target_cam = cam_id or camera_manager.hero_camera_id
            return camera_manager.get_camera_tracks(target_cam)

        if stage2_pipeline is None:
            return []
        return getattr(stage2_pipeline, "_latest_tracks_summary", [])

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
        return [
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

        ps.cameras.upsert_camera(existing)

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
        return d

    @app.get("/api/events", response_model=List[EventResponse])
    async def get_events(
        request: Request,
        risk_level: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ):
        if enforce_auth:
            _require_permission(request, Permission.MONITOR_VIEW)
        events = ev_manager.list_events(risk_level=risk_level, status=status, limit=limit)
        return [EventResponse(**_sanitize_event_dict(e.to_dict())) for e in events]

    @app.get("/api/events/{event_id}", response_model=EventResponse)
    async def get_event(event_id: str, request: Request):
        if enforce_auth:
            _require_permission(request, Permission.MONITOR_VIEW)
        event = ev_manager.get_event(event_id)
        if not event:
            raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found.")
        return EventResponse(**_sanitize_event_dict(event.to_dict()))

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
    async def get_current_session(request: Request):
        """Retrieve the currently active monitoring session."""
        if enforce_auth:
            _require_permission(request, Permission.MONITOR_VIEW)

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
    async def get_session_detail(session_id: str, request: Request):
        """Retrieve details of a single session."""
        if enforce_auth:
            _require_permission(request, Permission.HISTORY_VIEW)

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

        # Audit session update
        try:
            ps.audit.log_action(
                audit_id=f"aud_sess_upd_{int(time.time() * 1000) % 1000000}",
                action="SESSION_METADATA_UPDATED",
                session_id=session_id,
                actor_type="USER" if actor_id else "SYSTEM",
                actor_id=actor_id,
                actor_display_name=actor_name,
                details={"name": existing.name, "room": existing.room, "invigilator": existing.invigilator_name},
            )
        except Exception:
            pass

        summary_dict = ps.sessions.get_session_summary(session_id)
        return SessionResponse(
            session_id=existing.session_id,
            name=existing.name,
            room=existing.room,
            invigilator_name=existing.invigilator_name,
            started_at=existing.started_at,
            ended_at=existing.ended_at,
            status=existing.status,
            camera_count=existing.camera_count,
            created_at=existing.created_at,
            updated_at=existing.updated_at,
            last_heartbeat_at=existing.last_heartbeat_at,
            close_reason=existing.close_reason,
            summary=SessionSummary(**summary_dict),
        )

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


# Default module-level application instance for uvicorn run
app = create_app()
