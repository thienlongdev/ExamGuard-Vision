import { appState } from "../state.js";
import { ApiClient } from "../api.js";
import { getEventNameVi, getPostureNameVi } from "../localization.js";

export class CameraViewComponent {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.streamImg = null;
    this.overlaysLayer = null;
    this.placeholderEl = null;
    this.sessionStartOverlay = null;
    this.isStreaming = false;
    this.isFullscreen = false;
    this.streamState = "CONNECTING"; // CONNECTING | LIVE | RECONNECTING | ERROR
    this.canonicalCameraId = "webcam_0";
    this.cameraLabel = "CAM 01";
    this.reconnectTimer = null;
    this.healthMonitorTimer = null;
    this.reconnectAttempts = 0;
    this.init();
  }

  init() {
    this.render();
    this.bindEvents();
    this.startHealthMonitor();

    appState.subscribe((type, payload) => {
      if (type === "TRACKS_UPDATED") {
        this.renderTracks(payload);
      } else if (type === "CAMERA_UPDATED") {
        this.updateCameraMeta(payload);
      } else if (type === "TRACK_HIGHLIGHT_CHANGED") {
        this.highlightTrack(payload);
      } else if (type === "CURRENT_SESSION_UPDATED") {
        this.updateSessionState(payload);
      }
    });

    this.updateSessionState(appState.currentSession);
  }

  setStreamState(state, message = null) {
    this.streamState = state;
    const titleEl = document.getElementById("camera-placeholder-title");
    const descEl = document.getElementById("camera-placeholder-desc");
    const idChip = document.getElementById("cam-id-chip");

    if (state === "LIVE") {
      this.isStreaming = true;
      this.reconnectAttempts = 0;
      if (this.placeholderEl) this.placeholderEl.style.display = "none";
      if (idChip) {
        idChip.className = "cam-chip highlight";
        idChip.innerHTML = `<span style="color: var(--color-live, #10b981)">●</span><span>${this.cameraLabel} (TRỰC TIẾP)</span>`;
        idChip.title = `Camera ID gốc: ${this.canonicalCameraId}`;
      }
    } else if (state === "CONNECTING") {
      this.isStreaming = false;
      if (this.placeholderEl) this.placeholderEl.style.display = "flex";
      if (titleEl) titleEl.innerText = "Đang kết nối camera trực tiếp…";
      if (descEl) descEl.innerText = message || "Đang kết nối webcam vật lý trên ASUS TUF Gaming A17";
      if (idChip) {
        idChip.className = "cam-chip";
        idChip.innerHTML = `<span style="color: var(--color-warning, #f59e0b)">●</span><span>${this.cameraLabel} (ĐANG KẾT NỐI)</span>`;
      }
    } else if (state === "RECONNECTING") {
      this.isStreaming = false;
      if (this.placeholderEl) this.placeholderEl.style.display = "flex";
      if (titleEl) titleEl.innerText = "Không thể hiển thị luồng trực tiếp — đang kết nối lại";
      if (descEl) descEl.innerText = message || `Đang tự động thử lại luồng video (lần ${this.reconnectAttempts})…`;
      if (idChip) {
        idChip.className = "cam-chip";
        idChip.innerHTML = `<span style="color: var(--color-danger, #ef4444)">●</span><span>${this.cameraLabel} (KẾT NỐI LẠI)</span>`;
      }
    } else if (state === "ERROR") {
      this.isStreaming = false;
      if (this.placeholderEl) this.placeholderEl.style.display = "flex";
      if (titleEl) titleEl.innerText = "Mất kết nối camera";
      if (descEl) descEl.innerText = message || "Không tìm thấy thiết bị webcam khả dụng";
      if (idChip) {
        idChip.className = "cam-chip";
        idChip.innerHTML = `<span style="color: var(--text-dim, #64748b)">○</span><span>${this.cameraLabel} (NGOẠI TUYẾN)</span>`;
      }
    }
  }

  reconnectStream(reason = "auto") {
    if (!this.streamImg) return;
    this.reconnectAttempts++;
    this.setStreamState("RECONNECTING", `Đang tự động kết nối lại luồng video (lần ${this.reconnectAttempts})…`);
    const ts = Date.now();
    this.streamImg.src = `/api/cameras/stream?t=${ts}`;
  }

  startHealthMonitor() {
    if (this.healthMonitorTimer) clearInterval(this.healthMonitorTimer);
    this.healthMonitorTimer = setInterval(() => {
      // 1. If stream image has decoded pixels in DOM, transition to LIVE
      if (this.streamImg && this.streamImg.naturalWidth > 0) {
        if (this.streamState !== "LIVE") {
          this.setStreamState("LIVE");
        }
        return;
      }

      // 2. Check if backend reports capture is active, but DOM stream is not rendering
      const cam = appState.cameraInfo;
      const isBackendActive = cam && (cam.streaming || cam.connected || cam.is_active);

      if (isBackendActive) {
        if (this.streamState !== "RECONNECTING") {
          this.setStreamState("RECONNECTING", "Không thể hiển thị luồng trực tiếp — đang kết nối lại");
        }
        this.reconnectStream("backend_active_dom_empty");
      }
    }, 1500);
  }

  render() {
    this.container.innerHTML = `
      <div class="camera-viewport-card" id="camera-card">
        <div class="camera-top-bar">
          <div class="camera-meta-chips">
            <div class="cam-chip highlight" id="cam-id-chip" title="Camera ID gốc: ${this.canonicalCameraId}">
              <span style="color: var(--color-live)">●</span>
              <span>CAM 01 (TRỰC TIẾP)</span>
            </div>
            <div class="cam-chip" id="cam-res-chip">1280x720</div>
            <div class="cam-chip" id="cam-fps-chip">30.0 FPS</div>
            <div class="cam-chip" id="cam-tracks-chip">0 THÍ SINH</div>
          </div>

          <div class="camera-controls">
            <button class="cam-btn" id="btn-cam-refresh" title="Làm mới luồng video">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M23 4v6h-6"/><path d="M1 20v-6h6"/>
                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
              </svg>
            </button>
            <button class="cam-btn" id="btn-cam-fullscreen" title="Toàn màn hình">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/>
              </svg>
            </button>
          </div>
        </div>

        <div class="camera-feed-container" id="feed-container">
          <img class="camera-stream-img" id="camera-stream-img" src="/api/cameras/stream" alt="Luồng camera phòng thi" />
          <div class="camera-overlays-layer" id="camera-overlays-layer"></div>

          <div class="camera-placeholder-state" id="camera-placeholder">
            <div class="camera-placeholder-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
              </svg>
            </div>
            <div class="camera-placeholder-text">
              <h3 id="camera-placeholder-title">Đang kết nối camera trực tiếp…</h3>
              <p id="camera-placeholder-desc">Đang kết nối webcam vật lý trên ASUS TUF Gaming A17</p>
            </div>
          </div>

          <!-- Start Monitoring Session Overlay (Workstream 4) -->
          <div class="camera-no-session-overlay" id="camera-no-session-overlay" style="display: none; position: absolute; inset: 0; background: rgba(15, 23, 42, 0.88); backdrop-filter: blur(8px); z-index: 50; align-items: center; justify-content: center; padding: 20px;">
            <div style="background: #0f172a; border: 1px solid #1e293b; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7); border-radius: 12px; width: 100%; max-width: 480px; padding: 24px;">
              <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                <div style="width: 40px; height: 40px; border-radius: 8px; background: rgba(2, 132, 199, 0.15); border: 1px solid rgba(2, 132, 199, 0.3); display: flex; align-items: center; justify-content: center; color: #38bdf8;">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/>
                  </svg>
                </div>
                <div>
                  <h3 style="margin: 0; font-size: 1.1rem; font-weight: 700; color: #f8fafc;">Bắt đầu phiên giám sát</h3>
                  <p style="margin: 2px 0 0 0; font-size: 0.8rem; color: #94a3b8;">Thiết lập phòng thi để kích hoạt AI giám sát và lưu trữ bằng chứng</p>
                </div>
              </div>

              <form id="cam-start-session-form" style="display: flex; flex-direction: column; gap: 12px;">
                <div>
                  <label style="display: block; font-size: 0.8rem; font-weight: 500; color: #cbd5e1; margin-bottom: 4px;">Tên kỳ thi / Phiên (*):</label>
                  <input type="text" id="cam-input-session-name" required placeholder="Ví dụ: Thi Cuối Kỳ - Môn Cơ sở dữ liệu" style="width: 100%; background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 8px 12px; color: #f8fafc; font-size: 0.85rem; box-sizing: border-box;" />
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                  <div>
                    <label style="display: block; font-size: 0.8rem; font-weight: 500; color: #cbd5e1; margin-bottom: 4px;">Phòng thi (*):</label>
                    <input type="text" id="cam-input-room" required value="Phòng A203" style="width: 100%; background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 8px 12px; color: #f8fafc; font-size: 0.85rem; box-sizing: border-box;" />
                  </div>
                  <div>
                    <label style="display: block; font-size: 0.8rem; font-weight: 500; color: #cbd5e1; margin-bottom: 4px;">Lớp / Nhóm thi:</label>
                    <input type="text" id="cam-input-class" placeholder="20DTH01" style="width: 100%; background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 8px 12px; color: #f8fafc; font-size: 0.85rem; box-sizing: border-box;" />
                  </div>
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                  <div>
                    <label style="display: block; font-size: 0.8rem; font-weight: 500; color: #cbd5e1; margin-bottom: 4px;">Mã môn / Môn thi:</label>
                    <input type="text" id="cam-input-subject" placeholder="CSDL101" style="width: 100%; background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 8px 12px; color: #f8fafc; font-size: 0.85rem; box-sizing: border-box;" />
                  </div>
                  <div>
                    <label style="display: block; font-size: 0.8rem; font-weight: 500; color: #cbd5e1; margin-bottom: 4px;">Ghi chú phòng thi:</label>
                    <input type="text" id="cam-input-notes" placeholder="Không bắt buộc" style="width: 100%; background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 8px 12px; color: #f8fafc; font-size: 0.85rem; box-sizing: border-box;" />
                  </div>
                </div>

                <div style="margin-top: 10px; display: flex; justify-content: flex-end;">
                  <button type="submit" id="btn-cam-submit-session" style="background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%); color: white; border: none; border-radius: 6px; padding: 9px 20px; font-size: 0.85rem; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 8px; box-shadow: 0 4px 6px -1px rgba(2, 132, 199, 0.4);">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <polygon points="5 3 19 12 5 21 5 3"/>
                    </svg>
                    <span>Bắt đầu giám sát</span>
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>
    `;

    this.streamImg = document.getElementById("camera-stream-img");
    this.overlaysLayer = document.getElementById("camera-overlays-layer");
    this.placeholderEl = document.getElementById("camera-placeholder");
    this.sessionStartOverlay = document.getElementById("camera-no-session-overlay");

    // Handle stream load & error events
    this.streamImg.onload = () => {
      if (this.streamImg.naturalWidth > 0) {
        this.setStreamState("LIVE");
      }
    };

    this.streamImg.onerror = () => {
      const cam = appState.cameraInfo;
      const isBackendActive = cam && (cam.streaming || cam.connected);
      if (isBackendActive) {
        this.setStreamState("RECONNECTING", "Không thể hiển thị luồng trực tiếp — đang kết nối lại");
      } else {
        this.setStreamState("ERROR", "Lỗi luồng webcam vật lý");
      }
      if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
      this.reconnectTimer = setTimeout(() => {
        this.reconnectStream("onerror");
      }, 1500);
    };
  }

  updateSessionState(session) {
    if (!this.sessionStartOverlay) return;
    if (session && session.status === "ACTIVE") {
      this.sessionStartOverlay.style.display = "none";
    } else {
      this.sessionStartOverlay.style.display = "flex";
    }
  }

  bindEvents() {
    const btnRefresh = document.getElementById("btn-cam-refresh");
    const btnFullscreen = document.getElementById("btn-cam-fullscreen");
    const card = document.getElementById("camera-card");

    if (btnRefresh) {
      btnRefresh.addEventListener("click", () => {
        this.reconnectAttempts = 0;
        this.reconnectStream("manual_refresh");
      });
    }

    if (btnFullscreen) {
      btnFullscreen.addEventListener("click", () => {
        if (!document.fullscreenElement) {
          card.requestFullscreen?.().catch((err) => console.debug("Fullscreen error:", err));
        } else {
          document.exitFullscreen?.().catch((err) => console.debug("Exit fullscreen error:", err));
        }
      });
    }

    // Reconnect on tab visibility change or window focus
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) {
        if (!this.streamImg || this.streamImg.naturalWidth === 0) {
          this.reconnectStream("visibility_change");
        }
      }
    });

    window.addEventListener("focus", () => {
      if (!this.streamImg || this.streamImg.naturalWidth === 0) {
        this.reconnectStream("window_focus");
      }
    });

    const startForm = document.getElementById("cam-start-session-form");
    if (startForm) {
      startForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const name = document.getElementById("cam-input-session-name")?.value.trim();
        const room = document.getElementById("cam-input-room")?.value.trim();
        const class_name = document.getElementById("cam-input-class")?.value.trim() || null;
        const subject_code = document.getElementById("cam-input-subject")?.value.trim() || null;
        const notes = document.getElementById("cam-input-notes")?.value.trim() || null;

        if (!name || !room) {
          alert("Vui lòng điền tên phiên và phòng thi.");
          return;
        }

        const submitBtn = document.getElementById("btn-cam-submit-session");
        if (submitBtn) {
          submitBtn.disabled = true;
          submitBtn.innerHTML = `<span>Đang khởi tạo…</span>`;
        }

        try {
          const newSess = await ApiClient.startSession({
            name,
            room,
            class_name,
            subject_code,
            notes,
          });
          appState.resetForNewSession(newSess);
        } catch (err) {
          alert(`Không thể bắt đầu phiên: ${err.message}`);
          if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = `<span>Bắt đầu giám sát</span>`;
          }
        }
      });
    }
  }

  updateCameraMeta(cam) {
    if (!cam) return;
    if (cam.camera_id) {
      this.canonicalCameraId = cam.camera_id;
    }
    const resEl = document.getElementById("cam-res-chip");
    const fpsEl = document.getElementById("cam-fps-chip");
    const tracksEl = document.getElementById("cam-tracks-chip");
    const idChip = document.getElementById("cam-id-chip");

    if (resEl) {
      resEl.innerText = cam.observed_resolution || cam.configured_resolution || "1280x720";
    }
    if (fpsEl) {
      const fpsVal = cam.observed_capture_fps || cam.fps || 30.0;
      fpsEl.innerText = `${fpsVal.toFixed(1)} FPS`;
    }
    if (tracksEl) {
      tracksEl.innerText = `${appState.activeTracks.length} THÍ SINH`;
    }
    if (idChip) {
      idChip.title = `Camera ID gốc: ${this.canonicalCameraId}`;
    }

    if (cam.streaming === false && cam.connected === false && this.streamState !== "ERROR") {
      this.setStreamState("ERROR", "Webcam vật lý ngoại tuyến");
    } else if (cam.streaming && this.streamImg && this.streamImg.naturalWidth > 0 && this.streamState !== "LIVE") {
      this.setStreamState("LIVE");
    }
  }

  renderTracks(tracks) {
    if (!this.overlaysLayer || !this.streamImg) return;
    this.overlaysLayer.innerHTML = "";

    const tracksEl = document.getElementById("cam-tracks-chip");
    if (tracksEl) {
      tracksEl.innerText = `${tracks.length} THÍ SINH`;
    }

    if (!tracks || tracks.length === 0) return;

    // Get rendered dimensions of image inside container
    const imgRect = this.streamImg.getBoundingClientRect();
    const containerRect = this.overlaysLayer.getBoundingClientRect();

    // Source resolution reference (1280x720 = 16:9)
    const srcW = 1280;
    const srcH = 720;
    const srcAspect = srcW / srcH;
    const containerAspect = imgRect.width / (imgRect.height || 1);

    let renderW = imgRect.width;
    let renderH = imgRect.height;
    let padX = 0;
    let padY = 0;

    if (containerAspect > srcAspect) {
      renderW = imgRect.height * srcAspect;
      padX = (imgRect.width - renderW) / 2;
    } else {
      renderH = imgRect.width / srcAspect;
      padY = (imgRect.height - renderH) / 2;
    }

    const scaleX = renderW / srcW;
    const scaleY = renderH / srcH;
    const offsetX = (imgRect.left - containerRect.left) + padX;
    const offsetY = (imgRect.top - containerRect.top) + padY;

    // Gather active events by track ID
    const activeEventsByTrack = new Map();
    for (const ev of appState.events.values()) {
      if (ev.lifecycle === "open" || ev.lifecycle === "active") {
        if (!activeEventsByTrack.has(ev.trackId)) activeEventsByTrack.set(ev.trackId, []);
        activeEventsByTrack.get(ev.trackId).push(ev);
      }
    }

    tracks.forEach((t) => {
      const bbox = t.bbox || [0, 0, 0, 0];
      const x1 = bbox[0] * scaleX + offsetX;
      const y1 = bbox[1] * scaleY + offsetY;
      const w = (bbox[2] - bbox[0]) * scaleX;
      const h = (bbox[3] - bbox[1]) * scaleY;

      if (w <= 0 || h <= 0) return;

      // Determine 4-state per-track visual severity: SAFE (Green) | CANDIDATE (Blue) | ATTENTION (Amber) | HIGH_ALERT (Red)
      const trackEvents = activeEventsByTrack.get(t.track_id) || [];
      let visualState = "SAFE"; // SAFE | CANDIDATE | ATTENTION | HIGH_ALERT
      let activePrimaryEvent = null;

      // 1. Check active events first
      for (const ev of trackEvents) {
        if (ev.riskLevel === "HIGH") {
          visualState = "HIGH_ALERT";
          activePrimaryEvent = ev;
          break;
        } else if (ev.riskLevel === "MEDIUM" && visualState !== "HIGH_ALERT") {
          visualState = "ATTENTION";
          activePrimaryEvent = ev;
        } else if (ev.riskLevel === "LOW" && visualState === "SAFE") {
          if (ev.reviewStatus === "awaiting") {
            visualState = "ATTENTION";
          } else {
            visualState = "CANDIDATE";
          }
          activePrimaryEvent = ev;
        }
      }

      // 2. Check telemetry cues / candidate flags if not escalated by confirmed event
      const yawAbs = (t.yaw_deg !== null && t.yaw_deg !== undefined) ? Math.abs(t.yaw_deg) : null;
      if (visualState === "SAFE" || visualState === "CANDIDATE") {
        if (t.risk_level === "HIGH") {
          visualState = "HIGH_ALERT";
        } else if (
          t.risk_level === "MEDIUM" ||
          t.posture === "HEAD_REST_SLEEP" ||
          (activePrimaryEvent && (activePrimaryEvent.reviewStatus === "awaiting" || activePrimaryEvent.riskLevel === "MEDIUM"))
        ) {
          visualState = "ATTENTION";
        } else if (
          t.turn_candidate ||
          t.posture === "TURN_HEAD_CLEAR" ||
          (yawAbs !== null && yawAbs >= 24.0)
        ) {
          // Clear lateral head turn: leave GREEN immediately!
          // If turn is already confirmed sustained or awaiting review: AMBER; else preliminary observable: BLUE
          if (activePrimaryEvent && (activePrimaryEvent.duration >= 0.85 || activePrimaryEvent.reviewStatus === "awaiting")) {
            visualState = "ATTENTION";
          } else {
            visualState = "CANDIDATE";
          }
        }
      }

      // 3. Badge config and cue texts
      let boxClass = "track-box state-safe";
      let badgeHtml = `<span class="track-tag-badge safe">✓ BÌNH THƯỜNG</span>`;
      let bottomText = "Tư thế bình thường";

      if (visualState === "HIGH_ALERT") {
        boxClass = "track-box state-high";
        badgeHtml = `<span class="track-tag-badge high">⚠ CẢNH BÁO CAO</span>`;
        bottomText = activePrimaryEvent ? activePrimaryEvent.displayName : "Phát hiện điện thoại";
      } else if (visualState === "ATTENTION") {
        boxClass = "track-box state-attention";
        badgeHtml = `<span class="track-tag-badge attention">! CẦN CHÚ Ý</span>`;
        if (activePrimaryEvent) {
          const yawExtra = (yawAbs !== null && activePrimaryEvent.canonicalType && activePrimaryEvent.canonicalType.includes("LATERAL"))
            ? ` • ${t.yaw_deg}°`
            : "";
          bottomText = `${activePrimaryEvent.displayName}${yawExtra}`;
        } else if (t.posture === "TURN_HEAD_CLEAR" || (yawAbs !== null && yawAbs >= 24.0)) {
          bottomText = `Quay đầu • ${t.yaw_deg !== null ? `${t.yaw_deg}°` : "rõ"}`;
        } else if (t.posture === "HEAD_REST_SLEEP") {
          bottomText = "Gục đầu";
        } else {
          bottomText = "Cần chú ý";
        }
      } else if (visualState === "CANDIDATE") {
        boxClass = "track-box state-candidate";
        badgeHtml = `<span class="track-tag-badge candidate">👁 ĐANG QUAY ĐẦU</span>`;
        if (yawAbs !== null) {
          bottomText = `Góc quay • ${t.yaw_deg}°`;
        } else if (t.turn_candidate) {
          bottomText = "Đang quay đầu";
        } else {
          bottomText = "Quan sát";
        }
      } else {
        // SAFE / NORMAL
        if (t.posture === "NORMAL_READ_WRITE") {
          bottomText = "Đọc / viết";
        } else if (t.posture === "NORMAL_UPRIGHT") {
          bottomText = "Tư thế bình thường";
        } else {
          bottomText = getPostureNameVi(t.posture);
        }
      }

      const boxEl = document.createElement("div");
      boxEl.className = boxClass;
      boxEl.id = `track-box-${t.track_id}`;
      boxEl.style.left = `${Math.round(x1)}px`;
      boxEl.style.top = `${Math.round(y1)}px`;
      boxEl.style.width = `${Math.round(w)}px`;
      boxEl.style.height = `${Math.round(h)}px`;

      // Check for dense multi-student scene to prevent label collision
      const isDense = tracks.length > 5 || w < 120;
      const tagHtml = isDense
        ? `<div class="track-tag compact">
            <div class="track-tag-top">
              <span class="track-tag-id">#${t.track_id}</span>
              ${badgeHtml}
            </div>
          </div>`
        : `<div class="track-tag">
            <div class="track-tag-top">
              <span class="track-tag-id">THÍ SINH #${t.track_id}</span>
              ${badgeHtml}
            </div>
            <div class="track-tag-bottom">
              <span>${bottomText}</span>
            </div>
          </div>`;

      // Corner brackets + adaptive label
      boxEl.innerHTML = `
        <div class="track-box-corner tl"></div>
        <div class="track-box-corner tr"></div>
        <div class="track-box-corner bl"></div>
        <div class="track-box-corner br"></div>
        ${tagHtml}
      `;

      boxEl.addEventListener("mouseenter", () => {
        appState.setHighlightedTrack(t.track_id);
      });

      boxEl.addEventListener("mouseleave", () => {
        appState.setHighlightedTrack(null);
      });

      boxEl.addEventListener("click", () => {
        const evs = Array.from(appState.events.values()).filter((e) => e.trackId === t.track_id);
        if (evs.length > 0) {
          appState.selectEvent(evs[0].eventId);
        }
      });

      this.overlaysLayer.appendChild(boxEl);
    });

    if (appState.highlightedTrackId) {
      this.highlightTrack(appState.highlightedTrackId);
    }
  }

  highlightTrack(trackId) {
    if (!this.overlaysLayer) return;
    this.overlaysLayer.querySelectorAll(".track-box").forEach((box) => {
      box.classList.toggle("highlighted", box.id === `track-box-${trackId}`);
    });
  }
}
