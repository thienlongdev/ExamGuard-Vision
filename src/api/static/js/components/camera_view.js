import { appState } from "../state.js";
import { ApiClient } from "../api.js";
import { getEventNameVi, getPostureNameVi } from "../localization.js";

export class CameraViewComponent {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.streamImg = null;
    this.overlaysLayer = null;
    this.placeholderEl = null;
    this.isStreaming = false;
    this.isFullscreen = false;
    this.streamState = "IDLE"; // IDLE | CONNECTING | LIVE | RECONNECTING | ERROR
    // The MJPEG stream and track polling run only while an ACTIVE session is being monitored
    this.monitoringActive = false;
    this.canonicalCameraId = "webcam_0";
    this.cameraLabel = "CAM 01";
    this.reconnectTimer = null;
    this.healthMonitorTimer = null;
    this.trackPollTimer = null;
    this.resizeObserver = null;
    this.reconnectAttempts = 0;
    this.init();
  }

  init() {
    this.render();
    this.bindEvents();
    this.setupResizeObserver();
    this.startHealthMonitor();
    this.startTrackPolling();

    appState.subscribe((type, payload) => {
      if (type === "TRACKS_UPDATED") {
        this.renderTracks(payload);
      } else if (type === "CAMERA_UPDATED") {
        this.updateCameraMeta(payload);
      } else if (type === "TRACK_HIGHLIGHT_CHANGED") {
        this.highlightTrack(payload);
      } else if (type === "CURRENT_SESSION_UPDATED") {
        this.updateSessionState(payload);
      } else if (type === "MONITORING_ACTIVE_CHANGED") {
        this.setMonitoringActive(payload);
      }
    });

    this.updateSessionState(appState.currentSession);
    this.setMonitoringActive(appState.monitoringActive, true);
  }

  setMonitoringActive(active, force = false) {
    if (!force && active === this.monitoringActive) return;
    this.monitoringActive = !!active;
    const card = document.getElementById("camera-card");
    if (card) card.classList.toggle("camera-idle", !this.monitoringActive);

    if (this.monitoringActive) {
      this.reconnectAttempts = 0;
      this.setStreamState("CONNECTING");
      if (this.streamImg) this.streamImg.src = `/api/cameras/stream?t=${Date.now()}`;
      return;
    }

    // Stop immediately: drop the MJPEG connection, clear boxes, show the idle empty state
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.streamImg) {
      this.streamImg.removeAttribute("src");
    }
    if (this.overlaysLayer) this.overlaysLayer.innerHTML = "";
    const tracksEl = document.getElementById("cam-tracks-chip");
    if (tracksEl) tracksEl.innerText = "0 THÍ SINH";
    this.setStreamState("IDLE");
  }

  startTrackPolling() {
    if (this.trackPollTimer) clearInterval(this.trackPollTimer);
    this.trackPollTimer = setInterval(async () => {
      if (document.hidden) return;
      if (!this.monitoringActive) return;
      if (this.streamState === "ERROR") return;
      try {
        const tracks = await ApiClient.getCameraTracks(this.canonicalCameraId);
        if (tracks && this.monitoringActive) {
          appState.setActiveTracks(tracks);
        }
      } catch {
        // Silently continue on transient poll hiccup
      }
    }, 150);
  }

  setupResizeObserver() {
    if (window.ResizeObserver) {
      this.resizeObserver = new ResizeObserver(() => {
        if (appState.activeTracks && appState.activeTracks.length > 0) {
          this.renderTracks(appState.activeTracks);
        }
      });
      const feed = document.getElementById("feed-container");
      if (feed) this.resizeObserver.observe(feed);
      if (this.streamImg) this.resizeObserver.observe(this.streamImg);
    }

    window.addEventListener("resize", () => {
      if (appState.activeTracks && appState.activeTracks.length > 0) {
        this.renderTracks(appState.activeTracks);
      }
    });

    document.addEventListener("fullscreenchange", () => {
      setTimeout(() => {
        if (appState.activeTracks && appState.activeTracks.length > 0) {
          this.renderTracks(appState.activeTracks);
        }
      }, 50);
    });
  }

  setStreamState(state, message = null) {
    this.streamState = state;
    const titleEl = document.getElementById("camera-placeholder-title");
    const descEl = document.getElementById("camera-placeholder-desc");
    const idChip = document.getElementById("cam-id-chip");

    if (state === "IDLE") {
      this.isStreaming = false;
      if (this.placeholderEl) this.placeholderEl.style.display = "flex";
      if (titleEl) titleEl.innerText = "Chưa có phiên giám sát đang hoạt động";
      if (descEl) descEl.innerText = "Bắt đầu phiên để kích hoạt camera và AI.";
      if (idChip) {
        idChip.className = "cam-chip";
        idChip.innerHTML = `<span style="color: var(--text-dim, #64748b)">○</span><span>CAMERA · CHƯA HOẠT ĐỘNG</span>`;
      }
    } else if (state === "LIVE") {
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
    if (!this.streamImg || !this.monitoringActive) return;
    this.reconnectAttempts++;
    this.setStreamState("RECONNECTING", `Đang tự động kết nối lại luồng video (lần ${this.reconnectAttempts})…`);
    const ts = Date.now();
    this.streamImg.src = `/api/cameras/stream?t=${ts}`;
  }

  startHealthMonitor() {
    if (this.healthMonitorTimer) clearInterval(this.healthMonitorTimer);
    this.healthMonitorTimer = setInterval(() => {
      if (!this.monitoringActive) return;
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
      <div class="camera-viewport-card camera-idle" id="camera-card">
        <div class="camera-top-bar">
          <div class="camera-meta-chips">
            <div class="cam-chip" id="cam-id-chip" title="Camera ID gốc: ${this.canonicalCameraId}">
              <span style="color: var(--text-dim, #64748b)">○</span>
              <span>CAMERA · CHƯA HOẠT ĐỘNG</span>
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
          <img class="camera-stream-img" id="camera-stream-img" alt="Luồng camera phòng thi" />
          <div class="camera-overlays-layer" id="camera-overlays-layer"></div>

          <div class="camera-placeholder-state" id="camera-placeholder">
            <div class="camera-placeholder-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
              </svg>
            </div>
            <div class="camera-placeholder-text">
              <h3 id="camera-placeholder-title">Chưa có phiên giám sát đang hoạt động</h3>
              <p id="camera-placeholder-desc">Bắt đầu phiên để kích hoạt camera và AI.</p>
            </div>
          </div>

        </div>
      </div>
    `;

    this.streamImg = document.getElementById("camera-stream-img");
    this.overlaysLayer = document.getElementById("camera-overlays-layer");
    this.placeholderEl = document.getElementById("camera-placeholder");

    // Handle stream load & error events
    this.streamImg.onload = () => {
      if (this.streamImg.naturalWidth > 0) {
        this.setStreamState("LIVE");
      }
    };

    this.streamImg.onerror = () => {
      if (!this.monitoringActive) return;
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
    // Session start/resume is owned by the Session Gate (components/session_gate.js)
    this.currentSession = session;
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

  }

  updateCameraMeta(cam) {
    if (!cam) return;
    if (cam.camera_id) {
      this.canonicalCameraId = cam.camera_id;
    }
    if (cam.name) {
      this.cameraLabel = cam.name;
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

    if (!this.monitoringActive) return;
    if (cam.streaming === false && cam.connected === false && this.streamState !== "ERROR") {
      this.setStreamState("ERROR", "Webcam vật lý ngoại tuyến");
    } else if (cam.streaming && this.streamImg && this.streamImg.naturalWidth > 0 && this.streamState !== "LIVE") {
      this.setStreamState("LIVE");
    }
  }

  renderTracks(tracks) {
    if (!this.overlaysLayer) {
      this.overlaysLayer = document.getElementById("camera-overlays-layer");
    }
    if (!this.streamImg) {
      this.streamImg = document.getElementById("camera-stream-img");
    }
    if (!this.overlaysLayer || !this.streamImg) return;
    this.overlaysLayer.innerHTML = "";
    if (!this.monitoringActive) return;

    // Deduplicate tracks by track_id to avoid duplicate boxes
    const seenIds = new Set();
    const uniqueTracks = (tracks || []).filter((t) => {
      if (!t || t.track_id === undefined || t.track_id === null) return false;
      if (seenIds.has(t.track_id)) return false;
      seenIds.add(t.track_id);
      return true;
    });

    const tracksEl = document.getElementById("cam-tracks-chip");
    if (tracksEl) {
      tracksEl.innerText = `${uniqueTracks.length} THÍ SINH`;
    }

    if (uniqueTracks.length === 0) return;

    // Get rendered dimensions of image inside container
    const imgRect = this.streamImg.getBoundingClientRect();
    const containerRect = this.overlaysLayer.getBoundingClientRect();

    if (imgRect.width <= 0 || imgRect.height <= 0) return;

    // Source resolution reference (1280x720 = 16:9 standard, or actual frame dimensions)
    const srcW = (this.streamImg.naturalWidth && this.streamImg.naturalWidth > 0) ? this.streamImg.naturalWidth : 1280;
    const srcH = (this.streamImg.naturalHeight && this.streamImg.naturalHeight > 0) ? this.streamImg.naturalHeight : 720;
    const srcAspect = srcW / srcH;
    const containerAspect = imgRect.width / imgRect.height;

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

    uniqueTracks.forEach((t) => {
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
        if (
          t.risk_level === "HIGH" ||
          t.review_severity === "HIGH" ||
          t.severity === "HIGH" ||
          t.phone_status === "CONFIRMED" ||
          t.phone_status === "HIGH"
        ) {
          visualState = "HIGH_ALERT";
        } else if (
          t.risk_level === "MEDIUM" ||
          t.review_severity === "AMBER" ||
          t.severity === "AMBER" ||
          t.posture === "HEAD_REST_SLEEP" ||
          (activePrimaryEvent && (activePrimaryEvent.reviewStatus === "awaiting" || activePrimaryEvent.riskLevel === "MEDIUM"))
        ) {
          visualState = "ATTENTION";
        } else if (
          t.turn_candidate ||
          t.candidate ||
          t.review_severity === "LOW" ||
          t.severity === "LOW" ||
          t.posture === "TURN_HEAD_CLEAR" ||
          (yawAbs !== null && yawAbs >= 24.0)
        ) {
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
        } else if (t.turn_candidate || t.candidate) {
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
      const isDense = uniqueTracks.length > 5 || w < 120;
      const tagHtml = isDense
        ? `<div class="track-tag compact">
            <div class="track-tag-top">
              <span class="track-tag-id">#${t.track_id}</span>
              ${badgeHtml}
            </div>
          </div>`
        : `<div class="track-tag">
            <div class="track-tag-top">
              <span class="track-tag-id">Thí sinh #${t.track_id}</span>
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
