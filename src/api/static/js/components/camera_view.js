import { appState } from "../state.js";
import { getEventNameVi, getPostureNameVi } from "../localization.js";

export class CameraViewComponent {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.streamImg = null;
    this.overlaysLayer = null;
    this.placeholderEl = null;
    this.isStreaming = false;
    this.isFullscreen = false;
    this.init();
  }

  init() {
    this.render();
    this.bindEvents();

    appState.subscribe((type, payload) => {
      if (type === "TRACKS_UPDATED") {
        this.renderTracks(payload);
      } else if (type === "CAMERA_UPDATED") {
        this.updateCameraMeta(payload);
      } else if (type === "TRACK_HIGHLIGHT_CHANGED") {
        this.highlightTrack(payload);
      }
    });
  }

  render() {
    this.container.innerHTML = `
      <div class="camera-viewport-card" id="camera-card">
        <div class="camera-top-bar">
          <div class="camera-meta-chips">
            <div class="cam-chip highlight" id="cam-id-chip">
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
              <h3>Đang chờ luồng camera trực tiếp…</h3>
              <p>Đang kết nối webcam vật lý trên ASUS TUF Gaming A17</p>
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
      this.placeholderEl.style.display = "none";
      this.isStreaming = true;
    };

    this.streamImg.onerror = () => {
      this.placeholderEl.style.display = "flex";
      this.isStreaming = false;
      setTimeout(() => {
        if (!this.isStreaming && this.streamImg) {
          this.streamImg.src = `/api/cameras/stream?t=${Date.now()}`;
        }
      }, 3000);
    };
  }

  bindEvents() {
    const btnRefresh = document.getElementById("btn-cam-refresh");
    const btnFullscreen = document.getElementById("btn-cam-fullscreen");
    const card = document.getElementById("camera-card");

    if (btnRefresh) {
      btnRefresh.addEventListener("click", () => {
        if (this.streamImg) {
          this.streamImg.src = `/api/cameras/stream?t=${Date.now()}`;
        }
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
  }

  updateCameraMeta(cam) {
    if (!cam) return;
    const resEl = document.getElementById("cam-res-chip");
    const fpsEl = document.getElementById("cam-fps-chip");
    const tracksEl = document.getElementById("cam-tracks-chip");

    if (resEl) {
      resEl.innerText = cam.observed_resolution || cam.configured_resolution || "1280x720";
    }
    if (fpsEl) {
      fpsEl.innerText = cam.observed_capture_fps ? `${cam.observed_capture_fps.toFixed(1)} FPS` : "30.0 FPS";
    }
    if (tracksEl) {
      tracksEl.innerText = `${appState.activeTracks.length} THÍ SINH`;
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

      // Determine 3-state per-track visual severity
      const trackEvents = activeEventsByTrack.get(t.track_id) || [];
      let visualState = "SAFE"; // SAFE | ATTENTION | HIGH_ALERT
      let activePrimaryEvent = null;

      // Check active events first
      for (const ev of trackEvents) {
        if (ev.riskLevel === "HIGH") {
          visualState = "HIGH_ALERT";
          activePrimaryEvent = ev;
          break;
        } else if (ev.riskLevel === "MEDIUM" && visualState !== "HIGH_ALERT") {
          visualState = "ATTENTION";
          activePrimaryEvent = ev;
        }
      }

      // Check telemetry cues if no active event escalated
      if (visualState === "SAFE") {
        if (t.risk_level === "HIGH") {
          visualState = "HIGH_ALERT";
        } else if (
          t.risk_level === "MEDIUM" ||
          t.posture === "TURN_HEAD_CLEAR" ||
          t.posture === "HEAD_REST_SLEEP" ||
          (t.yaw_deg !== null && t.yaw_deg !== undefined && Math.abs(t.yaw_deg) >= 28.0)
        ) {
          visualState = "ATTENTION";
        }
      }

      // Badge config and cue texts
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
          const yawExtra = (t.yaw_deg !== null && t.yaw_deg !== undefined && activePrimaryEvent.canonicalType.includes("LATERAL"))
            ? ` • ${t.yaw_deg}°`
            : "";
          bottomText = `${activePrimaryEvent.displayName}${yawExtra}`;
        } else if (t.posture === "TURN_HEAD_CLEAR" || (t.yaw_deg !== null && Math.abs(t.yaw_deg) >= 25.0)) {
          bottomText = `Quay đầu • ${t.yaw_deg !== null ? `${t.yaw_deg}°` : "rõ"}`;
        } else if (t.posture === "HEAD_REST_SLEEP") {
          bottomText = "Gục đầu";
        } else {
          bottomText = "Cần chú ý";
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

      // Corner brackets + two-line label
      boxEl.innerHTML = `
        <div class="track-box-corner tl"></div>
        <div class="track-box-corner tr"></div>
        <div class="track-box-corner bl"></div>
        <div class="track-box-corner br"></div>
        <div class="track-tag">
          <div class="track-tag-top">
            <span class="track-tag-id">THÍ SINH #${t.track_id}</span>
            ${badgeHtml}
          </div>
          <div class="track-tag-bottom">
            <span>${bottomText}</span>
          </div>
        </div>
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
