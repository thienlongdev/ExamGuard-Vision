/**
 * ExamGuard Vision — Hero Camera Component
 * Dominant visual element with MJPEG stream and corner-bracket tracking overlays.
 */

import { appState } from "../state.js";

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
              <span>CAM 01 (PHYSICAL)</span>
            </div>
            <div class="cam-chip" id="cam-res-chip">1280x720</div>
            <div class="cam-chip" id="cam-fps-chip">30.0 FPS</div>
            <div class="cam-chip" id="cam-tracks-chip">0 TRACKS</div>
          </div>

          <div class="camera-controls">
            <button class="cam-btn" id="btn-cam-refresh" title="Refresh Stream">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M23 4v6h-6"/><path d="M1 20v-6h6"/>
                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
              </svg>
            </button>
            <button class="cam-btn" id="btn-cam-fullscreen" title="Toggle Fullscreen">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/>
              </svg>
            </button>
          </div>
        </div>

        <div class="camera-feed-container" id="feed-container">
          <img class="camera-stream-img" id="camera-stream-img" src="/api/cameras/stream" alt="Physical Exam Camera Feed" />
          <div class="camera-overlays-layer" id="camera-overlays-layer"></div>

          <div class="camera-placeholder-state" id="camera-placeholder">
            <div class="camera-placeholder-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
              </svg>
            </div>
            <div class="camera-placeholder-text">
              <h3>Waiting for physical camera stream…</h3>
              <p>Connecting physical UVC webcam on ASUS A17 runtime pipeline</p>
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
      // Retry stream connection after 3s
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
      tracksEl.innerText = `${appState.activeTracks.length} TRACK${appState.activeTracks.length === 1 ? "" : "S"}`;
    }
  }

  renderTracks(tracks) {
    if (!this.overlaysLayer || !this.streamImg) return;
    this.overlaysLayer.innerHTML = "";

    const tracksEl = document.getElementById("cam-tracks-chip");
    if (tracksEl) {
      tracksEl.innerText = `${tracks.length} TRACK${tracks.length === 1 ? "" : "S"}`;
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
      // Pillarbox (bars on left/right)
      renderW = imgRect.height * srcAspect;
      padX = (imgRect.width - renderW) / 2;
    } else {
      // Letterbox (bars on top/bottom)
      renderH = imgRect.width / srcAspect;
      padY = (imgRect.height - renderH) / 2;
    }

    const scaleX = renderW / srcW;
    const scaleY = renderH / srcH;
    const offsetX = (imgRect.left - containerRect.left) + padX;
    const offsetY = (imgRect.top - containerRect.top) + padY;

    tracks.forEach((t) => {
      const bbox = t.bbox || [0, 0, 0, 0];
      const x1 = bbox[0] * scaleX + offsetX;
      const y1 = bbox[1] * scaleY + offsetY;
      const w = (bbox[2] - bbox[0]) * scaleX;
      const h = (bbox[3] - bbox[1]) * scaleY;

      if (w <= 0 || h <= 0) return;

      const boxEl = document.createElement("div");
      const riskClass = `risk-${(t.risk_level || "normal").toLowerCase()}`;
      boxEl.className = `track-box ${riskClass}`;
      boxEl.id = `track-box-${t.track_id}`;
      boxEl.style.left = `${Math.round(x1)}px`;
      boxEl.style.top = `${Math.round(y1)}px`;
      boxEl.style.width = `${Math.round(w)}px`;
      boxEl.style.height = `${Math.round(h)}px`;

      // Corner brackets
      boxEl.innerHTML = `
        <div class="track-box-corner tl"></div>
        <div class="track-box-corner tr"></div>
        <div class="track-box-corner bl"></div>
        <div class="track-box-corner br"></div>
        <div class="track-tag">
          <span>Student #${t.track_id}</span>
          ${t.posture && t.posture !== "N/A" ? `· <span>${t.posture}</span>` : ""}
          ${t.yaw_deg !== null && t.yaw_deg !== undefined ? `· <span>${t.yaw_deg}°</span>` : ""}
        </div>
      `;

      boxEl.addEventListener("mouseenter", () => {
        appState.setHighlightedTrack(t.track_id);
      });

      boxEl.addEventListener("mouseleave", () => {
        appState.setHighlightedTrack(null);
      });

      boxEl.addEventListener("click", () => {
        // Find latest event for this track
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
