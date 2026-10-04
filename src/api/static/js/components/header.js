/**
 * ExamGuard Vision — Header Component
 * Top navigation bar with branding, tab routing, physical camera state, GPU memory, WS status, and system clock.
 */

import { appState } from "../state.js";

export class HeaderComponent {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.clockInterval = null;
    this.init();
  }

  init() {
    this.render();
    this.startClock();
    this.bindEvents();

    appState.subscribe((type, payload) => {
      if (type === "VIEW_CHANGED") {
        this.updateNavActive(payload);
      } else if (type === "CAMERA_UPDATED" || type === "SYSTEM_STATUS_UPDATED") {
        this.updateBadges();
      } else if (type === "WS_STATUS_CHANGED") {
        this.setWsStatus(payload);
      } else if (type === "CURRENT_SESSION_UPDATED") {
        this.updateSessionBadge(payload);
      }
    });
  }

  render() {
    this.container.innerHTML = `
      <div class="brand-section">
        <div class="brand-icon-wrapper" title="ExamGuard Vision AI Engine">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            <circle cx="12" cy="11" r="3"/>
          </svg>
        </div>
        <div class="brand-text">
          <h1>
            ExamGuard Vision
            <span class="product-badge">Trung tâm giám sát</span>
          </h1>
          <p class="brand-subtitle">Giám sát phòng thi theo thời gian thực · Hệ thống bằng chứng AI</p>
        </div>
      </div>

      <nav class="header-nav" aria-label="Main Navigation">
        <button class="nav-tab-btn active" data-view="monitor" id="tab-monitor">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>
          </svg>
          GIÁM SÁT
        </button>
        <button class="nav-tab-btn" data-view="review" id="tab-review">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>
          </svg>
          DUYỆT SỰ KIỆN
        </button>
        <button class="nav-tab-btn" data-view="history" id="tab-history">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
          </svg>
          LỊCH SỬ
        </button>
        <button class="nav-tab-btn" data-view="system" id="tab-system">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
          </svg>
          HỆ THỐNG
        </button>
      </nav>

      <div class="header-telemetry">
        <div class="status-pill session-pill" id="header-session-pill" title="Phiên giám sát đang ghi nhận">
          <span style="color: var(--color-live);">●</span>
          <span id="header-session-text">Phiên hiện tại</span>
        </div>

        <div id="live-indicator" class="live-badge">
          <span class="live-pulse-dot"></span>
          <span id="live-status-text">TRỰC TIẾP · CAM 01</span>
        </div>

        <div class="status-pill" id="vram-pill">
          <span>GPU</span>
          <strong id="vram-text">— MB</strong>
        </div>

        <div class="status-pill ws-offline" id="ws-pill">
          <span style="color: var(--color-high)">○</span>
          <span id="ws-text">WS MẤT KẾT NỐI</span>
        </div>

        <div class="header-clock" id="header-clock">--:--:--</div>
      </div>
    `;
  }

  bindEvents() {
    this.container.querySelectorAll(".nav-tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const view = btn.dataset.view;
        appState.setView(view);
      });
    });
  }

  updateNavActive(viewName) {
    this.container.querySelectorAll(".nav-tab-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.view === viewName);
    });
  }

  updateBadges() {
    const cam = appState.cameraInfo;
    const sys = appState.systemStatus;

    const liveInd = document.getElementById("live-indicator");
    const liveText = document.getElementById("live-status-text");
    const vramText = document.getElementById("vram-text");

    if (cam && cam.streaming) {
      if (liveInd) {
        liveInd.className = "live-badge";
        liveText.innerText = "TRỰC TIẾP · CAM 01";
      }
    } else if (cam && cam.connected) {
      if (liveInd) {
        liveInd.className = "live-badge connecting";
        liveText.innerText = "CHỜ · CAM 01";
      }
    } else {
      if (liveInd) {
        liveInd.className = "live-badge offline";
        liveText.innerText = "NGOẠI TUYẾN";
      }
    }

    if (sys && sys.gpu_vram_allocated_mb !== undefined && vramText) {
      vramText.innerText = `${sys.gpu_vram_allocated_mb.toFixed(0)} MB`;
    }
  }

  updateSessionBadge(session) {
    const el = document.getElementById("header-session-text");
    const pill = document.getElementById("header-session-pill");
    if (!el || !session) return;
    const name = session.name || "Phiên giám sát";
    const room = session.room ? ` · ${session.room}` : "";
    el.innerText = `${name}${room}`;
    if (pill) pill.title = `ID: ${session.session_id} | Giám thị: ${session.invigilator_name || "Chưa phân công"}`;
  }

  setWsStatus(status) {
    const wsPill = document.getElementById("ws-pill");
    if (!wsPill) return;

    if (status === "connected") {
      wsPill.className = "status-pill ws-live";
      wsPill.innerHTML = `<span class="ws-pulse-dot" style="color: var(--color-live)">●</span> <span id="ws-text">WS ĐÃ KẾT NỐI</span>`;
    } else if (status === "connecting") {
      wsPill.className = "status-pill ws-reconnecting";
      wsPill.innerHTML = `<span style="color: var(--color-medium)">◌</span> <span id="ws-text">WS ĐANG KẾT NỐI</span>`;
    } else if (status === "reconnecting") {
      wsPill.className = "status-pill ws-reconnecting";
      wsPill.innerHTML = `<span style="color: var(--color-medium)">◌</span> <span id="ws-text">WS THỬ LẠI...</span>`;
    } else {
      wsPill.className = "status-pill ws-offline";
      wsPill.innerHTML = `<span style="color: var(--color-high)">○</span> <span id="ws-text">WS MẤT KẾT NỐI</span>`;
    }
  }

  startClock() {
    const clockEl = document.getElementById("header-clock");
    const tick = () => {
      const d = new Date();
      if (clockEl) {
        clockEl.innerText = d.toLocaleTimeString([], { hour12: false });
      }
    };
    tick();
    this.clockInterval = setInterval(tick, 1000);
  }
}
