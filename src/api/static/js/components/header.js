/**
 * ExamGuard Vision — Header Component
 * Top navigation bar with branding, tab routing, physical camera state, GPU memory, WS status, and system clock.
 * Features an actionable session control dropdown and safe logout confirmation (Workstreams 5, 6, 7, 9).
 */

import { appState } from "../state.js";
import { ApiClient } from "../api.js";

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

      <div class="header-telemetry" style="position: relative;">
        <!-- Actionable Session Control Pill (Workstream 5) -->
        <div class="status-pill session-pill" id="header-session-pill" style="cursor: pointer; user-select: none; display: flex; align-items: center; gap: 6px;" title="Nhấn để quản lý phiên giám sát">
          <span style="color: var(--color-medium);" id="header-session-dot">○</span>
          <span id="header-session-text">Chưa có phiên</span>
          <span style="font-size: 0.68rem; opacity: 0.7;">▾</span>
        </div>

        <!-- Session Dropdown Menu -->
        <div id="session-dropdown-menu" style="display: none; position: absolute; top: calc(100% + 8px); left: 0; background: rgba(15, 23, 42, 0.96); border: 1px solid rgba(148, 163, 184, 0.2); backdrop-filter: blur(12px); border-radius: 8px; box-shadow: 0 15px 30px rgba(0,0,0,0.5); z-index: 1000; min-width: 200px; padding: 6px 0;">
          <div id="menu-view-session" class="session-menu-item" style="padding: 8px 16px; font-size: 0.8rem; color: #f1f5f9; cursor: pointer; display: flex; align-items: center; gap: 8px;">
            <span>ℹ</span> <span>Xem thông tin phiên</span>
          </div>
          <div id="menu-edit-session" class="session-menu-item" style="padding: 8px 16px; font-size: 0.8rem; color: #f1f5f9; cursor: pointer; display: flex; align-items: center; gap: 8px;">
            <span>✏</span> <span>Chỉnh sửa thông tin</span>
          </div>
          <div style="height: 1px; background: rgba(148, 163, 184, 0.15); margin: 4px 0;"></div>
          <div id="menu-new-session" class="session-menu-item" style="padding: 8px 16px; font-size: 0.8rem; color: #38bdf8; cursor: pointer; display: flex; align-items: center; gap: 8px;">
            <span>➕</span> <span>Bắt đầu phiên mới</span>
          </div>
          <div id="menu-end-session" class="session-menu-item" style="padding: 8px 16px; font-size: 0.8rem; color: #f87171; cursor: pointer; display: flex; align-items: center; gap: 8px;">
            <span>⏹</span> <span>Kết thúc phiên này</span>
          </div>
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

        <div class="user-profile-badge" id="header-user-badge" style="display: flex; align-items: center; gap: 8px; padding-left: 8px; border-left: 1px solid var(--color-border, #1e293b);">
          <div class="user-info" style="display: flex; flex-direction: column; text-align: right;">
            <span id="header-user-name" style="font-size: 0.8rem; font-weight: 600; color: #f1f5f9;">—</span>
            <span id="header-user-role" style="font-size: 0.7rem; color: #10b981; font-weight: 500;">—</span>
          </div>
          <button id="header-logout-btn" title="Đăng xuất khỏi hệ thống" style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.35); color: #f87171; border-radius: 4px; padding: 4px 8px; font-size: 0.75rem; cursor: pointer; display: flex; align-items: center; gap: 4px;">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path>
              <polyline points="16 17 21 12 16 7"></polyline>
              <line x1="21" y1="12" x2="9" y2="12"></line>
            </svg>
            <span>Đăng xuất</span>
          </button>
        </div>
      </div>

      <!-- Action Dialogs Container -->
      <div id="header-modals-host"></div>
    `;
  }

  bindEvents() {
    this.container.querySelectorAll(".nav-tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const view = btn.dataset.view;
        appState.setView(view);
      });
    });

    // Session dropdown toggle
    const sessionPill = document.getElementById("header-session-pill");
    const dropdown = document.getElementById("session-dropdown-menu");
    if (sessionPill && dropdown) {
      sessionPill.addEventListener("click", (e) => {
        e.stopPropagation();
        dropdown.style.display = dropdown.style.display === "none" ? "block" : "none";
      });

      document.addEventListener("click", (e) => {
        if (!sessionPill.contains(e.target) && !dropdown.contains(e.target)) {
          dropdown.style.display = "none";
        }
      });
    }

    // Dropdown menu actions
    document.getElementById("menu-view-session")?.addEventListener("click", () => {
      if (dropdown) dropdown.style.display = "none";
      this.showSessionInfoModal();
    });

    document.getElementById("menu-edit-session")?.addEventListener("click", () => {
      if (dropdown) dropdown.style.display = "none";
      this.showSessionEditModal();
    });

    document.getElementById("menu-end-session")?.addEventListener("click", () => {
      if (dropdown) dropdown.style.display = "none";
      this.showEndSessionConfirm();
    });

    document.getElementById("menu-new-session")?.addEventListener("click", () => {
      if (dropdown) dropdown.style.display = "none";
      this.showNewSessionPrompt();
    });

    // Safe Logout button (Workstream 9)
    const logoutBtn = this.container.querySelector("#header-logout-btn");
    if (logoutBtn) {
      logoutBtn.addEventListener("click", () => {
        this.handleSafeLogout();
      });
    }

    this.loadUserProfile();
  }

  async loadUserProfile() {
    try {
      const auth = await ApiClient.initAuth();
      if (auth && auth.user) {
        const nameEl = document.getElementById("header-user-name");
        const roleEl = document.getElementById("header-user-role");
        if (nameEl) nameEl.innerText = auth.user.display_name || auth.user.username;
        if (roleEl) roleEl.innerText = auth.role_display || auth.user.role;
      }
    } catch {}
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
    const dot = document.getElementById("header-session-dot");
    const pill = document.getElementById("header-session-pill");
    if (!el || !dot) return;

    if (session && session.status === "ACTIVE") {
      const name = session.name || "Phiên giám sát";
      const room = session.room ? ` · ${session.room}` : "";
      el.innerText = `${name}${room}`;
      dot.style.color = "var(--color-live)";
      dot.innerText = "●";
      if (pill) pill.title = `ID: ${session.session_id} | Giám thị: ${session.invigilator_name || "Chưa phân công"}`;
    } else {
      el.innerText = "Chưa có phiên";
      dot.style.color = "var(--color-medium)";
      dot.innerText = "○";
      if (pill) pill.title = "Nhấn để bắt đầu phiên giám sát mới";
    }
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

  // --- Modal Helpers ---

  showSessionInfoModal() {
    const sess = appState.currentSession;
    if (!sess) {
      alert("Hiện tại chưa có phiên giám sát nào đang hoạt động.");
      return;
    }
    const host = document.getElementById("header-modals-host");
    if (!host) return;

    host.innerHTML = `
      <div class="modal-backdrop" style="position: fixed; inset: 0; background: rgba(0,0,0,0.65); z-index: 2000; display: flex; align-items: center; justify-content: center;">
        <div class="modal-card" style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; width: 440px; padding: 20px; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.6);">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <h3 style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin: 0;">Thông tin phiên giám sát</h3>
            <button id="modal-close-btn" style="background: transparent; border: none; color: #94a3b8; font-size: 1.2rem; cursor: pointer;">✕</button>
          </div>
          <div style="display: flex; flex-direction: column; gap: 10px; font-size: 0.85rem; color: #cbd5e1;">
            <div><span style="color: #64748b;">Mã phiên:</span> <strong style="font-family: var(--font-mono); color: #38bdf8;">${sess.session_id}</strong></div>
            <div><span style="color: #64748b;">Tên kỳ thi/phiên:</span> <strong style="color: #f8fafc;">${sess.name || "—"}</strong></div>
            <div><span style="color: #64748b;">Phòng thi:</span> <strong style="color: #f8fafc;">${sess.room || "—"}</strong></div>
            <div><span style="color: #64748b;">Lớp / Nhóm thi:</span> <strong style="color: #f8fafc;">${sess.class_name || "—"}</strong></div>
            <div><span style="color: #64748b;">Mã môn / Môn thi:</span> <strong style="color: #f8fafc;">${sess.subject_code || "—"}</strong></div>
            <div><span style="color: #64748b;">Giám thị phụ trách:</span> <strong style="color: #f8fafc;">${sess.invigilator_name || "—"}</strong></div>
            <div><span style="color: #64748b;">Thời gian bắt đầu:</span> <span style="color: #94a3b8;">${sess.started_at || "—"}</span></div>
            <div><span style="color: #64748b;">Trạng thái:</span> <strong style="color: #10b981;">● ${sess.status}</strong></div>
          </div>
          <div style="margin-top: 20px; display: flex; justify-content: flex-end;">
            <button id="modal-ok-btn" style="background: #0284c7; color: white; border: none; border-radius: 4px; padding: 6px 16px; cursor: pointer; font-size: 0.85rem; font-weight: 600;">Đóng</button>
          </div>
        </div>
      </div>
    `;

    const close = () => { host.innerHTML = ""; };
    document.getElementById("modal-close-btn")?.addEventListener("click", close);
    document.getElementById("modal-ok-btn")?.addEventListener("click", close);
  }

  showSessionEditModal() {
    const sess = appState.currentSession;
    if (!sess) {
      alert("Hiện tại chưa có phiên giám sát nào đang hoạt động.");
      return;
    }
    const host = document.getElementById("header-modals-host");
    if (!host) return;

    host.innerHTML = `
      <div class="modal-backdrop" style="position: fixed; inset: 0; background: rgba(0,0,0,0.65); z-index: 2000; display: flex; align-items: center; justify-content: center;">
        <div class="modal-card" style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; width: 440px; padding: 20px; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.6);">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <h3 style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin: 0;">Chỉnh sửa thông tin phiên</h3>
            <button id="modal-edit-close" style="background: transparent; border: none; color: #94a3b8; font-size: 1.2rem; cursor: pointer;">✕</button>
          </div>
          <div style="display: flex; flex-direction: column; gap: 12px; font-size: 0.85rem;">
            <div>
              <label style="color: #94a3b8; display: block; margin-bottom: 4px;">Tên kỳ thi / phiên:</label>
              <input type="text" id="edit-session-name" value="${sess.name || ""}" style="width: 100%; background: #1e293b; border: 1px solid #475569; color: #f8fafc; padding: 6px 10px; border-radius: 4px; box-sizing: border-box;" />
            </div>
            <div>
              <label style="color: #94a3b8; display: block; margin-bottom: 4px;">Phòng thi:</label>
              <input type="text" id="edit-session-room" value="${sess.room || ""}" style="width: 100%; background: #1e293b; border: 1px solid #475569; color: #f8fafc; padding: 6px 10px; border-radius: 4px; box-sizing: border-box;" />
            </div>
            <div>
              <label style="color: #94a3b8; display: block; margin-bottom: 4px;">Giám thị:</label>
              <input type="text" id="edit-session-invigilator" value="${sess.invigilator_name || ""}" style="width: 100%; background: #1e293b; border: 1px solid #475569; color: #f8fafc; padding: 6px 10px; border-radius: 4px; box-sizing: border-box;" />
            </div>
          </div>
          <div style="margin-top: 20px; display: flex; justify-content: flex-end; gap: 8px;">
            <button id="modal-edit-cancel" style="background: transparent; color: #94a3b8; border: 1px solid #475569; border-radius: 4px; padding: 6px 12px; cursor: pointer;">Hủy</button>
            <button id="modal-edit-save" style="background: #0284c7; color: white; border: none; border-radius: 4px; padding: 6px 16px; cursor: pointer; font-weight: 600;">Lưu thay đổi</button>
          </div>
        </div>
      </div>
    `;

    const close = () => { host.innerHTML = ""; };
    document.getElementById("modal-edit-close")?.addEventListener("click", close);
    document.getElementById("modal-edit-cancel")?.addEventListener("click", close);

    document.getElementById("modal-edit-save")?.addEventListener("click", async () => {
      const name = document.getElementById("edit-session-name")?.value.trim();
      const room = document.getElementById("edit-session-room")?.value.trim();
      const invigilator_name = document.getElementById("edit-session-invigilator")?.value.trim();
      try {
        const updated = await ApiClient.updateSession(sess.session_id, { name, room, invigilator_name });
        appState.setCurrentSession(updated);
        close();
      } catch (err) {
        alert(`Không thể cập nhật: ${err.message}`);
      }
    });
  }

  showEndSessionConfirm() {
    const sess = appState.currentSession;
    if (!sess) {
      alert("Không có phiên giám sát nào đang hoạt động để kết thúc.");
      return;
    }
    const host = document.getElementById("header-modals-host");
    if (!host) return;

    host.innerHTML = `
      <div class="modal-backdrop" style="position: fixed; inset: 0; background: rgba(0,0,0,0.65); z-index: 2000; display: flex; align-items: center; justify-content: center;">
        <div class="modal-card" style="background: #0f172a; border: 1px solid #ef4444; border-radius: 8px; width: 440px; padding: 20px; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.6);">
          <h3 style="font-size: 1.05rem; font-weight: 700; color: #f87171; margin-top: 0; margin-bottom: 12px;">Xác nhận kết thúc phiên giám sát</h3>
          <p style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 16px;">
            Bạn có chắc chắn muốn kết thúc phiên <strong>${sess.name} (${sess.room})</strong>?<br/><br/>
            Hệ thống sẽ hoàn tất các đoạn video bằng chứng còn dang dở, chốt nhật ký kiểm toán và lưu trữ an toàn vào cơ sở dữ liệu.
          </p>
          <div style="display: flex; justify-content: flex-end; gap: 8px;">
            <button id="modal-end-cancel" style="background: transparent; color: #94a3b8; border: 1px solid #475569; border-radius: 4px; padding: 6px 12px; cursor: pointer;">Hủy</button>
            <button id="modal-end-confirm" style="background: #dc2626; color: white; border: none; border-radius: 4px; padding: 6px 16px; cursor: pointer; font-weight: 600;">Kết thúc phiên</button>
          </div>
        </div>
      </div>
    `;

    const close = () => { host.innerHTML = ""; };
    document.getElementById("modal-end-cancel")?.addEventListener("click", close);

    document.getElementById("modal-end-confirm")?.addEventListener("click", async () => {
      const btn = document.getElementById("modal-end-confirm");
      if (btn) { btn.disabled = true; btn.innerText = "Đang hoàn tất…"; }
      try {
        await ApiClient.endSession(sess.session_id, { reason: "COMPLETED" });
        appState.setCurrentSession(null);
        close();
      } catch (err) {
        alert(`Lỗi khi kết thúc phiên: ${err.message}`);
        close();
      }
    });
  }

  showNewSessionPrompt() {
    const active = appState.currentSession && appState.currentSession.status === "ACTIVE";
    if (active) {
      const confirmed = confirm(
        "Phiên hiện tại sẽ được kết thúc và lưu toàn bộ sự kiện/bằng chứng trước khi bắt đầu phiên mới.\n\nBạn có muốn tiếp tục?"
      );
      if (!confirmed) return;
    }
    this.showStartSessionModal();
  }

  showStartSessionModal() {
    const host = document.getElementById("header-modals-host");
    if (!host) return;

    host.innerHTML = `
      <div class="modal-backdrop" style="position: fixed; inset: 0; background: rgba(0,0,0,0.65); z-index: 2000; display: flex; align-items: center; justify-content: center;">
        <div class="modal-card" style="background: #0f172a; border: 1px solid #0284c7; border-radius: 8px; width: 480px; padding: 22px; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.6);">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <h3 style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin: 0;">Bắt đầu phiên giám sát mới</h3>
            <button id="modal-start-close" style="background: transparent; border: none; color: #94a3b8; font-size: 1.2rem; cursor: pointer;">✕</button>
          </div>
          <div style="display: flex; flex-direction: column; gap: 12px; font-size: 0.85rem;">
            <div>
              <label style="color: #94a3b8; display: block; margin-bottom: 4px;">Tên kỳ thi / Phiên (*):</label>
              <input type="text" id="start-session-name" placeholder="Ví dụ: Thi Cuối Kỳ - Môn Cơ sở dữ liệu" style="width: 100%; background: #1e293b; border: 1px solid #475569; color: #f8fafc; padding: 8px 10px; border-radius: 4px; box-sizing: border-box;" />
            </div>
            <div>
              <label style="color: #94a3b8; display: block; margin-bottom: 4px;">Phòng thi (*):</label>
              <input type="text" id="start-session-room" value="Phòng A203" style="width: 100%; background: #1e293b; border: 1px solid #475569; color: #f8fafc; padding: 8px 10px; border-radius: 4px; box-sizing: border-box;" />
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
              <div>
                <label style="color: #94a3b8; display: block; margin-bottom: 4px;">Lớp / Nhóm:</label>
                <input type="text" id="start-session-class" placeholder="20DTH01" style="width: 100%; background: #1e293b; border: 1px solid #475569; color: #f8fafc; padding: 8px 10px; border-radius: 4px; box-sizing: border-box;" />
              </div>
              <div>
                <label style="color: #94a3b8; display: block; margin-bottom: 4px;">Mã môn:</label>
                <input type="text" id="start-session-subject" placeholder="CSDL101" style="width: 100%; background: #1e293b; border: 1px solid #475569; color: #f8fafc; padding: 8px 10px; border-radius: 4px; box-sizing: border-box;" />
              </div>
            </div>
            <div>
              <label style="color: #94a3b8; display: block; margin-bottom: 4px;">Ghi chú phòng thi (nếu có):</label>
              <input type="text" id="start-session-notes" placeholder="Ghi chú thêm về ca thi..." style="width: 100%; background: #1e293b; border: 1px solid #475569; color: #f8fafc; padding: 8px 10px; border-radius: 4px; box-sizing: border-box;" />
            </div>
          </div>
          <div style="margin-top: 22px; display: flex; justify-content: flex-end; gap: 8px;">
            <button id="modal-start-cancel" style="background: transparent; color: #94a3b8; border: 1px solid #475569; border-radius: 4px; padding: 6px 12px; cursor: pointer;">Hủy</button>
            <button id="modal-start-submit" style="background: #0284c7; color: white; border: none; border-radius: 4px; padding: 7px 18px; cursor: pointer; font-weight: 600;">Bắt đầu giám sát</button>
          </div>
        </div>
      </div>
    `;

    const close = () => { host.innerHTML = ""; };
    document.getElementById("modal-start-close")?.addEventListener("click", close);
    document.getElementById("modal-start-cancel")?.addEventListener("click", close);

    document.getElementById("modal-start-submit")?.addEventListener("click", async () => {
      const name = document.getElementById("start-session-name")?.value.trim();
      const room = document.getElementById("start-session-room")?.value.trim() || "Phòng thi chính";
      const class_name = document.getElementById("start-session-class")?.value.trim() || null;
      const subject_code = document.getElementById("start-session-subject")?.value.trim() || null;
      const notes = document.getElementById("start-session-notes")?.value.trim() || null;

      if (!name) {
        alert("Vui lòng nhập tên phiên / kỳ thi.");
        return;
      }

      const btn = document.getElementById("modal-start-submit");
      if (btn) { btn.disabled = true; btn.innerText = "Đang khởi tạo…"; }

      try {
        const newSess = await ApiClient.startSession({
          name,
          room,
          class_name,
          subject_code,
          notes,
        });
        appState.resetForNewSession(newSess);
        close();
      } catch (err) {
        alert(`Không thể bắt đầu phiên: ${err.message}`);
        if (btn) { btn.disabled = false; btn.innerText = "Bắt đầu giám sát"; }
      }
    });
  }

  handleSafeLogout() {
    const active = appState.currentSession && appState.currentSession.status === "ACTIVE";
    if (!active) {
      ApiClient.logout();
      return;
    }

    const host = document.getElementById("header-modals-host");
    if (!host) return;

    host.innerHTML = `
      <div class="modal-backdrop" style="position: fixed; inset: 0; background: rgba(0,0,0,0.65); z-index: 2000; display: flex; align-items: center; justify-content: center;">
        <div class="modal-card" style="background: #0f172a; border: 1px solid #f59e0b; border-radius: 8px; width: 440px; padding: 20px; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.6);">
          <h3 style="font-size: 1.05rem; font-weight: 700; color: #fbbf24; margin-top: 0; margin-bottom: 12px;">Phiên giám sát vẫn đang hoạt động</h3>
          <p style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 18px;">
            Phiên <strong>${appState.currentSession.name}</strong> đang trực tiếp ghi nhận sự kiện phòng thi.<br/><br/>
            Bạn muốn tiếp tục chạy giám sát ở chế độ nền hay kết thúc phiên thi này trước khi đăng xuất?
          </p>
          <div style="display: flex; flex-direction: column; gap: 8px;">
            <button id="btn-logout-continue" style="background: #0284c7; color: white; border: none; border-radius: 4px; padding: 8px 12px; cursor: pointer; font-size: 0.85rem; font-weight: 600;">Đăng xuất và tiếp tục giám sát</button>
            <button id="btn-logout-end" style="background: #dc2626; color: white; border: none; border-radius: 4px; padding: 8px 12px; cursor: pointer; font-size: 0.85rem; font-weight: 600;">Kết thúc phiên và đăng xuất</button>
            <button id="btn-logout-cancel" style="background: transparent; color: #94a3b8; border: 1px solid #475569; border-radius: 4px; padding: 8px 12px; cursor: pointer; font-size: 0.85rem;">Hủy</button>
          </div>
        </div>
      </div>
    `;

    const close = () => { host.innerHTML = ""; };
    document.getElementById("btn-logout-cancel")?.addEventListener("click", close);

    document.getElementById("btn-logout-continue")?.addEventListener("click", () => {
      ApiClient.logout();
    });

    document.getElementById("btn-logout-end")?.addEventListener("click", async () => {
      try {
        await ApiClient.endSession(appState.currentSession.session_id, { reason: "COMPLETED" });
      } catch {}
      ApiClient.logout();
    });
  }
}
