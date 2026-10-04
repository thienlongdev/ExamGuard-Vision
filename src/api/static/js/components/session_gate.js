/**
 * ExamGuard Vision — Session Gate & Monitoring Session Bar
 * Login -> Session Gate (start new / resume active) -> Monitoring.
 * Owns the visible "Kết thúc phiên" control and the end-session confirmation.
 * A session is never created silently: only an explicit "Bắt đầu phiên giám sát" starts one.
 */

import { appState } from "../state.js";
import { ApiClient } from "../api.js";

// Per-tab acknowledgement: a refresh stays in the resumed session, a new login passes the gate again
const GATE_ACK_KEY = "eg_gate_session";

function readAck() {
  try {
    return sessionStorage.getItem(GATE_ACK_KEY);
  } catch {
    return null;
  }
}

function writeAck(sessionId) {
  try {
    if (sessionId) sessionStorage.setItem(GATE_ACK_KEY, sessionId);
    else sessionStorage.removeItem(GATE_ACK_KEY);
  } catch {}
}

function formatClock(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString("vi-VN", { hour: "2-digit", minute: "2-digit", day: "2-digit", month: "2-digit", year: "numeric" });
}

export function formatElapsed(sec) {
  const total = Math.max(0, Math.floor(sec || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const sPart = total % 60;
  if (h > 0) return `${h} giờ ${String(m).padStart(2, "0")} phút`;
  if (m > 0) return `${m} phút ${String(sPart).padStart(2, "0")} giây`;
  return `${sPart} giây`;
}

function elapsedSec(startIso, endIso = null) {
  const st = new Date(startIso).getTime();
  const et = endIso ? new Date(endIso).getTime() : Date.now();
  if (Number.isNaN(st) || Number.isNaN(et)) return 0;
  return Math.max(0, (et - st) / 1000);
}

export class SessionGateComponent {
  constructor(gateId, barId) {
    this.gate = document.getElementById(gateId);
    this.bar = document.getElementById(barId);
    this.mode = "checking"; // checking | start | resume | ended
    this.lastEnded = null;
    this.cameras = [];
    this.elapsedTimer = null;
    this.init();
  }

  init() {
    this.renderGate();
    this.updateVisibility();

    appState.subscribe((type, payload) => {
      if (type === "VIEW_CHANGED") {
        this.updateVisibility();
      } else if (type === "CURRENT_SESSION_UPDATED") {
        this.renderBar();
      } else if (type === "SESSION_ACTION_REQUESTED") {
        this.handleAction(payload);
      }
    });

    this.elapsedTimer = setInterval(() => this.tickElapsed(), 1000);
  }

  // Called once initial data (current session) has been loaded
  evaluate() {
    const sess = appState.currentSession;
    if (sess && sess.status === "ACTIVE") {
      this.mode = readAck() === sess.session_id ? "acknowledged" : "resume";
    } else {
      this.mode = "start";
    }
    this.renderGate();
    this.renderBar();
    this.updateVisibility();
  }

  isGateNeeded() {
    return this.mode !== "acknowledged";
  }

  updateVisibility() {
    if (!this.gate) return;
    const show = appState.currentView === "monitor" && this.isGateNeeded();
    this.gate.classList.toggle("open", show);
    this.gate.setAttribute("aria-hidden", show ? "false" : "true");
  }

  handleAction(action) {
    const active = appState.currentSession && appState.currentSession.status === "ACTIVE";
    if (action === "end") {
      if (active) this.showEndConfirm();
      return;
    }
    if (action === "start") {
      // Starting a new session while one is running goes through the explicit end confirmation first
      if (active) {
        this.showEndConfirm();
        return;
      }
      this.mode = "start";
      this.renderGate();
      appState.setView("monitor");
      this.updateVisibility();
    }
  }

  // ------------------------------------------------------------------
  // Gate overlay
  // ------------------------------------------------------------------

  renderGate() {
    if (!this.gate) return;
    if (this.mode === "checking") {
      this.gate.innerHTML = `
        <div class="session-gate-card">
          <p class="session-gate-muted">Đang kiểm tra phiên giám sát…</p>
        </div>`;
      return;
    }
    if (this.mode === "resume") {
      this.renderResume();
    } else if (this.mode === "ended") {
      this.renderEnded();
    } else if (this.mode === "start") {
      this.renderStart();
    } else {
      this.gate.innerHTML = "";
    }
  }

  renderStart() {
    const defaultName = `Phiên giám sát ${new Date().toLocaleDateString("vi-VN")}`;
    const invigilator = ApiClient.currentUser?.display_name || "";
    this.gate.innerHTML = `
      <div class="session-gate-card" role="dialog" aria-labelledby="gate-title">
        <h2 id="gate-title">Khởi tạo phiên giám sát</h2>
        <p class="session-gate-muted">Chưa có phiên nào đang hoạt động. Nhập thông tin phòng thi để bắt đầu giám sát.</p>
        <form id="gate-start-form" class="session-gate-form" autocomplete="off">
          <label>Tên phiên
            <input type="text" id="gate-session-name" required maxlength="120" placeholder="${escapeHtml(defaultName)}" />
          </label>
          <label>Phòng thi
            <input type="text" id="gate-session-room" required maxlength="60" placeholder="Ví dụ: Phòng A203" />
          </label>
          <label>Camera
            <select id="gate-session-camera"><option value="">Đang tải danh sách camera…</option></select>
          </label>
          <label>Giám thị phụ trách
            <input type="text" id="gate-session-invigilator" maxlength="80" value="${escapeHtml(invigilator)}" />
          </label>
          <div class="session-gate-error" id="gate-start-error" role="alert"></div>
          <div class="session-gate-actions">
            <button type="submit" class="gate-btn gate-btn-primary" id="gate-start-submit">Bắt đầu phiên giám sát</button>
          </div>
        </form>
      </div>`;

    this.loadCameraOptions();
    document.getElementById("gate-start-form")?.addEventListener("submit", (e) => {
      e.preventDefault();
      this.submitStart();
    });
    document.getElementById("gate-session-name")?.focus();
  }

  async loadCameraOptions() {
    const sel = document.getElementById("gate-session-camera");
    if (!sel) return;
    try {
      this.cameras = await ApiClient.getCameras();
    } catch {
      this.cameras = [];
    }
    if (!document.body.contains(sel)) return;
    if (!this.cameras || this.cameras.length === 0) {
      sel.innerHTML = `<option value="">Camera mặc định</option>`;
      return;
    }
    sel.innerHTML = this.cameras
      .map((c) => {
        const id = c.camera_id || c.id || "";
        const label = c.name || id;
        return `<option value="${escapeHtml(id)}">${escapeHtml(label)}</option>`;
      })
      .join("");
  }

  async submitStart() {
    const name = document.getElementById("gate-session-name")?.value.trim();
    const room = document.getElementById("gate-session-room")?.value.trim();
    const cameraId = document.getElementById("gate-session-camera")?.value || "";
    const invigilator = document.getElementById("gate-session-invigilator")?.value.trim();
    const errEl = document.getElementById("gate-start-error");
    const btn = document.getElementById("gate-start-submit");

    if (!name || !room) {
      if (errEl) errEl.innerText = "Vui lòng nhập tên phiên và phòng thi.";
      return;
    }
    if (btn) {
      btn.disabled = true;
      btn.innerText = "Đang khởi tạo…";
    }
    try {
      const newSess = await ApiClient.startSession({
        name,
        room,
        invigilator_name: invigilator || null,
        camera_ids: cameraId ? [cameraId] : null,
      });
      appState.resetForNewSession(newSess);
      writeAck(newSess.session_id);
      this.mode = "acknowledged";
      this.renderGate();
      appState.setView("monitor");
      this.updateVisibility();
    } catch (err) {
      if (errEl) errEl.innerText = `Không thể bắt đầu phiên: ${err.message}`;
      if (btn) {
        btn.disabled = false;
        btn.innerText = "Bắt đầu phiên giám sát";
      }
    }
  }

  renderResume() {
    const s = appState.currentSession || {};
    const summary = s.summary || {};
    const camLabel = appState.cameraInfo?.name || appState.cameraInfo?.camera_id || "Camera mặc định";
    this.gate.innerHTML = `
      <div class="session-gate-card" role="dialog" aria-labelledby="gate-title">
        <h2 id="gate-title">Bạn đang có một phiên giám sát đang hoạt động</h2>
        <dl class="session-gate-facts">
          <dt>Tên phiên</dt><dd>${escapeHtml(s.name || "—")}</dd>
          <dt>Phòng thi</dt><dd>${escapeHtml(s.room || "—")}</dd>
          <dt>Bắt đầu</dt><dd>${formatClock(s.started_at)}</dd>
          <dt>Camera</dt><dd>${escapeHtml(camLabel)}</dd>
        </dl>
        <div class="session-gate-details" id="gate-resume-details" hidden>
          <dl class="session-gate-facts">
            <dt>Mã phiên</dt><dd class="mono">${escapeHtml(s.session_id || "—")}</dd>
            <dt>Giám thị</dt><dd>${escapeHtml(s.invigilator_name || "Chưa phân công")}</dd>
            <dt>Thời lượng</dt><dd>${formatElapsed(elapsedSec(s.started_at))}</dd>
            <dt>Sự kiện</dt><dd>${summary.total_events || 0} (chờ duyệt: ${summary.awaiting_count || 0})</dd>
          </dl>
        </div>
        <div class="session-gate-actions">
          <button type="button" class="gate-btn gate-btn-ghost" id="gate-resume-details-btn">Xem chi tiết</button>
          <button type="button" class="gate-btn gate-btn-danger" id="gate-resume-end">Kết thúc phiên hiện tại</button>
          <button type="button" class="gate-btn gate-btn-primary" id="gate-resume-continue">Tiếp tục phiên</button>
        </div>
      </div>`;

    document.getElementById("gate-resume-continue")?.addEventListener("click", () => {
      writeAck(s.session_id);
      this.mode = "acknowledged";
      this.renderGate();
      this.updateVisibility();
    });
    document.getElementById("gate-resume-end")?.addEventListener("click", () => this.showEndConfirm());
    document.getElementById("gate-resume-details-btn")?.addEventListener("click", () => {
      const box = document.getElementById("gate-resume-details");
      if (box) box.hidden = !box.hidden;
    });
    document.getElementById("gate-resume-continue")?.focus();
  }

  renderEnded() {
    const s = this.lastEnded || {};
    const summary = s.summary || {};
    this.gate.innerHTML = `
      <div class="session-gate-card" role="dialog" aria-labelledby="gate-title">
        <h2 id="gate-title">Phiên đã kết thúc</h2>
        <dl class="session-gate-facts">
          <dt>Tên phiên</dt><dd>${escapeHtml(s.name || "—")}</dd>
          <dt>Phòng thi</dt><dd>${escapeHtml(s.room || "—")}</dd>
          <dt>Thời lượng</dt><dd>${formatElapsed(elapsedSec(s.started_at, s.ended_at))}</dd>
          <dt>Sự kiện</dt><dd>${summary.total_events || 0} (chờ duyệt: ${summary.awaiting_count || 0})</dd>
        </dl>
        <p class="session-gate-muted">Các sự kiện và bằng chứng của phiên đã được lưu trong Lịch sử.</p>
        <div class="session-gate-actions">
          <button type="button" class="gate-btn gate-btn-ghost" id="gate-ended-history">Xem lịch sử phiên</button>
          <button type="button" class="gate-btn gate-btn-primary" id="gate-ended-new">Bắt đầu phiên mới</button>
        </div>
      </div>`;

    document.getElementById("gate-ended-new")?.addEventListener("click", () => {
      this.mode = "start";
      this.renderGate();
    });
    document.getElementById("gate-ended-history")?.addEventListener("click", () => {
      this.mode = "start";
      this.renderGate();
      appState.setView("history");
    });
  }

  // ------------------------------------------------------------------
  // Monitoring session bar (always-visible session identity + end control)
  // ------------------------------------------------------------------

  renderBar() {
    if (!this.bar) return;
    const s = appState.currentSession;
    if (!s || s.status !== "ACTIVE") {
      this.bar.innerHTML = "";
      this.bar.hidden = true;
      return;
    }
    this.bar.hidden = false;
    this.bar.innerHTML = `
      <div class="monitor-session-facts">
        <span class="monitor-session-live" aria-hidden="true">●</span>
        <span><span class="msb-label">Phiên:</span> <strong id="msb-name">${escapeHtml(s.name || "—")}</strong></span>
        <span><span class="msb-label">Phòng:</span> <strong>${escapeHtml(s.room || "—")}</strong></span>
        <span><span class="msb-label">Bắt đầu:</span> <strong>${formatClock(s.started_at)}</strong></span>
        <span><span class="msb-label">Đã chạy:</span> <strong id="msb-elapsed">${formatElapsed(elapsedSec(s.started_at))}</strong></span>
      </div>
      <button type="button" class="gate-btn gate-btn-danger msb-end-btn" id="btn-end-session">Kết thúc phiên</button>`;
    document.getElementById("btn-end-session")?.addEventListener("click", () => this.showEndConfirm());
  }

  tickElapsed() {
    const s = appState.currentSession;
    const el = document.getElementById("msb-elapsed");
    if (el && s && s.status === "ACTIVE") {
      el.innerText = formatElapsed(elapsedSec(s.started_at));
    }
  }

  // ------------------------------------------------------------------
  // End-session confirmation
  // ------------------------------------------------------------------

  async showEndConfirm() {
    const s = appState.currentSession;
    if (!s) return;
    let summary = s.summary || {};
    try {
      const fresh = await ApiClient.getSession(s.session_id);
      if (fresh && fresh.summary) summary = fresh.summary;
    } catch {}

    const host = document.getElementById("session-end-modal-host");
    if (!host) return;
    host.innerHTML = `
      <div class="session-modal-backdrop">
        <div class="session-gate-card session-end-card" role="alertdialog" aria-labelledby="end-title">
          <h2 id="end-title">Kết thúc phiên giám sát?</h2>
          <dl class="session-gate-facts">
            <dt>Tên phiên</dt><dd>${escapeHtml(s.name || "—")}</dd>
            <dt>Phòng thi</dt><dd>${escapeHtml(s.room || "—")}</dd>
            <dt>Thời lượng</dt><dd>${formatElapsed(elapsedSec(s.started_at))}</dd>
            <dt>Tổng sự kiện</dt><dd>${summary.total_events || 0}</dd>
            <dt>Chờ duyệt</dt><dd>${summary.awaiting_count || 0}</dd>
          </dl>
          <p class="session-gate-muted">Các sự kiện và bằng chứng của phiên sẽ được lưu trong Lịch sử.</p>
          <div class="session-gate-error" id="end-session-error" role="alert"></div>
          <div class="session-gate-actions">
            <button type="button" class="gate-btn gate-btn-ghost" id="end-session-cancel">Hủy</button>
            <button type="button" class="gate-btn gate-btn-danger" id="end-session-confirm">Kết thúc phiên</button>
          </div>
        </div>
      </div>`;

    const close = () => {
      host.innerHTML = "";
    };
    document.getElementById("end-session-cancel")?.addEventListener("click", close);
    document.getElementById("end-session-confirm")?.addEventListener("click", async () => {
      const btn = document.getElementById("end-session-confirm");
      if (btn) {
        btn.disabled = true;
        btn.innerText = "Đang lưu…";
      }
      try {
        const closed = await ApiClient.endSession(s.session_id, { reason: "COMPLETED" });
        close();
        this.onSessionEnded(closed);
      } catch (err) {
        const errEl = document.getElementById("end-session-error");
        if (errEl) errEl.innerText = `Lỗi khi kết thúc phiên: ${err.message}`;
        if (btn) {
          btn.disabled = false;
          btn.innerText = "Kết thúc phiên";
        }
      }
    });
    document.getElementById("end-session-cancel")?.focus();
  }

  onSessionEnded(closedSession) {
    writeAck(null);
    this.lastEnded = closedSession;
    appState.clearAfterSessionEnd();
    this.mode = "ended";
    this.renderGate();
    appState.setView("monitor");
    this.updateVisibility();
  }

  // Another tab / client changed the session lifecycle
  onRemoteSessionEnded(session) {
    const cur = appState.currentSession;
    if (cur && session && cur.session_id === session.session_id) {
      this.onSessionEnded(session);
    }
  }

  onRemoteSessionStarted(session) {
    const cur = appState.currentSession;
    if (session && (!cur || cur.session_id !== session.session_id)) {
      appState.resetForNewSession(session);
      this.evaluate();
    }
  }
}

function escapeHtml(text) {
  if (text === null || text === undefined) return "";
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
