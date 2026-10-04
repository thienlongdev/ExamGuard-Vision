/**
 * ExamGuard Vision — Historical Sessions View Component
 * Multi-session history browser, detail inspector, metadata editing, and audit trail inspection.
 */

import { appState } from "../state.js";
import { ApiClient } from "../api.js";
import { normalizeEvent } from "../adapter.js";
import {
  EVENT_NAMES_VI,
  SEVERITY_BANDS_VI,
  REVIEW_STATUS_VI,
  SESSION_STATUS_VI,
  ALERT_SEMANTICS_TOOLTIP,
  formatDurationVi,
} from "../localization.js";

export class HistoryViewComponent {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.currentMode = "list"; // "list" | "detail"
    this.selectedSessionId = null;
    this.selectedSession = null;
    this.sessionEvents = [];
    this.sessionAudits = [];
    this.searchTerm = "";
    this.statusFilter = "all";
    this.dateFilter = "all";
    this.includeTest = false;
    this.page = 1;
    this.pageSize = 10;
    this.searchDebounce = null;
    this.loadSeq = 0;
    this.init();
  }

  init() {
    this.render();

    appState.subscribe((type) => {
      if (type === "VIEW_CHANGED" && appState.currentView === "history") {
        if (this.currentMode === "list") {
          this.loadSessions();
        } else if (this.selectedSessionId) {
          this.loadSessionDetail(this.selectedSessionId);
        }
      } else if (type === "EVENT_REVIEWED" && this.currentMode === "detail" && this.selectedSessionId) {
        this.loadSessionDetail(this.selectedSessionId);
      }
    });
  }

  render() {
    if (!this.container) return;
    if (this.currentMode === "list") {
      this.renderSessionList();
    } else {
      this.renderSessionDetail();
    }
  }

  // ==========================================
  // MODE 1: SESSION LIST VIEW
  // ==========================================

  renderSessionList() {
    this.container.innerHTML = `
      <div class="history-view-container">
        <!-- 1. Header Toolbar -->
        <div class="history-toolbar">
          <div>
            <h2>Lịch sử phiên giám sát</h2>
            <p>Mỗi dòng là một phiên giám sát. Chọn "Xem lại" để mở sự kiện và bằng chứng của riêng phiên đó.</p>
          </div>

          <div class="history-filters">
            <input
              type="search"
              class="history-search-input"
              id="input-history-search"
              placeholder="Tìm theo tên phiên hoặc phòng..."
              value="${escapeHtml(this.searchTerm)}"
            />

            <select class="history-select" id="select-history-status" aria-label="Lọc theo trạng thái">
              <option value="all" ${this.statusFilter === "all" ? "selected" : ""}>Tất cả</option>
              <option value="ACTIVE" ${this.statusFilter === "ACTIVE" ? "selected" : ""}>Đang hoạt động</option>
              <option value="CLOSED" ${this.statusFilter === "CLOSED" ? "selected" : ""}>Đã kết thúc</option>
              <option value="INTERRUPTED" ${this.statusFilter === "INTERRUPTED" ? "selected" : ""}>Bị gián đoạn</option>
            </select>

            <select class="history-select" id="select-history-date" aria-label="Lọc theo thời gian">
              <option value="today" ${this.dateFilter === "today" ? "selected" : ""}>Hôm nay</option>
              <option value="7d" ${this.dateFilter === "7d" ? "selected" : ""}>7 ngày</option>
              <option value="30d" ${this.dateFilter === "30d" ? "selected" : ""}>30 ngày</option>
              <option value="all" ${this.dateFilter === "all" ? "selected" : ""}>Tất cả</option>
            </select>

            <label class="history-test-toggle">
              <input type="checkbox" id="chk-history-include-test" ${this.includeTest ? "checked" : ""} />
              Hiển thị phiên kiểm thử
            </label>

            <button class="btn-history-refresh" id="btn-history-refresh" title="Tải lại danh sách">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>
                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
              </svg>
            </button>
          </div>
        </div>

        <!-- 2. One row per session -->
        <div class="history-table-card">
          <div class="history-table-wrapper history-table-paged">
            <table class="history-table">
              <thead>
                <tr>
                  <th>Tên phiên</th>
                  <th>Phòng thi</th>
                  <th>Giám thị</th>
                  <th>Bắt đầu</th>
                  <th>Kết thúc</th>
                  <th>Thời lượng</th>
                  <th title="Tổng sự kiện">Tổng</th>
                  <th title="Chờ duyệt">Chờ</th>
                  <th title="Đã xác nhận">Xác nhận</th>
                  <th title="Đã bỏ qua">Bỏ qua</th>
                  <th>Trạng thái</th>
                  <th style="text-align: right; width: 90px;">Thao tác</th>
                </tr>
              </thead>
              <tbody id="history-sessions-tbody">
                <tr>
                  <td colspan="12" style="text-align: center; padding: 48px; color: var(--text-dim);">
                    Đang tải danh sách phiên giám sát…
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <div class="history-pagination" id="history-pagination"></div>
        </div>
      </div>
    `;

    this.bindListEvents();
    this.loadSessions();
  }

  bindListEvents() {
    const inputSearch = document.getElementById("input-history-search");
    const selStatus = document.getElementById("select-history-status");
    const selDate = document.getElementById("select-history-date");
    const chkTest = document.getElementById("chk-history-include-test");
    const btnRefresh = document.getElementById("btn-history-refresh");

    inputSearch?.addEventListener("input", (e) => {
      this.searchTerm = e.target.value.trim();
      clearTimeout(this.searchDebounce);
      this.searchDebounce = setTimeout(() => {
        this.page = 1;
        this.loadSessions();
      }, 250);
    });

    selStatus?.addEventListener("change", (e) => {
      this.statusFilter = e.target.value;
      this.page = 1;
      this.loadSessions();
    });

    selDate?.addEventListener("change", (e) => {
      this.dateFilter = e.target.value;
      this.page = 1;
      this.loadSessions();
    });

    chkTest?.addEventListener("change", (e) => {
      this.includeTest = e.target.checked;
      this.page = 1;
      this.loadSessions();
    });

    btnRefresh?.addEventListener("click", () => this.loadSessions());
  }

  async loadSessions() {
    const tbody = document.getElementById("history-sessions-tbody");
    if (!tbody) return;
    const seq = ++this.loadSeq;

    try {
      const result = await ApiClient.getSessionsPage({
        page: this.page,
        pageSize: this.pageSize,
        status: this.statusFilter === "all" ? null : this.statusFilter,
        search: this.searchTerm || null,
        dateRange: this.dateFilter === "all" ? null : this.dateFilter,
        includeTest: this.includeTest,
      });
      if (seq !== this.loadSeq || !result) return;

      this.page = result.page;
      const sessions = result.items || [];
      this.renderPagination(result);

      if (sessions.length === 0) {
        tbody.innerHTML = `
          <tr>
            <td colspan="12" style="text-align: center; padding: 48px; color: var(--text-dim); font-size: 0.85rem;">
              Không tìm thấy phiên giám sát nào phù hợp.
            </td>
          </tr>
        `;
        return;
      }

      tbody.innerHTML = sessions.map((s) => this.renderSessionRow(s)).join("");

      tbody.querySelectorAll("button[data-reopen]").forEach((btn) => {
        btn.addEventListener("click", (e) => {
          e.stopPropagation();
          this.openSessionDetail(btn.dataset.reopen);
        });
      });
    } catch (err) {
      if (seq !== this.loadSeq) return;
      tbody.innerHTML = `
        <tr>
          <td colspan="12" style="text-align: center; padding: 48px; color: var(--state-high); font-size: 0.85rem;">
            Lỗi khi tải lịch sử phiên: ${escapeHtml(err.message)}
          </td>
        </tr>
      `;
    }
  }

  renderSessionRow(s) {
    const statusInfo = SESSION_STATUS_VI[s.status] || { label: s.status, class: "status-active" };
    const sum = s.summary || {};
    const fmt = (iso) =>
      iso
        ? new Date(iso).toLocaleString("vi-VN", { hour: "2-digit", minute: "2-digit", day: "2-digit", month: "2-digit", year: "numeric" })
        : "—";
    const kindBadge = s.session_kind && s.session_kind !== "OPERATIONAL"
      ? `<span class="session-kind-badge">${s.session_kind === "TEST" ? "Kiểm thử" : "Thẩm định"}</span>`
      : "";

    return `
      <tr class="history-table-row" data-session-id="${escapeHtml(s.session_id)}">
        <td>
          <div class="session-name-cell">
            <strong>${escapeHtml(s.name)}</strong> ${kindBadge}
            <span class="session-id-subtext">${escapeHtml(s.session_id)}</span>
          </div>
        </td>
        <td><span class="session-room-badge">${escapeHtml(s.room || "—")}</span></td>
        <td><span class="session-invigilator-text">${escapeHtml(s.invigilator_name || "Chưa phân công")}</span></td>
        <td style="font-family: var(--font-mono); font-size: 0.76rem;">${fmt(s.started_at)}</td>
        <td style="font-family: var(--font-mono); font-size: 0.76rem;">${s.status === "ACTIVE" ? "Đang chạy" : fmt(s.ended_at)}</td>
        <td>${formatSessionDuration(sum.duration_sec)}</td>
        <td><strong style="color: var(--text-primary);">${sum.total_events || 0}</strong></td>
        <td style="color: var(--color-medium);">${sum.awaiting_count || 0}</td>
        <td style="color: var(--color-live);">${sum.confirmed_count || 0}</td>
        <td style="color: var(--text-dim);">${sum.dismissed_count || 0}</td>
        <td><span class="session-status-badge ${statusInfo.class}">${statusInfo.label}</span></td>
        <td style="text-align: right;">
          <button class="btn-inspect-session" data-reopen="${escapeHtml(s.session_id)}">Xem lại</button>
        </td>
      </tr>
    `;
  }

  renderPagination(result) {
    const box = document.getElementById("history-pagination");
    if (!box) return;
    const { total, page, page_size: size, total_pages: pages, hidden_test_count: hidden } = result;
    const from = total === 0 ? 0 : (page - 1) * size + 1;
    const to = Math.min(total, page * size);

    const nums = [];
    for (let p = 1; p <= pages; p++) {
      if (p === 1 || p === pages || Math.abs(p - page) <= 1) nums.push(p);
      else if (nums[nums.length - 1] !== "…") nums.push("…");
    }

    const hiddenNote = hidden > 0 && !this.includeTest ? ` · đã ẩn ${hidden} phiên kiểm thử` : "";
    box.innerHTML = `
      <span class="history-page-info">Hiển thị ${from}–${to} / ${total} phiên${hiddenNote}</span>
      <div class="history-page-controls">
        <button class="history-page-btn" data-page="${page - 1}" ${page <= 1 ? "disabled" : ""}>Trước</button>
        ${nums
          .map((p) =>
            p === "…"
              ? `<span class="history-page-gap">…</span>`
              : `<button class="history-page-btn ${p === page ? "active" : ""}" data-page="${p}" ${p === page ? 'aria-current="page"' : ""}>${p}</button>`
          )
          .join("")}
        <button class="history-page-btn" data-page="${page + 1}" ${page >= pages ? "disabled" : ""}>Sau</button>
      </div>
    `;
    box.querySelectorAll("button[data-page]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = Number(btn.dataset.page);
        if (!target || target === this.page) return;
        this.page = target;
        this.loadSessions();
        this.container.querySelector(".history-view-container")?.scrollTo(0, 0);
      });
    });
  }

  // ==========================================
  // MODE 2: SESSION DETAIL VIEW
  // ==========================================

  async openSessionDetail(sessionId) {
    this.selectedSessionId = sessionId;
    this.currentMode = "detail";
    this.render();
    await this.loadSessionDetail(sessionId);
  }

  renderSessionDetail() {
    this.container.innerHTML = `
      <div class="history-detail-container">
        <!-- Top navigation breadcrumb -->
        <div class="history-detail-topbar">
          <button class="btn-back-to-list" id="btn-back-to-sessions">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/>
            </svg>
            Quay lại danh sách phiên
          </button>
          <div class="session-detail-id-badge" id="detail-session-id">—</div>
        </div>

        <!-- Session Editable Info Header -->
        <div class="session-detail-header-card">
          <div class="session-meta-edit-grid">
            <div class="meta-field">
              <label>Tên kỳ thi / Phiên:</label>
              <input type="text" id="edit-session-name" class="detail-edit-input" placeholder="Tên phiên giám sát" />
            </div>
            <div class="meta-field">
              <label>Phòng thi:</label>
              <input type="text" id="edit-session-room" class="detail-edit-input" placeholder="Phòng thi (vd: P.302)" />
            </div>
            <div class="meta-field">
              <label>Cán bộ giám thị:</label>
              <input type="text" id="edit-session-invigilator" class="detail-edit-input" placeholder="Họ tên giám thị" />
            </div>
            <div class="meta-field-action">
              <button class="btn-action btn-save-session-meta" id="btn-save-session-meta">Lưu thông tin</button>
            </div>
          </div>
        </div>

        <!-- Session Summary KPI Chips -->
        <div class="session-summary-kpis" id="session-summary-kpis-container">
          <!-- Rendered dynamically -->
        </div>

        <!-- Detail Sub-views: Events and Audit Log -->
        <div class="session-detail-content-grid">
          <!-- Left/Main Column: Recorded Observable Events -->
          <div class="session-events-card">
            <div class="card-section-title">
              <h3>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>
                </svg>
                Sự kiện ghi nhận trong phiên
              </h3>
              <select id="history-detail-cam-select" style="margin-left: auto; background: #1e293b; color: #cbd5e1; border: 1px solid #334155; border-radius: 4px; padding: 2px 6px; font-size: 0.75rem; margin-right: 8px;">
                <option value="all">Tất cả camera</option>
              </select>
              <span id="session-events-count-badge" class="count-badge">0 sự kiện</span>
            </div>

            <div class="session-events-table-wrapper">
              <table class="session-events-table">
                <thead>
                  <tr>
                    <th style="width: 70px;">Bằng chứng</th>
                    <th>Sự kiện</th>
                    <th>Camera</th>
                    <th>Thí sinh</th>
                    <th>Thời điểm</th>
                    <th>Thời lượng</th>
                    <th>Mức cảnh báo</th>
                    <th>Trạng thái duyệt</th>
                    <th style="text-align: right; width: 90px;">Thao tác</th>
                  </tr>
                </thead>
                <tbody id="session-events-tbody">
                  <tr>
                    <td colspan="9" style="text-align: center; padding: 36px; color: var(--text-dim);">
                      Đang tải danh sách sự kiện...
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <!-- Right/Secondary Column: Session Audit Trail -->
          <div class="session-audit-card">
            <div class="card-section-title">
              <h3>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                </svg>
                Nhật ký kiểm toán (Audit Trail)
              </h3>
              <span id="session-audit-count-badge" class="count-badge">0 mục</span>
            </div>

            <div class="session-audit-list" id="session-audit-list">
              <!-- Rendered audit records -->
            </div>
          </div>
        </div>
      </div>
    `;

    document.getElementById("btn-back-to-sessions")?.addEventListener("click", () => {
      this.currentMode = "list";
      this.render();
    });

    document.getElementById("btn-save-session-meta")?.addEventListener("click", async () => {
      if (!this.selectedSessionId) return;
      const name = document.getElementById("edit-session-name")?.value?.trim();
      const room = document.getElementById("edit-session-room")?.value?.trim();
      const invigilator_name = document.getElementById("edit-session-invigilator")?.value?.trim();

      const btn = document.getElementById("btn-save-session-meta");
      if (btn) btn.disabled = true;
      try {
        const updated = await ApiClient.updateSession(this.selectedSessionId, {
          name,
          room,
          invigilator_name,
        });
        this.selectedSession = updated;
        alert("Đã lưu thông tin phiên giám sát thành công!");
      } catch (err) {
        alert(`Lỗi lưu thông tin: ${err.message}`);
      } finally {
        if (btn) btn.disabled = false;
      }
    });
  }

  async loadSessionDetail(sessionId) {
    try {
      const [session, events, audits] = await Promise.all([
        ApiClient.getSession(sessionId),
        ApiClient.getSessionEvents(sessionId),
        ApiClient.getSessionAudit(sessionId),
      ]);

      this.selectedSession = session;
      this.sessionEvents = events || [];
      this.sessionAudits = audits || [];

      // Update Form Inputs
      const nameInput = document.getElementById("edit-session-name");
      const roomInput = document.getElementById("edit-session-room");
      const invInput = document.getElementById("edit-session-invigilator");
      const idBadge = document.getElementById("detail-session-id");

      if (nameInput) nameInput.value = session.name || "";
      if (roomInput) roomInput.value = session.room || "";
      if (invInput) invInput.value = session.invigilator_name || "";
      if (idBadge) idBadge.innerText = `PHIÊN: ${session.session_id}`;

      // Render KPIs
      this.renderSessionKPIs(session, events);

      // Render Events Table
      this.renderSessionEventsTable(events);

      // Render Audit Trail
      this.renderSessionAuditList(audits);
    } catch (err) {
      console.error("Failed to load session detail:", err);
      alert(`Lỗi khi mở phiên giám sát: ${err.message}`);
    }
  }

  renderSessionKPIs(session, events) {
    const kpiBox = document.getElementById("session-summary-kpis-container");
    if (!kpiBox) return;

    const total = events.length;
    const high = events.filter((e) => (e.severity || e.risk_level || "").toUpperCase() === "HIGH").length;
    const confirmed = events.filter((e) => (e.review_status || "").toLowerCase() === "confirmed").length;
    const dismissed = events.filter((e) => (e.review_status || "").toLowerCase() === "dismissed").length;
    const awaiting = events.filter((e) => (e.review_status || "").toLowerCase() === "awaiting").length;

    const dur = formatSessionDuration(session.summary?.duration_sec);
    const statusInfo = SESSION_STATUS_VI[session.status] || { label: session.status };

    kpiBox.innerHTML = `
      <div class="session-kpi-chip">
        <span class="chip-label">TRẠNG THÁI</span>
        <span class="chip-val" style="color: var(--accent-cyan);">${statusInfo.label}</span>
      </div>
      <div class="session-kpi-chip">
        <span class="chip-label">THỜI LƯỢNG</span>
        <span class="chip-val">${dur}</span>
      </div>
      <div class="session-kpi-chip">
        <span class="chip-label">TỔNG SỰ KIỆN</span>
        <span class="chip-val">${total}</span>
      </div>
      <div class="session-kpi-chip high">
        <span class="chip-label">CẢNH BÁO CAO</span>
        <span class="chip-val">${high}</span>
      </div>
      <div class="session-kpi-chip confirmed">
        <span class="chip-label">ĐÃ XÁC NHẬN</span>
        <span class="chip-val">${confirmed}</span>
      </div>
      <div class="session-kpi-chip dismissed">
        <span class="chip-label">ĐÃ BỎ QUA</span>
        <span class="chip-val">${dismissed}</span>
      </div>
      <div class="session-kpi-chip awaiting">
        <span class="chip-label">CHỜ DUYỆT</span>
        <span class="chip-val">${awaiting}</span>
      </div>
    `;
  }

  renderSessionEventsTable(events) {
    const tbody = document.getElementById("session-events-tbody");
    const countBadge = document.getElementById("session-events-count-badge");
    if (countBadge) countBadge.innerText = `${events.length} sự kiện`;
    if (!tbody) return;

    if (events.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="8" style="text-align: center; padding: 36px; color: var(--text-dim); font-size: 0.85rem;">
            Không có sự kiện nào được ghi nhận trong phiên này.
          </td>
        </tr>
      `;
      return;
    }

    tbody.innerHTML = events
      .map((e) => {
        const norm = normalizeEvent(e);
        const sevInfo = SEVERITY_BANDS_VI[norm.riskLevel] || { label: norm.riskLevel, color: "var(--text-secondary)" };
        const timeStr = new Date(norm.timestamp * 1000).toLocaleTimeString([], { hour12: false });
        const thumbHtml = norm.snapshotUrl
          ? `<img src="${norm.snapshotUrl}" class="history-table-thumb" alt="Ảnh bằng chứng" loading="lazy" onerror="this.style.display='none';" />`
          : `<div class="thumb-empty">—</div>`;
        const revInfo = REVIEW_STATUS_VI[norm.reviewStatus] || { label: "Chờ duyệt", class: "awaiting" };

        return `
          <tr class="session-event-row" data-event-id="${norm.eventId}">
            <td style="text-align: center;">${thumbHtml}</td>
            <td>
              <div class="review-cell-event">
                <strong>${norm.displayName}</strong>
                <span class="cell-subtext">${norm.canonicalType}</span>
              </div>
            </td>
            <td>
              <span class="cam-badge" style="background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 3px; font-size: 0.72rem; color: #94a3b8;">
                ${escapeHtml(norm.cameraId || 'cam01')}
              </span>
            </td>
            <td>
              <span class="student-track-badge" title="Mã theo dõi tạm thời của camera; có thể thay đổi khi đối tượng rời khung hình lâu.">
                Thí sinh #${norm.trackId}
              </span>
            </td>
            <td style="font-family: var(--font-mono); font-size: 0.78rem;">${timeStr}</td>
            <td>${formatDurationVi(norm.duration)}</td>
            <td>
              <span class="risk-pill ${norm.riskLevel.toLowerCase()}">${sevInfo.label} · ${norm.score}</span>
            </td>
            <td>
              <span class="status-tag ${revInfo.class}">${revInfo.label}</span>
            </td>
            <td style="text-align: right;">
              <button class="btn-inspect-row" data-open-event="${norm.eventId}">Xem</button>
            </td>
          </tr>
        `;
      })
      .join("");

    tbody.querySelectorAll("button[data-open-event]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const eid = btn.dataset.openEvent;
        this.openHistoricalEvent(eid);
      });
    });

    tbody.querySelectorAll(".session-event-row").forEach((row) => {
      row.addEventListener("click", () => {
        const eid = row.dataset.eventId;
        this.openHistoricalEvent(eid);
      });
    });
  }

  openHistoricalEvent(eventId) {
    const rawEv = this.sessionEvents.find((e) => (e.event_id || e.eventId) === eventId);
    if (rawEv) {
      // Inspect only: a historical event must never enter the live session's state/KPIs
      appState.inspectEvent(normalizeEvent(rawEv));
    }
  }

  renderSessionAuditList(audits) {
    const container = document.getElementById("session-audit-list");
    const countBadge = document.getElementById("session-audit-count-badge");
    if (countBadge) countBadge.innerText = `${audits.length} mục`;
    if (!container) return;

    if (audits.length === 0) {
      container.innerHTML = `
        <div style="text-align: center; padding: 24px; color: var(--text-dim); font-size: 0.8rem;">
          Chưa có nhật ký kiểm toán cho phiên này.
        </div>
      `;
      return;
    }

    container.innerHTML = audits
      .slice(0, 50)
      .map((a) => {
        const timeStr = new Date(a.created_at).toLocaleTimeString([], { hour12: false });
        const hashSub = a.entry_hash ? a.entry_hash.slice(0, 8) : "—";
        return `
          <div class="audit-item">
            <div class="audit-header-line">
              <span class="audit-action-tag">${escapeHtml(a.action)}</span>
              <span class="audit-time">${timeStr}</span>
            </div>
            <div class="audit-details-text">${escapeHtml(a.details_json || "Không có chi tiết")}</div>
            <div class="audit-hash-code" title="Băm kiểm toán chuỗi: ${a.entry_hash || ''}">Mã băm: ${hashSub}</div>
          </div>
        `;
      })
      .join("");
  }
}

function formatSessionDuration(sec) {
  if (!sec || sec < 1) return "—";
  const total = Math.floor(sec);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m > 0) return `${m}m ${String(s).padStart(2, "0")}s`;
  return `${s}s`;
}

function escapeHtml(text) {
  if (!text) return "";
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
