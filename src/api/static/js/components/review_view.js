/**
 * ExamGuard Vision — Historical Review View Component
 * Dedicated full-screen review interface for invigilator event adjudication.
 */

import { appState } from "../state.js";
import { formatRelativeTime, formatDuration } from "../adapter.js";
import { ApiClient } from "../api.js";

export class ReviewViewComponent {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.statusFilter = "all";
    this.riskFilter = "all";
    this.searchTerm = "";
    this.init();
  }

  init() {
    this.render();
    this.bindEvents();

    appState.subscribe((type) => {
      if (
        type === "EVENTS_UPDATED" ||
        type === "EVENTS_RESET" ||
        type === "EVENT_STATUS_CHANGED" ||
        type === "EVENT_LIFECYCLE_CHANGED" ||
        type === "VIEW_CHANGED"
      ) {
        if (appState.currentView === "review") {
          this.renderTable();
          this.renderKPIs();
        }
      }
    });
  }

  render() {
    this.container.innerHTML = `
      <div class="review-view-container">
        <!-- 1. Header Toolbar -->
        <div class="review-toolbar">
          <div>
            <h2>Exam Event Review Log</h2>
            <p>Historical record of all AI-detected observable behavior evidence for invigilator adjudication.</p>
          </div>

          <div class="review-filters">
            <input 
              type="text" 
              class="review-search-input" 
              id="input-review-search" 
              placeholder="Search student #, event name…" 
            />

            <select class="review-select" id="select-review-status">
              <option value="all">All Review Statuses</option>
              <option value="awaiting">Awaiting Review</option>
              <option value="confirmed">Human Confirmed</option>
              <option value="dismissed">Dismissed</option>
            </select>

            <select class="review-select" id="select-review-risk">
              <option value="all">All Risk Levels</option>
              <option value="high">High Risk</option>
              <option value="medium">Medium Risk</option>
              <option value="low">Low Risk</option>
            </select>
          </div>
        </div>

        <!-- 2. Review KPI Summary Bar -->
        <div class="review-kpi-summary-row" id="review-kpi-summary-row">
          <!-- Rendered dynamically -->
        </div>

        <!-- 3. Review Table Card (Content-adaptive height) -->
        <div class="review-table-card">
          <div class="review-table-wrapper">
            <table class="review-table">
              <thead>
                <tr>
                  <th style="width: 70px;">Evidence</th>
                  <th>Observable Event</th>
                  <th>Student Track</th>
                  <th>Camera</th>
                  <th>Time</th>
                  <th>Duration</th>
                  <th>Evidence Risk</th>
                  <th>Review Status</th>
                  <th style="text-align: right; width: 100px;">Action</th>
                </tr>
              </thead>
              <tbody id="review-table-tbody">
                <!-- Rendered rows -->
              </tbody>
            </table>
          </div>
        </div>

        <!-- 4. Session Adjudication Status Card (fills lower space meaningfully) -->
        <div class="review-session-adjudication-card" id="review-session-card">
          <div class="session-adjudication-header">
            <h4>Invigilator Adjudication Summary</h4>
            <span class="session-badge">AUDIT READY</span>
          </div>
          <div class="session-adjudication-body">
            <div class="adjudication-stat-box">
              <span class="stat-box-num" id="stat-total-events">0</span>
              <span class="stat-box-desc">Total Recorded Anomalies</span>
            </div>
            <div class="adjudication-stat-box">
              <span class="stat-box-num" id="stat-awaiting-review" style="color: var(--color-medium);">0</span>
              <span class="stat-box-desc">Pending Adjudication</span>
            </div>
            <div class="adjudication-stat-box">
              <span class="stat-box-num" id="stat-confirmed-review" style="color: var(--color-live);">0</span>
              <span class="stat-box-desc">Confirmed Observations</span>
            </div>
            <div class="adjudication-stat-box">
              <span class="stat-box-num" id="stat-dismissed-review" style="color: var(--color-dismissed);">0</span>
              <span class="stat-box-desc">Dismissed Anomalies</span>
            </div>
          </div>
        </div>
      </div>
    `;

    this.renderKPIs();
    this.renderTable();
  }

  bindEvents() {
    const selStatus = document.getElementById("select-review-status");
    const selRisk = document.getElementById("select-review-risk");
    const inputSearch = document.getElementById("input-review-search");

    if (selStatus) {
      selStatus.addEventListener("change", (e) => {
        this.statusFilter = e.target.value;
        this.renderTable();
      });
    }

    if (selRisk) {
      selRisk.addEventListener("change", (e) => {
        this.riskFilter = e.target.value;
        this.renderTable();
      });
    }

    if (inputSearch) {
      inputSearch.addEventListener("input", (e) => {
        this.searchTerm = e.target.value.toLowerCase().trim();
        this.renderTable();
      });
    }
  }

  renderKPIs() {
    const kpis = appState.getKPIs();
    const row = document.getElementById("review-kpi-summary-row");
    if (row) {
      row.innerHTML = `
        <div class="review-kpi-card">
          <span class="review-kpi-label">TOTAL EVENTS</span>
          <span class="review-kpi-val">${kpis.total}</span>
        </div>
        <div class="review-kpi-card awaiting">
          <span class="review-kpi-label">AWAITING REVIEW</span>
          <span class="review-kpi-val">${kpis.awaiting}</span>
        </div>
        <div class="review-kpi-card confirmed">
          <span class="review-kpi-label">HUMAN CONFIRMED</span>
          <span class="review-kpi-val">${kpis.confirmed}</span>
        </div>
        <div class="review-kpi-card dismissed">
          <span class="review-kpi-label">DISMISSED</span>
          <span class="review-kpi-val">${kpis.dismissed}</span>
        </div>
      `;
    }

    const tEl = document.getElementById("stat-total-events");
    const aEl = document.getElementById("stat-awaiting-review");
    const cEl = document.getElementById("stat-confirmed-review");
    const dEl = document.getElementById("stat-dismissed-review");
    if (tEl) tEl.innerText = kpis.total;
    if (aEl) aEl.innerText = kpis.awaiting;
    if (cEl) cEl.innerText = kpis.confirmed;
    if (dEl) dEl.innerText = kpis.dismissed;
  }

  renderTable() {
    const tbody = document.getElementById("review-table-tbody");
    if (!tbody) return;

    let events = Array.from(appState.events.values()).sort(
      (a, b) => (b.startTime || b.timestamp) - (a.startTime || a.timestamp)
    );

    // Apply filters
    if (this.statusFilter === "awaiting") {
      events = events.filter((e) => e.reviewStatus === "awaiting");
    } else if (this.statusFilter === "confirmed") {
      events = events.filter((e) => e.reviewStatus === "confirmed");
    } else if (this.statusFilter === "dismissed") {
      events = events.filter((e) => e.reviewStatus === "dismissed");
    }

    if (this.riskFilter !== "all") {
      events = events.filter((e) => e.riskLevel.toLowerCase() === this.riskFilter);
    }

    if (this.searchTerm) {
      events = events.filter((e) => {
        const text = `${e.displayName} ${e.canonicalType} student #${e.trackId} ${e.trackId}`.toLowerCase();
        return text.includes(this.searchTerm);
      });
    }

    if (events.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="9" style="text-align: center; padding: 48px; color: var(--text-dim); font-size: 0.85rem;">
            No events match the selected criteria.
          </td>
        </tr>
      `;
      return;
    }

    tbody.innerHTML = events
      .map((ev) => {
        const riskLower = ev.riskLevel.toLowerCase();
        const timeStr = new Date(ev.timestamp * 1000).toLocaleTimeString([], { hour12: false });
        const thumbHtml = ev.snapshotUrl
          ? `<img src="${ev.snapshotUrl}" class="review-table-thumb" alt="Snap" loading="lazy" onerror="this.style.display='none'; const el = this.parentElement.querySelector('.thumb-empty'); if (el) el.style.display='flex';" />
             <div class="thumb-empty" style="display: none;">—</div>`
          : `<div class="thumb-empty">—</div>`;

        const rStatus = ev.reviewStatus || "awaiting";
        let statusBadge = `<span class="status-tag awaiting">Awaiting Review</span>`;
        if (rStatus === "confirmed") {
          statusBadge = `<span class="status-tag confirmed">✓ Confirmed</span>`;
        } else if (rStatus === "dismissed") {
          statusBadge = `<span class="status-tag dismissed">Dismissed</span>`;
        }

        return `
          <tr class="review-table-row" data-id="${ev.eventId}">
            <td style="text-align: center;">
              <div class="review-thumb-wrapper">${thumbHtml}</div>
            </td>
            <td>
              <div class="review-cell-event">
                <strong>${ev.displayName}</strong>
                <span class="cell-subtext">${ev.canonicalType}</span>
              </div>
            </td>
            <td>
              <span class="student-track-badge">Student #${ev.trackId}</span>
            </td>
            <td>
              <span style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-secondary);">${ev.cameraId}</span>
            </td>
            <td>
              <span style="font-family: var(--font-mono); font-size: 0.78rem;">${timeStr}</span>
            </td>
            <td>
              <span>${formatDuration(ev.duration)}</span>
            </td>
            <td>
              <div class="review-risk-cell">
                <span class="risk-pill ${riskLower}">${ev.riskLevel}</span>
                <span class="risk-score-num">${ev.score}/100</span>
              </div>
            </td>
            <td>
              ${statusBadge}
            </td>
            <td style="text-align: right;">
              <button class="btn-inspect-row" data-inspect="${ev.eventId}">Inspect</button>
            </td>
          </tr>
        `;
      })
      .join("");

    tbody.querySelectorAll("button[data-inspect]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const eid = btn.dataset.inspect;
        appState.selectEvent(eid);
      });
    });

    tbody.querySelectorAll(".review-table-row").forEach((row) => {
      row.addEventListener("click", () => {
        const eid = row.dataset.id;
        appState.selectEvent(eid);
      });
    });
  }
}
