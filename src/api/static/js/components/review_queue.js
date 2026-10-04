import { appState } from "../state.js";
import { formatRelativeTime, formatDuration } from "../adapter.js";
import { ApiClient } from "../api.js";
import { getSeverityInfo, ALERT_SEMANTICS_TOOLTIP } from "../localization.js";

export class ReviewQueueComponent {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.init();
  }

  init() {
    this.render();
    this.bindEvents();

    appState.subscribe((type) => {
      if (
        type === "EVENTS_UPDATED" ||
        type === "EVENTS_RESET" ||
        type === "FILTER_CHANGED" ||
        type === "EVENT_STATUS_CHANGED" ||
        type === "EVENT_LIFECYCLE_CHANGED" ||
        type === "EVENT_SELECTED"
      ) {
        this.renderQueue();
      }
    });

    // Refresh relative times every 5s
    setInterval(() => {
      this.updateRelativeTimes();
    }, 5000);
  }

  render() {
    this.container.innerHTML = `
      <div class="review-queue-panel">
        <div class="queue-header">
          <div class="queue-title-row">
            <h2>Sự kiện cần xem</h2>
            <span class="queue-subtext">Quan sát theo thời gian thực</span>
          </div>

          <div class="queue-filter-tabs" style="display: flex; flex-wrap: wrap; gap: 4px; align-items: center;">
            <button class="filter-tab active" data-filter="all" id="tab-filter-all">Tất cả</button>
            <button class="filter-tab" data-filter="awaiting" id="tab-filter-awaiting">Chờ duyệt</button>
            <button class="filter-tab" data-filter="reviewed" id="tab-filter-reviewed">Đã xác nhận</button>
            <button class="filter-tab" data-filter="dismissed" id="tab-filter-dismissed">Đã bỏ qua</button>
            <select id="queue-camera-select" style="background: #1e293b; color: #cbd5e1; border: 1px solid #334155; border-radius: 4px; padding: 2px 6px; font-size: 0.75rem; margin-left: auto;">
              <option value="all">Tất cả camera</option>
            </select>
          </div>
        </div>

        <div class="queue-scroll-area" id="queue-scroll-area">
          <div class="queue-empty-state">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
            <h4>Không có sự kiện chờ duyệt</h4>
            <p>Camera trực tiếp đang hoạt động. Các sự kiện cần chú ý sẽ xuất hiện tại đây.</p>
          </div>
        </div>
      </div>
    `;
    this.renderQueue();
  }

  bindEvents() {
    this.container.querySelectorAll(".filter-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        const filter = tab.dataset.filter;
        this.container.querySelectorAll(".filter-tab").forEach((t) => t.classList.toggle("active", t === tab));
        appState.setFilter(filter);
      });
    });

    const camSel = this.container.querySelector("#queue-camera-select");
    if (camSel) {
      camSel.addEventListener("change", (e) => {
        appState.setCameraFilter(e.target.value);
      });
    }
  }

  renderQueue() {
    const scrollArea = document.getElementById("queue-scroll-area");
    if (!scrollArea) return;

    const prevScrollTop = scrollArea.scrollTop;

    const events = appState.getFilteredEvents();

    if (events.length === 0) {
      scrollArea.innerHTML = `
        <div class="queue-empty-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
          </svg>
          <h4>Không có sự kiện</h4>
          <p>Hệ thống giám sát liên tục đang hoạt động trên camera trực tiếp.</p>
        </div>
      `;
      return;
    }

    scrollArea.innerHTML = events
      .map((ev) => {
        const sev = getSeverityInfo(ev.riskLevel);
        const riskLower = sev.key;
        const isSelected = appState.selectedEventId === ev.eventId;
        const relativeTime = formatRelativeTime(ev.timestamp);
        const rStatus = ev.reviewStatus || "awaiting";

        let actionHtml = "";
        if (rStatus === "awaiting") {
          actionHtml = `
            <div class="card-actions-group" onclick="event.stopPropagation()">
              <button class="btn-action btn-confirm" data-action="confirm" data-id="${ev.eventId}" title="Giám thị xác nhận sự kiện">Xác nhận</button>
              <button class="btn-action btn-dismiss" data-action="dismiss" data-id="${ev.eventId}" title="Bỏ qua sự kiện">Bỏ qua</button>
            </div>
          `;
        } else if (rStatus === "confirmed") {
          actionHtml = `<span class="status-tag confirmed">✓ Đã xác nhận</span>`;
        } else {
          actionHtml = `<span class="status-tag dismissed">Đã bỏ qua</span>`;
        }

        const thumbHtml = ev.snapshotUrl
          ? `<img class="card-thumbnail-img" src="${ev.snapshotUrl}" alt="Ảnh bằng chứng" loading="lazy" onerror="this.style.display='none'; const el = this.parentElement.querySelector('.card-thumbnail-empty'); if (el) el.style.display='flex';" />
             <div class="card-thumbnail-empty" style="display: none;">
               <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                 <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>
               </svg>
               <span>Bằng chứng</span>
             </div>`
          : `<div class="card-thumbnail-empty">
               <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                 <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>
               </svg>
               <span>Bằng chứng</span>
             </div>`;

        const isLive = ev.lifecycle === "open" || ev.lifecycle === "active";
        const lifecycleBadge = isLive
          ? `<span class="event-live-indicator" title="Sự kiện đang diễn ra"><span class="live-pulse-dot small"></span> TRỰC TIẾP</span>`
          : "";

        return `
          <div class="event-card risk-${riskLower} ${isSelected ? "selected" : ""}" id="event-card-${ev.eventId}" data-id="${ev.eventId}">
            <div class="card-top">
              <span class="card-event-name" title="${ev.displayName}">
                ${ev.displayName}
                ${lifecycleBadge}
              </span>
              <div class="card-badges">
                <span class="risk-pill ${riskLower}">${sev.badge}</span>
              </div>
            </div>

            <div class="card-mid">
              <div class="card-thumbnail-wrapper">
                ${thumbHtml}
              </div>

              <div class="card-details">
                <div class="card-meta-line">
                  <span>Thí sinh: <strong>#${ev.trackId}</strong></span>
                  <span class="cam-badge" style="background: rgba(255,255,255,0.08); padding: 1px 5px; border-radius: 3px; font-size: 0.72rem; color: #94a3b8;">${ev.cameraId || 'cam01'}</span>
                  ${ev.duration ? `<span>·</span><span>${formatDuration(ev.duration)}</span>` : ""}
                </div>

                <div class="risk-bar-container" title="${ALERT_SEMANTICS_TOOLTIP}">
                  <div class="risk-bar-track">
                    <div class="risk-bar-fill ${riskLower}" style="width: ${ev.score}%"></div>
                  </div>
                  <span class="risk-bar-score">${ev.score}/100</span>
                </div>
              </div>
            </div>

            <div class="card-footer">
              <span class="card-time" data-ts="${ev.timestamp}">${relativeTime}</span>
              ${actionHtml}
            </div>
          </div>
        `;
      })
      .join("");

    // Bind card clicks & actions
    scrollArea.querySelectorAll(".event-card").forEach((card) => {
      const eid = card.dataset.id;
      const ev = appState.events.get(eid);

      card.addEventListener("click", () => {
        appState.selectEvent(eid);
      });

      card.addEventListener("mouseenter", () => {
        if (ev) appState.setHighlightedTrack(ev.trackId);
      });

      card.addEventListener("mouseleave", () => {
        appState.setHighlightedTrack(null);
      });
    });

    scrollArea.querySelectorAll("button[data-action]").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const action = btn.dataset.action;
        const eid = btn.dataset.id;
        const newStatus = action === "confirm" ? "confirmed" : "dismissed";

        btn.disabled = true;
        btn.innerText = "...";

        try {
          await ApiClient.updateEventStatus(eid, newStatus);
          appState.updateEventReviewStatus(eid, newStatus);
        } catch (err) {
          console.error("Failed to update status:", err);
          btn.disabled = false;
          btn.innerText = action === "confirm" ? "Confirm" : "Dismiss";
        }
      });
    });

    if (prevScrollTop > 5) {
      scrollArea.scrollTop = prevScrollTop;
    }
  }

  updateRelativeTimes() {
    this.container.querySelectorAll(".card-time").forEach((el) => {
      const ts = Number(el.dataset.ts);
      if (ts) {
        el.innerText = formatRelativeTime(ts);
      }
    });
  }
}
