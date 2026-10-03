/**
 * ExamGuard Vision — Review Queue Component
 * Right panel displaying real-time observable events requiring human review.
 */

import { appState } from "../state.js";
import { formatRelativeTime, formatDuration } from "../adapter.js";
import { ApiClient } from "../api.js";

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
            <h2>Review Queue</h2>
            <span class="queue-subtext">Realtime Observable Events</span>
          </div>

          <div class="queue-filter-tabs">
            <button class="filter-tab active" data-filter="all" id="tab-filter-all">All</button>
            <button class="filter-tab" data-filter="awaiting" id="tab-filter-awaiting">Awaiting</button>
            <button class="filter-tab" data-filter="reviewed" id="tab-filter-reviewed">Reviewed</button>
            <button class="filter-tab" data-filter="dismissed" id="tab-filter-dismissed">Dismissed</button>
          </div>
        </div>

        <div class="queue-scroll-area" id="queue-scroll-area">
          <div class="queue-empty-state">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
            <h4>No observable events awaiting review</h4>
            <p>Physical camera feed active. Behavioral anomalies will be queued here.</p>
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
  }

  renderQueue() {
    const scrollArea = document.getElementById("queue-scroll-area");
    if (!scrollArea) return;

    const events = appState.getFilteredEvents();

    if (events.length === 0) {
      scrollArea.innerHTML = `
        <div class="queue-empty-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
          </svg>
          <h4>No observable events</h4>
          <p>Continuous monitoring active on physical camera.</p>
        </div>
      `;
      return;
    }

    scrollArea.innerHTML = events
      .map((ev) => {
        const riskLower = ev.riskLevel.toLowerCase();
        const isSelected = appState.selectedEventId === ev.eventId;
        const relativeTime = formatRelativeTime(ev.timestamp);
        const rStatus = ev.reviewStatus || "awaiting";

        let actionHtml = "";
        if (rStatus === "awaiting") {
          actionHtml = `
            <div class="card-actions-group" onclick="event.stopPropagation()">
              <button class="btn-action btn-confirm" data-action="confirm" data-id="${ev.eventId}" title="Mark as confirmed review">Confirm</button>
              <button class="btn-action btn-dismiss" data-action="dismiss" data-id="${ev.eventId}" title="Dismiss event">Dismiss</button>
            </div>
          `;
        } else if (rStatus === "confirmed") {
          actionHtml = `<span class="status-tag confirmed">✓ Human Confirmed</span>`;
        } else {
          actionHtml = `<span class="status-tag dismissed">Dismissed</span>`;
        }

        const thumbHtml = ev.snapshotUrl
          ? `<img class="card-thumbnail-img" src="${ev.snapshotUrl}" alt="Evidence" loading="lazy" onerror="this.style.display='none'; const el = this.parentElement.querySelector('.card-thumbnail-empty'); if (el) el.style.display='flex';" />
             <div class="card-thumbnail-empty" style="display: none;">
               <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                 <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>
               </svg>
               <span>Evidence</span>
             </div>`
          : `<div class="card-thumbnail-empty">
               <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                 <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>
               </svg>
               <span>Evidence</span>
             </div>`;

        const isLive = ev.lifecycle === "open" || ev.lifecycle === "active";
        const lifecycleBadge = isLive
          ? `<span class="event-live-indicator" title="Active physical event"><span class="live-pulse-dot small"></span> LIVE</span>`
          : "";

        return `
          <div class="event-card risk-${riskLower} ${isSelected ? "selected" : ""}" id="event-card-${ev.eventId}" data-id="${ev.eventId}">
            <div class="card-top">
              <span class="card-event-name" title="${ev.displayName}">
                ${ev.displayName}
                ${lifecycleBadge}
              </span>
              <div class="card-badges">
                <span class="risk-pill ${riskLower}">${ev.riskLevel}</span>
              </div>
            </div>

            <div class="card-mid">
              <div class="card-thumbnail-wrapper">
                ${thumbHtml}
              </div>

              <div class="card-details">
                <div class="card-meta-line">
                  <span>Student: <strong>#${ev.trackId}</strong></span>
                  <span>·</span>
                  <span>${ev.cameraId}</span>
                  ${ev.duration ? `<span>·</span><span>${formatDuration(ev.duration)}</span>` : ""}
                </div>

                <div class="risk-bar-container">
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
