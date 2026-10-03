/**
 * ExamGuard Vision — Activity Timeline Component
 * Adaptive temporal visualization of observable events across tracked students.
 */

import { appState } from "../state.js";
import { formatDuration } from "../adapter.js";

export class TimelineComponent {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.init();
  }

  init() {
    this.render();
    appState.subscribe((type) => {
      if (
        type === "EVENTS_UPDATED" ||
        type === "EVENTS_RESET" ||
        type === "EVENT_STATUS_CHANGED" ||
        type === "EVENT_LIFECYCLE_CHANGED"
      ) {
        this.renderTimeline();
      }
    });

    // Re-render timeline every 5s to keep window rolling
    setInterval(() => {
      this.renderTimeline();
    }, 5000);
  }

  render() {
    this.container.innerHTML = `
      <div class="timeline-strip-card">
        <div class="timeline-header">
          <div class="timeline-title">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
            </svg>
            Activity Timeline <span id="timeline-window-label" class="timeline-window-tag">2m Window</span>
          </div>
          <div class="timeline-legend">
            <div class="legend-item"><span class="legend-dot high"></span> High Risk</div>
            <div class="legend-item"><span class="legend-dot medium"></span> Medium</div>
            <div class="legend-item"><span class="legend-dot low"></span> Low</div>
          </div>
        </div>

        <div class="timeline-canvas-container" id="timeline-canvas-container">
          <div class="timeline-tracks-area" id="timeline-tracks-area">
            <div class="timeline-empty-notice">
              Continuous monitoring active. Behavioral observations will display here across temporal lanes.
            </div>
          </div>

          <div class="timeline-time-axis" id="timeline-time-axis">
            <!-- Dynamic adaptive ticks -->
          </div>
        </div>
      </div>
    `;
    this.renderTimeline();
  }

  renderTimeline() {
    const tracksArea = document.getElementById("timeline-tracks-area");
    const axisEl = document.getElementById("timeline-time-axis");
    const windowLabel = document.getElementById("timeline-window-label");
    if (!tracksArea || !axisEl) return;

    const now = Date.now() / 1000;
    const sessionAgeSec = Math.max(10, now - appState.sessionStartTime / 1000);

    // Adaptive window calculation
    let windowSec = 120; // 2m
    let ticks = ["-2m", "-90s", "-60s", "-30s", "Now"];
    let tag = "2m Window";

    if (sessionAgeSec > 600) {
      windowSec = 900; // 15m
      ticks = ["-15m", "-12m", "-9m", "-6m", "-3m", "Now"];
      tag = "15m Rolling Window";
    } else if (sessionAgeSec > 300) {
      windowSec = 600; // 10m
      ticks = ["-10m", "-8m", "-6m", "-4m", "-2m", "Now"];
      tag = "10m Window";
    } else if (sessionAgeSec > 120) {
      windowSec = 300; // 5m
      ticks = ["-5m", "-4m", "-3m", "-2m", "-1m", "Now"];
      tag = "5m Window";
    }

    if (windowLabel) windowLabel.innerText = tag;

    // Render time axis
    axisEl.innerHTML = ticks.map((t) => `<span>${t}</span>`).join("");

    const windowStart = now - windowSec;
    const events = Array.from(appState.events.values());

    if (events.length === 0) {
      tracksArea.innerHTML = `
        <div class="timeline-empty-notice">
          Continuous monitoring active on physical camera. No observable events in this session.
        </div>
      `;
      return;
    }

    // Group events by track ID
    const byTrack = new Map();
    for (const ev of events) {
      const evEnd = ev.lifecycle === "closed" ? (ev.endTime || ev.timestamp) : now;
      if (evEnd < windowStart) continue; // Outside window
      if (!byTrack.has(ev.trackId)) byTrack.set(ev.trackId, []);
      byTrack.get(ev.trackId).push(ev);
    }

    if (byTrack.size === 0) {
      tracksArea.innerHTML = `
        <div class="timeline-empty-notice">
          No observable events in the active ${tag.toLowerCase()}.
        </div>
      `;
      return;
    }

    // Display max 4 active student lanes
    const trackIds = Array.from(byTrack.keys()).sort((a, b) => a - b).slice(0, 4);

    tracksArea.innerHTML = trackIds
      .map((tid) => {
        const trEvs = byTrack.get(tid);
        const markersHtml = trEvs
          .map((ev) => {
            const startClamped = Math.max(windowStart, ev.startTime || ev.timestamp);
            const evEnd = ev.lifecycle === "closed" ? (ev.endTime || ev.timestamp) : now;
            const endClamped = Math.min(now, Math.max(startClamped + 2.0, evEnd));

            const leftPct = Math.max(0, Math.min(99, ((startClamped - windowStart) / windowSec) * 100));
            const widthPct = Math.max(1.8, Math.min(100 - leftPct, ((endClamped - startClamped) / windowSec) * 100));
            const riskLower = ev.riskLevel.toLowerCase();

            // Compact event label if block is wide enough (> 6%)
            const showLabel = widthPct > 6;
            const shortLabel = getShortEventLabel(ev.canonicalType);

            const tooltip = `${ev.displayName} (Student #${ev.trackId})\n` +
              `Duration: ${formatDuration(ev.duration || (endClamped - startClamped))}\n` +
              `Evidence Risk: ${ev.score}/100 (${ev.riskLevel})\n` +
              `Status: ${ev.reviewStatus.toUpperCase()} · ${ev.lifecycle.toUpperCase()}`;

            return `
              <div 
                class="timeline-event-marker ${riskLower} ${ev.lifecycle === "closed" ? "closed" : "active"}" 
                style="left: ${leftPct.toFixed(1)}%; width: ${widthPct.toFixed(1)}%;" 
                title="${tooltip}"
                data-id="${ev.eventId}"
              >
                ${showLabel ? `<span class="marker-label">${shortLabel}</span>` : ""}
              </div>
            `;
          })
          .join("");

        return `
          <div class="timeline-track-lane">
            <span class="timeline-lane-label">Student #${tid}</span>
            <div class="timeline-markers-track">
              ${markersHtml}
            </div>
          </div>
        `;
      })
      .join("");

    tracksArea.querySelectorAll(".timeline-event-marker").forEach((marker) => {
      marker.addEventListener("click", () => {
        const eid = marker.dataset.id;
        appState.selectEvent(eid);
      });
    });
  }
}

function getShortEventLabel(type) {
  if (type === "SUSTAINED_HEAD_REST") return "HEAD REST";
  if (type === "PHONE_ASSOCIATED") return "PHONE";
  if (type === "SUSTAINED_LATERAL_HEAD_ORIENTATION") return "HEAD TURN";
  if (type === "STANDING") return "STANDING";
  if (type === "DISCUSSION_CANDIDATE") return "DISCUSS";
  return "EVENT";
}
