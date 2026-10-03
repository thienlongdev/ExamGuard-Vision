import { appState } from "../state.js";
import { formatDuration } from "../adapter.js";
import { getSeverityInfo, REVIEW_STATUS_VI } from "../localization.js";

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
            Dòng thời gian hoạt động <span id="timeline-window-label" class="timeline-window-tag">Khung 2 phút</span>
          </div>
          <div class="timeline-legend">
            <div class="legend-item"><span class="legend-dot high"></span> Cảnh báo cao</div>
            <div class="legend-item"><span class="legend-dot medium"></span> Cần chú ý</div>
            <div class="legend-item"><span class="legend-dot low"></span> Thông tin</div>
          </div>
        </div>

        <div class="timeline-canvas-container" id="timeline-canvas-container">
          <div class="timeline-tracks-area" id="timeline-tracks-area">
            <div class="timeline-empty-notice">
              Hệ thống giám sát liên tục đang hoạt động. Các sự kiện quan sát sẽ hiển thị tại đây theo từng thí sinh.
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
    let ticks = ["-2p", "-90s", "-60s", "-30s", "Hiện tại"];
    let tag = "Khung 2 phút";

    if (sessionAgeSec > 600) {
      windowSec = 900; // 15m
      ticks = ["-15p", "-12p", "-9p", "-6p", "-3p", "Hiện tại"];
      tag = "Khung 15 phút";
    } else if (sessionAgeSec > 300) {
      windowSec = 600; // 10m
      ticks = ["-10p", "-8p", "-6p", "-4p", "-2p", "Hiện tại"];
      tag = "Khung 10 phút";
    } else if (sessionAgeSec > 120) {
      windowSec = 300; // 5m
      ticks = ["-5p", "-4p", "-3p", "-2p", "-1p", "Hiện tại"];
      tag = "Khung 5 phút";
    }

    if (windowLabel) windowLabel.innerText = tag;

    // Render time axis
    axisEl.innerHTML = ticks.map((t) => `<span>${t}</span>`).join("");

    const windowStart = now - windowSec;
    const events = Array.from(appState.events.values());

    if (events.length === 0) {
      tracksArea.innerHTML = `
        <div class="timeline-empty-notice">
          Hệ thống giám sát liên tục đang hoạt động trên camera trực tiếp. Chưa có sự kiện quan sát trong phiên này.
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
          Không có sự kiện quan sát nào trong ${tag.toLowerCase()}.
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
            const sev = getSeverityInfo(ev.riskLevel);
            const riskLower = sev.key;

            const showLabel = widthPct > 6;
            const shortLabel = getShortEventLabel(ev.canonicalType);

            const rStatusVi = REVIEW_STATUS_VI[ev.reviewStatus] || ev.reviewStatus;
            const lifecycleVi = ev.lifecycle === "closed" ? "Đã kết thúc" : "Đang diễn ra";

            const tooltip = `${ev.displayName} (Thí sinh #${ev.trackId})\n` +
              `Thời lượng: ${formatDuration(ev.duration || (endClamped - startClamped))}\n` +
              `Mức cảnh báo: ${ev.score}/100 (${sev.badge})\n` +
              `Trạng thái: ${rStatusVi} · ${lifecycleVi}`;

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
            <span class="timeline-lane-label">Thí sinh #${tid}</span>
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
  if (type === "SUSTAINED_HEAD_REST") return "GỤC ĐẦU";
  if (type === "PHONE_ASSOCIATED") return "ĐIỆN THOẠI";
  if (type === "SUSTAINED_LATERAL_HEAD_ORIENTATION") return "QUAY ĐẦU";
  if (type === "STANDING") return "ĐỨNG DẬY";
  if (type === "DISCUSSION_CANDIDATE") return "TRAO ĐỔI";
  return "SỰ KIỆN";
}
