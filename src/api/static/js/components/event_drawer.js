/**
 * ExamGuard Vision — Event Details Drawer Component
 * Comprehensive inspector for human review, behavioral evidence, and status updates.
 */

import { appState } from "../state.js";
import { formatDuration } from "../adapter.js";
import { ApiClient } from "../api.js";
import {
  EVENT_NAMES_VI,
  POSTURE_NAMES_VI,
  SEVERITY_BANDS_VI,
  REVIEW_STATUS_VI,
  SOURCE_ORIGIN_VI,
  PHONE_ASSOCIATION_VI,
  ALERT_SEMANTICS_TOOLTIP,
  formatDurationVi
} from "../localization.js";

export class EventDrawerComponent {
  constructor(backdropId, drawerId) {
    this.backdrop = document.getElementById(backdropId);
    this.drawer = document.getElementById(drawerId);
    this.currentEvent = null;
    this.init();
  }

  init() {
    this.bindEvents();

    appState.subscribe((type, payload) => {
      if (type === "EVENT_SELECTED") {
        if (payload) {
          const ev = appState.events.get(payload);
          if (ev) this.open(ev);
        } else {
          this.close();
        }
      } else if (
        (type === "EVENT_STATUS_CHANGED" || type === "EVENT_LIFECYCLE_CHANGED" || type === "EVENTS_UPDATED") &&
        this.currentEvent &&
        payload &&
        payload.eventId === this.currentEvent.eventId
      ) {
        this.currentEvent = payload;
        this.renderContent();
      }
    });
  }

  bindEvents() {
    this.backdrop.addEventListener("click", () => {
      appState.selectEvent(null);
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && this.drawer.classList.contains("open")) {
        appState.selectEvent(null);
      }
    });
  }

  open(event) {
    this.currentEvent = event;
    this.renderContent();
    this.backdrop.classList.add("open");
    this.drawer.classList.add("open");
  }

  close() {
    this.backdrop.classList.remove("open");
    this.drawer.classList.remove("open");
    this.currentEvent = null;
  }

  renderContent() {
    if (!this.currentEvent) return;
    const ev = this.currentEvent;
    const sevInfo = SEVERITY_BANDS_VI[ev.riskLevel] || { label: ev.riskLevel, color: "var(--text-secondary)" };
    const sevLabel = sevInfo.label;
    const timeStr = new Date(ev.timestamp * 1000).toLocaleTimeString([], { hour12: false });
    const durationStr = formatDurationVi(ev.duration);

    // Behavioral cues from event-time snapshot (immutable fact)
    const obs = ev.observationSnapshot || {};
    const evObj = ev.evidence || {};

    // Posture cue
    let postureClass = obs.posture?.class || evObj.posture;
    if (!postureClass || postureClass === "UNKNOWN") {
      if (ev.canonicalType === "SUSTAINED_HEAD_REST") postureClass = "HEAD_REST_SLEEP";
      else if (ev.canonicalType === "SUSTAINED_LATERAL_HEAD_ORIENTATION") postureClass = "TURN_HEAD_CLEAR";
      else if (ev.canonicalType === "STANDING") postureClass = "STANDING";
      else postureClass = "NOT_EVALUATED";
    }
    const postureConfVal = obs.posture?.confidence !== undefined ? obs.posture.confidence : evObj.posture_confidence;
    const postureConf = postureConfVal !== undefined && postureConfVal !== null ? `${(postureConfVal * 100).toFixed(0)}%` : null;
    const postureVi = POSTURE_NAMES_VI[postureClass] || postureClass;
    const postureStr = postureConf ? `${postureVi} (${postureConf})` : postureVi;

    // Head yaw cue
    const rawYaw = obs.headpose?.yaw_deg !== undefined && obs.headpose?.yaw_deg !== null ? obs.headpose.yaw_deg : (evObj.yaw_deg !== undefined ? evObj.yaw_deg : null);
    const yawStr = rawYaw !== null ? `${Number(rawYaw).toFixed(1)}°` : "Chưa có dữ liệu";

    // Phone cue
    const phoneAssoc = obs.phone?.association_status || evObj.phone_status || (ev.canonicalType === "PHONE_ASSOCIATED" ? "CLEAR_ASSOCIATION" : "NO_PHONE");
    const phoneVi = PHONE_ASSOCIATION_VI[phoneAssoc] || (phoneAssoc === "NO_PHONE" ? "Không phát hiện" : phoneAssoc);

    // Macro cue
    const rawMacro = obs.macro?.cue || evObj.macro_behavior || "NOT_EVALUATED";
    const macroVi = POSTURE_NAMES_VI[rawMacro] || (rawMacro === "NOT_EVALUATED" ? "Chưa đánh giá" : rawMacro);

    // Status separation:
    const lifecycleLabel = (ev.lifecycle === "closed") ? "Sự kiện đã kết thúc" : "Đang diễn ra (Theo dõi)";
    const lifecycleClass = (ev.lifecycle === "closed") ? "lifecycle-closed" : "lifecycle-active";

    const rStatus = ev.reviewStatus || "awaiting";
    const revInfo = REVIEW_STATUS_VI[rStatus] || { label: "Chờ duyệt", class: "review-awaiting" };
    const reviewDisplay = revInfo.label;
    const reviewClass = revInfo.class;

    // Detection origin
    const originRaw = ev.raw?.event_origin || ev.eventOrigin || "PHYSICAL_LIVE_CAMERA";
    const originVi = SOURCE_ORIGIN_VI[originRaw] || "Camera trực tiếp";

    const displayNameVi = EVENT_NAMES_VI[ev.canonicalType] || ev.displayName;

    this.drawer.innerHTML = `
      <div class="drawer-header">
        <div>
          <h3>${displayNameVi}</h3>
          <span style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-dim);">${ev.canonicalType}</span>
        </div>
        <button class="drawer-close-btn" id="drawer-btn-close" title="Đóng bảng chi tiết (Esc)">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        </button>
      </div>

      <div class="drawer-content">
        <!-- 1. Primary Metadata -->
        <div class="drawer-section">
          <span class="drawer-section-title">Thông tin sự kiện</span>
          <div class="drawer-meta-grid">
            <div class="drawer-meta-item">
              <div class="meta-label">Thí sinh</div>
              <div class="meta-value">Thí sinh #${ev.trackId}</div>
            </div>
            <div class="drawer-meta-item">
              <div class="meta-label">Camera</div>
              <div class="meta-value">${ev.cameraId}</div>
            </div>
            <div class="drawer-meta-item" title="${ALERT_SEMANTICS_TOOLTIP}">
              <div class="meta-label">Mức cảnh báo</div>
              <div class="meta-value" style="color: ${sevInfo.color}">${sevLabel} · ${ev.score}/100</div>
            </div>
            <div class="drawer-meta-item">
              <div class="meta-label">Thời lượng</div>
              <div class="meta-value">${durationStr}</div>
            </div>
            <div class="drawer-meta-item">
              <div class="meta-label">Thời điểm</div>
              <div class="meta-value">${timeStr}</div>
            </div>
            <div class="drawer-meta-item">
              <div class="meta-label">Nguồn phát hiện</div>
              <div class="meta-value" style="font-size: 0.72rem;">${originVi}</div>
            </div>
          </div>
        </div>

        <!-- 2. Large Evidence Snapshot -->
        <div class="drawer-section">
          <span class="drawer-section-title">Ảnh bằng chứng</span>
          <div class="drawer-snapshot-box" id="drawer-snapshot-container">
            ${
              ev.snapshotUrl
                ? `<img class="drawer-snapshot-img" src="${ev.snapshotUrl}" alt="Ảnh bằng chứng" loading="lazy" onerror="this.style.display='none'; document.getElementById('drawer-snap-placeholder').style.display='flex';" />
                   <div class="drawer-snapshot-placeholder" id="drawer-snap-placeholder" style="display: none;">
                     <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                       <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
                     </svg>
                     <span>Chưa có ảnh bằng chứng cho sự kiện này</span>
                   </div>`
                : `<div class="drawer-snapshot-placeholder">
                     <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                       <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>
                     </svg>
                     <span>Chưa có ảnh bằng chứng cho sự kiện này</span>
                   </div>`
            }
          </div>
        </div>

        <!-- 3. Observable Behavioral Cues (Why this event appeared) -->
        <div class="drawer-section">
          <span class="drawer-section-title">Vì sao có cảnh báo?</span>
          <div class="drawer-cues-list">
            <div class="cue-row">
              <span class="cue-name">Tư thế</span>
              <span class="cue-val">${postureStr}</span>
            </div>
            <div class="cue-row">
              <span class="cue-name">Góc quay đầu</span>
              <span class="cue-val">${yawStr}</span>
            </div>
            <div class="cue-row">
              <span class="cue-name">Liên kết điện thoại</span>
              <span class="cue-val">${phoneVi}</span>
            </div>
            <div class="cue-row">
              <span class="cue-name">Hành vi tổng quát</span>
              <span class="cue-val">${macroVi}</span>
            </div>
          </div>
        </div>

        <!-- 4. Separate Lifecycle & Human Review Status -->
        <div class="drawer-section">
          <span class="drawer-section-title">Trạng thái & xử lý</span>
          <div class="drawer-status-split">
            <div class="status-block">
              <span class="status-block-label">Vòng đời sự kiện</span>
              <span class="status-block-value ${lifecycleClass}">${lifecycleLabel}</span>
            </div>
            <div class="status-block">
              <span class="status-block-label">Giám thị duyệt</span>
              <span class="status-block-value ${reviewClass}">${reviewDisplay}</span>
            </div>
          </div>
        </div>

        <!-- 5. Invigilator Notes -->
        <div class="drawer-section">
          <span class="drawer-section-title">Ghi chú của giám thị</span>
          <textarea 
            class="drawer-notes-area" 
            id="drawer-notes" 
            maxlength="500" 
            placeholder="Ghi chú thêm về hành vi quan sát được để lưu vào nhật ký rà soát…"
          >${escapeHtml(ev.reviewerNotes || "")}</textarea>
          <div style="font-size: 0.68rem; color: var(--text-dim); text-align: right; margin-top: 3px;">Tối đa 500 ký tự · Lưu vào nhật ký sự kiện</div>
        </div>
      </div>

      <div class="drawer-footer">
        <div class="drawer-btn-group">
          <button class="btn-action btn-confirm" id="btn-drawer-confirm" ${rStatus === "confirmed" ? 'disabled style="opacity: 0.6;"' : ""}>Xác nhận</button>
          <button class="btn-action btn-dismiss" id="btn-drawer-dismiss" ${rStatus === "dismissed" ? 'disabled style="opacity: 0.6;"' : ""}>Bỏ qua</button>
        </div>
      </div>
    `;

    document.getElementById("drawer-btn-close")?.addEventListener("click", () => {
      appState.selectEvent(null);
    });

    document.getElementById("btn-drawer-confirm")?.addEventListener("click", () => {
      this.handleAction("confirmed");
    });

    document.getElementById("btn-drawer-dismiss")?.addEventListener("click", () => {
      this.handleAction("dismissed");
    });
  }

  async handleAction(newStatus) {
    if (!this.currentEvent) return;
    const eid = this.currentEvent.eventId;
    const rawNotes = document.getElementById("drawer-notes")?.value || "";
    const notes = rawNotes.trim().slice(0, 500) || null;

    const btnConfirm = document.getElementById("btn-drawer-confirm");
    const btnDismiss = document.getElementById("btn-drawer-dismiss");
    if (btnConfirm) btnConfirm.disabled = true;
    if (btnDismiss) btnDismiss.disabled = true;

    try {
      await ApiClient.updateEventStatus(eid, newStatus, notes);
      appState.updateEventReviewStatus(eid, newStatus, notes);
      this.close();
    } catch (err) {
      console.error("Failed to update status from drawer:", err);
      if (btnConfirm) btnConfirm.disabled = false;
      if (btnDismiss) btnDismiss.disabled = false;
      alert(`Lỗi cập nhật trạng thái sự kiện: ${err.message}`);
    }
  }
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
