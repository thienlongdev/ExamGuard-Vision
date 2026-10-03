/**
 * ExamGuard Vision — Event Details Drawer Component
 * Comprehensive inspector for human review, behavioral evidence, and status updates.
 */

import { appState } from "../state.js";
import { formatDuration } from "../adapter.js";
import { ApiClient } from "../api.js";

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
    const riskLower = ev.riskLevel.toLowerCase();
    const timeStr = new Date(ev.timestamp * 1000).toLocaleTimeString([], { hour12: false });
    const durationStr = formatDuration(ev.duration);

    // Behavioral cues from event-time snapshot (immutable fact)
    const obs = ev.observationSnapshot || {};
    const evObj = ev.evidence || {};

    // Posture cue
    let postureClass = obs.posture?.class || evObj.posture;
    if (!postureClass || postureClass === "UNKNOWN") {
      if (ev.canonicalType === "SUSTAINED_HEAD_REST") postureClass = "HEAD_REST_SLEEP";
      else if (ev.canonicalType === "SUSTAINED_LATERAL_HEAD_ORIENTATION") postureClass = "TURN_HEAD_CLEAR";
      else if (ev.canonicalType === "STANDING") postureClass = "STANDING";
      else postureClass = "Unavailable";
    }
    const postureConfVal = obs.posture?.confidence !== undefined ? obs.posture.confidence : evObj.posture_confidence;
    const postureConf = postureConfVal !== undefined && postureConfVal !== null ? `${(postureConfVal * 100).toFixed(0)}%` : "Captured";
    const postureStr = `${postureClass} (${postureConf})`;

    // Head yaw cue
    const rawYaw = obs.headpose?.yaw_deg !== undefined && obs.headpose?.yaw_deg !== null ? obs.headpose.yaw_deg : (evObj.yaw_deg !== undefined ? evObj.yaw_deg : null);
    const yawStr = rawYaw !== null ? `${Number(rawYaw).toFixed(1)}°` : "Unavailable";

    // Phone cue
    const phoneAssoc = obs.phone?.association_status || evObj.phone_status || (ev.canonicalType === "PHONE_ASSOCIATED" ? "ASSOCIATED" : "None");
    const phoneStr = (phoneAssoc === "NO_PHONE" || phoneAssoc === "NONE") ? "None" : phoneAssoc;

    // Macro cue
    const macroStr = obs.macro?.cue || evObj.macro_behavior || "None Observed";

    // Status separation:
    const lifecycleLabel = (ev.lifecycle === "closed") ? "Closed" : "Active (Observing)";
    const lifecycleClass = (ev.lifecycle === "closed") ? "lifecycle-closed" : "lifecycle-active";

    const rStatus = ev.reviewStatus || "awaiting";
    let reviewDisplay = "Awaiting Review";
    let reviewClass = "review-awaiting";
    if (rStatus === "confirmed") {
      reviewDisplay = "Human Confirmed";
      reviewClass = "review-confirmed";
    } else if (rStatus === "dismissed") {
      reviewDisplay = "Dismissed";
      reviewClass = "review-dismissed";
    }

    this.drawer.innerHTML = `
      <div class="drawer-header">
        <div>
          <h3>${ev.displayName}</h3>
          <span style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-dim);">${ev.canonicalType}</span>
        </div>
        <button class="drawer-close-btn" id="drawer-btn-close" title="Close Drawer (Esc)">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        </button>
      </div>

      <div class="drawer-content">
        <!-- 1. Primary Metadata -->
        <div class="drawer-section">
          <span class="drawer-section-title">Event Overview</span>
          <div class="drawer-meta-grid">
            <div class="drawer-meta-item">
              <div class="meta-label">Student ID</div>
              <div class="meta-value">Student #${ev.trackId}</div>
            </div>
            <div class="drawer-meta-item">
              <div class="meta-label">Camera</div>
              <div class="meta-value">${ev.cameraId}</div>
            </div>
            <div class="drawer-meta-item">
              <div class="meta-label">Evidence Risk</div>
              <div class="meta-value" style="color: var(--color-${riskLower})">${ev.score} / 100 (${ev.riskLevel})</div>
            </div>
            <div class="drawer-meta-item">
              <div class="meta-label">Duration</div>
              <div class="meta-value">${durationStr}</div>
            </div>
            <div class="drawer-meta-item">
              <div class="meta-label">Timestamp</div>
              <div class="meta-value">${timeStr}</div>
            </div>
            <div class="drawer-meta-item">
              <div class="meta-label">Detection Origin</div>
              <div class="meta-value" style="font-size: 0.72rem;">${ev.raw?.event_origin || "PHYSICAL_CAMERA"}</div>
            </div>
          </div>
        </div>

        <!-- 2. Large Evidence Snapshot -->
        <div class="drawer-section">
          <span class="drawer-section-title">Visual Evidence Snapshot</span>
          <div class="drawer-snapshot-box" id="drawer-snapshot-container">
            ${
              ev.snapshotUrl
                ? `<img class="drawer-snapshot-img" src="${ev.snapshotUrl}" alt="Evidence Snapshot" loading="lazy" onerror="this.style.display='none'; document.getElementById('drawer-snap-placeholder').style.display='flex';" />
                   <div class="drawer-snapshot-placeholder" id="drawer-snap-placeholder" style="display: none;">
                     <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                       <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
                     </svg>
                     <span>Evidence snapshot not captured for this event</span>
                   </div>`
                : `<div class="drawer-snapshot-placeholder">
                     <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                       <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>
                     </svg>
                     <span>Evidence snapshot not captured for this event</span>
                   </div>`
            }
          </div>
        </div>

        <!-- 3. Observable Behavioral Cues (Why this event appeared) -->
        <div class="drawer-section">
          <span class="drawer-section-title">Why This Event Appeared (Event-Time Cues)</span>
          <div class="drawer-cues-list">
            <div class="cue-row">
              <span class="cue-name">Posture State</span>
              <span class="cue-val">${postureStr}</span>
            </div>
            <div class="cue-row">
              <span class="cue-name">Head Yaw Deviation</span>
              <span class="cue-val">${yawStr}</span>
            </div>
            <div class="cue-row">
              <span class="cue-name">Phone Association</span>
              <span class="cue-val">${phoneStr}</span>
            </div>
            <div class="cue-row">
              <span class="cue-name">Macro Behavior Cue</span>
              <span class="cue-val">${macroStr}</span>
            </div>
          </div>
        </div>

        <!-- 4. Separate Lifecycle & Human Review Status -->
        <div class="drawer-section">
          <span class="drawer-section-title">Status & Adjudication</span>
          <div class="drawer-status-split">
            <div class="status-block">
              <span class="status-block-label">Event Lifecycle</span>
              <span class="status-block-value ${lifecycleClass}">${lifecycleLabel}</span>
            </div>
            <div class="status-block">
              <span class="status-block-label">Human Review</span>
              <span class="status-block-value ${reviewClass}">${reviewDisplay}</span>
            </div>
          </div>
        </div>

        <!-- 5. Invigilator Notes -->
        <div class="drawer-section">
          <span class="drawer-section-title">Invigilator Notes</span>
          <textarea 
            class="drawer-notes-area" 
            id="drawer-notes" 
            maxlength="500" 
            placeholder="Document notes regarding this observable behavior for audit history…"
          >${escapeHtml(ev.reviewerNotes || "")}</textarea>
          <div style="font-size: 0.68rem; color: var(--text-dim); text-align: right; margin-top: 3px;">Max 500 characters · Persisted to audit log</div>
        </div>
      </div>

      <div class="drawer-footer">
        <div class="drawer-btn-group">
          <button class="btn-action btn-confirm" id="btn-drawer-confirm" ${rStatus === "confirmed" ? 'disabled style="opacity: 0.6;"' : ""}>Confirm Event</button>
          <button class="btn-action btn-dismiss" id="btn-drawer-dismiss" ${rStatus === "dismissed" ? 'disabled style="opacity: 0.6;"' : ""}>Dismiss</button>
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
      alert(`Error updating event status: ${err.message}`);
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
