import { normalizeSnapshotUrl } from "./adapter.js";

/**
 * Authoritative human-review queue sorting comparator (Section 37 & 38)
 * 1. HIGH (CẢNH BÁO CAO) first
 * 2. MEDIUM (CẦN CHÚ Ý) second
 * 3. LOW (THÔNG TIN) last
 * Within same severity:
 * - ACTIVE / OPEN incidents first
 * - Newest event first (opened_at / timestamp DESC)
 */
export function compareEventsForReview(a, b) {
  const rank = (sev) => (sev === "HIGH" ? 3 : (sev === "MEDIUM" ? 2 : 1));
  const diffSev = rank(b.riskLevel) - rank(a.riskLevel);
  if (diffSev !== 0) return diffSev;

  const activeRank = (lc) => (lc === "active" || lc === "open" ? 1 : 0);
  const diffActive = activeRank(b.lifecycle) - activeRank(a.lifecycle);
  if (diffActive !== 0) return diffActive;

  const tA = a.startTime || a.timestamp || 0;
  const tB = b.startTime || b.timestamp || 0;
  return tB - tA;
}

class DashboardState {
  constructor() {
    this.currentView = "monitor"; // monitor | review | system
    this.events = new Map();
    this.activeFilter = "all"; // all | awaiting | reviewed | dismissed
    this.cameraFilter = "all";
    this.selectedEventId = null;
    this.highlightedTrackId = null;
    this.activeTracks = [];
    this.cameraInfo = null;
    this.systemStatus = null;
    this.systemModels = null;
    this.sessionStartTime = Date.now();
    this.currentSession = null;
    this.listeners = new Set();
    this.wsStatus = "disconnected";

    // Bounded rolling telemetry history for System view (max 60 samples ~ 2.5m)
    this.telemetryHistory = {
      timestamps: [],
      captureFps: [],
      processedFps: [],
      p50LatencyMs: [],
      queueDepth: [],
      vramMb: [],
    };
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  notify(eventType, payload) {
    for (const listener of this.listeners) {
      try {
        listener(eventType, payload, this);
      } catch (e) {
        console.error("State listener error:", e);
      }
    }
  }

  setView(viewName) {
    if (this.currentView !== viewName) {
      this.currentView = viewName;
      this.notify("VIEW_CHANGED", viewName);
    }
  }

  setWsStatus(status) {
    this.wsStatus = status;
    this.notify("WS_STATUS_CHANGED", status);
  }

  setFilter(filterName) {
    if (this.activeFilter !== filterName) {
      this.activeFilter = filterName;
      this.notify("FILTER_CHANGED", filterName);
    }
  }

  selectEvent(eventId) {
    this.selectedEventId = eventId;
    if (eventId && this.events.has(eventId)) {
      this.highlightedTrackId = this.events.get(eventId).trackId;
    } else {
      this.highlightedTrackId = null;
    }
    this.notify("EVENT_SELECTED", eventId);
  }

  setHighlightedTrack(trackId) {
    if (this.highlightedTrackId !== trackId) {
      this.highlightedTrackId = trackId;
      this.notify("TRACK_HIGHLIGHT_CHANGED", trackId);
    }
  }

  upsertEvent(normEvent) {
    if (!normEvent || !normEvent.eventId) return;
    const existing = this.events.get(normEvent.eventId);
    if (existing) {
      // Preserve user-adjudicated review status
      if (existing.reviewStatus === "confirmed" || existing.reviewStatus === "dismissed") {
        normEvent.reviewStatus = existing.reviewStatus;
        normEvent.status = existing.reviewStatus;
      }
      if (existing.reviewerNotes && !normEvent.reviewerNotes) {
        normEvent.reviewerNotes = existing.reviewerNotes;
      }
      if (existing.snapshotUrl && !normEvent.snapshotUrl) {
        normEvent.snapshotUrl = existing.snapshotUrl;
      }
      if (existing.observationSnapshot && Object.keys(existing.observationSnapshot).length > 0) {
        if (!normEvent.observationSnapshot || Object.keys(normEvent.observationSnapshot).length === 0) {
          normEvent.observationSnapshot = existing.observationSnapshot;
        }
      }
    }
    this.events.set(normEvent.eventId, normEvent);
    this.notify("EVENTS_UPDATED", normEvent);
  }

  setAllEvents(eventList) {
    for (const ev of eventList) {
      if (ev && ev.eventId) {
        const existing = this.events.get(ev.eventId);
        if (existing) {
          if (existing.reviewStatus === "confirmed" || existing.reviewStatus === "dismissed") {
            ev.reviewStatus = existing.reviewStatus;
            ev.status = existing.reviewStatus;
          }
          if (existing.reviewerNotes && !ev.reviewerNotes) {
            ev.reviewerNotes = existing.reviewerNotes;
          }
        }
        this.events.set(ev.eventId, ev);
      }
    }
    this.notify("EVENTS_RESET", this.events);
  }

  updateEventLifecycle(eventId, lifecycle) {
    const ev = this.events.get(eventId);
    if (ev) {
      ev.lifecycle = lifecycle;
      this.notify("EVENT_LIFECYCLE_CHANGED", ev);
      this.notify("EVENTS_UPDATED", ev);
    }
  }

  updateEventReviewStatus(eventId, newStatus, reviewerNotes = null) {
    const ev = this.events.get(eventId);
    if (ev) {
      const norm = newStatus === "confirmed" ? "confirmed" : (newStatus === "dismissed" ? "dismissed" : "awaiting");
      ev.reviewStatus = norm;
      ev.status = norm;
      if (reviewerNotes !== null && reviewerNotes !== undefined) {
        ev.reviewerNotes = reviewerNotes;
      }
      this.notify("EVENT_STATUS_CHANGED", ev);
      this.notify("EVENTS_UPDATED", ev);
    }
  }

  updateEventStatus(eventId, newStatus, reviewerNotes = null) {
    if (newStatus === "closed") {
      this.updateEventLifecycle(eventId, "closed");
      return;
    }
    this.updateEventReviewStatus(eventId, newStatus, reviewerNotes);
  }

  setActiveTracks(tracks) {
    this.activeTracks = tracks || [];
    this.notify("TRACKS_UPDATED", this.activeTracks);
  }

  setCameraInfo(camera) {
    this.cameraInfo = camera;
    this.notify("CAMERA_UPDATED", camera);
  }

  setSystemStatus(status) {
    this.systemStatus = status;

    // Add rolling sample to telemetry history (max 60)
    if (status) {
      const now = Date.now();
      const obs = status.observed_rates || {};
      const capFps = obs.capture_fps || 30.0;
      const procFps = obs.processed_fps || status.effective_fps || 0.0;
      const p50 = obs.p50_latency_ms || 28.0;
      const q = status.queue_depth || 0;
      const vram = status.gpu_vram_allocated_mb || 0.0;

      const h = this.telemetryHistory;
      h.timestamps.push(now);
      h.captureFps.push(capFps);
      h.processedFps.push(procFps);
      h.p50LatencyMs.push(p50);
      h.queueDepth.push(q);
      h.vramMb.push(vram);

      const maxSamples = 60;
      if (h.timestamps.length > maxSamples) {
        h.timestamps.shift();
        h.captureFps.shift();
        h.processedFps.shift();
        h.p50LatencyMs.shift();
        h.queueDepth.shift();
        h.vramMb.shift();
      }
    }

    this.notify("SYSTEM_STATUS_UPDATED", status);
  }

  setSystemModels(models) {
    this.systemModels = models;
    this.notify("SYSTEM_MODELS_UPDATED", models);
  }

  setCurrentSession(session) {
    this.currentSession = session;
    this.notify("CURRENT_SESSION_UPDATED", session);
  }

  resetForNewSession(session) {
    this.events.clear();
    this.currentSession = session;
    this.selectedEventId = null;
    this.notify("EVENTS_RESET", null);
    this.notify("CURRENT_SESSION_UPDATED", session);
  }

  // Derived KPIs based on human review status
  getKPIs() {
    const all = Array.from(this.events.values());
    const total = all.length;
    const awaiting = all.filter((e) => e.reviewStatus === "awaiting").length;
    const confirmed = all.filter((e) => e.reviewStatus === "confirmed").length;
    const dismissed = all.filter((e) => e.reviewStatus === "dismissed").length;
    return { total, awaiting, confirmed, dismissed };
  }

  setCameraFilter(camId) {
    if (this.cameraFilter !== camId) {
      this.cameraFilter = camId;
      this.notify("FILTER_CHANGED", camId);
    }
  }

  updateEventEvidence(eventId, evidenceType, filePath) {
    const ev = this.events.get(eventId);
    if (ev) {
      if (evidenceType === "SNAPSHOT") {
        ev.snapshotUrl = normalizeSnapshotUrl(filePath);
        ev.snapshotStatus = "READY";
      } else if (evidenceType === "VIDEO_CLIP") {
        ev.clipUrl = normalizeSnapshotUrl(filePath);
        ev.clipStatus = "READY";
      } else if (evidenceType === "VIDEO_CLIP_FAILED") {
        ev.clipStatus = "FAILED";
      }
      this.notify("EVIDENCE_UPDATED", { eventId, evidenceType, event: ev });
      this.notify("EVENTS_UPDATED", ev);
    }
  }

  // Monitor-side operational inbox: ONLY events awaiting human review (Section 34, 37, 39)
  getPendingReviewEvents() {
    let pending = Array.from(this.events.values()).filter(
      (e) => (e.reviewStatus || "awaiting") === "awaiting" && e.reviewStatus !== "internal"
    );

    if (this.cameraFilter && this.cameraFilter !== "all") {
      pending = pending.filter((e) => e.cameraId === this.cameraFilter);
    }

    pending.sort(compareEventsForReview);
    return pending.slice(0, 200);
  }

  getFilteredEvents() {
    let all = Array.from(this.events.values());
    all.sort(compareEventsForReview);

    if (this.cameraFilter && this.cameraFilter !== "all") {
      all = all.filter((e) => e.cameraId === this.cameraFilter);
    }

    if (this.activeFilter === "awaiting") {
      return all.filter((e) => (e.reviewStatus || "awaiting") === "awaiting" && e.reviewStatus !== "internal");
    }
    if (this.activeFilter === "reviewed") {
      return all.filter((e) => e.reviewStatus === "confirmed");
    }
    if (this.activeFilter === "dismissed") {
      return all.filter((e) => e.reviewStatus === "dismissed");
    }
    return all;
  }
}

export const appState = new DashboardState();
