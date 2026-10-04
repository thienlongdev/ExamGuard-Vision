/**
 * ExamGuard Vision — Reactive State Store
 * Centralized state management for real-time monitoring and review.
 */

class DashboardState {
  constructor() {
    this.currentView = "monitor"; // monitor | review | system
    this.events = new Map();
    this.activeFilter = "all"; // all | awaiting | reviewed | dismissed
    this.selectedEventId = null;
    this.highlightedTrackId = null;
    this.activeTracks = [];
    this.cameraInfo = null;
    this.systemStatus = null;
    this.systemModels = null;
    this.sessionStartTime = Date.now();
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

  // Derived KPIs based on human review status
  getKPIs() {
    const all = Array.from(this.events.values());
    const total = all.length;
    const awaiting = all.filter((e) => e.reviewStatus === "awaiting").length;
    const confirmed = all.filter((e) => e.reviewStatus === "confirmed").length;
    const dismissed = all.filter((e) => e.reviewStatus === "dismissed").length;
    return { total, awaiting, confirmed, dismissed };
  }

  getFilteredEvents() {
    // Event flood control: Prioritize HIGH risk first, then MEDIUM, then LOW, then recency
    const priorityWeight = { HIGH: 3, MEDIUM: 2, LOW: 1 };
    const all = Array.from(this.events.values()).sort((a, b) => {
      const pDiff = (priorityWeight[b.riskLevel] || 1) - (priorityWeight[a.riskLevel] || 1);
      if (pDiff !== 0) return pDiff;
      return (b.startTime || b.timestamp) - (a.startTime || a.timestamp);
    });
    if (this.activeFilter === "awaiting") {
      return all.filter((e) => e.reviewStatus === "awaiting");
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
