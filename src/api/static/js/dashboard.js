/**
 * ExamGuard Vision — Master Dashboard Application Bootstrap
 * Product-grade AI Exam Monitoring & Human Review Control Center
 */

import { appState } from "./state.js";
import { normalizeEvent } from "./adapter.js";
import { ApiClient } from "./api.js";
import { WebSocketClient } from "./websocket.js";

import { HeaderComponent } from "./components/header.js";
import { CameraViewComponent } from "./components/camera_view.js";
import { KPIBarComponent } from "./components/kpi_bar.js";
import { HealthStripComponent } from "./components/health_strip.js";
import { ReviewQueueComponent } from "./components/review_queue.js";
import { EventDrawerComponent } from "./components/event_drawer.js";
import { TimelineComponent } from "./components/timeline.js";
import { ReviewViewComponent } from "./components/review_view.js";
import { HistoryViewComponent } from "./components/history_view.js";
import { SystemViewComponent } from "./components/system_view.js";
import { SessionGateComponent } from "./components/session_gate.js";

class DashboardApp {
  constructor() {
    this.wsClient = null;
    this.components = {};
    this.telemetryInterval = null;
  }

  async start() {
    console.log("ExamGuard Vision Dashboard initializing...");

    // Initialize all UI components
    this.components.header = new HeaderComponent("header-container");
    this.components.cameraView = new CameraViewComponent("camera-container");
    this.components.kpiBar = new KPIBarComponent("kpi-container");
    this.components.healthStrip = new HealthStripComponent("health-strip-container");
    this.components.reviewQueue = new ReviewQueueComponent("review-queue-container");
    this.components.eventDrawer = new EventDrawerComponent("drawer-backdrop", "event-drawer");
    this.components.timeline = new TimelineComponent("timeline-container");
    this.components.reviewView = new ReviewViewComponent("view-review");
    this.components.historyView = new HistoryViewComponent("view-history");
    this.components.systemView = new SystemViewComponent("view-system");
    this.components.sessionGate = new SessionGateComponent("session-gate", "monitor-session-bar");

    // Setup view routing
    appState.subscribe((type, payload) => {
      if (type === "VIEW_CHANGED") {
        this.switchView(payload);
      }
    });

    // Initial data fetch, then route through the session gate (start new / resume active)
    await this.loadInitialData();
    this.components.sessionGate.evaluate();

    // Setup WebSocket
    this.initWebSocket();

    // Start background telemetry polling (2.5s)
    this.telemetryInterval = setInterval(() => this.pollTelemetry(), 2500);

    // Deep-linking via URL query params (e.g. ?view=review or ?event=ev_001_phone)
    const urlParams = new URLSearchParams(window.location.search);
    const viewParam = urlParams.get("view");
    if (viewParam && ["monitor", "review", "history", "system"].includes(viewParam)) {
      appState.setView(viewParam);
    }
    const eventParam = urlParams.get("event");
    if (eventParam) {
      setTimeout(() => appState.selectEvent(eventParam), 150);
    }

    console.log("ExamGuard Vision Dashboard ready.");
  }

  switchView(viewName) {
    document.querySelectorAll(".view-section").forEach((sec) => {
      sec.classList.toggle("active", sec.id === `view-${viewName}`);
    });
  }

  async loadInitialData() {
    try {
      const [eventsRaw, camsRaw, statusRaw, modelsRaw, currentSessionRaw, tracksRaw] = await Promise.allSettled([
        ApiClient.getEvents(),
        ApiClient.getCameras(),
        ApiClient.getSystemStatus(),
        ApiClient.getSystemModels(),
        ApiClient.getCurrentSession(),
        ApiClient.getCameraTracks(),
      ]);

      if (eventsRaw.status === "fulfilled" && Array.isArray(eventsRaw.value)) {
        const norm = eventsRaw.value.map(normalizeEvent).filter(Boolean);
        appState.setAllEvents(norm);
      }

      if (camsRaw.status === "fulfilled" && Array.isArray(camsRaw.value) && camsRaw.value.length > 0) {
        appState.setCameraInfo(camsRaw.value[0]);
      }

      if (statusRaw.status === "fulfilled") {
        appState.setSystemStatus(statusRaw.value);
      }

      if (modelsRaw.status === "fulfilled") {
        appState.setSystemModels(modelsRaw.value);
      }

      if (currentSessionRaw.status === "fulfilled" && currentSessionRaw.value) {
        appState.setCurrentSession(currentSessionRaw.value);
      }

      if (appState.monitoringActive && tracksRaw.status === "fulfilled" && Array.isArray(tracksRaw.value)) {
        appState.setActiveTracks(tracksRaw.value);
      }
    } catch (err) {
      console.warn("Initial data load partial failure:", err);
    }
  }

  initWebSocket() {
    this.wsClient = new WebSocketClient("/ws/events");

    this.wsClient.addStatusListener((status) => {
      appState.setWsStatus(status);
      if (this.components.header) {
        this.components.header.setWsStatus(status);
      }
    });

    this.wsClient.addListener((msg) => {
      if (!msg) return;

      // Handle event lifecycle without clobbering human review status
      if (msg.type === "NEW_EVENT" && msg.event) {
        const norm = normalizeEvent(msg.event);
        if (norm) appState.upsertEvent(norm);
      } else if (msg.type === "EVENT_OPEN" && msg.event) {
        const evData = typeof msg.event === "object" ? msg.event : JSON.parse(msg.event);
        const norm = normalizeEvent(evData);
        if (norm) {
          norm.lifecycle = "open";
          appState.upsertEvent(norm);
        }
      } else if (msg.type === "EVENT_UPDATE" && msg.event) {
        const evData = typeof msg.event === "object" ? msg.event : JSON.parse(msg.event);
        const norm = normalizeEvent(evData);
        if (norm) {
          norm.lifecycle = "active";
          appState.upsertEvent(norm);
        }
      } else if (msg.type === "EVENT_CLOSE" && msg.event) {
        const evData = typeof msg.event === "object" ? msg.event : JSON.parse(msg.event);
        const eid = evData.event_id || evData.eventId;
        if (eid) {
          appState.updateEventLifecycle(eid, "closed");
        }
      } else if (msg.type === "EVENT_STATUS_UPDATED" && msg.event_id) {
        appState.updateEventReviewStatus(msg.event_id, msg.status, msg.reviewer_notes);
      } else if (msg.type === "EVIDENCE_READY" && msg.event_id) {
        appState.updateEventEvidence(msg.event_id, msg.evidence_type, msg.file_path);
      } else if (msg.type === "MONITORING_SESSION_ENDED" && msg.session) {
        this.components.sessionGate.onRemoteSessionEnded(msg.session);
      } else if (msg.type === "MONITORING_SESSION_STARTED" && msg.session) {
        this.components.sessionGate.onRemoteSessionStarted(msg.session);
      }
    });

    this.wsClient.connect();

    // Export connectWebSocket function globally for backward test compatibility
    window.connectWebSocket = () => {
      this.wsClient.connect();
    };
  }

  async pollTelemetry() {
    try {
      // Track polling only while monitoring: no session, no overlay
      const [status, tracks, cameras] = await Promise.all([
        ApiClient.getSystemStatus().catch(() => null),
        appState.monitoringActive ? ApiClient.getCameraTracks().catch(() => []) : Promise.resolve(null),
        ApiClient.getCameras().catch(() => []),
      ]);

      if (status) appState.setSystemStatus(status);
      if (tracks && appState.monitoringActive) appState.setActiveTracks(tracks);
      if (cameras && cameras.length > 0) appState.setCameraInfo(cameras[0]);
    } catch (e) {
      console.debug("Telemetry poll error:", e);
    }
  }
}

// Global bootstrap on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  const app = new DashboardApp();
  app.start();
  window.__app = app;
});
