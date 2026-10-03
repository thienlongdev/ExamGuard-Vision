/**
 * ExamGuard Vision — WebSocket Client Manager
 * Canonical real-time event subscription with robust reconnection and state tracking.
 * Supported status states: "connecting" | "connected" | "reconnecting" | "disconnected"
 */

export class WebSocketClient {
  constructor(endpoint = "/ws/events") {
    this.endpoint = endpoint;
    this.socket = null;
    this.listeners = new Set();
    this.statusListeners = new Set();
    this.reconnectTimer = null;
    this.pingInterval = null;
    this.status = "disconnected"; // disconnected | connecting | connected | reconnecting
    this.reconnectAttempts = 0;
    this._isExplicitClose = false;
  }

  connect() {
    this._isExplicitClose = false;

    // Avoid duplicate connection if socket is already open or connecting
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    // Clean up any stale socket instance
    if (this.socket) {
      try {
        this.socket.onopen = null;
        this.socket.onmessage = null;
        this.socket.onclose = null;
        this.socket.onerror = null;
        this.socket.close();
      } catch (_) {}
      this.socket = null;
    }

    const nextStatus = this.reconnectAttempts > 0 ? "reconnecting" : "connecting";
    this._setStatus(nextStatus);

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host || "127.0.0.1:8000";
    const wsUrl = `${protocol}//${host}${this.endpoint}`;

    try {
      this.socket = new WebSocket(wsUrl);
    } catch (e) {
      console.warn("WebSocket initialization error:", e);
      this._scheduleReconnect();
      return;
    }

    this.socket.onopen = () => {
      this._setStatus("connected");
      this.reconnectAttempts = 0;
      if (this.reconnectTimer) {
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
      }
      this._startPing();
    };

    this.socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this._dispatch(data);
      } catch (err) {
        console.debug("Non-JSON WS message received:", event.data);
      }
    };

    this.socket.onclose = (event) => {
      this._stopPing();
      if (this._isExplicitClose) {
        this._setStatus("disconnected");
        return;
      }
      this._setStatus(this.reconnectAttempts > 0 ? "reconnecting" : "disconnected");
      this._scheduleReconnect();
    };

    this.socket.onerror = (err) => {
      console.debug("WebSocket error:", err);
      try {
        this.socket.close();
      } catch (_) {}
    };
  }

  disconnect() {
    this._isExplicitClose = true;
    this._stopPing();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.socket) {
      try {
        this.socket.close();
      } catch (_) {}
      this.socket = null;
    }
    this._setStatus("disconnected");
  }

  _setStatus(newStatus) {
    if (this.status === newStatus && this.status !== "connected") return;
    this.status = newStatus;
    for (const cb of this.statusListeners) {
      try {
        cb(newStatus);
      } catch (e) {
        console.error("Error in WS status listener:", e);
      }
    }
  }

  _dispatch(message) {
    for (const listener of this.listeners) {
      try {
        listener(message);
      } catch (e) {
        console.error("Error in WS event listener:", e);
      }
    }
  }

  _startPing() {
    this._stopPing();
    this.pingInterval = setInterval(() => {
      if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        this.socket.send("PING");
      }
    }, 10000);
  }

  _stopPing() {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  _scheduleReconnect() {
    if (this.reconnectTimer || this._isExplicitClose) return;
    this.reconnectAttempts++;
    this._setStatus("reconnecting");
    // Bounded exponential backoff with jitter: 1s, 1.5s, 2.25s, max 4s
    const baseDelay = Math.min(4000, 1000 * Math.pow(1.5, Math.min(this.reconnectAttempts, 5)));
    const jitter = Math.random() * 300;
    const delay = Math.round(baseDelay + jitter);

    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }

  addListener(callback) {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  addStatusListener(callback) {
    this.statusListeners.add(callback);
    callback(this.status);
    return () => this.statusListeners.delete(callback);
  }
}
