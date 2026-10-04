/**
 * ExamGuard Vision — API Client
 * Clean HTTP client for REST endpoints with CSRF, error handling, and auth session support.
 */

export class ApiClient {
  static csrfToken = "";
  static currentUser = null;

  static async initAuth() {
    try {
      const res = await fetch("/api/auth/me");
      if (res.status === 401) {
        window.location.href = "/login";
        return null;
      }
      if (res.ok) {
        const data = await res.json();
        ApiClient.csrfToken = data.csrf_token || "";
        ApiClient.currentUser = data.user || null;
        return data;
      }
    } catch (e) {
      console.warn("Could not initialize auth context:", e);
    }
    return null;
  }

  static async logout() {
    try {
      // A fresh login must pass through the session gate again
      sessionStorage.removeItem("eg_gate_session");
    } catch {}
    try {
      await fetch("/api/auth/logout", {
        method: "POST",
        headers: ApiClient._headers({}),
      });
    } catch {}
    window.location.href = "/login";
  }

  static _headers(customHeaders = {}) {
    const headers = { ...customHeaders };
    if (ApiClient.csrfToken) {
      headers["X-CSRF-Token"] = ApiClient.csrfToken;
    }
    return headers;
  }

  static async getEvents(riskLevel = null, status = null, limit = 100) {
    const params = new URLSearchParams();
    if (riskLevel) params.append("risk_level", riskLevel);
    if (status) params.append("status", status);
    if (limit) params.append("limit", limit);

    const res = await fetch(`/api/events?${params.toString()}`);
    if (res.status === 401) {
      window.location.href = "/login";
      return [];
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch events`);
    return await res.json();
  }

  static async getEvent(eventId) {
    const res = await fetch(`/api/events/${encodeURIComponent(eventId)}`);
    if (res.status === 401) {
      window.location.href = "/login";
      return null;
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch event ${eventId}`);
    return await res.json();
  }

  static async updateEventStatus(eventId, status, reviewerNotes = null) {
    const payload = { status };
    if (reviewerNotes !== null && reviewerNotes !== undefined) {
      payload.reviewer_notes = reviewerNotes;
    }

    const res = await fetch(`/api/events/${encodeURIComponent(eventId)}`, {
      method: "PATCH",
      headers: ApiClient._headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    });

    if (res.status === 401) {
      window.location.href = "/login";
      return null;
    }
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`Failed to update event: ${err || res.status}`);
    }
    return await res.json();
  }

  static async getCameras() {
    const res = await fetch("/api/cameras");
    if (res.status === 401) {
      window.location.href = "/login";
      return [];
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch cameras`);
    return await res.json();
  }

  static async setHeroCamera(cameraId) {
    const res = await fetch("/api/cameras/hero", {
      method: "POST",
      headers: ApiClient._headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({ camera_id: cameraId }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to set hero camera`);
    return await res.json();
  }

  static async getCameraTracks(camId = null) {
    try {
      const url = camId ? `/api/cameras/${encodeURIComponent(camId)}/tracks` : "/api/cameras/tracks";
      const res = await fetch(url);
      if (!res.ok) return [];
      return await res.json();
    } catch {
      return [];
    }
  }

  static async getSystemStatus() {
    const res = await fetch("/api/system/status");
    if (res.status === 401) {
      window.location.href = "/login";
      return null;
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch system status`);
    return await res.json();
  }

  static async getSystemSecurityStatus() {
    const res = await fetch("/api/system/security");
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch security status`);
    return await res.json();
  }

  static async getSystemModels() {
    const res = await fetch("/api/system/models");
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch system models`);
    return await res.json();
  }

  // --- User Management APIs (ADMIN only) ---

  static async getUsers() {
    const res = await fetch("/api/users");
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch users`);
    return await res.json();
  }

  static async createUser(data) {
    const res = await fetch("/api/users", {
      method: "POST",
      headers: ApiClient._headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Không thể tạo người dùng");
    }
    return await res.json();
  }

  static async updateUser(userId, data) {
    const res = await fetch(`/api/users/${encodeURIComponent(userId)}`, {
      method: "PATCH",
      headers: ApiClient._headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Không thể cập nhật người dùng");
    }
    return await res.json();
  }

  static async resetUserPassword(userId, newPassword, mustChangePassword = false) {
    const res = await fetch(`/api/users/${encodeURIComponent(userId)}/reset-password`, {
      method: "POST",
      headers: ApiClient._headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        new_password: newPassword,
        must_change_password: !!mustChangePassword,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Không thể đặt lại mật khẩu");
    }
    return await res.json();
  }

  static async unlockUser(userId) {
    const res = await fetch(`/api/users/${encodeURIComponent(userId)}/unlock`, {
      method: "POST",
      headers: ApiClient._headers({ "Content-Type": "application/json" }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Không thể mở khóa tài khoản");
    }
    return await res.json();
  }

  // --- Camera Config APIs (ADMIN only) ---

  static async getCamerasConfig() {
    const res = await fetch("/api/cameras/config");
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch camera configs`);
    return await res.json();
  }

  static async updateCameraConfig(camId, data) {
    const res = await fetch(`/api/cameras/config/${encodeURIComponent(camId)}`, {
      method: "PATCH",
      headers: ApiClient._headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to update camera config`);
    return await res.json();
  }

  // --- Session, History & Backup APIs ---

  static async getSessions(status = null, search = null, limit = 100, offset = 0) {
    const params = new URLSearchParams();
    if (status) params.append("status", status);
    if (search) params.append("search", search);
    params.append("limit", limit);
    params.append("offset", offset);

    const res = await fetch(`/api/sessions?${params.toString()}`);
    if (res.status === 401) {
      window.location.href = "/login";
      return [];
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch sessions`);
    return await res.json();
  }

  static async getSessionsPage({ page = 1, pageSize = 10, status = null, search = null, dateRange = null, includeTest = false } = {}) {
    const params = new URLSearchParams();
    params.append("page", page);
    params.append("page_size", pageSize);
    if (status) params.append("status", status);
    if (search) params.append("search", search);
    if (dateRange) params.append("date_range", dateRange);
    if (includeTest) params.append("include_test", "true");

    const res = await fetch(`/api/sessions/page?${params.toString()}`);
    if (res.status === 401) {
      window.location.href = "/login";
      return null;
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch sessions`);
    return await res.json();
  }

  static async getCurrentSession() {
    const res = await fetch("/api/sessions/current");
    if (res.status === 401) {
      window.location.href = "/login";
      return null;
    }
    if (res.status === 404) {
      return null;
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch current session`);
    return await res.json();
  }

  static async startSession(data) {
    const res = await fetch("/api/sessions/start", {
      method: "POST",
      headers: ApiClient._headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || "Không thể bắt đầu phiên giám sát");
    }
    return await res.json();
  }

  static async endSession(sessionId, data = { reason: "COMPLETED" }) {
    const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/end`, {
      method: "POST",
      headers: ApiClient._headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || "Không thể kết thúc phiên giám sát");
    }
    return await res.json();
  }

  static async getSession(sessionId) {
    const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch session ${sessionId}`);
    return await res.json();
  }

  static async updateSession(sessionId, data) {
    const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
      method: "PATCH",
      headers: ApiClient._headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to update session ${sessionId}`);
    return await res.json();
  }

  static async getSessionEvents(sessionId, severity = null, reviewStatus = null, limit = 500) {
    const params = new URLSearchParams();
    if (severity) params.append("severity", severity);
    if (reviewStatus) params.append("review_status", reviewStatus);
    params.append("limit", limit);

    const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/events?${params.toString()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch session events`);
    return await res.json();
  }

  static async getSessionAudit(sessionId) {
    const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/audit`);
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch session audit`);
    return await res.json();
  }

  static async verifyEvidence(evidenceId) {
    const res = await fetch(`/api/evidence/verify/${encodeURIComponent(evidenceId)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to verify evidence`);
    return await res.json();
  }

  static async triggerBackup(recoveryPassphrase = null) {
    const body = recoveryPassphrase ? JSON.stringify({ recovery_passphrase: recoveryPassphrase }) : undefined;
    const res = await fetch("/api/backup", {
      method: "POST",
      headers: ApiClient._headers(body ? { "Content-Type": "application/json" } : {}),
      body: body,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to trigger backup`);
    return await res.json();
  }

  static async verifyBackup(backupIdOrPath, recoveryPassphrase = null) {
    const payload = { backup_id_or_path: backupIdOrPath };
    if (recoveryPassphrase) payload.recovery_passphrase = recoveryPassphrase;
    const res = await fetch("/api/backup/verify", {
      method: "POST",
      headers: ApiClient._headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to verify backup`);
    return await res.json();
  }

  static async getBackupStatus() {
    const res = await fetch("/api/backup/status");
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch backup status`);
    return await res.json();
  }

  static async previewRetention() {
    const res = await fetch("/api/retention/preview");
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch retention preview`);
    return await res.json();
  }
}
