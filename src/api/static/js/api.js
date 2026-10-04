/**
 * ExamGuard Vision — API Client
 * Clean HTTP client for REST endpoints with error handling.
 */

export class ApiClient {
  static async getEvents(riskLevel = null, status = null, limit = 100) {
    const params = new URLSearchParams();
    if (riskLevel) params.append("risk_level", riskLevel);
    if (status) params.append("status", status);
    if (limit) params.append("limit", limit);

    const res = await fetch(`/api/events?${params.toString()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch events`);
    return await res.json();
  }

  static async getEvent(eventId) {
    const res = await fetch(`/api/events/${encodeURIComponent(eventId)}`);
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
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`Failed to update event: ${err || res.status}`);
    }
    return await res.json();
  }

  static async getCameras() {
    const res = await fetch("/api/cameras");
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch cameras`);
    return await res.json();
  }

  static async getCameraTracks() {
    try {
      const res = await fetch("/api/cameras/tracks");
      if (!res.ok) return [];
      return await res.json();
    } catch {
      return [];
    }
  }

  static async getSystemStatus() {
    const res = await fetch("/api/system/status");
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch system status`);
    return await res.json();
  }

  static async getSystemModels() {
    const res = await fetch("/api/system/models");
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch system models`);
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
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch sessions`);
    return await res.json();
  }

  static async getCurrentSession() {
    const res = await fetch("/api/sessions/current");
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch current session`);
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
      headers: { "Content-Type": "application/json" },
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

  static async triggerBackup() {
    const res = await fetch("/api/backup", { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to trigger backup`);
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
