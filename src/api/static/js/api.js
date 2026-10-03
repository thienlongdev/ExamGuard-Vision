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
}
