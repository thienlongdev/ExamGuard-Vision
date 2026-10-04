/**
 * ExamGuard Vision — Data Normalization Adapter
 * Safe normalization layer between raw backend schemas and frontend components.
 * Strictly adheres to observable behavior semantics (No cheating words).
 */

import { getEventNameVi, formatRelativeTimeVi, formatDurationVi } from "./localization.js";

/**
 * Convert canonical or rule string into a polished Vietnamese display title.
 */
export function getEventDisplayName(rawType) {
  return getEventNameVi(rawType);
}

/**
 * Format raw evidence snapshot paths into safe web URLs without exposing server filesystem roots.
 */
export function normalizeSnapshotUrl(rawPath) {
  if (!rawPath) return null;
  // If already a web URL
  if (rawPath.startsWith("http://") || rawPath.startsWith("https://") || rawPath.startsWith("/api/")) {
    return rawPath;
  }
  // Strip Windows backslashes and drive letters (e.g. E:\... or storage\...)
  let clean = rawPath.replace(/\\/g, "/");

  // Keep subpaths within known storage roots
  const markers = ["storage/evidence/", "runs/local_live/", "evidence/"];
  for (const marker of markers) {
    const idx = clean.indexOf(marker);
    if (idx !== -1) {
      const sub = clean.substring(idx + marker.length).replace(/^\/+/, "");
      return `/api/evidence/${sub}`;
    }
  }

  const snapIdx = clean.indexOf("snapshots/");
  if (snapIdx !== -1) {
    return `/api/evidence/${clean.substring(snapIdx)}`;
  }
  const clipIdx = clean.indexOf("clips/");
  if (clipIdx !== -1) {
    return `/api/evidence/${clean.substring(clipIdx)}`;
  }

  // If just filename
  const filename = clean.split("/").pop();
  return `/api/evidence/snapshots/${filename}`;
}

/**
 * Canonical review status normalization: "awaiting" | "confirmed" | "dismissed"
 */
export function normalizeReviewStatus(rawStatus) {
  if (!rawStatus) return "awaiting";
  const s = String(rawStatus).toLowerCase().trim();
  if (s === "confirmed" || s === "confirmed_event" || s === "reviewed") return "confirmed";
  if (s === "dismissed") return "dismissed";
  return "awaiting";
}

/**
 * Canonical event lifecycle normalization: "open" | "active" | "closed"
 */
export function normalizeLifecycle(rawLifecycle, rawStatus) {
  if (rawLifecycle) {
    const l = String(rawLifecycle).toLowerCase().trim();
    if (l === "closed") return "closed";
    if (l === "active" || l === "update") return "active";
    if (l === "open") return "open";
  }
  if (rawStatus && String(rawStatus).toLowerCase() === "closed") {
    return "closed";
  }
  return "active";
}

/**
 * Normalize an event object into safe, structured UI state.
 */
export function normalizeEvent(raw) {
  if (!raw) return null;
  const eid = String(raw.event_id || raw.eventId || raw.id || "");
  if (!eid) return null;

  const tid = raw.track_id !== undefined ? Number(raw.track_id) : (raw.trackId !== undefined ? Number(raw.trackId) : 1);
  const cid = String(raw.camera_id || raw.cameraId || "webcam_0");

  // Timestamps
  const ts = Number(raw.timestamp || raw.last_update_timestamp || raw.end_time || Date.now() / 1000);
  const startTime = Number(raw.start_time || raw.start_timestamp || ts);
  const endTime = Number(raw.end_time || raw.end_timestamp || ts);
  const duration = Number(raw.duration !== undefined ? raw.duration : (endTime > startTime ? endTime - startTime : 0));

  // Risk & Score
  const rawScore = raw.score !== undefined ? raw.score : (raw.risk_score !== undefined ? raw.risk_score : 0.0);
  const score = Math.max(0, Math.min(100, Math.round(Number(rawScore))));
  let riskLevel = String(raw.risk_level || raw.riskLevel || "LOW").toUpperCase();
  if (!["HIGH", "MEDIUM", "LOW"].includes(riskLevel)) {
    riskLevel = score >= 75 ? "HIGH" : (score >= 40 ? "MEDIUM" : "LOW");
  }

  // Canonical type & Display label
  const canonicalType = String(raw.event_type || raw.canonicalType || raw.evidence?.rule_id || "OBSERVABLE_EVENT");
  const displayName = getEventDisplayName(canonicalType);

  // Review status vs Lifecycle separation
  const reviewStatus = normalizeReviewStatus(raw.review_status || raw.reviewStatus || raw.status);
  const lifecycle = normalizeLifecycle(raw.lifecycle_status || raw.lifecycle || raw.lifecycle_action, raw.status);

  // Evidence snapshot & clips
  const rawSnap = raw.snapshot_path || raw.snapshotUrl || raw.evidence?.snapshot_path || raw.evidence_summary?.open_snapshot_path || raw.evidence_summary?.snapshot_path;
  const snapshotUrl = normalizeSnapshotUrl(rawSnap);

  const rawClip = raw.clip_path || raw.clipUrl || raw.evidence?.clip_path || raw.evidence_summary?.clip_path;
  const clipUrl = normalizeSnapshotUrl(rawClip);

  // Event-time observation snapshot & cues
  const obs = raw.observation_snapshot || raw.observationSnapshot || {};
  const evSummary = raw.evidence_summary || raw.evidence || {};

  return {
    eventId: eid,
    trackId: tid,
    cameraId: cid,
    timestamp: ts,
    startTime: startTime,
    endTime: endTime,
    duration: duration,
    canonicalType: canonicalType,
    displayName: displayName,
    riskLevel: riskLevel,
    score: score,
    status: reviewStatus, // backward compatibility
    reviewStatus: reviewStatus,
    lifecycle: lifecycle,
    reviewerNotes: raw.reviewer_notes || raw.reviewerNotes || null,
    snapshotUrl: snapshotUrl,
    clipUrl: clipUrl,
    observationSnapshot: obs,
    evidence: evSummary,
    raw: raw,
  };
}

/**
 * Format relative seconds into human-readable compact string.
 */
export function formatRelativeTime(timestampSec) {
  return formatRelativeTimeVi(timestampSec);
}

/**
 * Format duration in seconds.
 */
export function formatDuration(sec) {
  return formatDurationVi(sec);
}
