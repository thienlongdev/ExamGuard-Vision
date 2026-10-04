/**
 * ExamGuard Vision — Centralized Vietnamese Localization Dictionary
 * Authoritative human-facing display mappings for school invigilator presentations.
 * Strictly adheres to observable behavior semantics (NO cheating words).
 */

export const ALERT_SEMANTICS_TOOLTIP = "Mức cảnh báo dùng để ưu tiên giám thị xem xét, không phải kết luận gian lận.";

export const POSTURE_NAMES_VI = {
  NORMAL_UPRIGHT: "Bình thường",
  NORMAL_READ_WRITE: "Đọc / viết",
  HEAD_REST_SLEEP: "Gục đầu",
  TURN_HEAD_CLEAR: "Quay đầu rõ",
  STANDING: "Đứng dậy",
  UNKNOWN: "Chưa xác định",
  "N/A": "Chưa xác định",
  AVAILABLE: "Có dữ liệu",
  UNAVAILABLE: "Không khả dụng",
  NOT_EVALUATED: "Chưa đánh giá",
};

export const EVENT_NAMES_VI = {
  PHONE_ASSOCIATED: "Phát hiện điện thoại",
  SUSTAINED_HEAD_REST: "Gục đầu kéo dài",
  SUSTAINED_LATERAL_HEAD_ORIENTATION: "Quay đầu kéo dài",
  DISCUSSION_CANDIDATE: "Có dấu hiệu trao đổi",
  STANDING: "Đứng dậy",
  MULTI_CUE_ATTENTION_SHIFT: "Chuyển hướng chú ý",
  turn_head: "Quay đầu",
  prolonged_phone_use: "Phát hiện điện thoại",
  repeated_turning: "Quay đầu nhiều lần",
  discussion: "Có dấu hiệu trao đổi",
  standing_up: "Rời chỗ / Đứng dậy",
  phone_and_head_down: "Cúi đầu sử dụng điện thoại",
  prolonged_head_down: "Gục đầu kéo dài",
  OBSERVABLE_EVENT: "Sự kiện quan sát",
};

export const SEVERITY_BANDS_VI = {
  LOW: {
    key: "low",
    code: "THẤP",
    title: "BÌNH THƯỜNG",
    badge: "THẤP",
    label: "THẤP",
    icon: "✓",
    tagClass: "safe",
    color: "var(--state-safe)",
  },
  MEDIUM: {
    key: "medium",
    code: "TRUNG BÌNH",
    title: "CẦN CHÚ Ý",
    badge: "CẦN CHÚ Ý",
    label: "CẦN CHÚ Ý",
    icon: "!",
    tagClass: "attention",
    color: "var(--state-attention)",
  },
  HIGH: {
    key: "high",
    code: "CAO",
    title: "CẢNH BÁO CAO",
    badge: "CẢNH BÁO CAO",
    label: "CẢNH BÁO CAO",
    icon: "⚠",
    tagClass: "high",
    color: "var(--state-high)",
  },
};

export const REVIEW_STATUS_VI = {
  awaiting: { label: "Chờ duyệt", class: "awaiting" },
  confirmed: { label: "Đã xác nhận", class: "confirmed" },
  dismissed: { label: "Đã bỏ qua", class: "dismissed" },
  new: { label: "Chờ duyệt", class: "awaiting" },
};

export const SOURCE_ORIGIN_VI = {
  PHYSICAL_LIVE_CAMERA: "Camera trực tiếp",
  PHYSICAL_CAMERA: "Camera trực tiếp",
  SOFTWARE_VALIDATION_FIXTURE: "Dữ liệu kiểm thử",
  REPLAY_STREAM: "Luồng phát lại",
  VIDEO_FILE: "Tệp video",
  RTSP_STREAM: "Camera RTSP",
  SYNTHETIC_TEST: "Dữ liệu mô phỏng",
  UNKNOWN: "Không xác định",
};

export const PHONE_ASSOCIATION_VI = {
  CLEAR_ASSOCIATION: "Liên kết rõ",
  AMBIGUOUS_ASSOCIATION: "Liên kết không rõ",
  NO_PHONE: "Không phát hiện",
  NONE: "Không có",
  DIRECT_CONTACT: "Cầm trên tay / Tiếp xúc",
  DESK_PROXIMITY: "Gần vị trí ngồi",
  NOT_EVALUATED: "Chưa đánh giá",
  ASSOCIATED: "Liên kết rõ",
  UNASSOCIATED: "Không liên kết",
};

export const MACRO_BEHAVIOR_VI = {
  stand: "Đứng dậy",
  discuss: "Trao đổi",
  normal: "Bình thường",
  AVAILABLE: "Có dữ liệu",
  NOT_EVALUATED: "Chưa đánh giá",
  UNAVAILABLE: "Không khả dụng",
  NONE: "Không ghi nhận",
  "None Observed": "Không ghi nhận",
  STANDING: "Đứng dậy",
  DISCUSSION_CANDIDATE: "Có dấu hiệu trao đổi",
};

/**
 * Helper to get human-friendly Vietnamese event name.
 */
export function getEventNameVi(rawType) {
  if (!rawType) return "Sự kiện quan sát";
  if (EVENT_NAMES_VI[rawType]) return EVENT_NAMES_VI[rawType];
  const cleaned = String(rawType).replace(/^ORIENTATION_/, "").replace(/_/g, " ").trim();
  return cleaned || "Sự kiện quan sát";
}

/**
 * Helper to get human-friendly Vietnamese posture name.
 */
export function getPostureNameVi(rawPosture) {
  if (!rawPosture) return "Bình thường";
  if (POSTURE_NAMES_VI[rawPosture]) return POSTURE_NAMES_VI[rawPosture];
  return rawPosture;
}

/**
 * Helper to get severity display object.
 */
export function getSeverityInfo(riskLevel) {
  const norm = String(riskLevel || "LOW").toUpperCase();
  return SEVERITY_BANDS_VI[norm] || SEVERITY_BANDS_VI.LOW;
}

/**
 * Helper to get source origin in Vietnamese.
 */
export function getSourceOriginVi(rawOrigin) {
  if (!rawOrigin) return "Camera trực tiếp";
  return SOURCE_ORIGIN_VI[rawOrigin] || rawOrigin;
}

/**
 * Helper to get phone association string in Vietnamese.
 */
export function getPhoneStatusVi(rawStatus) {
  if (!rawStatus) return "Không phát hiện";
  return PHONE_ASSOCIATION_VI[rawStatus] || rawStatus;
}

/**
 * Format relative seconds in concise natural Vietnamese.
 */
export function formatRelativeTimeVi(timestampSec) {
  if (!timestampSec) return "—";
  const now = Date.now() / 1000;
  const diff = Math.max(0, now - timestampSec);

  if (diff < 5) return "Vừa xong";
  if (diff < 60) return `${Math.floor(diff)} giây trước`;
  if (diff < 3600) return `${Math.floor(diff / 60)} phút trước`;
  const d = new Date(timestampSec * 1000);
  return d.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

/**
 * Format duration in Vietnamese.
 */
export function formatDurationVi(sec) {
  if (!sec || sec < 0.1) return "< 1s";
  return `${sec.toFixed(1).replace(".", ",")}s`;
}
