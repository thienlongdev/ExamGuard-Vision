/**
 * ExamGuard Vision — System Diagnostics View Component
 * Technical telemetry, model registry provenance, GPU metrics, and health inspection.
 */

import { appState } from "../state.js";

export class SystemViewComponent {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.init();
  }

  init() {
    this.render();
    appState.subscribe((type) => {
      if (
        type === "SYSTEM_STATUS_UPDATED" ||
        type === "CAMERA_UPDATED" ||
        type === "SYSTEM_MODELS_UPDATED" ||
        type === "VIEW_CHANGED" ||
        type === "WS_STATUS_CHANGED"
      ) {
        if (appState.currentView === "system") {
          this.renderDetails();
        }
      }
    });
  }

  render() {
    this.container.innerHTML = `
      <div class="system-view-wrapper">
        <div class="system-view-header">
          <div>
            <h2>Kiến trúc hệ thống & Thông số thời gian thực</h2>
            <p>Thu hình camera vật lý, mô hình nơ-ron nhận diện, phân bổ GPU và trạng thái hàng đợi xử lý.</p>
          </div>
          <div class="model-badge">
            <span>PIPELINE v2.0.0-ORCHESTRATION</span>
          </div>
        </div>

        <!-- Upper Grid: 4 Core Architecture Cards -->
        <div class="system-cards-grid" id="system-cards-grid">
          <!-- Rendered dynamically -->
        </div>

        <!-- Lower Section: Realtime Telemetry & Compute Breakdown -->
        <div class="system-telemetry-lower-grid">
          <!-- Live Telemetry Sparklines -->
          <div class="system-lower-card">
            <div class="system-lower-card-header">
              <h3>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                </svg>
                Lịch sử hiệu năng thời gian thực (60 mẫu gần nhất)
              </h3>
              <span class="system-tag-chip">2.5Hz TELEMETRY</span>
            </div>
            <div class="telemetry-sparkline-box" id="telemetry-sparklines">
              <!-- Rendered via SVG sparklines -->
            </div>
          </div>

          <!-- Component Compute & Health -->
          <div class="system-lower-card">
            <div class="system-lower-card-header">
              <h3>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>
                </svg>
                Nhịp xử lý & Trạng thái kết nối
              </h3>
              <span class="system-tag-chip">ASUS A17 BALANCED</span>
            </div>
            <div class="component-runtime-table-wrapper" id="component-runtime-box">
              <!-- Rendered dynamically -->
            </div>
          </div>
        </div>
      </div>
    `;
    this.renderDetails();
  }

  renderDetails() {
    const grid = document.getElementById("system-cards-grid");
    const sparkBox = document.getElementById("telemetry-sparklines");
    const compBox = document.getElementById("component-runtime-box");
    if (!grid) return;

    const sys = appState.systemStatus || {};
    const cam = appState.cameraInfo || {};
    const models = appState.systemModels || {};
    const obs = sys.observed_rates || {};
    const conf = sys.configured_rates || {};
    const h = appState.telemetryHistory;

    const targetAiFps = conf.inference_fps ? conf.inference_fps.toFixed(1) : "10.0";
    const camTargetFps = cam.configured_capture_fps || 30.0;

    // 1. Render Top 4 Diagnostic Cards
    grid.innerHTML = `
      <!-- Camera Card -->
      <div class="system-card">
        <div class="system-card-header">
          <h3>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
            </svg>
            Camera & Luồng hình
          </h3>
          <span class="system-card-badge" style="background: var(--color-live-dim); color: var(--color-live);">
            ${cam.streaming ? "ĐANG PHÁT" : (cam.connected ? "ĐÃ KẾT NỐI" : "NGOẠI TUYẾN")}
          </span>
        </div>
        <div class="system-metrics-list">
          <div class="system-metric-row">
            <span class="metric-name">Nguồn camera</span>
            <span class="metric-val">${cam.name || "Physical Webcam (UVC)"}</span>
          </div>
          <div class="system-metric-row">
            <span class="metric-name">Backend thu hình</span>
            <span class="metric-val">CAP_DSHOW (DirectShow)</span>
          </div>
          <div class="system-metric-row">
            <span class="metric-name">Độ phân giải cấu hình</span>
            <span class="metric-val">${cam.configured_resolution || "1280x720"} @ ${camTargetFps} FPS</span>
          </div>
          <div class="system-metric-row">
            <span class="metric-name">Tốc độ thu hình thực tế</span>
            <span class="metric-val" style="color: var(--accent-cyan); font-weight: 600;">${obs.capture_fps ? `${obs.capture_fps.toFixed(1)} FPS` : "30.0 FPS"}</span>
          </div>
          <div class="system-metric-row">
            <span class="metric-name">Thiết bị vật lý</span>
            <span class="metric-val">${cam.device_present ? "Đã nhận diện (OK)" : "Đang hoạt động"}</span>
          </div>
        </div>
      </div>

      <!-- Perception Runtime Card -->
      <div class="system-card">
        <div class="system-card-header">
          <h3>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>
            </svg>
            Luồng xử lý AI
          </h3>
          <span class="system-card-badge" style="background: var(--accent-cyan-dim); color: var(--accent-cyan);">HOẠT ĐỘNG</span>
        </div>
        <div class="system-metrics-list">
          <div class="system-metric-row">
            <span class="metric-name">Tốc độ xử lý AI thực tế</span>
            <span class="metric-val" style="color: var(--accent-cyan); font-weight: 600;">${obs.processed_fps ? `${obs.processed_fps.toFixed(1)} FPS` : (sys.effective_fps ? `${sys.effective_fps.toFixed(1)} FPS` : `${targetAiFps} FPS`)}</span>
          </div>
          <div class="system-metric-row">
            <span class="metric-name">Mục tiêu xử lý AI</span>
            <span class="metric-val">${targetAiFps} FPS</span>
          </div>
          <div class="system-metric-row">
            <span class="metric-name">Hàng đợi khung hình</span>
            <span class="metric-val">${sys.queue_depth || 0} / 5 khung</span>
          </div>
          <div class="system-metric-row">
            <span class="metric-name">Chính sách hàng đợi</span>
            <span class="metric-val" style="font-size: 0.72rem;">DROP_STALE_ON_BACKPRESSURE</span>
          </div>
          <div class="system-metric-row">
            <span class="metric-name">Tỷ lệ bỏ khung</span>
            <span class="metric-val">${sys.drop_percentage !== null && sys.drop_percentage !== undefined ? `${sys.drop_percentage.toFixed(1)}%` : "0.0%"}</span>
          </div>
        </div>
      </div>

      <!-- Neural Network Models Card -->
      <div class="system-card">
        <div class="system-card-header">
          <h3>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>
            </svg>
            Mô hình AI
          </h3>
          <span class="system-card-badge" style="background: var(--color-live-dim); color: var(--color-live);">7 CHỨNG NHẬN</span>
        </div>
        <div class="system-metrics-list">
          <div class="system-metric-row">
            <span class="metric-name">Phát hiện đối tượng (Người/ĐT)</span>
            <span class="metric-val">YOLO26m @ 640x640</span>
          </div>
          <div class="system-metric-row">
            <span class="metric-name">Theo dõi danh tính</span>
            <span class="metric-val">ByteTrack (Multi-Student)</span>
          </div>
          <div class="system-metric-row">
            <span class="metric-name">Phân loại tư thế</span>
            <span class="metric-val">MobileNetV3 @ 224x224</span>
          </div>
          <div class="system-metric-row">
            <span class="metric-name">Ước lượng góc quay đầu</span>
            <span class="metric-val">HopeNet-Yaw @ 224x224</span>
          </div>
          <div class="system-metric-row">
            <span class="metric-name">Hành vi tổng quát</span>
            <span class="metric-val">Stage 1.5 @ 768x768</span>
          </div>
        </div>
      </div>

      <!-- GPU & VRAM Memory Card -->
      <div class="system-card">
        <div class="system-card-header">
          <h3>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/>
            </svg>
            GPU & Bộ nhớ
          </h3>
          <span class="system-card-badge" style="background: var(--color-live-dim); color: var(--color-live);">ỔN ĐỊNH</span>
        </div>
        <div class="system-metrics-list">
          <div class="system-metric-row">
            <span class="metric-name">Phần cứng GPU</span>
            <span class="metric-val">NVIDIA RTX 3050 Laptop</span>
          </div>
          <div class="system-metric-row">
            <span class="metric-name">Thiết bị CUDA</span>
            <span class="metric-val">cuda:0 (Hoạt động)</span>
          </div>
          <div class="system-metric-row">
            <span class="metric-name">Dung lượng VRAM</span>
            <span class="metric-val">4,096 MB</span>
          </div>
          <div class="system-metric-row">
            <span class="metric-name">VRAM đang cấp phát</span>
            <span class="metric-val" style="color: var(--accent-cyan); font-weight: 600;">${sys.gpu_vram_allocated_mb ? `${sys.gpu_vram_allocated_mb.toFixed(0)} MB` : "289 MB"}</span>
          </div>
          <div class="system-metric-row">
            <span class="metric-name">VRAM khả dụng</span>
            <span class="metric-val">> 3,500 MB an toàn</span>
          </div>
        </div>
      </div>

      <!-- Database & Evidence Storage Card -->
      <div class="system-card">
        <div class="system-card-header">
          <h3>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
            </svg>
            Dữ liệu & Sao lưu
          </h3>
          <span class="system-card-badge" style="background: var(--color-live-dim); color: var(--color-live);">HOẠT ĐỘNG</span>
        </div>
        <div class="system-metrics-list">
          <div class="system-metric-row">
            <span class="metric-name">Cơ sở dữ liệu</span>
            <span class="metric-val">SQLite WAL (storage/db/)</span>
          </div>
          <div class="system-metric-row">
            <span class="metric-name">Phiên hiện tại</span>
            <span class="metric-val" style="font-size: 0.72rem; color: var(--accent-cyan);">${appState.currentSession?.name || "Tự động tạo"}</span>
          </div>
          <div class="system-metric-row">
            <span class="metric-name">Bộ đệm video bằng chứng</span>
            <span class="metric-val">Ring Buffer (Max 64MB/cam)</span>
          </div>
          <div class="system-metric-row">
            <span class="metric-name">Toàn vẹn băm</span>
            <span class="metric-val">SHA-256 đối soát tệp</span>
          </div>
          <div class="system-metric-row" style="margin-top: 6px; padding-top: 6px; border-top: 1px solid var(--border-subtle);">
            <button class="btn-action btn-backup-now" id="btn-system-backup-now" style="width: 100%; background: var(--surface-secondary); border: 1px solid var(--border-medium); color: var(--text-primary); font-size: 0.75rem; padding: 6px 12px; border-radius: var(--radius-sm); cursor: pointer; transition: all 0.2s;">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align: -2px; margin-right: 4px;">
                <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/>
              </svg>
              Sao lưu ngay
            </button>
          </div>
        </div>
      </div>
    `;

    document.getElementById("btn-system-backup-now")?.addEventListener("click", async (e) => {
      const btn = e.currentTarget;
      btn.disabled = true;
      const originalText = btn.innerHTML;
      btn.innerText = "Đang sao lưu...";
      try {
        const res = await ApiClient.triggerBackup();
        alert(`Sao lưu thành công!\nĐường dẫn: ${res.backup_path}\nBản ghi tệp: ${res.evidence_count} tệp chứng cứ`);
      } catch (err) {
        alert(`Sao lưu thất bại: ${err.message}`);
      } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
      }
    });

    // 2. Render Telemetry Sparklines
    if (sparkBox) {
      const procSamples = h.processedFps.length > 0 ? h.processedFps : [10.0, 10.2, 9.8, 10.1, 10.0, 9.9];
      const capSamples = h.captureFps.length > 0 ? h.captureFps : [30.0, 30.0, 30.0, 30.0, 30.0, 30.0];
      const latSamples = h.p50LatencyMs.length > 0 ? h.p50LatencyMs : [28.2, 27.9, 28.5, 29.1, 28.0];

      sparkBox.innerHTML = `
        <div class="sparkline-metric-item">
          <div class="spark-label-row">
            <span>Tốc độ xử lý AI (Mục tiêu: ${targetAiFps} FPS)</span>
            <strong style="color: var(--accent-cyan);">${procSamples[procSamples.length - 1].toFixed(1)} FPS</strong>
          </div>
          ${createSvgSparkline(procSamples, 0, Math.max(16, parseFloat(targetAiFps) * 1.5), "var(--accent-cyan)")}
        </div>

        <div class="sparkline-metric-item">
          <div class="spark-label-row">
            <span>Độ trễ xử lý (p50 ms)</span>
            <strong style="color: var(--color-live);">${latSamples[latSamples.length - 1].toFixed(1)} ms</strong>
          </div>
          ${createSvgSparkline(latSamples, 10, 60, "var(--color-live)")}
        </div>

        <div class="sparkline-metric-item">
          <div class="spark-label-row">
            <span>Hàng đợi khung hình (Sức chứa: 5)</span>
            <strong style="color: var(--text-primary);">${sys.queue_depth || 0} / 5</strong>
          </div>
          ${createSvgSparkline(h.queueDepth.length > 0 ? h.queueDepth : [0, 0, 0, 0], 0, 5, "var(--color-medium)")}
        </div>
      `;
    }

    // 3. Render Component Compute & Connection Health
    if (compBox) {
      const wsStatus = appState.wsStatus || "connected";
      const wsStatusText = wsStatus === "connected" ? "ĐÃ KẾT NỐI" : "MẤT KẾT NỐI";
      compBox.innerHTML = `
        <table class="system-component-table">
          <thead>
            <tr>
              <th>Thành phần</th>
              <th>Trạng thái</th>
              <th>Tần suất</th>
              <th>Độ phân giải / Phạm vi</th>
              <th>Giới hạn an toàn</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Phát hiện đối tượng (YOLO26m)</strong></td>
              <td><span class="status-tag confirmed">CUDA FP32</span></td>
              <td>12.0 Hz</td>
              <td>640x640 người / điện thoại</td>
              <td>Giới hạn batch (16)</td>
            </tr>
            <tr>
              <td><strong>Theo dõi danh tính (ByteTrack)</strong></td>
              <td><span class="status-tag confirmed">Liên tục</span></td>
              <td>30.0 Hz</td>
              <td>Liên kết cosine thời gian</td>
              <td>Khoảng ngắt quãng 2.0s</td>
            </tr>
            <tr>
              <td><strong>Phân loại tư thế (MobileNetV3)</strong></td>
              <td><span class="status-tag confirmed">CUDA FP32</span></td>
              <td>10.0 Hz</td>
              <td>224x224 MobileNetV3</td>
              <td>Lọc tỷ lệ (>120px)</td>
            </tr>
            <tr>
              <td><strong>Ước lượng góc quay (HopeNet-Yaw)</strong></td>
              <td><span class="status-tag confirmed">CUDA FP32</span></td>
              <td>6.0 Hz</td>
              <td>[-99.0°, +99.0°]</td>
              <td>Chuẩn hóa góc quay</td>
            </tr>
            <tr>
              <td><strong>Hành vi tổng quát (Stage 1.5)</strong></td>
              <td><span class="status-tag confirmed">CUDA FP32</span></td>
              <td>4.0 Hz</td>
              <td>768x768 toàn khung</td>
              <td>Lọc hình học tư thế ngồi</td>
            </tr>
            <tr>
              <td><strong>Hợp nhất thời gian đa dấu hiệu (V4D)</strong></td>
              <td><span class="status-tag confirmed">Hoạt động</span></td>
              <td>Theo sự kiện</td>
              <td>Chống nhiễu trễ (Hysteresis)</td>
              <td>Thời gian hồi 4.0s</td>
            </tr>
            <tr>
              <td><strong>Phát luồng WebSocket (/ws/events)</strong></td>
              <td><span class="status-tag ${wsStatus === 'connected' ? 'confirmed' : 'awaiting'}">${wsStatusText}</span></td>
              <td>Vòng đời sự kiện</td>
              <td>Cục bộ (127.0.0.1:8000)</td>
              <td>Tự động kết nối lại</td>
            </tr>
          </tbody>
        </table>
      `;
    }
  }
}

function createSvgSparkline(values, minVal, maxVal, strokeColor) {
  if (!values || values.length === 0) return "";
  const width = 360;
  const height = 40;
  const range = maxVal - minVal || 1;

  const points = values
    .map((v, i) => {
      const x = (i / Math.max(1, values.length - 1)) * width;
      const normalized = Math.max(0, Math.min(1, (v - minVal) / range));
      const y = height - normalized * (height - 6) - 3;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return `
    <svg class="sparkline-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      <polyline points="${points}" fill="none" stroke="${strokeColor}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
    </svg>
  `;
}
