/**
 * ExamGuard Vision — Health Strip Component
 * High-density horizontal telemetry strip differentiating configured vs measured values.
 */

import { appState } from "../state.js";

export class HealthStripComponent {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.init();
  }

  init() {
    this.render();
    appState.subscribe((type) => {
      if (type === "SYSTEM_STATUS_UPDATED" || type === "CAMERA_UPDATED" || type === "TRACKS_UPDATED") {
        this.updateHealth();
      }
    });
  }

  render() {
    this.container.innerHTML = `
      <div class="health-strip-card" id="health-strip">
        <div class="health-item" title="Trạng thái kết nối camera vật lý">
          <span class="label">Camera</span>
          <span class="val live" id="strip-cam-status">HOẠT ĐỘNG</span>
        </div>

        <div class="health-divider"></div>

        <div class="health-item" title="Tốc độ thu hình từ camera vật lý">
          <span class="label">Thu hình</span>
          <span class="val cyan" id="strip-cap-fps">— FPS</span>
        </div>

        <div class="health-divider"></div>

        <div class="health-item" title="Tốc độ xử lý thực tế của luồng AI">
          <span class="label">Xử lý AI</span>
          <span class="val cyan" id="strip-proc-fps">— FPS</span>
        </div>

        <div class="health-divider"></div>

        <div class="health-item" title="Số lượng thí sinh đang theo dõi">
          <span class="label">Thí sinh</span>
          <span class="val" id="strip-tracks">0</span>
        </div>

        <div class="health-divider"></div>

        <div class="health-item" title="Độ sâu hàng đợi khung hình (Tối đa 5)">
          <span class="label">Hàng đợi</span>
          <span class="val" id="strip-queue">0 / 5</span>
        </div>

        <div class="health-divider"></div>

        <div class="health-item" title="Tỷ lệ bỏ khung hình khi nghẽn">
          <span class="label">Tỷ lệ bỏ</span>
          <span class="val" id="strip-drops">0.0%</span>
        </div>

        <div class="health-divider"></div>

        <div class="health-item" title="Bộ nhớ GPU PyTorch đang cấp phát (torch.cuda.memory_allocated)">
          <span class="label">VRAM (Alloc)</span>
          <span class="val" id="strip-vram">— MB</span>
        </div>
      </div>
    `;
    this.updateHealth();
  }

  updateHealth() {
    const sys = appState.systemStatus;
    const cam = appState.cameraInfo;

    const elCam = document.getElementById("strip-cam-status");
    const elCap = document.getElementById("strip-cap-fps");
    const elProc = document.getElementById("strip-proc-fps");
    const elTracks = document.getElementById("strip-tracks");
    const elQueue = document.getElementById("strip-queue");
    const elDrops = document.getElementById("strip-drops");
    const elVram = document.getElementById("strip-vram");

    if (cam) {
      if (cam.streaming) {
        if (elCam) { elCam.innerText = "HOẠT ĐỘNG"; elCam.className = "val live"; }
      } else if (cam.connected) {
        if (elCam) { elCam.innerText = "CHỜ"; elCam.className = "val amber"; }
      } else {
        if (elCam) { elCam.innerText = "NGOẠI TUYẾN"; elCam.className = "val"; }
      }
    }

    if (sys) {
      const obs = sys.observed_rates || {};
      if (elCap) {
        elCap.innerText = obs.capture_fps !== null && obs.capture_fps !== undefined
          ? `${obs.capture_fps.toFixed(1)} FPS`
          : "— FPS";
      }
      if (elProc) {
        elProc.innerText = obs.processed_fps !== null && obs.processed_fps !== undefined
          ? `${obs.processed_fps.toFixed(1)} FPS`
          : (sys.effective_fps ? `${sys.effective_fps.toFixed(1)} FPS` : "— FPS");
      }
      if (elTracks) {
        elTracks.innerText = sys.active_students !== undefined ? sys.active_students : appState.activeTracks.length;
      }
      if (elQueue) {
        elQueue.innerText = `${sys.queue_depth || 0} / 5`;
      }
      if (elDrops) {
        elDrops.innerText = sys.drop_percentage !== null && sys.drop_percentage !== undefined
          ? `${sys.drop_percentage.toFixed(1)}%`
          : "0.0%";
      }
      if (elVram) {
        elVram.innerText = sys.gpu_vram_allocated_mb !== undefined
          ? `${sys.gpu_vram_allocated_mb.toFixed(0)} MB`
          : "— MB";
      }
    }
  }
}
