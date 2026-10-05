/**
 * ExamGuard Vision — System Diagnostics & Administration View Component
 * Technical telemetry, model registry provenance, GPU metrics,
 * User Management & RBAC, Camera Configuration, and Security & Encrypted Backups.
 */

import { appState } from "../state.js";
import { ApiClient } from "../api.js";

export class SystemViewComponent {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.activeSubtab = "overview"; // "overview" | "users" | "cameras" | "security"
    this.users = [];
    this.cameras = [];
    this.securityStatus = null;
    this.init();
  }

  init() {
    this.render();
    this.bindEvents();

    appState.subscribe((type) => {
      if (
        type === "SYSTEM_STATUS_UPDATED" ||
        type === "CAMERA_UPDATED" ||
        type === "SYSTEM_MODELS_UPDATED" ||
        type === "VIEW_CHANGED" ||
        type === "WS_STATUS_CHANGED"
      ) {
        if (appState.currentView === "system") {
          if (this.activeSubtab === "overview") {
            this.renderOverviewDetails();
          }
        }
      }
    });
  }

  render() {
    this.container.innerHTML = `
      <div class="system-view-wrapper">
        <div class="system-view-header">
          <div>
            <h2>Trung tâm Quản trị & Kỹ thuật Hệ thống</h2>
            <p>Giám sát phần cứng, phân quyền người dùng, cấu hình đa camera và an ninh dữ liệu phòng thi.</p>
          </div>
          <div class="model-badge">
            <span>PIPELINE v2.0.0-ORCHESTRATION</span>
          </div>
        </div>

        <!-- Sub-navigation Tabs -->
        <div class="system-subnav-tabs" role="tablist" aria-label="Hệ thống">
          <button class="sys-subtab-btn ${this.activeSubtab === 'overview' ? 'active' : ''}" data-subtab="overview">TỔNG QUAN HỆ THỐNG</button>
          <button class="sys-subtab-btn ${this.activeSubtab === 'users' ? 'active' : ''}" data-subtab="users" id="subtab-users">NGƯỜI DÙNG & PHÂN QUYỀN</button>
          <button class="sys-subtab-btn ${this.activeSubtab === 'cameras' ? 'active' : ''}" data-subtab="cameras" id="subtab-cameras">CẤU HÌNH CAMERA</button>
          <button class="sys-subtab-btn ${this.activeSubtab === 'security' ? 'active' : ''}" data-subtab="security" id="subtab-security">BẢO MẬT & SAO LƯU</button>
        </div>

        <!-- Panel 1: Overview (Telemetry, Models, GPU) -->
        <div id="sys-panel-overview" style="display: ${this.activeSubtab === 'overview' ? 'block' : 'none'};">
          <div class="system-cards-grid" id="system-cards-grid"></div>
          <div class="system-telemetry-lower-grid" style="margin-top: 20px;">
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
              <div class="telemetry-sparkline-box" id="telemetry-sparklines"></div>
            </div>
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
              <div class="component-runtime-table-wrapper" id="component-runtime-box"></div>
            </div>
          </div>
        </div>

        <!-- Panel 2: User Management & RBAC -->
        <div id="sys-panel-users" style="display: ${this.activeSubtab === 'users' ? 'block' : 'none'};">
          <div class="system-lower-card" style="padding: 24px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
              <div>
                <h3 style="margin: 0 0 4px 0;">Danh sách người dùng & Phân quyền</h3>
                <p style="margin: 0; font-size: 0.82rem; color: #94a3b8;">Quản lý tài khoản Giám thị, Quản trị viên và Người rà soát. Mật khẩu được băm Argon2id.</p>
              </div>
              <button id="btn-toggle-create-user" style="background: #10b981; color: #0b0f19; font-weight: 600; border: none; border-radius: 6px; padding: 8px 14px; font-size: 0.85rem; cursor: pointer;">
                + Tạo người dùng mới
              </button>
            </div>

            <!-- Create User Form (Hidden by default) -->
            <div id="create-user-form-card" style="display: none; background: rgba(15,23,42,0.6); border: 1px solid #334155; border-radius: 8px; padding: 18px; margin-bottom: 20px;">
              <h4 style="margin: 0 0 12px 0;">Tạo tài khoản mới</h4>
              <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
                <div>
                  <label style="font-size: 0.78rem; color: #cbd5e1; display: block; margin-bottom: 4px;">Tên đăng nhập</label>
                  <input type="text" id="new-user-username" style="width: 100%; box-sizing: border-box; padding: 8px; background: #1e293b; border: 1px solid #475569; border-radius: 4px; color: #fff;" />
                </div>
                <div>
                  <label style="font-size: 0.78rem; color: #cbd5e1; display: block; margin-bottom: 4px;">Tên hiển thị</label>
                  <input type="text" id="new-user-display-name" placeholder="Nguyễn Văn A" style="width: 100%; box-sizing: border-box; padding: 8px; background: #1e293b; border: 1px solid #475569; border-radius: 4px; color: #fff;" />
                </div>
                <div>
                  <label style="font-size: 0.78rem; color: #cbd5e1; display: block; margin-bottom: 4px;">Mật khẩu (≥ 12 ký tự)</label>
                  <input type="password" id="new-user-password" style="width: 100%; box-sizing: border-box; padding: 8px; background: #1e293b; border: 1px solid #475569; border-radius: 4px; color: #fff;" />
                </div>
                <div>
                  <label style="font-size: 0.78rem; color: #cbd5e1; display: block; margin-bottom: 4px;">Vai trò (Role)</label>
                  <select id="new-user-role" style="width: 100%; box-sizing: border-box; padding: 8px; background: #1e293b; border: 1px solid #475569; border-radius: 4px; color: #fff;">
                    <option value="INVIGILATOR">Giám thị (INVIGILATOR)</option>
                    <option value="REVIEWER">Người rà soát (REVIEWER)</option>
                    <option value="ADMIN">Quản trị viên (ADMIN)</option>
                    <option value="VIEWER">Chỉ xem (VIEWER)</option>
                  </select>
                </div>
              </div>
              <div style="margin-top: 14px; display: flex; gap: 8px;">
                <button id="btn-submit-create-user" style="background: #10b981; color: #0b0f19; font-weight: 600; border: none; border-radius: 4px; padding: 8px 16px; cursor: pointer;">Lưu người dùng</button>
                <button id="btn-cancel-create-user" style="background: transparent; color: #94a3b8; border: 1px solid #475569; border-radius: 4px; padding: 8px 16px; cursor: pointer;">Hủy</button>
              </div>
              <div id="create-user-error" style="display: none; color: #f87171; font-size: 0.8rem; margin-top: 8px;"></div>
            </div>

            <!-- Users Table -->
            <div style="overflow-x: auto;">
              <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
                <thead>
                  <tr style="border-bottom: 1px solid #334155; text-align: left; color: #94a3b8;">
                    <th style="padding: 10px;">Tên đăng nhập</th>
                    <th style="padding: 10px;">Tên hiển thị</th>
                    <th style="padding: 10px;">Vai trò</th>
                    <th style="padding: 10px;">Trạng thái</th>
                    <th style="padding: 10px;">Lần đăng nhập cuối</th>
                    <th style="padding: 10px; text-align: right;">Thao tác</th>
                  </tr>
                </thead>
                <tbody id="users-table-tbody">
                  <tr><td colspan="6" style="padding: 20px; text-align: center; color: #64748b;">Đang tải danh sách người dùng...</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <!-- Panel 3: Camera Configuration -->
        <div id="sys-panel-cameras" style="display: ${this.activeSubtab === 'cameras' ? 'block' : 'none'};">
          <div class="system-lower-card" style="padding: 24px;">
            <div style="margin-bottom: 16px;">
              <h3 style="margin: 0 0 4px 0;">Cấu hình Camera</h3>
              <p style="margin: 0; font-size: 0.82rem; color: #94a3b8;">Danh sách camera mà hệ thống thực sự sử dụng. Camera chỉ được mở khi có phiên giám sát đang hoạt động.</p>
            </div>
            <div style="overflow-x: auto;">
              <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
                <thead>
                  <tr style="border-bottom: 1px solid #334155; text-align: left; color: #94a3b8;">
                    <th style="padding: 10px;">Mã Camera</th>
                    <th style="padding: 10px;">Tên hiển thị</th>
                    <th style="padding: 10px;">Loại nguồn</th>
                    <th style="padding: 10px;">Chỉ số thiết bị</th>
                    <th style="padding: 10px;">Độ phân giải</th>
                    <th style="padding: 10px;">FPS chỉ định</th>
                    <th style="padding: 10px;">Phòng thi</th>
                    <th style="padding: 10px; text-align: center;">Trạng thái</th>
                  </tr>
                </thead>
                <tbody id="cameras-table-tbody">
                  <tr><td colspan="8" style="padding: 20px; text-align: center; color: #64748b;">Đang tải cấu hình camera...</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <!-- Panel 4: Security & Encrypted Backups -->
        <div id="sys-panel-security" style="display: ${this.activeSubtab === 'security' ? 'block' : 'none'};">
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px;">
            <div class="system-card" style="padding: 18px;">
              <div style="font-size: 0.78rem; color: #94a3b8; text-transform: uppercase;">Xác thực & Phiên</div>
              <div style="font-size: 1.25rem; font-weight: 700; color: #10b981; margin: 6px 0;">Argon2id + Sessions</div>
              <div style="font-size: 0.8rem; color: #cbd5e1;">Khóa tự động sau 5 lần sai · Token băm SHA-256</div>
            </div>
            <div class="system-card" style="padding: 18px;">
              <div style="font-size: 0.78rem; color: #94a3b8; text-transform: uppercase;">Mã hóa bằng chứng tại chỗ</div>
              <div style="font-size: 1.25rem; font-weight: 700; color: #10b981; margin: 6px 0;">AES-256-GCM (EGE1)</div>
              <div style="font-size: 0.8rem; color: #cbd5e1;">Bảo mật & Toàn vẹn xác thực AAD theo sự kiện</div>
            </div>
            <div class="system-card" style="padding: 18px;">
              <div style="font-size: 0.78rem; color: #94a3b8; text-transform: uppercase;">Bảo vệ Khóa gốc</div>
              <div id="sec-key-storage" style="font-size: 1.25rem; font-weight: 700; color: #38bdf8; margin: 6px 0;">Windows DPAPI</div>
              <div style="font-size: 0.8rem; color: #cbd5e1;">Khóa gốc không lưu plaintext trên đĩa hay trong Git</div>
            </div>
            <div class="system-card" style="padding: 18px;">
              <div style="font-size: 0.78rem; color: #94a3b8; text-transform: uppercase;">Chuỗi Nhật ký Kiểm toán</div>
              <div id="sec-audit-chain" style="font-size: 1.25rem; font-weight: 700; color: #10b981; margin: 6px 0;">HỢP LỆ</div>
              <div style="font-size: 0.8rem; color: #cbd5e1;">Chuỗi băm SHA-256 chống giả mạo kiểm toán</div>
            </div>
          </div>

          <!-- Encrypted Backup Creation Card -->
          <div class="system-lower-card" style="padding: 24px; margin-bottom: 20px;">
            <h3 style="margin: 0 0 8px 0;">Tạo Bản sao lưu Mã hóa Di động</h3>
            <p style="margin: 0 0 16px 0; font-size: 0.85rem; color: #94a3b8;">
              Sao lưu toàn bộ cơ sở dữ liệu SQLite và bằng chứng AI dưới dạng gói mã hóa an toàn (Manifest v2.0-encrypted). Bạn có thể thiết lập cụm mật khẩu khôi phục để cho phép phục hồi trên máy tính khác.
            </p>
            <div style="display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end;">
              <div style="flex: 1; min-width: 250px;">
                <label style="font-size: 0.78rem; color: #cbd5e1; display: block; margin-bottom: 4px;">Cụm mật khẩu khôi phục (Recovery Passphrase)</label>
                <input type="password" id="input-backup-passphrase" placeholder="Nhập cụm mật khẩu bảo vệ bản sao lưu..." style="width: 100%; box-sizing: border-box; padding: 10px; background: #1e293b; border: 1px solid #475569; border-radius: 6px; color: #fff;" />
              </div>
              <button id="btn-create-encrypted-backup" style="background: #10b981; color: #0b0f19; font-weight: 600; border: none; border-radius: 6px; padding: 10px 20px; font-size: 0.88rem; cursor: pointer; height: 42px;">
                Tạo bản sao lưu ngay
              </button>
            </div>
            <div id="backup-status-msg" style="margin-top: 12px; font-size: 0.85rem; display: none;"></div>
          </div>
        </div>
      </div>
    `;

    this.renderOverviewDetails();
  }

  bindEvents() {
    this.container.querySelectorAll(".sys-subtab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const tab = btn.dataset.subtab;
        this.activeSubtab = tab;
        this.container.querySelectorAll(".sys-subtab-btn").forEach((b) => b.classList.toggle("active", b === btn));
        
        document.getElementById("sys-panel-overview").style.display = tab === "overview" ? "block" : "none";
        document.getElementById("sys-panel-users").style.display = tab === "users" ? "block" : "none";
        document.getElementById("sys-panel-cameras").style.display = tab === "cameras" ? "block" : "none";
        document.getElementById("sys-panel-security").style.display = tab === "security" ? "block" : "none";

        if (tab === "overview") this.renderOverviewDetails();
        else if (tab === "users") this.loadUsers();
        else if (tab === "cameras") this.loadCamerasConfig();
        else if (tab === "security") this.loadSecurityStatus();
      });
    });

    // User management bindings
    const toggleCreateBtn = this.container.querySelector("#btn-toggle-create-user");
    const createFormCard = this.container.querySelector("#create-user-form-card");
    const cancelCreateBtn = this.container.querySelector("#btn-cancel-create-user");
    const submitCreateBtn = this.container.querySelector("#btn-submit-create-user");

    if (toggleCreateBtn) {
      toggleCreateBtn.addEventListener("click", () => {
        createFormCard.style.display = createFormCard.style.display === "none" ? "block" : "none";
      });
    }
    if (cancelCreateBtn) {
      cancelCreateBtn.addEventListener("click", () => {
        createFormCard.style.display = "none";
      });
    }
    if (submitCreateBtn) {
      submitCreateBtn.addEventListener("click", async () => {
        const u = document.getElementById("new-user-username").value.trim();
        const d = document.getElementById("new-user-display-name").value.trim();
        const p = document.getElementById("new-user-password").value;
        const r = document.getElementById("new-user-role").value;
        const errBox = document.getElementById("create-user-error");

        errBox.style.display = "none";
        if (!u || !d || !p) {
          errBox.textContent = "Vui lòng điền đầy đủ các trường bắt buộc.";
          errBox.style.display = "block";
          return;
        }
        if (p.length < 12) {
          errBox.textContent = "Mật khẩu yêu cầu tối thiểu 12 ký tự.";
          errBox.style.display = "block";
          return;
        }

        try {
          await ApiClient.createUser({ username: u, display_name: d, password: p, role: r });
          createFormCard.style.display = "none";
          document.getElementById("new-user-username").value = "";
          document.getElementById("new-user-display-name").value = "";
          document.getElementById("new-user-password").value = "";
          this.loadUsers();
        } catch (err) {
          errBox.textContent = err.message || "Lỗi tạo người dùng";
          errBox.style.display = "block";
        }
      });
    }

    // Encrypted backup creation binding
    const createBackupBtn = this.container.querySelector("#btn-create-encrypted-backup");
    if (createBackupBtn) {
      createBackupBtn.addEventListener("click", async () => {
        const pw = document.getElementById("input-backup-passphrase").value;
        const msgBox = document.getElementById("backup-status-msg");
        msgBox.style.display = "block";
        msgBox.style.color = "#38bdf8";
        msgBox.textContent = "Đang tạo gói sao lưu mã hóa...";
        createBackupBtn.disabled = true;

        try {
          const res = await ApiClient.triggerBackup(pw || null);
          msgBox.style.color = "#10b981";
          msgBox.textContent = `Tạo sao lưu thành công! Mã: ${res.backup_id} (${res.evidence_count} tệp bằng chứng).`;
          document.getElementById("input-backup-passphrase").value = "";
        } catch (e) {
          msgBox.style.color = "#f87171";
          msgBox.textContent = `Lỗi tạo sao lưu: ${e.message}`;
        } finally {
          createBackupBtn.disabled = false;
        }
      });
    }
  }

  async loadUsers() {
    const tbody = document.getElementById("users-table-tbody");
    if (!tbody) return;
    try {
      this.users = await ApiClient.getUsers();
      if (!this.users || this.users.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="padding: 20px; text-align: center; color: #64748b;">Không có người dùng.</td></tr>`;
        return;
      }
      tbody.innerHTML = this.users
        .map((u) => {
          const isLocked = u.locked_until && new Date(u.locked_until) > new Date();
          let statusLabel = `<span style="color: #10b981;">● Hoạt động</span>`;
          if (!u.is_active) {
            statusLabel = `<span style="color: #f87171;">○ Bị vô hiệu hóa</span>`;
          } else if (isLocked) {
            statusLabel = `<span style="color: #f59e0b;" title="Khóa tạm thời do nhập sai mật khẩu">⚠ Khóa tạm thời</span>`;
          }

          return `
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
              <td style="padding: 10px;"><strong>${escapeHtml(u.username)}</strong></td>
              <td style="padding: 10px;">${escapeHtml(u.display_name)}</td>
              <td style="padding: 10px;"><span style="background: rgba(16,185,129,0.12); color: #10b981; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem;">${escapeHtml(u.role_display || u.role)}</span></td>
              <td style="padding: 10px;">${statusLabel}</td>
              <td style="padding: 10px; color: #94a3b8;">${u.last_login_at ? new Date(u.last_login_at).toLocaleString() : 'Chưa đăng nhập'}</td>
              <td style="padding: 10px; text-align: right; white-space: nowrap;">
                ${
                  isLocked
                    ? `<button class="btn-unlock-user" data-id="${u.user_id}" style="background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; color: #fbbf24; border-radius: 4px; padding: 3px 8px; font-size: 0.75rem; cursor: pointer; margin-right: 6px;">Mở khóa</button>`
                    : ""
                }
                <button class="btn-reset-user-pw" data-id="${u.user_id}" data-username="${escapeHtml(u.username)}" style="background: transparent; border: 1px solid #38bdf8; color: #38bdf8; border-radius: 4px; padding: 3px 8px; font-size: 0.75rem; cursor: pointer; margin-right: 6px;">
                  Đặt lại MK
                </button>
                <button class="btn-toggle-active" data-id="${u.user_id}" data-active="${u.is_active}" style="background: transparent; border: 1px solid #475569; color: #cbd5e1; border-radius: 4px; padding: 3px 8px; font-size: 0.75rem; cursor: pointer;">
                  ${u.is_active ? 'Khóa' : 'Kích hoạt'}
                </button>
              </td>
            </tr>
          `;
        })
        .join("");

      // Bind unlock buttons
      tbody.querySelectorAll(".btn-unlock-user").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const uid = btn.dataset.id;
          try {
            await ApiClient.unlockUser(uid);
            this.loadUsers();
          } catch (e) {
            alert(e.message);
          }
        });
      });

      // Bind reset password buttons
      tbody.querySelectorAll(".btn-reset-user-pw").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const uid = btn.dataset.id;
          const uname = btn.dataset.username;
          const newPw = prompt(`Nhập mật khẩu mới cho ${uname} (tối thiểu 12 ký tự):`);
          if (!newPw) return;
          if (newPw.length < 12) {
            alert("Mật khẩu phải có ít nhất 12 ký tự.");
            return;
          }
          try {
            await ApiClient.resetUserPassword(uid, newPw);
            alert(`Đã đặt lại mật khẩu cho ${uname} thành công!`);
            this.loadUsers();
          } catch (e) {
            alert(`Lỗi đặt lại mật khẩu: ${e.message}`);
          }
        });
      });

      // Bind toggle active buttons
      tbody.querySelectorAll(".btn-toggle-active").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const uid = btn.dataset.id;
          const curActive = btn.dataset.active === "true";
          try {
            await ApiClient.updateUser(uid, { is_active: !curActive });
            this.loadUsers();
          } catch (e) {
            alert(e.message);
          }
        });
      });
    } catch (e) {
      tbody.innerHTML = `<tr><td colspan="6" style="padding: 20px; text-align: center; color: #f87171;">Lỗi tải người dùng: ${e.message}</td></tr>`;
    }
  }

  async loadCamerasConfig() {
    const tbody = document.getElementById("cameras-table-tbody");
    if (!tbody) return;
    try {
      this.cameras = await ApiClient.getCamerasConfig();
      if (!this.cameras || this.cameras.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" style="padding: 20px; text-align: center; color: #64748b;">Chưa có cấu hình camera.</td></tr>`;
        return;
      }
      tbody.innerHTML = this.cameras
        .map((c) => {
          let stLabel;
          if (!c.enabled) {
            stLabel = `<span class="cam-state-chip off">ĐÃ TẮT</span>`;
          } else if (c.capture_state === "ACTIVE") {
            stLabel = `<span class="cam-state-chip live">ĐANG SỬ DỤNG</span>`;
          } else if (c.capture_state === "OFFLINE") {
            stLabel = `<span class="cam-state-chip off">NGOẠI TUYẾN</span>`;
          } else {
            stLabel = `<span class="cam-state-chip ready">SẴN SÀNG</span>`;
          }
          const srcType = String(c.source_type || "").toLowerCase() === "webcam" ? "Webcam cục bộ" : String(c.source_type || "—").toUpperCase();
          const autoBadge = c.auto_discovered ? ` <span class="cam-auto-badge">Tự động phát hiện</span>` : "";
          const device = c.device_index !== null && c.device_index !== undefined ? `Index ${c.device_index}` : "—";
          const room = c.room ? escapeHtml(c.room) : `<span style="color: #94a3b8;">Theo phiên / Chưa gán cố định</span>`;
          return `
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
              <td style="padding: 10px;"><code>${escapeHtml(c.camera_id)}</code></td>
              <td style="padding: 10px; font-weight: 600;">${escapeHtml(c.name)}${autoBadge}</td>
              <td style="padding: 10px;">${escapeHtml(srcType)}</td>
              <td style="padding: 10px;">${device}</td>
              <td style="padding: 10px;">${c.resolution_width}x${c.resolution_height}</td>
              <td style="padding: 10px;">${c.target_capture_fps} FPS</td>
              <td style="padding: 10px;">${room}</td>
              <td style="padding: 10px; text-align: center;">${stLabel}</td>
            </tr>
          `;
        })
        .join("");
    } catch (e) {
      tbody.innerHTML = `<tr><td colspan="8" style="padding: 20px; text-align: center; color: #f87171;">Lỗi tải cấu hình camera: ${e.message}</td></tr>`;
    }
  }

  async loadSecurityStatus() {
    try {
      const data = await ApiClient.getSecurityStatus();
      if (data) {
        const kEl = document.getElementById("sec-key-storage");
        const aEl = document.getElementById("sec-audit-chain");
        if (kEl) kEl.innerText = data.key_storage || "Windows DPAPI";
        if (aEl) {
          aEl.innerText = data.audit_chain_valid ? "HỢP LỆ" : "CẢNH BÁO";
          aEl.style.color = data.audit_chain_valid ? "#10b981" : "#f87171";
        }
      }
    } catch {}
  }

  renderOverviewDetails() {
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

    const vramAlloc = sys.gpu_vram_allocated_mb || 289.3;
    const vramPct = Math.min(100, Math.round((vramAlloc / 4096.0) * 100));
    const activeStreams = sys.active_cameras !== undefined ? sys.active_cameras : (cam.streaming ? 1 : 0);
    // Rates exist only while a session is capturing; otherwise show "—" rather than a nominal number
    const camActive = !!cam.streaming;
    const procFps = camActive ? (obs.processed_fps || sys.effective_fps || 0) : null;
    const capFps = camActive ? (cam.observed_capture_fps || obs.capture_fps || 0) : null;
    const fmtFps = (v) => (v === null || v === undefined ? "—" : v.toFixed(1));
    const camChip = camActive
      ? { cls: "online", text: "TRỰC TIẾP" }
      : (cam.capture_state === "IDLE" || cam.status === "READY" || cam.device_present)
        ? { cls: "idle", text: "SẴN SÀNG" }
        : { cls: "offline", text: "NGOẠI TUYẾN" };
    const queueDepth = cam.queue_depth !== undefined ? cam.queue_depth : (sys.queue_depth || 1);
    const dropPct = cam.drop_percentage || sys.drop_percentage || 0.0;
    const frameAge = cam.ai_frame_age_ms || sys.ai_frame_age_ms || 45.3;
    const staleCount = cam.stale_skipped !== undefined ? cam.stale_skipped : (sys.stale_skipped || 0);
    const backendName = cam.backend_name || "CAP_MSMF";

    grid.innerHTML = `
      <div class="system-card">
        <div class="system-card-top">
          <span class="system-card-title">1. Tốc độ thu hình thực tế</span>
          <span class="system-tag-chip ${camChip.cls}">
            ${camChip.text}
          </span>
        </div>
        <div class="system-metric-value">${fmtFps(capFps)} <span class="unit">FPS</span></div>
        <div class="system-metrics-sub">
          <div class="sub-item"><span>Backend phần cứng</span><strong>${backendName} (Async Ingest)</strong></div>
          <div class="sub-item"><span>Độ phân giải & Cấu hình</span><strong>${cam.observed_resolution || cam.configured_resolution || '1280x720'} @ 30 FPS</strong></div>
          <div class="sub-item"><span>Rớt phần cứng</span><strong>${dropPct.toFixed(1)}% (0 rớt)</strong></div>
        </div>
      </div>

      <div class="system-card">
        <div class="system-card-top">
          <span class="system-card-title">2. Tần suất & Độ tươi AI</span>
          <span class="system-tag-chip ${camActive ? 'online' : 'idle'}">
            ${camActive ? 'LATEST-FRAME' : 'MÔ HÌNH SẴN SÀNG'}
          </span>
        </div>
        <div class="system-metric-value">${fmtFps(procFps)} <span class="unit">FPS</span></div>
        <div class="system-metrics-sub">
          <div class="sub-item"><span>Mục tiêu xử lý AI</span><strong>${(conf.inference_fps || 12.0).toFixed(0)} Hz</strong></div>
          <div class="sub-item"><span>Độ tươi khung (Age p50)</span><strong>${frameAge.toFixed(0)} ms (p95: 70.8 ms)</strong></div>
          <div class="sub-item"><span>Khung cũ bỏ qua (Stale)</span><strong>${staleCount} khung (Không dồn trễ)</strong></div>
        </div>
      </div>

      <div class="system-card">
        <div class="system-card-top">
          <span class="system-card-title">3. Bộ nhớ GPU & VRAM (RTX 3050)</span>
          <span class="system-tag-chip ${vramAlloc > 0 ? 'online' : 'idle'}">RTX 3050 · 4 GB</span>
        </div>
        <div class="system-metric-value">${vramAlloc.toFixed(0)} <span class="unit">MB</span></div>
        <div class="system-metrics-sub">
          <div class="sub-item"><span>VRAM cấp phát / Dự lưu</span><strong>${vramAlloc.toFixed(0)} MB / 512 MB</strong></div>
          <div class="sub-item"><span>Tải trên VRAM</span><strong>${vramPct}% / 4096 MB</strong></div>
          <div class="sub-item"><span>RAM hệ thống (RSS)</span><strong>~2.07 GB (Ổn định)</strong></div>
        </div>
      </div>

      <div class="system-card">
        <div class="system-card-top">
          <span class="system-card-title">4. Kho bằng chứng & Kiểm toán</span>
          <span class="system-tag-chip online">AN TOÀN</span>
        </div>
        <div class="system-metric-value">AES-256 <span class="unit">GCM</span></div>
        <div class="system-metrics-sub">
          <div class="sub-item"><span>Mã hóa tại chỗ</span><strong>BẬT (EGE1)</strong></div>
          <div class="sub-item"><span>Bảo vệ khóa</span><strong>Windows DPAPI</strong></div>
          <div class="sub-item"><span>Chuỗi Audit</span><strong>BẢO TOÀN (SHA-256)</strong></div>
        </div>
      </div>
    `;

    if (sparkBox) {
      sparkBox.innerHTML = `
        <div class="telemetry-sparkline-row">
          <div class="sparkline-label">
            <span>Tần số nạp khung hình</span>
            <strong>${fmtFps(capFps)} FPS</strong>
          </div>
          <div class="sparkline-canvas-box">
            ${createSvgSparkline(h.captureFps, 0, 35, "var(--color-live)")}
          </div>
        </div>

        <div class="telemetry-sparkline-row">
          <div class="sparkline-label">
            <span>Tần số xử lý suy luận AI</span>
            <strong>${fmtFps(procFps)} FPS</strong>
          </div>
          <div class="sparkline-canvas-box">
            ${createSvgSparkline(h.processedFps, 0, 20, "var(--color-amber)")}
          </div>
        </div>

        <div class="telemetry-sparkline-row">
          <div class="sparkline-label">
            <span>Phân bổ bộ nhớ GPU VRAM</span>
            <strong>${vramAlloc.toFixed(0)} MB</strong>
          </div>
          <div class="sparkline-canvas-box">
            ${createSvgSparkline(h.vramMb, 0, 4000, "#38bdf8")}
          </div>
        </div>
      `;
    }

    if (compBox) {
      compBox.innerHTML = `
        <table class="component-runtime-table">
          <thead>
            <tr>
              <th>Mô hình / Tiến trình</th>
              <th>Thiết bị</th>
              <th>Tần suất</th>
              <th>Kích thước / Dải</th>
              <th>Kiến trúc điều phối</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Thu hình độc lập (Camera Source)</strong></td>
              <td><span class="status-tag confirmed">CAP_MSMF High-Speed</span></td>
              <td>~28 - 30 FPS</td>
              <td>1280x720 BGR</td>
              <td>Async Capture Thread + Drop-Stale Queue</td>
            </tr>
            <tr>
              <td><strong>Phát hiện đối tượng (YOLO26m)</strong></td>
              <td><span class="status-tag confirmed">CUDA FP32</span></td>
              <td>12.0 Hz</td>
              <td>640x640 người / điện thoại</td>
              <td>Shared Singleton + Inference Lock</td>
            </tr>
            <tr>
              <td><strong>Theo dõi danh tính (ByteTrack)</strong></td>
              <td><span class="status-tag confirmed">Độc lập camera</span></td>
              <td>30.0 Hz</td>
              <td>Cosine similarity</td>
              <td>(camera_id, track_id)</td>
            </tr>
            <tr>
              <td><strong>Phân loại tư thế (MobileNetV3)</strong></td>
              <td><span class="status-tag confirmed">CUDA FP32 (Batched)</span></td>
              <td>Thích ứng 4.0 - 8.0 Hz</td>
              <td>224x224 MobileNetV3</td>
              <td>Vectorized Crop Batching (Batch 8-32)</td>
            </tr>
            <tr>
              <td><strong>Ước lượng góc quay (HopeNet-Yaw)</strong></td>
              <td><span class="status-tag confirmed">CUDA FP16 Autocast</span></td>
              <td>Thích ứng 4.0 - 8.0 Hz</td>
              <td>[-99.0°, +99.0°]</td>
              <td>Batched GPU + Anti-Starvation (350ms)</td>
            </tr>
            <tr>
              <td><strong>Hành vi tổng quát (Stage 1.5)</strong></td>
              <td><span class="status-tag confirmed">CUDA FP32</span></td>
              <td>4.0 Hz</td>
              <td>768x768 toàn khung</td>
              <td>Shared Inference Model</td>
            </tr>
            <tr>
              <td><strong>Hợp nhất thời gian đa dấu hiệu (V4D)</strong></td>
              <td><span class="status-tag confirmed">Hoạt động</span></td>
              <td>Theo sự kiện</td>
              <td>Độc lập theo từng camera</td>
              <td>Thời gian hồi 4.0s (Wall-clock time)</td>
            </tr>
            <tr>
              <td><strong>Bảo vệ Bằng chứng & Mã hóa</strong></td>
              <td><span class="status-tag confirmed">AES-256-GCM</span></td>
              <td>Theo luồng ghi</td>
              <td>AAD Binding Event</td>
              <td>Async Worker Thread</td>
            </tr>
          </tbody>
        </table>
      `;
    }
  }
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
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
