# ExamGuard Vision — Deployment & Operations Guide

This guide details the setup, hardware requirements, configuration options, and runtime operations for ExamGuard Vision on local laptops and exam room workstations.

---

## 1. System Requirements

### Hardware Requirements
- **Recommended GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) or desktop equivalent (GTX 1660 / RTX 2060 / 3060 / 4060).
- **Minimum CPU:** 4 physical cores (AMD Ryzen 5/7 or Intel Core i5/i7).
- **RAM:** Minimum 8 GB DDR4/DDR5 (16 GB recommended).
- **Webcam / Camera:** DirectShow-compatible UVC USB Webcam (720p or 1080p @ 30 FPS) or RTSP H.264 IP Camera stream.
- **Operating System:** Windows 10/11 64-bit or Ubuntu Linux 22.04 LTS.

### Software Prerequisites
- **Python:** 3.10 to 3.13 (64-bit).
- **CUDA:** 12.1+ / 12.6 supported via PyTorch wheels.
- **Git & Git LFS:** Required for checkpoint tracking.

---

## 2. Fresh Installation Procedure

```powershell
# 1. Clone the repository and fetch LFS model checkpoints
git lfs install
git clone https://github.com/thienlongdev/ExamGuard-Vision.git
cd ExamGuard-Vision
git lfs pull

# 2. Create isolated Python virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Linux:
source .venv/bin/activate

# 4. Install PyTorch with CUDA support (for NVIDIA GPUs)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 5. Install runtime dependencies
pip install -r requirements-runtime.txt

# 6. Verify hardware and model checkpoints
python scripts/laptop_preflight.py
```

---

## 3. Running the Demo Application

### Standard Single-Command Launcher
```powershell
python scripts/run_asus_a17_demo.py
```

### CLI Command Options
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--camera-index` | `0` | Physical camera device index (0 for default laptop webcam) |
| `--port` | `8000` | HTTP port for the FastAPI server and web dashboard |
| `--host` | `127.0.0.1` | Network bind address (defaults to localhost) |
| `--headless` | `False` | Disables local OpenCV popup window (runs headless backend) |
| `--burn-in-web-hud` | `False` | Overlays diagnostic telemetry text directly into the MJPEG web stream |
| `--duration` | `None` | Automatically shuts down cleanly after N seconds (useful for soak tests) |

---

## 4. Accessing the Dashboard

Once the launcher displays `Dashboard: http://127.0.0.1:8000/`:
1. Open any modern browser (Chrome, Edge, Firefox, Safari).
2. Navigate to [http://127.0.0.1:8000/](http://127.0.0.1:8000/).
3. The dashboard connects via WebSocket (`ws://127.0.0.1:8000/ws/events`) to stream real-time events, KPI counters, and telemetry metrics.

---

## 5. Troubleshooting & FAQ

### Camera Device Busy (`Could not open physical webcam`)
- **Cause:** Another application (e.g. Windows Camera app, Zoom, Teams, browser) has locked the webcam device handle.
- **Resolution:** Close all camera-consuming applications. In Windows Settings, verify camera access permissions. Test device accessibility with:
  ```powershell
  python scripts/run_local_live_validation.py --check-camera
  ```

### Port 8000 Already in Use
- **Cause:** Another service is listening on port 8000.
- **Resolution:** Pass a different port parameter, e.g.:
  ```powershell
  python scripts/run_asus_a17_demo.py --port 8080
  ```
  Then access [http://127.0.0.1:8080/](http://127.0.0.1:8080/).

### Model Checkpoint File Size Small (~130 bytes)
- **Cause:** Git LFS pointers were cloned without the actual binary payloads.
- **Resolution:** Run `git lfs pull` from the repository root. Ensure `models/trained/v4_headpose_yaw_best.pt` is ~284 MB.

---

## 6. Hướng dẫn vận hành một click trên Windows (One-Click Operations)

ExamGuard Vision cung cấp trải nghiệm khởi động và dừng một click hoàn chỉnh trên Windows, phục vụ trình diễn và vận hành thực tế tại phòng thi mà không cần gõ lệnh dòng lệnh hay kích hoạt môi trường ảo thủ công.

### 6.1 Khởi động một click (Start ExamGuard Vision)
- **Cách thực hiện:** Double-click file `Start ExamGuard Vision.bat` tại thư mục gốc repository (hoặc shortcut **ExamGuard Vision** trên Desktop).
- **Quy trình tự động:**
  1. Kiểm tra môi trường Python `.venv`, file cấu hình `configs/runtime/asus_a17_demo.yaml` và sự hiện diện của 7 checkpoint mô hình AI.
  2. Kiểm tra tính khả dụng của cổng 8000 và phát hiện nếu hệ thống đã đang chạy (tránh xung đột tạo 2 phiên bản).
  3. Khởi động runtime camera (DirectShow 1280x720 @ 30 FPS) và pipeline nhận diện AI trên card đồ họa rời NVIDIA RTX 3050.
  4. Lắng nghe endpoint `/health`. Ngay khi hệ thống sẵn sàng, trình duyệt mặc định sẽ tự động mở trang Bảng điều khiển: `http://127.0.0.1:8000/`.

### 6.2 Dừng an toàn một click (Stop ExamGuard Vision)
- **Cách thực hiện:** Double-click file `Stop ExamGuard Vision.bat` (hoặc shortcut **Stop ExamGuard Vision** trên Desktop).
- **Quy trình tự động:**
  1. Đọc PID đã ghi nhận trong `.runtime/examguard.pid` và xác minh đúng tiến trình ExamGuard.
  2. Gửi tín hiệu dừng an toàn (graceful stop) qua file cờ `.runtime/stop.signal` và thông điệp đóng cửa sổ.
  3. Runtime tiến hành đóng luồng camera vật lý, giải phóng handle phần cứng, dừng backend FastAPI, dọn dẹp bộ nhớ VRAM CUDA và lưu trữ bằng chứng.
  4. Sau khi tiến trình dừng hoàn toàn, cổng 8000 được giải phóng và dọn sạch file trạng thái tạm thời.
  5. Tuyệt đối không can thiệp hay buộc dừng các tiến trình Python khác trên máy tính.

### 6.3 Cài đặt lối tắt ra màn hình Desktop (Desktop Shortcuts)
- **Cách thực hiện:** Double-click file `Install ExamGuard Shortcuts.bat` (chỉ cần chạy một lần duy nhất).
- **Kết quả:** Tự động tạo 2 lối tắt chuẩn Windows trên Desktop của người dùng:
  - `ExamGuard Vision.lnk`: Trỏ vào `Start ExamGuard Vision.bat` với Working Directory chuẩn.
  - `Stop ExamGuard Vision.lnk`: Trỏ vào `Stop ExamGuard Vision.bat`.
- **Lưu ý:** Không yêu cầu quyền Administrator.

### 6.4 Xử lý sự cố thường gặp (Troubleshooting)

#### 1. Xung đột cổng 8000 (Port 8000 Occupied)
- **Hiện tượng:** Launcher báo `[LỖI] Không thể khởi động ExamGuard Vision vì cổng 8000 đang được chương trình khác sử dụng.` kèm PID chiếm cổng.
- **Bảo vệ an toàn:** Hệ thống **tuyệt đối không tự ý ngắt (kill)** tiến trình lạ này.
- **Xử lý:** Kiểm tra PID được hiển thị trong thông báo lỗi (ví dụ dùng Task Manager hoặc `Get-Process -Id <PID>`) để tắt ứng dụng xung đột, hoặc khởi chạy với cổng khác.

#### 2. Camera đang bị ứng dụng khác khóa (Camera Occupied)
- **Hiện tượng:** Log báo lỗi `Could not open physical webcam index 0`.
- **Nguyên nhân:** Windows Camera, Teams, Zoom hoặc trình duyệt đang chiếm giữ camera UVC.
- **Xử lý:** Đóng các ứng dụng đang dùng webcam. Double-click `Stop ExamGuard Vision.bat` để đảm bảo sạch trạng thái, sau đó double-click `Start ExamGuard Vision.bat` lại.

#### 3. Tra cứu nhật ký runtime (Startup Logs)
- File nhật ký hoạt động được tự động ghi tại:
  `.runtime/logs/examguard-runtime.log`
- Khi gặp sự cố khởi động, launcher sẽ tự động trích xuất 15 dòng log gần nhất để người dùng nắm được nguyên nhân trực tiếp.

#### 4. Chẩn đoán phần cứng toàn diện (Full Preflight Certification)
- Khi cần kiểm tra SHA-256 toàn vẹn của tất cả 7 model, đo đạc thông số phần cứng CPU/RAM/VRAM chi tiết và kiểm tra khả năng DirectShow:
  ```powershell
  .\.venv\Scripts\python.exe scripts\laptop_preflight.py
  ```
