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
