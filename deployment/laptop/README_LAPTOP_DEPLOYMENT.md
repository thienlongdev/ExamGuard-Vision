# ASUS TUF Gaming A17 (FA707RC) Live-Demo Deployment Guide

This document specifies the exact, step-by-step Windows 11 deployment procedure for setting up the portable live-demo environment on the target **ASUS TUF Gaming A17 (FA707RC)** laptop.

---

## Target Hardware Specifications

| Component | Target Specification |
|---|---|
| **System** | ASUS TUF Gaming A17 (Model/Board Identifier: `FA707RC`) |
| **OS** | Windows 11 Home Single Language (64-bit) |
| **CPU** | AMD Ryzen 7 6800H with Radeon Graphics (8C / 16T, 3.20 GHz base) |
| **RAM** | 16 GB DDR5 4800 MT/s SODIMM (2 / 2 slots occupied) |
| **Discrete GPU** | NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB Dedicated VRAM) |
| **Integrated GPU** | AMD Radeon Graphics |
| **Storage** | NVMe SSD ~512 GB class (~477 GB usable device capacity) |
| **Camera** | Integrated HD Webcam (expected index: `0`, DirectShow / MediaFoundation) |
| **Target Role** | `PORTABLE_LIVE_DEMO_TARGET` (Integrated camera capture, real human validation, FastAPI, WebSocket, Dashboard) |

> [!IMPORTANT]
> The target laptop has **4.0 GB dedicated VRAM**. Static 4-model loading requires approximately **388 MB**, leaving over **3.5 GB headroom** for inference context and video buffers. The laptop workload is strictly sized for **1 to 2 persons**, not dense 20-person 4K stress.

---

## 12-Step Windows Deployment Procedure

Open **PowerShell** (Run as Administrator or standard user with write permissions in your chosen development directory).

### STEP 1: Clone Repository
Clone the repository from GitHub:
```powershell
git clone <GITHUB_REPOSITORY_URL> DETECTOR-YOLO
```

### STEP 2: Navigate to Repository Root
```powershell
cd DETECTOR-YOLO
```

### STEP 3: Ensure Git LFS is Installed
Git LFS is required because `models/trained/v4_headpose_yaw_best.pt` (271 MB) and `runs/v4c/headpose_resnet18_yaw/best_model.pt` (128 MB) exceed GitHub's 100 MB hard limit:
```powershell
git lfs install
```

### STEP 4: Pull Git LFS Artifacts
Pull the actual model binary checkpoints:
```powershell
git lfs pull
```
*Verification*: Ensure `models/trained/v4_headpose_yaw_best.pt` is ~271 MB and NOT a 130-byte text pointer.

### STEP 5: Create a Fresh Python Virtual Environment
> [!WARNING]
> DO NOT copy or transfer the desktop's `.venv` directory. The laptop requires its own clean virtual environment built against its local Python and CUDA drivers.

```powershell
python -m venv .venv
```
*(Recommended: Python 3.10 - 3.13 64-bit)*

### STEP 6: Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If execution policies block scripts, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### STEP 7: Install Dependencies
Install the PyTorch stack compatible with the laptop's NVIDIA RTX 3050 driver, followed by the runtime dependencies:
```powershell
# Install PyTorch with CUDA support (e.g., CUDA 12.x / 13.x compatible with driver)
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install runtime dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
```

### STEP 8: Verify NVIDIA Driver & GPU State
Verify that the RTX 3050 Laptop GPU is recognized by NVIDIA Management Library:
```powershell
nvidia-smi
```
Confirm:
- GPU name: `NVIDIA GeForce RTX 3050 Laptop GPU`
- Driver Version: >= 535.xx
- CUDA Version supported by driver

### STEP 9: Run Automated Laptop Preflight
Execute the comprehensive laptop preflight script:
```powershell
.\.venv\Scripts\python.exe scripts/laptop_preflight.py
```
This script automatically performs:
1. System hardware inventory (CPU, RAM, GPU, VRAM, NVMe free space)
2. PyTorch CUDA capability and driver handshake
3. 7-checkpoint cryptographic hash certification & Git LFS pointer detection
4. Integrated webcam discovery (DirectShow / MediaFoundation)
5. Localhost port 8000 availability
6. OOM-safe sequential model memory test measuring actual VRAM delta

### STEP 10: Verify Cryptographic Checkpoint Integrity
The preflight script will verify the SHA-256 signatures of all required checkpoints:
- `yolo26m.pt`: `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`
- `models/trained/stage1_5_best.pt`: `68690cf82715dc6dad2eb35a08711531170e935eb7dbe1c30fafabae341e2c2c`
- `models/trained/v4_posture_best.pt`: `529a23f96ebec6051ad29279fba3cd5279e7aa8fe2e46292ab4bb9c9523ca180`
- `models/trained/v4_headpose_yaw_best.pt`: `5d15eec5941cfc8d2468d09c27bff8eca2014e62bb12fc9a95c0c6239fbbca55`

If any hash mismatches or file is an LFS pointer, the preflight script will halt immediately with actionable remediation steps.

### STEP 11: Probe Physical Integrated Camera
Ensure the preflight output confirms:
- `CAMERA_DISCOVERED = YES`
- `CAMERA_INDEX = 0` (or detected index)
- `CAMERA_ACTUAL_RESOLUTION = 1280x720` (or native sensor mode)
- `CAMERA_ACTUAL_FPS = 30.0`

### STEP 12: Launch Physical Live-Camera Demo & Validation
**Only after Step 9 outputs `READY_FOR_LAPTOP_LIVE_VALIDATION = YES`**, proceed to launch the live demo:

```powershell
# Launch FastAPI backend with integrated camera stream
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```
Then open your web browser to:
[http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)

---

## Runtime Profiles for Live Inference

The orchestrator supports 4 profiles:
- `BALANCED` (*Recommended starting bootstrap candidate*): 720p @ 30 FPS capture, YOLO @ 12 Hz, ByteTrack every frame, Posture @ 10 Hz, Headpose @ 5 Hz, Macro @ 4 Hz.
- `FULL`: Full sensor resolution, YOLO @ 30 Hz, all branches evaluated every frame.
- `LOW_POWER`: 720p @ 15 FPS capture, YOLO @ 8 Hz, Headpose @ 3 Hz.
- `AUTO`: Dynamically governed by measured latency P95/P99, queue depth, and VRAM margin.

Scientific behavior and fusion thresholds remain **100% identical** across all profiles.
