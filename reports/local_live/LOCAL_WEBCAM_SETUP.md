# Local Live Camera Validation: Hardware & Camera Setup Report

**Phase:** LOCAL LIVE CAMERA VALIDATION  
**Execution Timestamp:** 2026-10-03T18:35:00+07:00  
**Host Architecture:** ASUS Desktop Workstation (Desktop Chassis Type 3)  
**Processor:** AMD Ryzen 9 9950X 16-Core Processor (32 logical threads)  
**GPU:** NVIDIA GeForce RTX 5070 (12,226.6 MB VRAM, CUDA 13.0, PyTorch 2.14.1+cu130)  
**Operating System:** Windows 11 Pro (10.0.26200-SP0)  
**Python Environment:** Python 3.13.9 (`.\.venv\Scripts\python.exe`)  

---

## 1. Hardware Enumeration & Probe Results

Probing was conducted across device indices `[0, 1, 2, 3]` using candidate Windows backends: DirectShow (`CAP_DSHOW`), Microsoft Media Foundation (`CAP_MSMF`), and Auto (`CAP_ANY`).

| Index | Backend Tested | Device Status | Reported Resolution | Reported FPS | Verdict |
|---|---|---|---|---|---|
| **0** | `CAP_DSHOW`, `CAP_MSMF`, `CAP_ANY` | Not Found | N/A | N/A | `NO_PHYSICAL_CAMERA` |
| **1** | `CAP_DSHOW`, `CAP_MSMF`, `CAP_ANY` | Not Found | N/A | N/A | `NO_PHYSICAL_CAMERA` |
| **2** | `CAP_DSHOW`, `CAP_MSMF`, `CAP_ANY` | Not Found | N/A | N/A | `NO_PHYSICAL_CAMERA` |
| **3** | `CAP_DSHOW`, `CAP_MSMF`, `CAP_ANY` | Not Found | N/A | N/A | `NO_PHYSICAL_CAMERA` |

### System PnP & USB Diagnostics
- **Windows Device Manager Query:** `Get-PnpDevice -Class Camera, Image` returned **0 devices present**.
- **USB Bus Inspection:** All connected devices are standard desktop peripherals (Sino Wealth Keyboard/Mouse dongle, Glorious Wireless receiver, ASUS Aura LED controller, AzureWave Bluetooth). An unconfigured device previously reporting `Device Descriptor Request Failed` on port 7 was disconnected cleanly.
- **Privacy Permission Audit:** `ConsentStore\webcam` checked at HKLM and HKCU: Both set to `Allow`.

---

## 2. Scientific Decision & Deferral Governance

Per explicit user authorization and strict scientific integrity:
1. **Zero Metric Fabrication:** No artificial synthetic frames, virtual webcam software (e.g. DroidCam/Iriun/OBS), or pre-recorded video loops were substituted as a webcam.
2. **Cameraless Software Verification:** The entire application and perception pipeline software path was validated independently using certified local test fixtures.
3. **Hardware Status:** `PHYSICAL_WEBCAM_PRESENT = NO`.
4. **Validation Status:** `LOCAL_LIVE_CAMERA_VALIDATION = DEFERRED_NO_CAMERA_HARDWARE`.
5. **Software Readiness:** `CAMERALESS_SOFTWARE_PREFLIGHT = PASS`, `READY_FOR_PHYSICAL_WEBCAM_TEST = YES`.

---

## 3. Configuration Verification

The local live camera configuration was audited and validated against the `Stage2Pipeline` schema in `configs/local_live_camera.yaml`:

```yaml
version: "2.1.0-local-live"
video_source:
  default_type: "webcam"
  default_path: 0
  camera_id: "laptop_webcam_0"
  target_fps: 30.0
  width: 1280
  height: 720
camera:
  source_type: "webcam"
  camera_index: 0
  camera_id: "laptop_webcam_0"
  requested_resolution: [1280, 720]
  requested_fps: 30.0
server:
  host: "127.0.0.1" # Strictly localhost
  port: 8000
```

All network interfaces are strictly bound to `127.0.0.1`.
