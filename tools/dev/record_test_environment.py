"""
Record Test Environment and Platform Metadata
=============================================
Records comprehensive hardware, software, framework, and tool versions.
Saves to runs/stage2_integrity/test_environment.json
"""

import json
import os
import platform
import sys
import time
from pathlib import Path
import psutil
import torch

try:
    import ultralytics
    ultralytics_ver = ultralytics.__version__
except Exception as e:
    ultralytics_ver = str(e)

try:
    import cv2
    cv2_ver = cv2.__version__
except Exception as e:
    cv2_ver = str(e)

try:
    import fastapi
    fastapi_ver = fastapi.__version__
except Exception as e:
    fastapi_ver = str(e)

try:
    import pytest
    pytest_ver = pytest.__version__
except Exception as e:
    pytest_ver = str(e)

OUT_DIR = Path("runs/stage2_integrity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

env_data = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "python": {
        "executable": sys.executable,
        "version": platform.python_version(),
        "build": platform.python_build(),
        "compiler": platform.python_compiler(),
    },
    "os": {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    },
    "hardware": {
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "system_ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0.0,
    },
    "frameworks": {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "ultralytics": ultralytics_ver,
        "opencv": cv2_ver,
        "fastapi": fastapi_ver,
        "psutil": psutil.__version__,
        "pytest": pytest_ver,
    },
}

with open(OUT_DIR / "test_environment.json", "w", encoding="utf-8") as f:
    json.dump(env_data, f, indent=2)

print("test_environment.json generated successfully:")
print(json.dumps(env_data, indent=2))
