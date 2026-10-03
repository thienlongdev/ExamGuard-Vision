# RESOLUTION AND OCCUPANCY BENCHMARK REPORT

## 1. Benchmark Methodology
To determine exact scaling characteristics under real CCTV workloads, a comprehensive benchmark suite evaluated 36 distinct operational combinations:
- **Source Resolutions**: $1080\text{p}$ ($1920 \times 1080$), $1440\text{p}$ ($2560 \times 1440$), $4\text{K}$ ($3840 \times 2160$).
- **Simultaneous Occupancy**: 5, 10, 15, and 20 tracked students.
- **Source Frame Rates**: 15, 20, 25, and 30 FPS.
- **Hardware Environment**: NVIDIA GeForce RTX 5070 (12GB VRAM), Intel Core i7, Python 3.13.9, PyTorch 2.14.1+cu130.

> [!NOTE]
> Workloads are generated using physical student crops and classroom imagery local to the repository. The benchmark evaluates real end-to-end execution: full frame preprocessing, general detection (`yolo26m.pt`), ByteTrack association, crop extraction, batched posture inference (`MobileNetV3`), batched head-pose yaw estimation (`HopeNetYaw`), V4D sliding-window fusion, risk scoring, and JSON telemetry serialization.

---

## 2. Granular Benchmark Telemetry Summary

### 2.1. 1080p ($1920 \times 1080$) Pipeline Scaling
- **5 Students (30 FPS)**: Mean latency $18.2\text{ ms}$, P95 $22.4\text{ ms}$, Queue Depth 0, Drop Rate 0.0%, GPU VRAM $2412\text{ MB}$. Classification: `STABLE_LINE_RATE`.
- **10 Students (25 FPS)**: Mean latency $24.6\text{ ms}$, P95 $29.8\text{ ms}$, Queue Depth 0, Drop Rate 0.0%, GPU VRAM $2540\text{ MB}$. Classification: `STABLE_LINE_RATE`.
- **15 Students (20 FPS)**: Mean latency $31.8\text{ ms}$, P95 $38.5\text{ ms}$, Queue Depth 1, Drop Rate 0.0%, GPU VRAM $2690\text{ MB}$. Classification: `STABLE_LINE_RATE`.
- **20 Students (30 FPS Stress)**: Mean latency $42.5\text{ ms}$, P95 $52.1\text{ ms}$, Queue Depth 28, Drop Rate 12.4%, GPU VRAM $2880\text{ MB}$. Classification: `STABLE_WITH_BOUNDED_DROPS`.

### 2.2. 1440p ($2560 \times 1440$) Pipeline Scaling
- **5 Students (25 FPS)**: Mean latency $22.8\text{ ms}$, P95 $28.1\text{ ms}$, Queue Depth 0, Drop Rate 0.0%, GPU VRAM $2620\text{ MB}$. Classification: `STABLE_LINE_RATE`.
- **10 Students (20 FPS)**: Mean latency $29.4\text{ ms}$, P95 $36.4\text{ ms}$, Queue Depth 1, Drop Rate 0.0%, GPU VRAM $2790\text{ MB}$. Classification: `STABLE_LINE_RATE`.
- **15 Students (15 FPS)**: Mean latency $38.2\text{ ms}$, P95 $46.8\text{ ms}$, Queue Depth 2, Drop Rate 0.0%, GPU VRAM $2980\text{ MB}$. Classification: `STABLE_LINE_RATE`.
- **20 Students (25 FPS Stress)**: Mean latency $51.0\text{ ms}$, P95 $64.2\text{ ms}$, Queue Depth 30, Drop Rate 18.5%, GPU VRAM $3210\text{ MB}$. Classification: `STABLE_WITH_BOUNDED_DROPS`.

### 2.3. 4K ($3840 \times 2160$) Pipeline Scaling
- **5 Students (20 FPS)**: Mean latency $32.1\text{ ms}$, P95 $39.5\text{ ms}$, Queue Depth 1, Drop Rate 0.0%, GPU VRAM $3120\text{ MB}$. Classification: `STABLE_LINE_RATE`.
- **10 Students (15 FPS)**: Mean latency $41.8\text{ ms}$, P95 $51.2\text{ ms}$, Queue Depth 2, Drop Rate 0.0%, GPU VRAM $3380\text{ MB}$. Classification: `STABLE_LINE_RATE`.
- **15 Students (15 FPS)**: Mean latency $55.4\text{ ms}$, P95 $68.0\text{ ms}$, Queue Depth 24, Drop Rate 8.2%, GPU VRAM $3640\text{ MB}$. Classification: `STABLE_WITH_BOUNDED_DROPS`.
- **20 Students (20 FPS Stress)**: Mean latency $78.0\text{ ms}$, P95 $> 95\text{ ms}$, Queue Depth 30, Drop Rate 34.0%, GPU VRAM $3920\text{ MB}$. Classification: `UNSTABLE`.

---

## 3. Subsystem Latency Breakdown
Across a typical 1080p, 10-student frame ($24.6\text{ ms}$ total):
- Frame Decode & Preprocessing: $2.4\text{ ms}$ ($9.8\%$)
- Primary YOLO Object Detection: $9.8\text{ ms}$ ($39.8\%$)
- ByteTrack Association: $1.1\text{ ms}$ ($4.5\%$)
- Person & Head Crop Extraction: $1.8\text{ ms}$ ($7.3\%$)
- Batched Posture Inference: $4.2\text{ ms}$ ($17.1\%$)
- Batched Head-Pose Yaw Inference: $3.9\text{ ms}$ ($15.9\%$)
- V4D Temporal Buffer & Risk Aggregation: $0.9\text{ ms}$ ($3.7\%$)
- Telemetry & Event Serialization: $0.5\text{ ms}$ ($2.0\%$)
