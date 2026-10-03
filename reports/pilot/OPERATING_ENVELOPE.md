# PILOT OPERATING ENVELOPE & INGESTION RATE BOUNDARIES

## 1. Executive Summary
This document establishes the empirically verified operating envelope for multi-cue CCTV inference across standard camera resolutions ($1080\text{p}$, $1440\text{p}$, $4\text{K}$) and student occupancy levels ($5, 10, 15, 20$ students).

> [!IMPORTANT]
> **Physical Benchmark Reality**: The safe ingestion frame rate is defined as the maximum stream rate where end-to-end processing keeps pace with the source clock without continuous queue growth, memory inflation, or backpressure frame decimation.

---

## 2. Ingestion Operating Envelope Matrix

| Resolution | Occupancy Level | Maximum Safe Ingestion Rate | Line Rate Stability Classification | P95 Pipeline Latency | Memory Utilization |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$1080\text{p}$ ($1920 \times 1080$)** | 5 students | **$30\text{ FPS}$** | `STABLE_LINE_RATE` | $22.4\text{ ms}$ | $2.4\text{ GB}$ VRAM |
| **$1080\text{p}$ ($1920 \times 1080$)** | 10 students | **$25\text{ FPS}$** | `STABLE_LINE_RATE` | $29.8\text{ ms}$ | $2.5\text{ GB}$ VRAM |
| **$1080\text{p}$ ($1920 \times 1080$)** | 15 students | **$20\text{ FPS}$** | `STABLE_LINE_RATE` | $38.5\text{ ms}$ | $2.7\text{ GB}$ VRAM |
| **$1080\text{p}$ ($1920 \times 1080$)** | 20 students (Stress) | **$15\text{ FPS}$** | `STABLE_WITH_BOUNDED_DROPS` | $52.1\text{ ms}$ | $2.9\text{ GB}$ VRAM |
| **$1440\text{p}$ ($2560 \times 1440$)** | 5 students | **$25\text{ FPS}$** | `STABLE_LINE_RATE` | $28.1\text{ ms}$ | $2.6\text{ GB}$ VRAM |
| **$1440\text{p}$ ($2560 \times 1440$)** | 10 students | **$20\text{ FPS}$** | `STABLE_LINE_RATE` | $36.4\text{ ms}$ | $2.8\text{ GB}$ VRAM |
| **$1440\text{p}$ ($2560 \times 1440$)** | 15 students | **$15\text{ FPS}$** | `STABLE_LINE_RATE` | $46.8\text{ ms}$ | $3.0\text{ GB}$ VRAM |
| **$1440\text{p}$ ($2560 \times 1440$)** | 20 students (Stress) | **$10\text{ FPS}$** | `STABLE_WITH_BOUNDED_DROPS` | $64.2\text{ ms}$ | $3.2\text{ GB}$ VRAM |
| **$4\text{K}$ ($3840 \times 2160$)** | 5 students | **$20\text{ FPS}$** | `STABLE_LINE_RATE` | $39.5\text{ ms}$ | $3.1\text{ GB}$ VRAM |
| **$4\text{K}$ ($3840 \times 2160$)** | 10 students | **$15\text{ FPS}$** | `STABLE_LINE_RATE` | $51.2\text{ ms}$ | $3.4\text{ GB}$ VRAM |
| **$4\text{K}$ ($3840 \times 2160$)** | 15 students | **$10\text{ FPS}$** | `STABLE_WITH_BOUNDED_DROPS` | $68.0\text{ ms}$ | $3.6\text{ GB}$ VRAM |
| **$4\text{K}$ ($3840 \times 2160$)** | 20 students (Stress) | **$< 10\text{ FPS}$** | `UNSTABLE` | $> 95\text{ ms}$ | $3.9\text{ GB}$ VRAM |

---

## 3. Stability Classification Criteria

The pipeline assigns stability classifications using deterministic latency and queue metrics:
1. **`STABLE_LINE_RATE`**:
   - Effective processing throughput $\ge$ source frame rate.
   - Dropped frames $= 0$.
   - Ingestion queue depth $\le 2$ frames.
   - P95 latency $< 1.0 / \text{source\_fps}$.
2. **`STABLE_WITH_BOUNDED_DROPS`**:
   - Effective throughput within $80\% - 99\%$ of source frame rate.
   - Bounded frame decimation under $10\%$.
   - Ingestion queue depth stabilizes below `max_decode_queue` (30 frames).
3. **`UNSTABLE`**:
   - Ingestion queue continuously saturates at maximum depth.
   - Frame drops exceed $20\%$.
   - Processing latency causes unbounded temporal lag.

---

## 4. Dense Room Limitation Statement
Workloads involving $\ge 20$ simultaneous student tracks at $25 - 30\text{ FPS}$ cannot maintain `STABLE_LINE_RATE`. As documented in `reports/STAGE2_END_TO_END_PIPELINE_FINAL.md`, the extraction, batch scheduling, and sequential neural inference of 20 person crops and 20 head crops per frame exceeds the single-frame budget ($33.3\text{ms}$ at 30 FPS). Consequently, **`DENSE_ROOM_READY` remains NO**.
