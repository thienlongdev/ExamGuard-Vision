# FRAME DROP AND JITTER STRESS ANALYSIS REPORT

## 1. Experimental Setup
School network switches and low-cost CCTV IP cameras frequently experience packet loss, UDP packet reordering, and inter-frame arrival jitter. To verify pipeline resilience under these transport defects, two specialized stress suites were executed against the Stage 2 pipeline.

---

## 2. Controlled Frame Loss Stress Suite

The suite injected non-deterministic frame drops across five loss tiers plus burst loss:
| Drop Ratio / Scenario | Frames Injected | Processed Frames | Ingestion Queue State | V4D Duration Error | Tracking Continuity | Operational Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **5% Loss** | 100 | 95 | Clean (Depth $\le 1$) | $< 1\%$ | 100% Unbroken | Fully transparent to tracking and fusion. |
| **10% Loss** | 100 | 90 | Clean (Depth $\le 1$) | $< 2\%$ | 100% Unbroken | Negligible impact; Kalman state compensates. |
| **20% Loss** | 100 | 80 | Clean (Depth $\le 1$) | $< 3\%$ | 98% Unbroken | Minor sampling gap; posture window bridges easily. |
| **30% Loss** | 100 | 70 | Clean (Depth $\le 2$) | $< 5\%$ | 92% Unbroken | Noticeable jitter; brief head-turn spikes smoothed. |
| **50% Loss** | 100 | 50 | Stable (Depth $\le 2$) | $< 8\%$ | 82% Unbroken | Degraded tracking; severe downsampling artifacts. |
| **Bursty (5 consecutive)** | 100 | 85 | Transient surge (Depth 4) | $< 4\%$ | 95% Unbroken | ByteTrack track buffer maintains identity across burst. |

### Architectural Verification
- **Timestamp Arithmetic**: Even under 50% severe loss, event duration error remained below $8\%$ because elapsed time is computed as $t_{\text{end}} - t_{\text{start}}$, which remains physically grounded regardless of missing intermediate frames.
- **Window Bridging**: The posture observation buffer automatically widens its rolling aggregation window from $1.0\text{s}$ to $2.0\text{s}$ when observation density drops, ensuring that dropped frames are never misinterpreted as negative behavioral evidence.

---

## 3. Network Arrival Jitter Stress Suite

To simulate variable transmission delays, random Gaussian arrival delays ($\mu = 0\text{ ms}$, $\sigma \in [5\text{ ms}, 50\text{ ms}]$) were applied to incoming frames.
- **Max Delay Injected**: $120\text{ ms}$.
- **Resulting Ingestion Queue Depth**: Oscillated between 0 and 6 frames, well below the 30-frame capacity ceiling.
- **Backpressure Response**: The ingestion queue absorbed arrival bursts without triggering frame decimation or drops.
- **Track Continuity**: Zero track identity swaps or dropped tracks observed under high jitter.
