# V4D Lateral Head Orientation & Multi-Cue Fusion Analysis

**Document ID**: `reports/v4d/V4D_TURN_HEAD_FUSION_ANALYSIS.md`  
**Phase**: Phase V4D — Temporal & Multi-Cue Fusion Layer  
**Date**: 2026-10-03  
**Status**: ARCHITECTURAL ANALYSIS & LIMITATION SPECIFICATION  

---

## 1. Classroom Yaw Bridge Finding

In Phase V4C, an extensive forensic evaluation of continuous yaw was conducted across $N=5,466$ SCBehavior classroom person crops ($N=4,465$ upright vs $N=1,001$ turn-head):
- `NORMAL_UPRIGHT` mean $|\text{yaw}| = 17.46^\circ$
- `TURN_HEAD_CLEAR` mean $|\text{yaw}| = 18.73^\circ$
- Mean Separation: **+1.27°**
- Cohen's d: **0.087**
- Empirical Distribution Overlap: **87.64%**

Because classroom students frequently rotate their heads within $\pm 20^\circ$ during ordinary note-taking, and because camera perspective angles compress horizontal head rotation, **continuous yaw alone is statistically incapable of reliably classifying turn behavior**.

---

## 2. Multi-Cue Fusion Design for Turn Events

To overcome this physical limitation, V4D implements **multi-cue agreement with correlated cue protection**:

1. **Posture as Primary Anchor**:
   - The discrete posture classifier (`TURN_HEAD_CLEAR`) evaluates overall head and torso orientation.
2. **Yaw as Supporting Evidence**:
   - Continuous yaw ($|\text{yaw}| \ge 25^\circ$) reinforces a high posture score ($P_{\text{turn}} \ge 0.40$), yielding:
     $$ \text{Evidence}_{\text{fused}} = 0.70 \times P_{\text{turn}} + 0.30 \times \min\left(1.0, \frac{|\text{yaw}| - 20^\circ}{40^\circ}\right) $$
3. **Correlated Cue Discounting**:
   - Because posture turn and yaw both physically reflect head rotation, their evidence weights are not naively summed. When posture turn is active, head-pose reliability is discounted by $0.30$ to prevent artificial confidence inflation.
4. **Frontal Yaw Attenuation**:
   - When head-pose indicates a clearly frontal face ($|\text{yaw}| < 20^\circ$), it acts as mild negative evidence ($0.85\times$ attenuation) against noisy posture turn predictions.

---

## 3. Strict Scientific Limitation

> [!WARNING]
> **`TEMPORAL_TURN_HEAD_POSITIVE_VALIDATION = NOT_SUPPORTED`**
>
> In the underlying datasets, `TURN_HEAD_CLEAR` samples are 100% sourced from `SCBehavior`, which consists of static frame crops. The temporal holdout set (`temporal_holdout.jsonl`) contains 42 video clips, all 100% sourced from `EduAction` (containing sleep, writing, and lecture clips, but **zero positive temporal turn clips**).
> 
> Consequently:
> - The multi-cue turn fusion logic has been validated through **functional test fixtures and unit tests**.
> - Empirical temporal recall and latency for lateral turn events are **NOT scientifically validated on continuous video**.
> - Video validation of sustained lateral turns is an explicit responsibility of Stage 2 using target-school classroom CCTV footage.
