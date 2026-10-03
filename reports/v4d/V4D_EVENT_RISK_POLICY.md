# V4D Event Lifecycle, Risk Scoring & Explainability Policy

**Document ID**: `reports/v4d/V4D_EVENT_RISK_POLICY.md`  
**Phase**: Phase V4D — Temporal & Multi-Cue Fusion Layer  
**Date**: 2026-10-03  
**Status**: AUTHORITATIVE POLICY SPECIFICATION (RISK TERMINOLOGY RECONCILED)  

---

## 1. Ethical Governance & Core Principles

1. **Risk Is Observable Evidence Degree, NOT Guilt or Probability**:
   - Risk levels `LOW`, `MEDIUM`, `HIGH` represent the degree, temporal persistence, and concurrence of observable physical indicators.
   - The system **never** claims a student is "cheating", "guilty", or committing "fraud".
   - Terms such as "calibrated risk score", "calibrated probability", or "cheating probability" are strictly prohibited. The system outputs a **`CONFIGURED_RISK_SCORE`** or **`PROVISIONAL_EVIDENCE_SCORE`**.
2. **Provisional Engineering Thresholds**:
   - Numerical category boundaries (`LOW`: 0–39, `MEDIUM`: 40–74, `HIGH`: 75–100) are **`PROVISIONAL_ENGINEERING_THRESHOLDS`**. They are deterministic heuristic scoring policies, NOT empirical Bayesian probabilities.
3. **Single-Cue Safety Guardrail**:
   - An isolated noisy frame, single-frame object detection, or transient sensor glitch is **strictly capped at a score of $\le 25.0$** and categorized as **`LOW`** risk.
   - Escalation to `MEDIUM` ($\ge 40.0$) or `HIGH` ($\ge 75.0$) requires persistent temporal duration, multi-cue concurrence, or repeated recurrence.
4. **Strict Explainability**:
   - Every event object outputs structured factual metrics: cue probabilities, duration in seconds, competing negative evidence scores, and reliability weights.
   - Subjective narrative judgments (e.g. "AI detected student cheating") are permanently prohibited.

---

## 2. Risk Scoring Formulation

The continuous configured risk score $S \in [0.0, 100.0]$ is computed as:

$$ S = \min\left(100.0, \left( B_{\text{family}} + D(t) + C_{\text{multi}} + R_{\text{recur}} \right) \times W_{\text{reliability}} \right) $$

Where:
1. **$B_{\text{family}}$ (Base Family Score)**:
   - `PHONE_ASSOCIATED`: $45.0$
   - `MULTI_CUE_ATTENTION_SHIFT`: $35.0$
   - `DISCUSSION_CANDIDATE`: $30.0$
   - `SUSTAINED_LATERAL_HEAD_ORIENTATION`: $25.0$
   - `SUSTAINED_HEAD_REST`: $20.0$
   - `STANDING`: $15.0$
2. **$D(t)$ (Duration Factor)**:
   - For duration $t < 0.5\text{ s}$, $S \le 25.0$ (Single-cue safety cap).
   - For $t \ge 0.5\text{ s}$, $D(t) = \min(25.0, 10.0 \times \ln(1 + t))$.
3. **$C_{\text{multi}}$ (Independent Multi-Cue Concurrence Bonus)**:
   - $+20.0 \times (N_{\text{independent}} - 1)$ when multiple uncorrelated cues agree.
4. **$R_{\text{recur}}$ (Recurrence Bonus)**:
   - $+5.0$ per prior closed event on the same track ID (capped at $+15.0$).
5. **$W_{\text{reliability}}$ (Composite Reliability Weight)**:
   - Scaled by crop resolution ($H < 150\text{ px}$ penalty) and blur score ($\text{blur} < 40$ penalty).

---

## 3. Risk Level Assignment

- **`LOW` ($0.0 \le S < 40.0$)**: Routine posture variation or transient glance; logged for statistics, no immediate alert.
- **`MEDIUM` ($40.0 \le S < 75.0$)**: Sustained singular behavior (e.g., unbroken head-rest $> 3.0\text{ s}$); flagged for routine invigilator visual check.
- **`HIGH` ($75.0 \le S \le 100.0$)**: Multi-cue concurrence (e.g., persistent turn + phone association) or repeated high-confidence events; prioritizes human review queue.

---

## 4. Event Schema Compatibility

All fused events emit a `.to_dict()` structure 100% compatible with the existing `/api/events` endpoint:

```json
{
  "event_id": "eb0eac66-00de-44fb-a08d-8c8956af9eab",
  "track_id": 1,
  "camera_id": "cam_0",
  "event_type": "SUSTAINED_HEAD_REST",
  "start_timestamp": 1.08,
  "last_update_timestamp": 4.32,
  "end_timestamp": null,
  "duration": 3.24,
  "risk_level": "LOW",
  "risk_score": 26.46,
  "evidence_summary": {
    "posture_sleep_prob": 0.9883,
    "competing_read_write_score": 0.0025,
    "read_write_suppression": false,
    "candidate_duration_sec": 1.5,
    "risk_assessment": {
      "base_score": 20.0,
      "duration_factor": 14.45,
      "concurrence_bonus": 0.0,
      "recurrence_bonus": 0.0,
      "mean_reliability": 1.0,
      "final_score": 26.46,
      "risk_level": "LOW"
    }
  },
  "cue_availability": {
    "posture": "AVAILABLE",
    "headpose": "UNAVAILABLE",
    "phone": "UNAVAILABLE",
    "macro": "UNAVAILABLE"
  },
  "status": "active",
  "reviewer_notes": null,
  "fusion_config_version": "4.0.0-v4d"
}
```
