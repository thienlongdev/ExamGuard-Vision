# V4D Raw Traceability Reconciliation & Operating Point Governance

## 1. Traceability Defect & Resolution
Prior Stage 2 claim traceability tables referenced JSON paths that do not exist in the repository (e.g. `isolated_evaluation.*`, `recommended_operating_point.*`). 

To establish absolute provenance, all temporal fusion and sleep calibration claims are reconciled directly against the physical machine-readable artifact:
`runs/v4d/V4D_EVALUATION_RESULTS.json` (SHA256: `18f77ea6bfa1dc860ea605df0434224750f64be8723cbf9ba51e70776785ceeb`).

---

## 2. Canonical Physical Schema vs Fictional Keys

| Referenced Entity | Fictional Claim Key (Disallowed) | Physical Ground-Truth JSON Path |
| :--- | :--- | :--- |
| **Sleep Temporal Calibration** | `isolated_evaluation.sleep.*` | `sleep_temporal_calibration["V4D_FULL_HYSTERESIS_DEBOUNCE_1.5S"]` |
| **Read/Write FP Audit** | `isolated_evaluation.false_positives.*` | `read_write_false_positive_audit` |
| **Duration Parameter Sweep** | `recommended_operating_point.sweep.*` | `temporal_threshold_sweep["1.5"]` |
| **Ablation Evidence** | `ablation_table.*` | `ablation_study` |
| **Track Continuity** | `continuity_stats.*` | `track_continuity_audit` |

---

## 3. Canonical Sleep Calibration Metrics

Extracted directly from `runs/v4d/V4D_EVALUATION_RESULTS.json`:

```json
"V4D_FULL_HYSTERESIS_DEBOUNCE_1.5S": {
  "strategy_id": "V4D_FULL_HYSTERESIS_DEBOUNCE_1.5S",
  "strategy": "v4d_hysteresis_debounce",
  "positive_clips": 20,
  "negative_normal_clips": 20,
  "negative_writing_clips": 20,
  "frame_precision": 1.0,
  "frame_recall": 0.8223,
  "frame_f1": 0.9025,
  "positive_clip_recall": 0.85,
  "normal_false_alarm_clips": 0,
  "writing_false_alarm_clips": 0,
  "total_negative_false_alarm_clips": 0,
  "fragmentation": 1.0,
  "activation_delay_mean": 1.556,
  "activation_delay_median": 1.5,
  "flicker": 0.0315,
  "hysteresis_enabled": true,
  "read_write_veto_enabled": true
}
```

### Verified Scientific Truth:
- **Positive Clip Recall**: $17 / 20 = 85.0\%$ (0.8500).
- **Negative Control False Event Clips**: $0 / 40 = 0.0\%$ (20 normal sitting clips + 20 active exam writing clips).
- **Read/Write False Positive Suppression**: 100.0% of writing false-positive events vetoed.
- **Mean Activation Delay**: 1.556 seconds.
- **Event Fragmentation**: 1.0 (each detected episode generated exactly one continuous event, zero fragmentation).

---

## 4. 1.5 Second Duration Governance & Wording

The physical threshold sweep in `temporal_threshold_sweep` recorded:

| Candidate Duration | Detected Clips (Recall) | Normal False Alarms | Writing False Alarms | Total False Alarms | Mean Activation Delay | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.5 s** | 17 / 20 (85.0%) | 0 | 0 | 0 / 40 | 0.556 s | `EQUIVALENT_PARETO_PLATEAU` |
| **1.0 s** | 17 / 20 (85.0%) | 0 | 0 | 0 / 40 | 1.056 s | `EQUIVALENT_PARETO_PLATEAU` |
| **1.5 s** | **17 / 20 (85.0%)** | **0** | **0** | **0 / 40** | **1.556 s** | `RECOMMENDED_ENGINEERING_OPERATING_POINT` |
| **2.0 s** | 17 / 20 (85.0%) | 0 | 0 | 0 / 40 | 2.056 s | `EQUIVALENT_PARETO_PLATEAU` |
| **2.5 s** | 17 / 20 (85.0%) | 0 | 0 | 0 / 40 | 2.556 s | `EQUIVALENT_PARETO_PLATEAU` |
| **3.0 s** | 14 / 20 (70.0%) | 0 | 0 | 0 / 40 | 3.000 s | Sensitivity Drop |

### Governance Rules Enforced:
1. **No "Optimal" Claims**: Because recall (85.0%) and false alarms (0) are identical across $0.5\text{ s} \le \tau \le 2.5\text{ s}$, 1.5 seconds is **not** mathematically or statistically optimal.
2. **Standard Terminology**: Designate as:
   $$\textbf{RECOMMENDED\_PROVISIONAL\_CANDIDATE\_DURATION} = 1.5\text{ s}$$
3. **Engineering Rationale**: 1.5 seconds provides a safety buffer against transient head-down glances (e.g., dipping head to inspect pencil or desk) while preserving full 85.0% clip recall and zero false alarms on the test corpus.
