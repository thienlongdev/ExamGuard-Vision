# V4D Temporal Threshold Sweep & Calibration Study

**Document ID**: `reports/v4d/V4D_TEMPORAL_CALIBRATION.md`  
**Phase**: Phase V4D — Temporal & Multi-Cue Fusion Layer  
**Date**: 2026-10-03  
**Status**: AUDITED RECONCILED REPORT (SOURCE ARTIFACT: `runs/v4d/threshold_sweep.json`)  

---

## 1. Executive Summary & Protocol Reconciliation

This report documents the empirical evaluation of temporal duration persistence thresholds across the 60 physical surveillance clips (768 frames) from the frozen V4C test benchmark.

### 1.1 Resolution of Prior Metric Discrepancy
Historical exploratory drafts contained mutually incompatible numbers (e.g. 20/20 clips with 1 false alarm vs 17/20 clips with 0 false alarms). Forensic audit of the execution traces revealed the exact cause:
- **Continuous Stream Concatenation (Historical Draft Artifact)**: When all 20 sleep clips were streamed into a single unreset `ReplayEngine` on `track_id=1`, active events from earlier clips bridged across the 3 perception-negative clips (`sleep (15)`, `sleep (5)`, `sleep (4)`), yielding an apparent 20/20 detection. When this unreset engine continued into the negative control set, the trailing open event from the 20th sleep clip closed on the first negative clip (`writing (12).mp4`), registering as "1 false event on negatives".
- **Strict Episodic Evaluation (Canonical Source of Truth)**: In production surveillance, each video clip represents an independent observation. When each clip is evaluated with fresh track state:
  - **Positive clip recall** is **17 / 20 (85.0%)**, bounded strictly by the upstream perception model (`MobileNetV3-Small` output $p \le 0.09$ on 3 clips).
  - **Negative false events** are strictly **0 / 40 (0 false events)** across both normal upright and writing control sets.

---

## 2. Canonical Empirical Threshold Sweep Results

All evaluations were executed on the physical temporal test clips using `scripts/evaluate_v4d_fusion.py` under identical hysteresis parameters ($\tau_{\text{enter}} = 0.55$, $\tau_{\text{exit}} = 0.35$, $\tau_{\text{veto}} = 0.50$):

| Candidate Duration ($D_{\text{cand}}$) | Detected True Positive Clips | Clip Recall (%) | Normal False Alarms | Writing False Alarms | Total False Alarms | Mean Activation Delay (s) | Median Activation Delay (s) | Mean Event Duration (s) | Event Fragmentation | Operating Point Classification |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.5 s** | 17 / 20 | 85.0% | 0 | 0 | **0** | 0.56 s | 0.50 s | 3.68 s | 1.00 | `EQUIVALENT_PARETO_PLATEAU` |
| **1.0 s** | 17 / 20 | 85.0% | 0 | 0 | **0** | 1.06 s | 1.00 s | 3.68 s | 1.00 | `EQUIVALENT_PARETO_PLATEAU` |
| **1.5 s** | **17 / 20** | **85.0%** | **0** | **0** | **0** | **1.56 s** | **1.50 s** | **3.68 s** | **1.00** | **`RECOMMENDED_ENGINEERING_OPERATING_POINT`** |
| **2.0 s** | 17 / 20 | 85.0% | 0 | 0 | **0** | 2.06 s | 2.00 s | 3.68 s | 1.00 | `EQUIVALENT_PARETO_PLATEAU` |
| **2.5 s** | 17 / 20 | 85.0% | 0 | 0 | **0** | 2.56 s | 2.50 s | 3.68 s | 1.00 | `EQUIVALENT_PARETO_PLATEAU` |
| **3.0 s** | 14 / 20 | 70.0% | 0 | 0 | **0** | 3.00 s | 3.00 s | 3.85 s | 1.00 | `SUBOPTIMAL_HIGH_LATENCY_OR_TRUNCATION` |
| **4.0 s** | 4 / 20 | 20.0% | 0 | 0 | **0** | 4.00 s | 4.00 s | 4.32 s | 1.00 | `SUBOPTIMAL_HIGH_LATENCY_OR_TRUNCATION` |

---

## 3. Scientific Operating Point Justification

### 3.1 Formal Selection Objective
Operating points are evaluated under a lexicographical multi-objective criterion:
1. **Primary Priority**: Zero writing false events ($F_{\text{writing}} = 0$).
2. **Secondary Priority**: Minimize total negative false alarms ($F_{\text{total}} = 0$).
3. **Tertiary Priority**: Maximize positive clip recall ($R_{\text{clip}} \ge 85.0\%$).
4. **Quaternary Priority**: Minimize event fragmentation ($\text{Frag} = 1.00$).
5. **Tie-Breaker Priority**: Robust noise margin against momentary physical posture dips vs. activation latency.

### 3.2 Evaluation of Pareto Plateau
Durations $[0.5\text{ s}, 1.0\text{ s}, 1.5\text{ s}, 2.0\text{ s}, 2.5\text{ s}]$ **tie exactly** on the top four criteria:
- Clip Recall = 85.0% (17/20)
- Normal False Alarms = 0
- Writing False Alarms = 0
- Fragmentation = 1.00

Because multiple thresholds tie, **there is NO unique mathematical optimum**. 

### 3.3 Operating Point Recommendation
- **$D_{\text{cand}} = 1.5\text{ s}$** is designated as the **`RECOMMENDED_ENGINEERING_OPERATING_POINT`** (or **`RECOMMENDED_CANDIDATE_DURATION`**).
- **Physical Rationale**:
  - $0.5\text{ s}$ and $1.0\text{ s}$ possess minimal safety margin against transient physical head-dips or natural micro-movements in continuous classroom feeds.
  - $2.0\text{ s}$ and $2.5\text{ s}$ add unnecessary detection latency ($2.06\text{ s}$ and $2.56\text{ s}$) and risk truncating shorter resting bouts.
  - $1.5\text{ s}$ provides balanced debounce stability with a modest, predictable activation latency ($1.56\text{ s}$).
- **Status Classification**: Classified strictly as **`PROVISIONAL_ENGINEERING_THRESHOLD`**, pending live target-school CCTV pilot data in Stage 2.
