# EduAction Temporal Redundancy Analysis & Frame Sampling Policy

**Document ID**: `reports/v4b/EDUACTION_TEMPORAL_SAMPLING.md`  
**Phase**: V4B — Clean Person-Crop / Head-Pose Dataset Construction  
**Author**: Data Engineering & Audit Pipeline  
**Date**: 2026-10-02  
**Status**: APPROVED & EMPIRICALLY GROUNDED  

---

## 1. Executive Summary

EduAction contains **350 MP4 video clips** across 7 behavioral categories, recorded at 25.0 FPS with 3–5 seconds per clip ($70\text{--}160$ frames per clip).  
Because adjacent video frames captured at 25 FPS exhibit massive temporal autocorrelation, blind frame extraction would introduce over 39,000 highly redundant, near-duplicate images that artificially inflate dataset size and cause catastrophic data leakage if split across sets.

To solve this, an empirical temporal correlation study was executed on all 350 clips (`reports/v4b/eduaction_temporal_redundancy.json`), establishing an evidence-based temporal sampling policy that extracts **8 to 18 diverse, non-redundant frames per clip** while rejecting ~86–90% of redundant frames.

---

## 2. Empirical Frame-to-Frame Redundancy Measurements

Mean normalized absolute pixel differences $\Delta \in [0, 1]$ were measured at multiple temporal lags $L \in \{1, 2, 5, 10, 15, 20, 25\}$ frames across all categories:

| Category | Total Clips | Total Raw Frames | Mean $\Delta$ (Lag 1, 0.04s) | Mean $\Delta$ (Lag 5, 0.20s) | Mean $\Delta$ (Lag 15, 0.60s) | Mean $\Delta$ (Lag 25, 1.00s) | Motion Characteristic |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`sleeping`** | 50 | 4,987 | **0.0022** | 0.0056 | 0.0091 | 0.0105 | **Extremely Static** (99.8% visual overlap frame-to-frame) |
| **`writing`** | 50 | 7,991 | **0.0034** | 0.0110 | 0.0214 | 0.0273 | **Moderate Motion** (Hand/wrist movements, static torso) |
| **`lecture`** | 50 | 5,411 | **0.0042** | 0.0122 | 0.0222 | 0.0273 | **Low Motion** (Seated attentive listening) |
| **`play_phone`** | 50 | 5,050 | **0.0031** | 0.0086 | 0.0153 | 0.0187 | **Low-Moderate Motion** (Thumb manipulation) |
| **`talking`** | 50 | 5,118 | **0.0084** | 0.0275 | 0.0531 | 0.0673 | **High Motion** (Facial articulation, lateral nods) |
| **`drinking`** | 50 | 5,469 | **0.0087** | 0.0271 | 0.0515 | 0.0663 | **High Motion** (Arm reaching, bottle tipping) |
| **`watch_computer`** | 50 | 5,537 | **0.0040** | 0.0117 | 0.0217 | 0.0276 | **Low Motion** (Gaze focused on screen) |
| **Total Corpus** | **350** | **39,563** | **0.0049** | **0.0148** | **0.0278** | **0.0350** | ~88% Consecutive Redundancy |

### Key Empirical Findings
1. At 25 FPS (Lag 1, 40 ms), consecutive frame differences for `sleeping` are $0.0022$—visually indistinguishable to both human reviewers and convolutional feature maps.
2. Significant postural and action-phase transitions emerge only at $\ge 15$ frames ($\ge 0.60$ seconds).
3. A static stride alone is insufficient: active activities (`writing`, `talking`) produce new information faster than static postures (`sleeping`).

---

## 3. Evidence-Based Sampling Policy

To adaptively capture motion diversity without hardcoding arbitrary intervals, the sampling pipeline implements:

1. **Adaptive Stride Window:**
   $$\text{Stride } S = \max\left(4, \left\lfloor \frac{N_{\text{frames}}}{12} \right\rfloor \right)$$
   For a typical 100-frame clip, $S \approx 8$ frames ($0.32\text{ s}$ interval, ~3.1 FPS).
2. **Perceptual Difference Screening:**
   A candidate frame $F_t$ is evaluated against the last accepted frame $F_{\text{last}}$:
   $$\Delta(F_t, F_{\text{last}}) = \frac{1}{W \cdot H} \sum |F_t(x, y) - F_{\text{last}}(x, y)|$$
   - If $\Delta \ge 0.003$: **ACCEPT** (observable motion transition).
   - If $\Delta < 0.003$: Check elapsed interval $t - t_{\text{last}}$. If $t - t_{\text{last}} \ge 25$ frames ($1.0\text{ s}$), **ACCEPT** (preserves temporal anchor across sustained still poses); otherwise **REJECT** (discard redundant near-duplicate frame).

---

## 4. Semantic Ontology Mapping & Context Isolation

The sampled frames are mapped strictly according to the V4B ontology:

| EduAction Class | Target Observable Ontology Label | Future Canonical Collapse | Handling in V4B Posture Classifier |
| :--- | :--- | :--- | :--- |
| **`sleeping`** | `HEAD_REST_SLEEP` | `HEAD_DOWN` | **Supervised Positive** (Deep head down / resting on desk) |
| **`writing`** | `NORMAL_READ_WRITE` | `NORMAL` | **Supervised Positive** (Hard negative against head_down) |
| **`lecture`** | `NORMAL_UPRIGHT` | `NORMAL` | **Supervised Positive** (Upright classroom attention) |
| **`talking`** | `TALKING_CONTEXT` | `DISCUSS` / Interaction | **Context / Quarantined** (Do NOT map to `TURN_HEAD_CLEAR`) |
| **`play_phone`** | `PHONE_INTERACTION_CONTEXT` | Secondary Object Rule | **Context / Quarantined** (Do NOT map to `HEAD_DOWN` or `TURN_HEAD`) |
| **`drinking`** | `DRINKING_CONTEXT` | Normal Participation | **Context / Quarantined** |
| **`watch_computer`**| `COMPUTER_CONTEXT` | Normal Participation | **Context / Quarantined** |

---

## 5. Split-First Clip Assignment (Zero Video Leakage)

To satisfy the zero-leakage invariant, whole clips are assigned to splits **prior to frame extraction**:

| Split | Clip Allocation per Category | Total Clips Across 7 Classes | Sampling Method | Leakage Barrier |
| :--- | :--- | :--- | :--- | :--- |
| **`train`** | 35 clips (70.0%) | 245 clips | Adaptive Stride + Difference Filter | Clip ID isolated |
| **`same_domain_val`** | 8 clips (16.0%) | 56 clips | Adaptive Stride + Difference Filter | Independent clip IDs |
| **`temporal_holdout`** | 7 clips (14.0%) | 49 clips | Adaptive Stride + Difference Filter | Completely unseen clips |
| **Total** | **50 clips (100.0%)** | **350 clips** | — | **Zero cross-clip leakage** |

---

## 6. Frame Selection & Redundancy Statistics

| Category | Raw Video Frames | Selected Frames | Rejected Near-Duplicates | Selection Ratio | Frames per Clip |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`sleeping`** | 4,987 | 648 | 4,339 | 13.0% | 13.0 |
| **`writing`** | 7,991 | 962 | 7,029 | 12.0% | 19.2 |
| **`lecture`** | 5,411 | 647 | 4,764 | 12.0% | 12.9 |
| **`talking`** | 5,118 | 586 | 4,532 | 11.4% | 11.7 |
| **`play_phone`** | 5,050 | 649 | 4,401 | 12.8% | 13.0 |
| **Primary 5 Classes** | **28,557** | **3,492** | **25,065** | **12.2%** | **14.0** |
| **All 7 Classes** | **39,563** | **4,792** | **34,771** | **12.1%** | **13.7** |

By rejecting $34,771$ near-duplicate frames ($87.9\%$ redundancy reduction), the resulting sample set provides maximum visual and postural diversity while eliminating artificial overfitting.
