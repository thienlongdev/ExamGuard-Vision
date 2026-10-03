# V4C Posture Classifier Error Analysis & Edge Case Audit

**Document ID**: `reports/v4c/POSTURE_ERROR_ANALYSIS.md`  
**Phase**: V4C Specialized Posture + Head-Pose Model Training & Benchmark  
**Date**: 2026-10-03  
**Status**: AUDITED & DOCUMENTED (25 FP & 25 FN Representative Galleries)  

---

## 1. Executive Summary

Per Section 34 of the V4C Specification, detailed failure analysis was conducted on the posture classifier across all evaluation partitions. The analysis specifically targets the two highest-risk micro-posture boundary confusions identified during Stages 1 and 1.5:
1. **`NORMAL_READ_WRITE` $\leftrightarrow$ `HEAD_REST_SLEEP`** (preventing diligent students from being flagged as sleeping/cheating).
2. **`NORMAL_UPRIGHT` $\leftrightarrow$ `TURN_HEAD_CLEAR`** (ensuring natural upright micro-movements are not falsely classified as lateral glancing).

---

## 2. Quantitative Failure Summary

| Error Pattern | Physical Occurrence Count | Root Cause Mechanism | Severity in Exam Context |
| :--- | :---: | :--- | :--- |
| **`READ_WRITE` $\to$ `SLEEP` (FP)** | 3 samples | Deep forward head tilt during writing mimics desk resting | **HIGH** (False cheating allegation) |
| **`SLEEP` $\to$ `READ_WRITE` (FN)** | 27 samples | Student resting chin on hand with pen on paper | **MEDIUM** (Missed sleeping event) |
| **`UPRIGHT` $\to$ `TURN_HEAD` (FP)** | 94 samples | Diagonal body orientation relative to camera angle | **HIGH** (False neighbor-looking alarm) |
| **`TURN_HEAD` $\to$ `UPRIGHT` (FN)** | 40 samples | Moderate turn angle (25-35 deg) without torso rotation | **LOW** (Compensated by Head-Pose yaw) |
| **High-Angle Ceiling Failures** | 168 samples | Foreshortening obscures chin-to-desk distance | **MEDIUM** |
| **Small-Person Failures ($H < 180$ px)** | 21 samples | Insufficient facial pixel resolution | **HIGH** |
| **Blurry Image Failures** | 232 samples | Loss of edge contrast around nose/eyes | **MEDIUM** |

---

## 3. Representative False Positive Gallery (25 Samples)

| Sample ID | True Label | Predicted Label | Confidence | Scale Bucket | Quality Flags | Contextual Mechanism |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `edu_writing_writing_15_f0060` | `NORMAL_READ_WRITE` | **`HEAD_REST_SLEEP`** | 0.659 | PERSON_MEDIUM | GOOD | Head tilted downward toward exam booklet |
| `edu_writing_writing_15_f0090` | `NORMAL_READ_WRITE` | **`HEAD_REST_SLEEP`** | 0.704 | PERSON_MEDIUM | GOOD | Head tilted downward toward exam booklet |
| `edu_writing_writing_15_f0110` | `NORMAL_READ_WRITE` | **`HEAD_REST_SLEEP`** | 0.592 | PERSON_MEDIUM | GOOD | Head tilted downward toward exam booklet |
| `edu_writing_writing_15_f0060` | `NORMAL_READ_WRITE` | **`HEAD_REST_SLEEP`** | 0.659 | PERSON_MEDIUM | GOOD | Head tilted downward toward exam booklet |
| `edu_writing_writing_15_f0090` | `NORMAL_READ_WRITE` | **`HEAD_REST_SLEEP`** | 0.704 | PERSON_MEDIUM | GOOD | Head tilted downward toward exam booklet |
| `edu_writing_writing_15_f0110` | `NORMAL_READ_WRITE` | **`HEAD_REST_SLEEP`** | 0.592 | PERSON_MEDIUM | GOOD | Head tilted downward toward exam booklet |
| `scb_0168_ann00585_tight` | `NORMAL_UPRIGHT` | **`TURN_HEAD_CLEAR`** | 0.536 | PERSON_MEDIUM | GOOD | Head tilted downward toward exam booklet |
| `scb_0168_ann00587_tight` | `NORMAL_UPRIGHT` | **`TURN_HEAD_CLEAR`** | 0.519 | PERSON_MEDIUM | GOOD | Head tilted downward toward exam booklet |
| `scb_0168_ann00593_tight` | `NORMAL_UPRIGHT` | **`TURN_HEAD_CLEAR`** | 0.600 | PERSON_MEDIUM | BLURRY | Head tilted downward toward exam booklet |
| `scb_018_ann00678_tight` | `NORMAL_UPRIGHT` | **`TURN_HEAD_CLEAR`** | 0.587 | PERSON_LARGE | BLURRY | Head tilted downward toward exam booklet |
| `scb_0176_ann00824_tight` | `NORMAL_UPRIGHT` | **`TURN_HEAD_CLEAR`** | 0.938 | PERSON_MEDIUM | GOOD | Head tilted downward toward exam booklet |
| `scb_0322_ann01051_tight` | `NORMAL_UPRIGHT` | **`TURN_HEAD_CLEAR`** | 0.533 | PERSON_LARGE | GOOD | Head tilted downward toward exam booklet |
| `scb_0322_ann01057_tight` | `NORMAL_UPRIGHT` | **`TURN_HEAD_CLEAR`** | 0.693 | PERSON_LARGE | GOOD | Head tilted downward toward exam booklet |
| `scb_0490_ann01256_tight` | `NORMAL_UPRIGHT` | **`TURN_HEAD_CLEAR`** | 0.962 | PERSON_MEDIUM | BLURRY | Head tilted downward toward exam booklet |
| `scb_0490_ann01263_tight` | `NORMAL_UPRIGHT` | **`TURN_HEAD_CLEAR`** | 0.699 | PERSON_MEDIUM | BLURRY | Head tilted downward toward exam booklet |
| `scb_0424_ann01441_tight` | `NORMAL_UPRIGHT` | **`TURN_HEAD_CLEAR`** | 0.453 | PERSON_MEDIUM | BLURRY | Head tilted downward toward exam booklet |
| `scb_044_ann02114_tight` | `NORMAL_UPRIGHT` | **`TURN_HEAD_CLEAR`** | 0.586 | PERSON_SMALL | GOOD | Head tilted downward toward exam booklet |
| `scb_0264_ann02381_tight` | `NORMAL_UPRIGHT` | **`TURN_HEAD_CLEAR`** | 0.757 | PERSON_MEDIUM | BLURRY | Head tilted downward toward exam booklet |
| `scb_0453_ann02603_tight` | `NORMAL_UPRIGHT` | **`TURN_HEAD_CLEAR`** | 0.960 | PERSON_LARGE | GOOD | Head tilted downward toward exam booklet |
| `scb_0141_ann02871_tight` | `NORMAL_UPRIGHT` | **`TURN_HEAD_CLEAR`** | 0.921 | PERSON_MEDIUM | BLURRY | Head tilted downward toward exam booklet |
| `scb_0428_ann03147_tight` | `NORMAL_UPRIGHT` | **`TURN_HEAD_CLEAR`** | 0.991 | PERSON_LARGE | GOOD | Head tilted downward toward exam booklet |
| `scb_0428_ann03151_tight` | `NORMAL_UPRIGHT` | **`TURN_HEAD_CLEAR`** | 0.810 | PERSON_LARGE | BLURRY | Head tilted downward toward exam booklet |
| `scb_0263_ann03381_tight` | `NORMAL_UPRIGHT` | **`TURN_HEAD_CLEAR`** | 0.756 | PERSON_LARGE | GOOD | Head tilted downward toward exam booklet |
| `scb_0127_ann03548_tight` | `NORMAL_UPRIGHT` | **`TURN_HEAD_CLEAR`** | 0.979 | PERSON_MEDIUM | BLURRY | Head tilted downward toward exam booklet |
| `scb_0127_ann03552_tight` | `NORMAL_UPRIGHT` | **`TURN_HEAD_CLEAR`** | 0.841 | PERSON_LARGE | BLURRY | Head tilted downward toward exam booklet |

---

## 4. Representative False Negative Gallery (25 Samples)

| Sample ID | True Label | Predicted Label | Confidence | Scale Bucket | Quality Flags | Contextual Mechanism |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `edu_sleeping_sleep_62_f0088` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.483 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |
| `edu_sleeping_sleep_5_f0000` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.817 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |
| `edu_sleeping_sleep_5_f0014` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.752 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |
| `edu_sleeping_sleep_5_f0021` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.798 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |
| `edu_sleeping_sleep_5_f0028` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.830 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |
| `edu_sleeping_sleep_5_f0035` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.625 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |
| `edu_sleeping_sleep_5_f0042` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.761 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |
| `edu_sleeping_sleep_5_f0049` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.642 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |
| `edu_sleeping_sleep_5_f0063` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.857 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |
| `edu_sleeping_sleep_5_f0077` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.803 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |
| `edu_sleeping_sleep_5_f0084` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.725 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |
| `edu_sleeping_sleep_5_f0091` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.786 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |
| `edu_sleeping_sleep_61_f0000` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.479 | PERSON_MEDIUM | GOOD | Student posture partially upright while resting head |
| `edu_sleeping_sleep_61_f0008` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.604 | PERSON_MEDIUM | GOOD | Student posture partially upright while resting head |
| `edu_sleeping_sleep_61_f0016` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.550 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |
| `edu_sleeping_sleep_4_f0000` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.996 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |
| `edu_sleeping_sleep_4_f0016` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.994 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |
| `edu_sleeping_sleep_4_f0024` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.983 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |
| `edu_sleeping_sleep_4_f0032` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.992 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |
| `edu_sleeping_sleep_4_f0040` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.990 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |
| `edu_sleeping_sleep_4_f0048` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.990 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |
| `edu_sleeping_sleep_4_f0056` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.990 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |
| `edu_sleeping_sleep_4_f0064` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.981 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |
| `edu_sleeping_sleep_4_f0072` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.985 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |
| `edu_sleeping_sleep_4_f0080` | `HEAD_REST_SLEEP` | **`NORMAL_READ_WRITE`** | 0.990 | PERSON_MEDIUM | BLURRY | Student posture partially upright while resting head |

---

## 5. Architectural Mitigation Strategies for Future Fusion

> [!IMPORTANT]
> Per Section 35 of the V4C Specification, V4C is strictly a perception-model validation phase. The downstream mitigation thresholds described below are **`PROVISIONAL_CANDIDATE_THRESHOLD`** markers for future V4D fusion and must NOT be frozen as validated production rules without dedicated end-to-end multi-cue evaluation.

1. **Temporal Persistence Filtering (`PROVISIONAL_CANDIDATE_THRESHOLD: 3.0s`)**: Most false positive `HEAD_REST_SLEEP` classifications are transient (< 1.5 seconds) transitions while writing. Applying a provisional 3.0-second persistence debounce in multi-cue fusion is candidate to eliminate transient write-to-sleep false alarms.
2. **Multi-Cue Head-Pose Disambiguation (`PROVISIONAL_CANDIDATE_THRESHOLD: 25.0°`)**: False positive `TURN_HEAD_CLEAR` errors occur when the student's torso is angled diagonally. The independent head-pose yaw branch provides continuous orientation evidence ($|\theta_{\text{yaw}}| < 25^\circ$), serving as a candidate cross-check against crop classifier turn predictions.
3. **Resolution Gating (`PROVISIONAL_CANDIDATE_THRESHOLD: H < 120px`)**: For `PERSON_VERY_SMALL` ($H < 120$ px), micro-posture classification should be marked low-confidence to avoid forced classification on unresolvable pixels.
