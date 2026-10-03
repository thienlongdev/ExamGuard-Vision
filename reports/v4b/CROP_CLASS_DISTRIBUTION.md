# Crop Class Distribution & Balance Safety Report

**Document ID**: `reports/v4b/CROP_CLASS_DISTRIBUTION.md`  
**Phase**: V4B — Clean Person-Crop / Head-Pose Dataset Construction  
**Author**: Data Engineering & Audit Pipeline  
**Date**: 2026-10-03  
**Status**: PHYSICALLY VERIFIED  

---

## 1. Executive Summary

A core principle established for V4B is that **data purity and split independence strictly supersede artificial numerical class balance**.  
In previous iterations or naive machine learning workflows, practitioner teams frequently force equal class counts through oversampling, naive duplication, or artificial synthetic augmentation. This introduces memorization shortcuts and destroys the natural occurrence priors of the classroom environment.

V4B adheres to the mandate:
> *Do not force perfect class balance. Do not duplicate files to make counts equal. The objective is clean, diverse, group-independent, cross-camera capable data.*

---

## 2. Global Class Distribution ($N = 20,490$)

The physical breakdown across all ontology classes in `datasets/v4_crop/manifest.jsonl` is:

| Class Group | Ontology Label | Physical Crop Count | % of Dataset | Semantic Role in V4 Architecture |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Posture Positives** | **`NORMAL_UPRIGHT`** | **9,583** | **46.77%** | Sitting upright, attentive classroom focus |
| | **`NORMAL_READ_WRITE`** | **3,825** | **18.67%** | Active paper reading & writing (**CRITICAL Hard Negative**) |
| | **`TURN_HEAD_CLEAR`** | **2,002** | **9.77%** | Clear lateral head turn ($35^\circ \text{--} 75^\circ$ yaw) |
| | **`HEAD_REST_SLEEP`** | **604** | **2.95%** | Head resting on desk / deep slump |
| **Primary Supervised Subtotal** | — | **16,014** | **78.16%** | Clean supervised training/validation corpus |
| **Quarantined Context & Ambiguous** | `AMBIGUOUS_LOOKUP` | 1,284 | 6.27% | Small/distant rear students with ambiguous gaze |
| | `COMPUTER_CONTEXT` | 656 | 3.20% | Screen/keyboard interaction context |
| | `TALKING_CONTEXT` | 651 | 3.18% | Conversational facial articulation context |
| | `PHONE_INTERACTION_CONTEXT` | 636 | 3.10% | Secondary phone manipulation context |
| | `DRINKING_CONTEXT` | 633 | 3.09% | Incidental drinking behavior context |
| | `DISCUSS_PAIR` | 484 | 2.36% | Macro full-frame paired interaction |
| | `STAND_MACRO` | 132 | 0.64% | Macro standing mobility state |
| **Quarantined Subtotal** | — | **4,476** | **21.84%** | Excluded from primary 5-class posture supervision |
| **Total Global Manifest** | — | **20,490** | **100.00%** | Full audited corpus |

---

## 3. Class Counts across Partitions

| Ontology Label | `train` | `same_domain_val` | `high_angle_holdout` | `cross_source_holdout` | `temporal_holdout` | Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`NORMAL_UPRIGHT`** | 6,719 | 1,475 | 1,230 | 79 | 80 | **9,583** |
| **`NORMAL_READ_WRITE`** | 2,925 | 564 | 176 | 83 | 77 | **3,825** |
| **`TURN_HEAD_CLEAR`** | 1,546 | 264 | 192 | 0 | 0 | **2,002** |
| **`HEAD_REST_SLEEP`** | 362 | 91 | 0 | 76 | 75 | **604** |
| `AMBIGUOUS_LOOKUP` (Q) | 1,172 | 108 | 4 | 0 | 0 | **1,284** |
| `TALKING_CONTEXT` (Q) | 390 | 105 | 0 | 78 | 78 | **651** |
| `PHONE_INTERACTION` (Q) | 379 | 103 | 0 | 80 | 74 | **636** |
| `COMPUTER_CONTEXT` (Q) | 400 | 104 | 0 | 77 | 75 | **656** |
| `DRINKING_CONTEXT` (Q) | 382 | 105 | 0 | 70 | 76 | **633** |
| `DISCUSS_PAIR` (Q) | 448 | 34 | 2 | 0 | 0 | **484** |
| `STAND_MACRO` (Q) | 98 | 20 | 14 | 0 | 0 | **132** |
| **Total Crops** | **14,821** | **2,973** | **1,618** | **543** | **535** | **20,490** |

---

## 4. Canonical Collapse Plan (For Future Fusion Reference Only)

V4B preserves all fine-grained observable semantic states without premature loss of information. When model training or multi-cue fusion is initiated in downstream phases, the canonical collapse mapping is defined as:

$$\begin{aligned}
\text{NORMAL\_UPRIGHT} + \text{NORMAL\_READ\_WRITE} &\longrightarrow \mathbf{NORMAL} \quad (13,408\text{ crops}) \\
\text{HEAD\_DOWN\_DEEP} + \text{HEAD\_REST\_SLEEP} &\longrightarrow \mathbf{HEAD\_DOWN} \quad (604\text{ crops}) \\
\text{TURN\_HEAD\_CLEAR} &\longrightarrow \mathbf{TURN\_HEAD} \quad (2,002\text{ crops})
\end{aligned}$$

- **Zero Overwriting in V4B:** This mapping is recorded in metadata field `future_canonical_label` but is **NOT** executed as a destructive collapse in V4B files.
- **Handling Hard Negatives:** By keeping `NORMAL_READ_WRITE` explicitly distinct from `NORMAL_UPRIGHT`, future models can compute class-specific loss weights (e.g. focal loss or margin loss) to prevent the detector from ever penalizing normal writing as head-down cheating.
