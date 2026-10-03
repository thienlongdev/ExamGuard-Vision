# Temporal Leakage & Near-Duplicate Video Audit

**Document ID**: `reports/v4b/TEMPORAL_LEAKAGE_AUDIT.md`  
**Phase**: V4B — Clean Person-Crop / Head-Pose Dataset Construction  
**Author**: Data Engineering & Audit Pipeline  
**Date**: 2026-10-03  
**Status**: ZERO TEMPORAL LEAKAGE VERIFIED  

---

## 1. Executive Summary

Continuous video data (such as the 350 MP4 clips of EduAction) represents the most acute source of subtle data leakage in deep learning pipelines. When adjacent frames $F_t$ and $F_{t+\delta}$ (separated by only 40 ms at 25 FPS) are randomly distributed between train and validation sets, convolutional networks memorize background clothing, desk textures, and lighting patterns rather than learning generalizeable behavior semantics.

To permanently eradicate this failure mode, the V4B pipeline enforced two complementary safeguards:
1. **Clip-Level Split Isolation:** Whole video clips are assigned to partitions **prior to any frame extraction**.
2. **Adaptive Perceptual Stride Sampling:** Within each clip, consecutive near-duplicate frames are screened out, reducing raw frames by 87.9%.

---

## 2. Temporal Partition Architecture

Each EduAction category (50 clips each, $N_{\text{total}} = 350$ clips) was partitioned strictly at the clip boundary:

| Split Destination | Clips Allocated per Class | Total Clips in Split | Raw Video Frames | Extracted Sampled Crops | Selection Ratio | Rejection of Redundant Frames |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`train`** | 30 clips (60.0%) | 210 clips | 23,738 frames | 2,682 crops | 11.3% | 21,056 discarded |
| **`same_domain_val`** | 8 clips (16.0%) | 56 clips | 6,330 frames | 715 crops | 11.3% | 5,615 discarded |
| **`cross_source_holdout`**| 6 clips (12.0%) | 42 clips | 4,748 frames | 543 crops | 11.4% | 4,205 discarded |
| **`temporal_holdout`** | 6 clips (12.0%) | 42 clips | 4,747 frames | 535 crops | 11.3% | 4,212 discarded |
| **Total EduAction** | **50 clips (100%)** | **350 clips** | **39,563 frames** | **4,475 crops** | **11.3%** | **35,088 discarded** |

---

## 3. Temporal Leakage Verification Tests

### 3.1 Clip Disjointness Across Partitions
Let $\mathcal{C}_{\text{train}}, \mathcal{C}_{\text{val}}, \mathcal{C}_{\text{cross}}, \mathcal{C}_{\text{temporal}}$ be the sets of clip IDs in each partition:

$$\mathcal{C}_{\text{train}} \cap \mathcal{C}_{\text{val}} = \emptyset$$
$$\mathcal{C}_{\text{train}} \cap \mathcal{C}_{\text{cross}} = \emptyset$$
$$\mathcal{C}_{\text{train}} \cap \mathcal{C}_{\text{temporal}} = \emptyset$$
$$\mathcal{C}_{\text{val}} \cap \mathcal{C}_{\text{temporal}} = \emptyset$$

- **Physical Verification:** Automated script scanned all 4,475 EduAction manifest records. **Zero clip IDs appear in more than one split.**

### 3.2 Intra-Clip Temporal Stride Audit
Within each individual clip, adjacent sampled frames were analyzed:
- **Minimum Temporal Stride:** 5 frames ($0.20\text{ seconds}$).
- **Mean Temporal Stride:** $8.8\text{ frames}$ ($0.35\text{ seconds}$).
- **Maximum Temporal Stride:** 25 frames ($1.00\text{ seconds}$ idle cap).
- **Near-Duplicate Screening Result:** Zero extracted frames exhibit normalized pixel difference $\Delta < 0.003$ unless separated by $\ge 1.0\text{ second}$.

---

## 4. Verification Check
- Temporal leakage across all evaluation holdouts: **0% (Zero Leakage)**.
- Over-representation of static resting postures (`sleeping`): Reduced from 4,987 raw frames down to 604 non-redundant, diverse crops.
- The `temporal_holdout` split provides 42 completely unseen, continuous action sequences for final model evaluation in V4C.
