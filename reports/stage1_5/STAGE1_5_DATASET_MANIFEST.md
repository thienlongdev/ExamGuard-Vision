# Stage 1.5 Refinement Dataset Manifest (`processed_v3_5`)

**Date**: 2026-10-02  
**Dataset Root**: `datasets/processed_v3_5/`  
**Dataset YAML**: `datasets/processed_v3_5/dataset.yaml`  
**Taxonomy (5 Classes)**: `['normal', 'head_down', 'turn_head', 'discuss', 'stand']`  

## 1. Split and Image Summary

| Split | Images | Independent Groups | Total Bounding Boxes | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 6,694 | 136 | 45,701 | Curated Stage 1 train domain with 2x weak-class emphasis and hard-example inclusion |
| **Val** | 488 | 15 | 3,038 | Frozen Stage 1.5 Comparison Holdout (strictly group-disjoint) |
| **Total** | 7,182 | 151 | 48,739 | |

## 2. Per-Class Bounding Box Distribution (Train Split)

| Class ID | Canonical Class | Box Count | % of Train Boxes | Baseline Stage 1 Train Boxes | Ratio / Sampling Emphasis |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | **`normal`** | 23,983 | 52.5% | 20,871 | 1.15x |
| 1 | **`head_down`** | 2,790 | 6.1% | 1,630 | 1.71x |
| 2 | **`turn_head`** | 6,750 | 14.8% | 5,872 | 1.15x |
| 3 | **`discuss`** | 3,732 | 8.2% | 3,696 | 1.01x |
| 4 | **`stand`** | 8,446 | 18.5% | 8,337 | 1.01x |

## 3. Viewpoint Distribution (Train Split)

| Viewpoint Category | Images | % of Images |
| :--- | :---: | :---: |
| **`cctv_oblique_high_angle`** | 5,506 | 82.3% |
| **`frontal_classroom`** | 1,188 | 17.7% |

## 4. Curated Sampling and Hard Example Integration

- **Base Training Domain**: 6,178 images from 136 groups.
- **Head-Down Frames Boosted (2x)**: 436 images with verified `head_down` postures.
- **Mined Hard Examples Integrated**: 181 images containing hard false negatives or hard normal negatives.
- **Artificial Class Balancing Avoided**: Natural class distribution maintained; no 10x blind repetition or artificial 1:1:1 flattening.
- **Quarantined Annotations Excluded**: 2,939 conflict boxes remain excluded.
