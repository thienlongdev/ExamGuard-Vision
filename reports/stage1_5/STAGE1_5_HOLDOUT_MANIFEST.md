# Stage 1.5 Comparison Holdout Manifest

**Date**: 2026-10-02  
**Dataset Location**: `datasets\stage1_5_holdout/`  
**Source Origin**: Strictly derived from Stage 1 `train` split only (zero leakage with original test/val)  
**Total Images**: 488  
**Total Groups**: 15 groups (9.93% of 151 Stage 1 train groups)  
**Total Bounding Boxes**: 3,038  

## 1. Class Distribution in Frozen Holdout

| Class ID | Canonical Class | Box Count | % of Holdout Boxes |
| :---: | :--- | :---: | :---: |
| 0 | **normal** | 975 | 32.09% |
| 1 | **head_down** | 235 | 7.74% |
| 2 | **turn_head** | 1,035 | 34.07% |
| 3 | **discuss** | 221 | 7.27% |
| 4 | **stand** | 572 | 18.83% |

## 2. Group Breakdown and Viewpoint Provenance

| Group ID | Images | Viewpoint | Provenance Subsets | HD | TH | Normal | Discuss | Stand | Total Boxes |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `grp_0002` | 68 | `cctv_oblique_high_angle` | scb_dataset5_full/SCB5-Discuss-2024-9-17<br>scb_dataset5_full/SCB5-Handrise-Read-write-2024-9-17<br>scb_dataset5_full/SCB5-Stand-2024-9-17<br>scbehavior/SCB_BowTurnHead_20250509 | 9 | 136 | 33 | 2 | 77 | 257 |
| `grp_0003` | 13 | `cctv_oblique_high_angle` | scb_dataset5_full/SCB5-Discuss-2024-9-17<br>scb_dataset5_full/SCB5-Handrise-Read-write-2024-9-17<br>scb_dataset5_full/SCB5-Stand-2024-9-17<br>scbehavior/SCB_BowTurnHead_20250509 | 6 | 17 | 21 | 20 | 11 | 75 |
| `grp_0011` | 34 | `cctv_oblique_high_angle` | scb_dataset5_full/SCB5-Handrise-Read-write-2024-9-17<br>scb_dataset5_full/SCB5-Stand-2024-9-17<br>scbehavior/SCB_BowTurnHead_20250509 | 11 | 23 | 222 | 0 | 18 | 274 |
| `grp_0014` | 32 | `cctv_oblique_high_angle` | scb_dataset5_full/SCB5-Discuss-2024-9-17<br>scb_dataset5_full/SCB5-Handrise-Read-write-2024-9-17<br>scb_dataset5_full/SCB5-Stand-2024-9-17<br>scbehavior/SCB_BowTurnHead_20250509 | 13 | 45 | 136 | 18 | 19 | 231 |
| `grp_0600` | 5 | `frontal_classroom` | scb_dataset5_full/SCB5-Handrise-Read-write-2024-9-17 | 0 | 0 | 9 | 0 | 0 | 9 |
| `grp_0601` | 9 | `frontal_classroom` | scb_dataset5_full/SCB5-Handrise-Read-write-2024-9-17 | 0 | 0 | 27 | 0 | 0 | 27 |
| `grp_0800` | 6 | `frontal_classroom` | scb_dataset5_full/SCB5-Handrise-Read-write-2024-9-17 | 0 | 0 | 34 | 0 | 0 | 34 |
| `grp_1000` | 13 | `frontal_classroom` | scb_dataset5_full/SCB5-Handrise-Read-write-2024-9-17 | 0 | 0 | 95 | 0 | 0 | 95 |
| `grp_1121` | 3 | `frontal_classroom` | scb_dataset5_full/SCB5-Handrise-Read-write-2024-9-17 | 0 | 0 | 11 | 0 | 0 | 11 |
| `grp_131` | 63 | `cctv_oblique_high_angle` | scb_dataset5_full/SCB5-Handrise-Read-write-2024-9-17<br>scb_dataset5_full/SCB5-Stand-2024-9-17<br>scbehavior/SCB_BowTurnHead_20250509 | 16 | 15 | 133 | 0 | 82 | 246 |
| `grp_135` | 48 | `cctv_oblique_high_angle` | scb_dataset5_full/SCB5-Stand-2024-9-17<br>scbehavior/SCB_BowTurnHead_20250509 | 40 | 3 | 0 | 0 | 68 | 111 |
| `grp_137` | 56 | `cctv_oblique_high_angle` | scb_dataset5_full/SCB5-Discuss-2024-9-17<br>scb_dataset5_full/SCB5-Stand-2024-9-17<br>scbehavior/SCB_BowTurnHead_20250509 | 84 | 261 | 0 | 156 | 63 | 564 |
| `grp_2` | 54 | `cctv_oblique_high_angle` | scb_dataset5_full/SCB5-Discuss-2024-9-17<br>scb_dataset5_full/SCB5-Handrise-Read-write-2024-9-17<br>scb_dataset5_full/SCB5-Stand-2024-9-17<br>scbehavior/SCB_BowTurnHead_20250509 | 6 | 112 | 73 | 17 | 105 | 313 |
| `grp_22` | 58 | `cctv_oblique_high_angle` | scb_dataset5_full/SCB5-Discuss-2024-9-17<br>scb_dataset5_full/SCB5-Handrise-Read-write-2024-9-17<br>scb_dataset5_full/SCB5-Stand-2024-9-17<br>scbehavior/SCB_BowTurnHead_20250509 | 10 | 81 | 99 | 8 | 78 | 276 |
| `grp_9` | 26 | `cctv_oblique_high_angle` | scb_dataset5_full/SCB5-Handrise-Read-write-2024-9-17<br>scb_dataset5_full/SCB5-Stand-2024-9-17<br>scbehavior/SCB_BowTurnHead_20250509 | 40 | 342 | 82 | 0 | 51 | 515 |

## 3. Viewpoint Summary

| Viewpoint Category | Groups | Images | Boxes |
| :--- | :---: | :---: | :---: |
| `cctv_oblique_high_angle` | 10 | 452 | 2862 |
| `frontal_classroom` | 5 | 36 | 176 |
