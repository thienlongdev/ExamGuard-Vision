# Global Conflict & Duplicate Safety Audit for Stage 1.5

**Date**: 2026-10-02  
**Target Dataset**: `datasets/processed_v3_5/`  
**Verification Engine**: Cryptographic SHA-256 Hashing & Disjoint Set Union Group Verification  

## 1. Split Isolation and Boundary Protection

| Audit Item | Threshold | Observed Value | Verdict |
| :--- | :---: | :---: | :---: |
| **Refinement Train vs Stage 1.5 Holdout Group Leakage** | 0 groups | `0` | **PASS** |
| **Refinement Train vs Stage 1.5 Holdout Hash Collision** | 0 images | `0` | **PASS** |
| **Refinement Train vs Original V3 Val Hash Collision** | 0 images | `0` | **PASS** |
| **Refinement Train vs Original V3 Test Hash Collision** | 0 images | `0` | **PASS** |

## 2. Quarantined Annotation Conflict Policy
- All 2,939 conflicting annotation instances cataloged in `reports/annotation_conflicts_v3.json` remain strictly quarantined.
- Ambiguous forward slump annotations where reading/writing conflicted with BowHead remain excluded from positive head_down supervision.
- Zero ambiguous or cross-conflict labels were admitted into `processed_v3_5`.

## 3. Summary of Disjoint Groups
- **Refinement Train Groups**: 136 independent recording groups
- **Frozen Stage 1.5 Holdout Groups**: 15 independent recording groups
- Zero sequence, recording, or duplicate overlap exists across the partition boundaries.
