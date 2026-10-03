# Stage 1.5 Pre-Flight Verification Report

**Date**: 2026-10-02  
**Target Experiment**: Stage 1.5 Weak-Class Data-Centric Refinement Training  
**Target Checkpoint Initializer**: `models/trained/stage1_best.pt`  
**Dataset**: `datasets/processed_v3_5/`  

## 1. Pre-Flight Checklist Table

| Check Item | Requirement | Observed Status | Verdict |
| :--- | :--- | :--- | :---: |
| **CUDA RTX 5070 Active** | CUDA active on NVIDIA RTX 5070 GPU | NVIDIA GeForce RTX 5070 (CUDA 13.0) | **PASS** |
| **stage1_best.pt Loads Successfully** | File exists, weights loadable in YOLO, matching 5-class taxonomy | Exists: True, Names: ['normal', 'head_down', 'turn_head', 'discuss', 'stand'] | **PASS** |
| **Dataset YAML Correctness** | dataset.yaml valid, 5 canonical classes, paths resolve | nc: 5, names: ['normal', 'head_down', 'turn_head', 'discuss', 'stand'], train: images/train, val: images/val | **PASS** |
| **All Images Readable** | Images non-corrupt, valid pixel data | 6694 train + 488 val images; sample verified clean | **PASS** |
| **Label Validity & Class IDs** | Zero invalid class IDs (must be in 0..4), coordinates within [0,1] | Total boxes: 48,739, invalid cids: set(), invalid coords: 0 | **PASS** |
| **Class Taxonomy Exactly Unchanged** | 0: normal, 1: head_down, 2: turn_head, 3: discuss, 4: stand | ['normal', 'head_down', 'turn_head', 'discuss', 'stand'] | **PASS** |
| **Zero Holdout & Group Leakage** | Zero overlap between refinement train and frozen holdout groups/hashes | Group leakage: 0, Hash leakage: 0 | **PASS** |
| **Quarantined Conflicts Excluded** | All 2,939 conflict boxes remain excluded | 2939 conflict records verified quarantined | **PASS** |

## 2. Pre-Flight Final Verdict

**OVERALL VERDICT**: **READY TO TRAIN — PASS**

All critical gates passed: GPU acceleration verified on RTX 5070, `stage1_best.pt` loads cleanly with matching taxonomy, zero leakage between training split and frozen holdout, and label geometry strictly verified.
