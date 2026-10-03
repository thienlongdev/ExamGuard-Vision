# Data License and Usage Governance Matrix
**Date:** 2026-10-02  
**Report ID:** V4A-LIC-MATRIX-01  
**Governance Scope:** Academic Compliance, Redistribution Rights, and Commercial Boundaries  

---

## 1. Compliance Framework

Under the V4A directive, every candidate source is categorized into three strict compliance tiers:
- **`SAFE_FOR_ACADEMIC_PROJECT`**: Explicitly permits non-commercial academic research, reproducible locally, unencumbered by restrictive NDA/EULA.
- **`CONDITIONAL`**: Permitted under strict conditions (e.g. requires individual manual agreement, institutional signing, or strictly prohibits redistribution).
- **`DO_NOT_USE`**: Prohibited due to terms of service violation, copyright infringement, commercial prohibition on open release, or withheld source data.

---

## 2. Complete License Matrix

| Dataset | Official Source / Publisher | License Stated | Academic Use | Commercial Use | Redistribution Allowed? | Attribution Required? | Manual Agreement Required? | Recommended Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SCBehavior-Dataset** | GitHub (`20191864136`) | Public Open Academic Access | **YES** | Prohibited / Unclear | Allowed (Public GitHub LFS) | Yes (Cite Repository) | **NO** (Publicly cloned) | **`SAFE_FOR_ACADEMIC_PROJECT`** |
| **EduAction** | GitHub (`hhhhha-ops`) | Public Academic Research | **YES** | Prohibited | Allowed (Public GitHub repo) | Yes (Cite Paper) | **NO** (Publicly cloned) | **`SAFE_FOR_ACADEMIC_PROJECT`** |
| **AFLW2000-3D** | CASIA / CVPR 2016 (`Xiangyu Zhu et al.`) | Non-Commercial Academic Research | **YES** | **PROHIBITED** | Fair use for research benchmarks | Yes (Cite CVPR 2016) | **NO** | **`SAFE_FOR_ACADEMIC_PROJECT`** |
| **AFLW_GT** | Graz University / CASIA | Non-Commercial Academic Research | **YES** | **PROHIBITED** | Research benchmarks only | Yes (Cite AFLW) | **NO** | **`SAFE_FOR_ACADEMIC_PROJECT`** |
| **Smart Classroom** | National Public Resource Service / `master-weixiao` | Academic Research Only (Strictly non-commercial) | **YES** | **STRICTLY PROHIBITED** | Prohibited without permission | Yes | **YES** (Baidu Netdisk authentication) | **`CONDITIONAL`** |
| **CStudentAct / StudentAct** | Hanoi Univ of Science & Technology (HUST) | HUST Copyright Academic Research | **YES** | **STRICTLY PROHIBITED** | **STRICTLY FORBIDDEN** without written approval | Yes (Mandatory HUST citation) | **YES** (Must email signed commitment to `lan.lethi1@hust.edu.vn`) | **`CONDITIONAL`** |
| **Hashemite Cheating** | Hashemite Univ / Kaggle | Academic Research | **YES** | Prohibited | Kaggle platform terms | Yes | **YES** (Kaggle account login) | **`CONDITIONAL`** |
| **AIRC-SMARTCLASS** | Univ of Transport Technology, Hanoi / Mendeley | Creative Commons **CC BY 4.0** | **YES** | **YES** (Under CC BY 4.0) | **YES** (With attribution) | Yes (Mandatory DOI citation) | **NO** | **`SAFE_FOR_ACADEMIC_PROJECT`** |
| **NCBD** | NanNing Normal University | Research Privacy Agreement | **YES** | **STRICTLY PROHIBITED** | Forbidden | Yes | **YES** (Signed privacy agreement for feature vectors) | **`DO_NOT_USE`** (Raw video withheld) |
| **ClassBehavior** | MDPI Sensors 2023 / `weniu` | Withheld | Unclear | Unclear | Unclear | Yes | **YES** (Contact authors) | **`DO_NOT_USE`** (Empty repository) |
| **Class Pose** | Sensors 2022 (`Wang et al.`) | Privacy-Restricted Surveillance | Unclear | **PROHIBITED** | Forbidden | Yes | **YES** (Direct author contact) | **`DO_NOT_USE`** (No public repo) |
| **BIWI Kinect** | ETH Zurich CVL | Non-commercial university research/education | **YES** | **PROHIBITED** | Forbidden | Yes | **NO** (Server currently 403) | **`DO_NOT_USE`** (Access blocked) |

---

## 3. Strict Compliance Guidelines for V4 Execution

1. **Academic Research Sandbox:**
   - This project operates strictly as a non-commercial academic research initiative into automated classroom behavior understanding.
   - None of the acquired datasets will be distributed for commercial purposes.
2. **Zero Redistribution Policy:**
   - Raw video files and images from restricted repositories (e.g. `EduAction`, `CStudentAct`) must remain in local workspace directories (`datasets/raw_v4/`) and must NEVER be pushed to public git remotes.
   - Only trained checkpoint weights and aggregated evaluation reports may be published.
3. **Attribution Commitment:**
   - All benchmark reports and final research documentation will include full academic citations for all utilized datasets (SCBehavior, EduAction, AFLW2000-3D, and SCB).
