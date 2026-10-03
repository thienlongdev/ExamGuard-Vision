# Data License Derivation & Attribution Matrix

**Document ID**: `reports/v4b/DATA_LICENSE_DERIVATION.md`  
**Phase**: V4B — Clean Person-Crop / Head-Pose Dataset Construction  
**Author**: Data Engineering & Legal Compliance  
**Date**: 2026-10-02  
**Status**: AUDITED & COMPLIANT  

---

## 1. Executive Summary

This report establishes the intellectual property, licensing, and derivative use compliance for all datasets incorporated into the V4B Person-Crop (`datasets/v4_crop/`) and Head-Pose (`datasets/v4_head_pose/`) pipelines.

### Zero-Redistribution Guarantee
In compliance with strict data governance rules:
1. No raw proprietary dataset archives are redistributed or committed into public version control.
2. All generated crops and manifests carry an immutable `license_status` field.
3. All derived crops originating from academic or restricted sources are tagged as `ACADEMIC_ONLY` or `ACADEMIC_NON_COMMERCIAL`.

---

## 2. Source-to-Derivative License Matrix

| Source Dataset | Source Identity / Repository | Original License / Stated Terms | Derivative Crop Classification | Derivative Target Directory | Permitted Usage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SCBehavior High-Res** | `datasets/raw_v4/scbehavior_highres/repo` (GitCode / GitHub 20191864136) | Academic Research / Educational Evaluation | `ACADEMIC_ONLY` | `datasets/v4_crop/raw_crops/`<br>`datasets/v4_crop/normalized_224/`<br>`datasets/v4_crop/normalized_320/` | Strictly non-commercial research, academic benchmarking, and thesis evaluation. |
| **EduAction** | `datasets/raw_v4/other_candidates/eduaction` (HuggingFace `hhhhha-ops/EduAction`) | Open Academic Dataset (CC-BY-NC / Academic) | `ACADEMIC_ONLY` | `datasets/v4_crop/raw_crops/`<br>`datasets/v4_crop/normalized_224/`<br>`datasets/v4_crop/normalized_320/` | Educational activity recognition research. |
| **AFLW2000-3D** | `datasets/raw_v4/head_pose_aflw2000` (Cleardusk 3DDFA / CVPR 2016) | Academic Non-Commercial Research License | `ACADEMIC_NON_COMMERCIAL` | `datasets/v4_head_pose/images/aflw2000_3d/`<br>`datasets/v4_head_pose/landmarks/` | Research benchmark for 3D facial analysis and head-pose estimation. |
| **AFLW-GT** | `datasets/raw_v4/head_pose_aflw2000` (AFLW / Köstinger et al., ICCV Workshops 2011) | Academic Non-Commercial Research License | `ACADEMIC_NON_COMMERCIAL` | `datasets/v4_head_pose/images/aflw_gt/` | Research benchmark for head-pose regression. |

---

## 3. Academic Attribution Citations

All academic publications and project documentations utilizing these datasets must preserve the following formal citations:

### 3.1 SCBehavior High-Resolution Dataset
```bibtex
@misc{scbehavior2024,
  author = {SCBehavior Project Authors},
  title = {SCBehavior: High-Resolution Classroom Student Behavior Dataset},
  year = {2024},
  howpublished = {GitCode / GitHub repository 20191864136}
}
```

### 3.2 EduAction College Student Behavior Dataset
```bibtex
@misc{eduaction2024,
  author = {hhhhha-ops},
  title = {EduAction: Video Dataset for Classroom Action Recognition},
  year = {2024},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/datasets/hhhhha-ops/EduAction}}
}
```

### 3.3 AFLW2000-3D & 3DDFA
```bibtex
@inproceedings{zhu2016face,
  title={Face alignment across large poses: A 3D solution},
  author={Zhu, Xiangyu and Lei, Zhen and Liu, Xiaoming and Shi, Hailin and Li, Stan Z},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={146--155},
  year={2016}
}
```

### 3.4 AFLW
```bibtex
@inproceedings{kostinger2011annotated,
  title={Annotated facial landmarks in the wild: A large-scale, versatile database for facial analysis},
  author={K{\"o}stinger, Martin and Wohlhart, Paul and Roth, Peter M and Bischof, Horst},
  booktitle={2011 IEEE international conference on computer vision workshops (ICCV workshops)},
  pages={2144--2151},
  year={2011},
  organization={IEEE}
}
```

---

## 4. Verification Check
Every line in `datasets/v4_crop/manifest.jsonl` and `datasets/v4_head_pose/manifest.jsonl` includes:
- `license_status: "ACADEMIC_ONLY"` or `"ACADEMIC_NON_COMMERCIAL"`
- `source_dataset`
- `source_identity`
- `source_file`
Ensuring full machine-readable provenance and license adherence across all downstream processing.
