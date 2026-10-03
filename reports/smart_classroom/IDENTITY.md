# Smart Classroom Student Behavior Dataset — Identity Report

**Audit Date**: 2026-10-02  
**Dataset Name**: Smart-Classroom-Student-Behavior-Dataset  
**Upstream Repository**: https://github.com/master-weixiao/Smart-Classroom-Student-Behavior-Dataset  
**Storage Origin**: Baidu Netdisk (`https://pan.baidu.com/s/1uSdGbXAyZKbxD4fQdOrZzA`, extraction code: `1234`)  
**Data Source**: National Public Resource Service Platform (2019 Elementary School Chinese Language Grade 1-6 Minister-level Exemplary Classes)  
**License**: Strictly Academic Research Only. Commercial application prohibited.

---

## 1. Identity & Upstream Availability

| Dimension | Upstream Status | Local Status | Download Mechanism | Usable for YOLO Behavior Detector |
| :--- | :--- | :--- | :--- | :---: |
| Metadata & Documentation | Present on GitHub (`master-weixiao`) | Present (`reports/smart_classroom/`) | Git clone / HTTP fetch | Documentation only |
| Raw Video Clips | Hosted on Baidu Netdisk | Not downloaded | Manual SMS verification / Baidu client required | **NO (Pending Manual Download)** |
| Extracted BBox Labels | Embedded in upstream YOLOv5 training results | Inaccessible without Baidu download | Baidu Netdisk | **NO** |

## 2. Upstream Viewpoint Statistics

The dataset captures 12,836 video clips across 4 viewpoints:

| Viewpoint | Chinese Label | Clip Count | Duration Range (s) | Size | Resolution | FPS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Front View** | 正方视角 | 593 | 1–174 | 574 MB | 800×450 | 25 fps |
| **Diagonal Upper View** | 斜上方视角 | 4,781 | 1–115 | 5.30 GB | 800×450 | 25 fps |
| **Rear View** | 后方视角 | 4,544 | 1–59 | 3.48 GB | 800×450 | 25 fps |
| **Teacher View** | 教师视角 | 2,918 | 1–106 | 1.93 GB | 800×450 | 25 fps |

## 3. Physical Identity Assessment

- **Is this actually the official Smart-Classroom-Student-Behavior-Dataset?**: **YES** (Upstream identity confirmed via GitHub documentation and author citation).
- **Direct YOLO usability in current environment**: **NO**.
- **Reason**: The raw media files and box annotations reside behind Baidu Netdisk authentication. Automated pipelines cannot bypass Chinese SMS / Baidu client authentication without user interaction.
