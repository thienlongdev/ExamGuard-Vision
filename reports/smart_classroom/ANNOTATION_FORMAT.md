# Smart Classroom Student Behavior Dataset — Annotation Format & Semantics

**Audit Date**: 2026-10-02  

## 1. Upstream Annotation Taxonomy

The dataset annotates 10 behavioral categories in elementary classroom video recordings:

| Dimension | Behavior Category | Chinese Name | Code | Upstream Annotated Count | YOLO Compatibility |
| :--- | :--- | :--- | :---: | :---: | :---: |
| Individual | Bow head writing | 低头写字 | `dx` | 72,462 | Potential candidate for `normal` |
| Individual | Bow head reading | 低头看书 | `dk` | 58,932 | Potential candidate for `normal` |
| Individual | Head up listening | 抬头听课 | `tt` | 117,528 | Potential candidate for `normal` |
| Individual | Turn head | 转头 | `zt` | 5,339 | Potential candidate for `turn_head` |
| Individual | Hand raising | 举手 | `js` | 4,183 | Ignored (non-exam specific) |
| Individual | Standing | 站立 | `zl` | 4,101 | Potential candidate for `stand` |
| Individual | Walking | 走动 | `zd` | 0 | 0 instances |
| Individual | Abnormal posture | 异常状态 | `yc` | 0 | 0 instances |
| Group | Group discussion | 小组讨论 | `xt` | 4,663 | Potential candidate for `discuss` |
| Group | Teacher guidance | 教师指导 | `jz` | 680 | Ignored |

## 2. Format & Usability Analysis

1. **Source Format**: Video clip segments (800×450, 25 fps) with frame-level bounding boxes trained using YOLOv5s (300 epochs).
2. **Current Availability**: The raw media files cannot be fetched automatically due to Baidu Netdisk access barrier.
3. **Usage Policy**:
   - Per Section 11 of project directives: *"If conversion would lose semantics or if inaccessible: keep it out of V1 training."*
   - Because physical label files are not locally accessible, Smart-Classroom data **must NOT be fabricated or assumed**.
   - It is retained as a documented external data source for future manual acquisition if upper/rear camera angle diversity is needed.
