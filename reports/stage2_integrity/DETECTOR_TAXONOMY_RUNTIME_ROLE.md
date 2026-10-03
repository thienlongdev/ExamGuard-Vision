# Detector Taxonomy & Runtime Role Forensic Audit

## 1. Executive Summary
A forensic inspection of the physical checkpoint assets in `models/trained/` revealed that historical Stage 2 configurations operated under erroneous taxonomy assumptions—specifically assuming that `models/trained/stage1_best.pt` was a COCO detector with `0: person` and `67: cell phone`.

Direct inspection using the physical Ultralytics runtime disproved this hypothesis. Both `stage1_best.pt` and `stage1_5_best.pt` are custom 5-class student macro-behavior classification models, not general object detectors. Class 0 is `normal`, not `person`. Class 67 does not physically exist in either checkpoint.

To restore system integrity without violating the zero-retraining and zero-external-download mandates, the runtime roles were restructured using verified local physical assets.

---

## 2. Checkpoint Forensic Verification

| Metric / Attribute | `stage1_best.pt` | `stage1_5_best.pt` | `yolo26m.pt` (General Detector) |
| :--- | :--- | :--- | :--- |
| **Physical File Path** | `models/trained/stage1_best.pt` | `models/trained/stage1_5_best.pt` | `yolo26m.pt` |
| **SHA-256 Hash** | `6d713808f0bc670e8bf06e01b006fac73d8d6e5db7130584cec1658b003ce98a` | `68690cf82715dc6dad2eb35a08711531170e935eb7dbe1c30fafabae341e2c2c` | `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7` |
| **Model Task** | `detect` (custom behavior) | `detect` (custom behavior) | `detect` (general object) |
| **Architecture Provenance** | `yolo26m.yaml` (custom) | `yolo26m.yaml` (custom) | `yolo26m.yaml` (COCO pretrain) |
| **Parameter Count** | 20,495,295 | 20,495,295 | 21,894,768 |
| **Model Stride** | 32 | 32 | 32 |
| **Number of Classes** | 5 | 5 | 80 |
| **Physical Taxonomy (`model.names`)** | `{0: 'normal', 1: 'head_down', 2: 'turn_head', 3: 'discuss', 4: 'stand'}` | `{0: 'normal', 1: 'head_down', 2: 'turn_head', 3: 'discuss', 4: 'stand'}` | Full 80-class COCO (`0: 'person'`, `67: 'cell phone'`, ...) |
| **Certified Image Size** | 768 px | 768 px | 640 px |

---

## 3. Disproved Assumptions & Prohibited Logic Removed

1. **COCO Class 0 Assumption Prohibited**:
   - `stage1_best.pt` and `stage1_5_best.pt` class 0 is `normal` student behavior.
   - Any runtime code mapping class 0 to "person" from these models was creating false bounding box interpretations.
2. **COCO Class 67 Assumption Prohibited**:
   - Class 67 does not exist in `stage1_best.pt` or `stage1_5_best.pt` (both have only 5 classes: indices 0–4).
   - Calling `class_id == 67` on custom behavior models resulted in silent zero-detections or index errors.
3. **Stage 1 Runtime Redundancy**:
   - `stage1_best.pt` was an early behavior baseline.
   - `stage1_5_best.pt` represents the enhanced 5-class behavioral model.
   - Loading `stage1_best.pt` in the live inference loop was redundant and wasted GPU resources. It is retained as a protected baseline but excluded from live runtime orchestration.

---

## 4. Runtime Role Architecture & Allocation

```
+-----------------------------------------------------------------------------------+
|                              FRAME INGESTION                                      |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| GENERAL OBJECT DETECTOR (yolo26m.pt @ 640px)                                     |
| Role: Primary Scene Bounding Boxes & Object Detections                           |
| Classes resolved dynamically from model.names:                                    |
|   - Person: Class ID 0 ('person')                                                 |
|   - Phone:  Class ID 67 ('cell phone')                                            |
+-----------------------------------------------------------------------------------+
         |                                                   |
         | (person boxes)                                    | (phone detections)
         v                                                   v
+-----------------------------+                     +-------------------------------+
| ByteTrack Tracker           |                     | Phone Spatial Associator      |
| Filter: class_name == person|                     | Associates detected phones to |
| Tracks: Persistent ID       |                     | active person bounding boxes  |
+-----------------------------+                     +-------------------------------+
         |
         +---------------------------------------+
         |                                       |
         v (crop schedule: 15 Hz / 10 Hz)        v (cadence gated: 6.0 Hz)
+-----------------------------+     +-----------------------------------------------+
| Crop Scheduler & Batching   |     | MACRO BEHAVIOR DETECTOR (stage1_5_best.pt)    |
| - Posture: v4_posture_best  |     | Resolution: 768 px certified                  |
| - Head-Pose: v4_headpose_yaw|     | Taxonomy: {0: normal, 1: head_down,           |
+-----------------------------+     |            2: turn_head, 3: discuss, 4: stand}|
                                    | Output Cues: 'stand', 'discuss'               |
                                    +-----------------------------------------------+
```

---

## 5. Verification Flags

- `CHECKPOINT_INTEGRITY_OK = YES`
- `DETECTOR_TAXONOMY_VERIFIED = YES`
- `PERSON_DETECTION_RUNTIME_ROLE_VERIFIED = YES`
- `PHONE_OBJECT_DETECTION_AVAILABLE = YES`
- `MACRO_BEHAVIOR_ROLE_VERIFIED = YES`
- `PRODUCTION_READY = NO`
