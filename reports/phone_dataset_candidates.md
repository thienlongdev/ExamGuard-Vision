# Phone dataset candidates and decision

Audit date: 2026-10-02. Counts below are physical unless explicitly marked remote/unverified.

## Selected for local audit: Student Behaviour Detection v5

- Source: https://www.kaggle.com/datasets/youssefehab23/student-behaviour-detection-v5i-yolov8
- Upstream: https://universe.roboflow.com/mywork-lkwz4/student-behaviour-detection-neazg/dataset/5
- Physical root: datasets/raw/phone_candidate_student_behaviour/Student Behaviour Detection.v5i.yolov8
- 5,661 images and 5,661 YOLO label files; 216,098 annotations, 12 actual classes.
- Source class 0 Using_phone: 17,799 boxes; source class 5 phone: 6,942 boxes.
- Local YAML and README declare CC BY 4.0. Kaggle declares Unknown. License status: needs_manual_verification
  because underlying imagery overlaps visually/perceptually with SCB, whose author restricts commercial use.
- Camera domain: medium-distance and high-angle multi-student classrooms, not just webcam faces.
- Bbox style: Using_phone tightly bounds device/hand contact regions; phone tightly bounds devices,
  including devices resting on desks. Neither is consistently a student/person behavior box.
- Visual audit: all four pages for each phone-related class inspected: 48 Using_phone + 48 phone samples.
  Remaining class sheets are generated but NOT visually approved.
- Quality: 640x640 stretch, pre-augmented brightness/exposure/blur/noise, repeated frame variants;
  very small devices in distant rows; plausible active interaction in some samples but not reliable
  person-level supervision without reviewed reassociation/reannotation.
- Decision: reject Using_phone -> canonical use_phone AS-IS; ignore phone for behavior training.
  Retain original data for possible object/contact detection research only. Do not expand device boxes.
- No parsing issues under existing validator; this is not a blanket guarantee of annotation correctness.

## Existing local SCB subsets

- Author source: https://github.com/Whiffe/SCB-dataset
- Physical subsets: discuss (864 images), hand-raising/read/write (6,864), BowHead/TurnHead (2,410).
- Phone annotations: 0. Domain: classrooms; behavior boxes already reviewed in prior phase.
- Decision: reject as a phone source. Do not infer phone availability from the full benchmark paper.
- Licensing: author repository states academic/personal/non-commercial restrictions; production clearance needs review.

## Direct Roboflow source

- https://universe.roboflow.com/mywork-lkwz4/student-behaviour-detection-neazg
- Listing declares CC BY 4.0 and both phone and Using_phone, but listing alone is not label evidence.
- Direct version-6 YOLO export probe returned HTTP 401: API key required. No credentials invented.
- Version-5 mirror above supplies physical files; no second copy downloaded from Roboflow.
- Manual route if desired: sign into Roboflow, open version 5, choose Download Dataset -> YOLOv8,
  preserve README/license and export. This would not by itself fix person-box semantics.

## CStudentAct (not acquired)

- Official source: https://sigmlab.com/datasets/CStudentAct/
- Remote documentation reports 3,125 using-phone images and 11,998 boxes. LOCAL counts: unavailable.
- Classroom spatial/temporal activity annotations; actual boxes and semantics remain unverified.
- Restricted research terms: no commercial use or modification; access requires signed commitment
  emailed to the dataset administrator. No agreement signed or email sent.
- Decision: needs_review / access-and-license blocked; not included.

## Other remote leads (not accepted)

- wintonYF/SCB-Dataset: public API lists a 5.33 GB YOLO.zip, but contents not downloaded/verified;
  existing modular releases contain no phone class. Do not claim full-archive phone data.
- Kaggle classroom-student-behaviors: 28.42 GB listing with Other license; no local verification.
- Cropped classroom classification mirrors: not a replacement for person-level scene annotations.

## Phone strategy resolved: Option B

Keep COCO cell-phone detection -> person/track association -> duration/posture evidence.
Report phone-related suspicious evidence, not proof of active phone use or cheating.
The behavior class use_phone is omitted from the proposed V2 taxonomy due to insufficient verified
person-level annotations. Existing inference architecture and risk rules were NOT redesigned.
Fallback selection is a data decision, not a claim that a new standalone phone-presence rule was deployed.

