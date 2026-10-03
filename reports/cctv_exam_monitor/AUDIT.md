# CCTV visual and annotation audit

Status: ACQUIRED; VISUAL DOMAIN CONFIRMED; UNLABELED; EXCLUDED FROM TRAINING.

All three unlabeled_domain contact sheets were inspected (36 deterministic random images, seed 42).
They show fixed elevated room-corner views, several classrooms/computer labs, rows of desks,
multiple distant seated people, and some standing people. This is useful target-domain imagery.
Exam intent cannot be established for every frame from appearance alone.
Repeated rooms, timestamps, brightness/noise augmentation and adjacent-frame similarity are apparent.

## Source class audit

There are no annotation-defined source classes to sample. Class-specific sheets cannot be generated.
Each domain tile explicitly says bbox UNAVAILABLE, with filename and image size; no boxes were invented.
Correct Posture, LeftSideMove, RightSideMove, ForwardMove, BackwardMove and Stand remain
unverified historical claims, NOT source inventory entries. No mapping is approved for any of them.
In particular, visible standing people do not create a physical Stand annotation class.
Lean is not enabled. No directional movement classes are introduced.

## Grouping and holdout

Preserve archive split folders only as provenance, not as leakage-safe train/validation/test.
Names contain numeric frame-like stems and Roboflow hashes; these do not prove session boundaries.
Request original video/camera/session IDs and annotation files. Group entire identified recordings,
connect all exact and reviewed perceptual duplicate components, then reserve full CCTV groups
for validation and test. Never random-split adjacent frames or treat unlabeled images as negatives.

## Required action

Obtain the matching YOLO/COCO/VOC bounding-box export plus class definitions from the publisher,
or provide a reviewed human annotation export for these frames and camera/session provenance.
Re-downloading this same version will not add labels. The images are already present; do not duplicate them.
No publisher contact or new manual labeling campaign was initiated.

