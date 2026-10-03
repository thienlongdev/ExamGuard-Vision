# Acquisition and Audit Final

Audit date: 2026-10-02 (Asia/Bangkok).

**READY FOR FULL STAGE 1 TRAINING: NO**

Acquisition/audit work completed as far as source files permit. Final dataset rebuild and sanity training were NOT completed.

## 1–6. CCTV acquisition, identity, classes, statistics, visuals and mappings

Downloaded the known public Kaggle archive (627,054,407 bytes), preserved original ZIP and extracted JPEGs.
8,156 images; 8,143 unique SHA256 images; 0 label files; 0 physically defined classes. Every image is 640x640.
Archive splits: train 7,734; valid 202; test 220. These are not approved target-domain holdouts.
36 sample frames reviewed: high-angle multi-student desk/classroom and computer-lab scenes; useful visual domain.
No source YAML exists to print. No class-specific annotation audit can pass without annotations.
Correct Posture, Stand and movement names in historical metadata are not verified local classes. No CCTV mappings approved.
See cctv_exam_monitor/IDENTITY.md, AUDIT.md, statistics.json and class_samples/.

## 7–12. Phone source, physical labels, visual audit, licensing and decision

Downloaded Student Behaviour Detection v5 (5,661 images, 5,661 label files, 216,098 boxes).
Physical class 0 Using_phone has 17,799 boxes; class 5 phone has 6,942 boxes.
Reviewed 48 samples per phone-related class. Boxes mark devices/hands, not consistent student behavior boxes;
phone also labels devices resting on desks. Do not relabel these as person-level use_phone.
Option B selected: existing COCO phone association plus temporal/posture evidence. No new inference architecture implemented.
Local export/upstream declare CC BY 4.0, but mirror Unknown and possible SCB rights conflict require manual verification.
See phone_dataset_candidates.md and acquisition_v2/provenance.json for candidates, access limits, citation/version/date/license.

## 13–18. Source inventory, proposed taxonomy and excluded mappings

SOURCE_CLASS_INVENTORY_FINAL.md lists only annotation classes physically found. Source semantics are retained.
Supported behavior set: normal, head_down, turn_head, discuss. Proposed compact IDs after rebuild: 0,1,2,3 respectively.
Desired fallback set after clean CCTV Stand labels arrive: normal, head_down, turn_head, discuss, stand.
Neither V1-A nor V1-B can be finalized today: stand has no verified labels; use_phone is deliberately omitted; lean not enabled.
Existing verified mapping decisions: read/write -> normal; BowHead -> head_down; TurnHead -> turn_head; discuss -> discuss.
Existing hand-raising remains ignored. New phone candidate phone/book/hand-raising ignored;
Using_phone rejected as-is for behavior supervision; other new classes remain needs_review and excluded.
Subjective single-frame cheating remains prohibited. No such class was found in these downloaded annotations.
configs/acquisition_v2_decisions.yaml records the new decisions. configs/dataset_mapping.yaml and processed files remain
the legacy matched pair, NOT a final V2 mapping. Changing IDs before a complete rebuild would break their consistency.

Raw SCB hand-raising/read/write validation reports 12 annotation issues; raw counts above include parsed problematic boxes.
The new phone candidate reports zero issues under the existing validator. Strict physical checks pass on legacy processed boxes.
Issue records remain in acquisition_v2/source_statistics.json; no original labels were changed.

## 19–23. Global duplicates, fusion, grouping and leakage

Fresh physical SHA256 scan covers all five source roots, including the excluded new data. No corrupt images found.
```json
{
  "images": 23955,
  "corrupt_images": 0,
  "unique_sha256": 21920,
  "exact_duplicate_clusters": 1905,
  "duplicate_excess": 2035,
  "cross_source_exact_clusters": 1891,
  "same_phash_different_sha256_clusters": 3013,
  "perceptual_method": "32x32 grayscale DCT, 64-bit median hash; distance=0 candidate screen, not exhaustive"
}
```
Exact cluster source combinations:
```json
{
  "scb_discuss + scb_handrise_read_write + scbehavior_bow_turn": 130,
  "scb_discuss + scb_handrise_read_write": 424,
  "scb_discuss + scbehavior_bow_turn": 158,
  "scb_handrise_read_write + scbehavior_bow_turn": 1179,
  "scb_handrise_read_write": 1,
  "cctv_exam_monitor": 13
}
```
522 same-perceptual-hash clusters span phone and SCB sources despite differing bytes.
Perceptual candidates use a 64-bit DCT hash at distance zero, not an exhaustive near-duplicate search or automatic equivalence proof.
No new annotations fused or discarded. Global cluster membership and all source paths preserved in acquisition_v2/global_duplicates.json.
Future fusion must deduplicate compatible boxes with provenance lists and quarantine overlapping contradictory labels.
Existing builder merges rounded class/box keys but does not explicitly quarantine class conflicts and keeps only one
source entry for identical boxes; do not assume it satisfies the stronger V2 fusion requirements.
CCTV needs source camera/video/session metadata. Roboflow hash suffixes and numeric filenames are not verified sessions.
Keep exact clusters, reviewed near duplicates and all augmented versions in one recording-group component.
Reserve entire CCTV recordings for validation/test; do not trust archive split folders as independent scenes.
Legacy processed exact duplicate leakage: 0; shared group-ID leakage: 0. Actual unseen-sequence independence is NOT verified.

## 24–27. Split/class/source distributions and target-domain coverage

No final V2 splits exist. The following are measured LEGACY counts, not a new final dataset.
Source counts overlap for fused frames and must not be summed as image totals.
```json
{
  "train": {
    "images": 5665,
    "boxes": 38293,
    "class_boxes": {
      "discuss": 4130,
      "turn_head": 7159,
      "head_down": 3349,
      "normal": 23655
    },
    "class_images": {
      "discuss": 651,
      "turn_head": 1475,
      "head_down": 772,
      "normal": 2880
    },
    "sources": {
      "scb_discuss": 651,
      "scb_handrise_read_write": 4720,
      "scbehavior_bow_turn": 1723
    },
    "cctv_images": 0,
    "phone_images": 0
  },
  "val": {
    "images": 1218,
    "boxes": 6077,
    "class_boxes": {
      "normal": 4714,
      "discuss": 446,
      "turn_head": 775,
      "head_down": 142
    },
    "class_images": {
      "normal": 610,
      "discuss": 120,
      "turn_head": 248,
      "head_down": 89
    },
    "sources": {
      "scb_handrise_read_write": 1126,
      "scb_discuss": 120,
      "scbehavior_bow_turn": 270
    },
    "cctv_images": 0,
    "phone_images": 0
  },
  "test": {
    "images": 1233,
    "boxes": 7787,
    "class_boxes": {
      "discuss": 530,
      "normal": 4024,
      "turn_head": 2627,
      "head_down": 606
    },
    "class_images": {
      "discuss": 93,
      "normal": 580,
      "turn_head": 375,
      "head_down": 140
    },
    "sources": {
      "scb_discuss": 93,
      "scb_handrise_read_write": 1017,
      "scbehavior_bow_turn": 417
    },
    "cctv_images": 0,
    "phone_images": 0
  }
}
```
CCTV and phone contributions are zero in every processed split. Normal dominates; head_down is smallest overall
and especially sparse in validation. Missing stand/use_phone/lean must not retain empty output slots in a final model.

## 28. Sampling strategy

No final numerical sampling weights approved: labeled CCTV unique frames and recording counts are zero.
Do not reuse 40/30/20/10. Weight candidates only after reviewed grouped splits exist, using duplicate-adjusted
frames, class instances and camera/session counts; cap per-session repetition. Excluded sources receive no samples.
The 8,156 unlabeled CCTV images cannot justify a supervised sampling weight or be treated as empty-label negatives.

## 29. Stage 1 configuration

Hardware defaults already match: yolo26m.pt, imgsz 768, batch 8, device 0, AMP true, epochs 80, patience 15,
fliplr/flipud 0, close_mosaic 10. Conservative augmentations retained. Added a blocked-state warning and corrected batch comment.
80 epochs is a future configuration value, NOT permission or evidence of a training run.

## 30–34. Sanity, CUDA, losses, validation and tests

Final one-epoch training: NOT RUN because labeled CCTV, final taxonomy, rebuilt splits and preflight prerequisites failed.
Epoch time, training VRAM peak, box/cls/dfl losses, validation result, best.pt and last.pt: NOT AVAILABLE for V2.
No checkpoint resumed, overwritten or reused as new evidence; no Stage 1 or Stage 2 training launched.
GPU check passed: CUDA True, NVIDIA GeForce RTX 5070, 11.94 GB, cuda:0.
Separate YOLO26m inference check passed: predictor cuda:0 and result tensors cuda:0; 338 MB reserved inference peak
(this is NOT training calibration). Wrapper model parameters can remain on CPU while predictor backend uses CUDA.
Initial bare pytest used system Anaconda and failed collection (cv2 missing). Repository .venv: 38 tests passed before changes.
Final .venv test suite: 42 passed, 1 existing FastAPI/Starlette deprecation warning. JUnit: acquisition_v2/tests.xml.

## 35. Blockers and exact next action

Supply the matching labeled CCTV export plus class definitions and original camera/video/session grouping metadata.
Source page: https://www.kaggle.com/datasets/cctvdataset/cctv-exam-monitor-dataset . Version 1 is image-only;
another download of it is unnecessary. Request the annotation export from its publisher or provide reviewed custom annotations.
The credential-free command already used was:
```powershell
.\.venv\Scripts\python.exe scripts/acquire_public_archive.py https://www.kaggle.com/api/v1/datasets/download/cctvdataset/cctv-exam-monitor-dataset datasets/raw/cctv_exam_monitor
```
No credentials needed for that public archive; Kaggle CLI was not used. No account credentials stored in code.
Additional work after labels arrive: CCTV class audit; final compact mapping; conflict-aware fusion with full provenance;
real recording-group split reconstruction; target-domain held-out checks; full preflight; exactly one fresh sanity epoch.
Production licensing also requires review: SCB author repository restricts commercial use; mirrored CC licenses may not resolve that.

## 36. Readiness

**READY FOR FULL STAGE 1 TRAINING: NO**

Stopped without training. Existing application architecture, inference, tracking, temporal engine and API were not redesigned.
