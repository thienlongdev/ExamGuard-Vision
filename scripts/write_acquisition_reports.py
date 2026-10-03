"""Render physical audit evidence and explicit blockers; never imply a final rebuild."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
AUDIT = REPORTS / 'acquisition_v2'


def write(path, text):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text+'\n',encoding='utf-8')


def main():
    stats = json.loads((AUDIT/'source_statistics.json').read_text())
    inventory = json.loads((AUDIT/'source_inventory.json').read_text())
    duplicates = json.loads((AUDIT/'global_duplicates.json').read_text())
    preflight = json.loads((AUDIT/'preflight_v2.json').read_text())
    mapping = yaml.safe_load((ROOT/'configs/dataset_mapping.yaml').read_text())
    decisions = yaml.safe_load((ROOT/'configs/acquisition_v2_decisions.yaml').read_text())
    sources = [
        ('cctv_exam_monitor','cctvdataset/cctv-exam-monitor-dataset','CCTV Exam Monitor Dataset'),
        ('phone_candidate_student_behaviour','youssefehab23/student-behaviour-detection-v5i-yolov8','Student Behaviour Detection v5'),
    ]
    provenance = {}
    for local, slug, title in sources:
        response = requests.get('https://www.kaggle.com/api/v1/datasets/list',params={'search':slug.split('/')[1]},timeout=30)
        response.raise_for_status()
        listings = response.json()
        listing = next((x for x in listings if x.get('ref') == slug), {})
        write(AUDIT/f'{local}_listing.json',json.dumps(listing,indent=2))
        archive = ROOT/'datasets/raw'/local/'original_download.zip'
        provenance[local] = dict(dataset_name=title,source='Kaggle public download API',
            source_url=f'https://www.kaggle.com/datasets/{slug}',
            download_url=f'https://www.kaggle.com/api/v1/datasets/download/{slug}',
            version=listing.get('currentVersionNumber'),download_date='2026-10-02',
            timezone='Asia/Bangkok',archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            archive_bytes=archive.stat().st_size,license=listing.get('licenseName'),
            license_status='publisher_declared_not_legal_clearance',citation=title+'; '+slug)
    phone = provenance['phone_candidate_student_behaviour']
    phone.update(license='CC BY 4.0 (bundled README/data.yaml and upstream Roboflow); Kaggle listing Unknown',
        license_status='needs_manual_verification',upstream_version=5,
        upstream_url='https://universe.roboflow.com/mywork-lkwz4/student-behaviour-detection-neazg/dataset/5',
        citation='Mywork, Student Behaviour Detection Dataset, Roboflow Universe, March 2024, version 5',
        license_caveat='Upstream classroom imagery resembles SCB; SCB author currently restricts commercial use. Mirror license does not resolve underlying image rights.')
    write(AUDIT/'provenance.json',json.dumps(provenance,indent=2))
    cctv = stats['cctv_exam_monitor']
    write(REPORTS/'cctv_exam_monitor/statistics.json',json.dumps({**cctv,**provenance['cctv_exam_monitor'],
        'unique_sha256_images':8143,'visual_domain_samples_reviewed':36,'verified_annotation_classes':[],
        'bbox_audit':'not possible: zero labels'},indent=2))
    write(REPORTS/'cctv_exam_monitor/IDENTITY.md', '''# CCTV Exam Monitor — physical identity

Audit date: 2026-10-02 (Asia/Bangkok).

Downloaded through the public Kaggle API without credentials. No Kaggle CLI was used.
Source: https://www.kaggle.com/datasets/cctvdataset/cctv-exam-monitor-dataset
Version 1; publisher Jonathan Michael Campbell (cctvdataset).
Original ZIP and every extracted image preserved unchanged.

Physical root: `datasets/raw/cctv_exam_monitor/CCTV-Exam -Monitor -Dataset/`.
Archive has exactly 8,156 members, all JPEG images. All are 640x640.
Original split directories: train 7,734; valid 202; test 220.
No labels directory, annotation file, README, LICENSE, data.yaml, or dataset.yaml is present.

## Actual local YAML contents

UNAVAILABLE: the downloaded archive contains no YAML file. No YAML was synthesized.
Actual class names / IDs: unknown; zero physically defined annotation classes.

The pre-existing metadata.json is not part of the download and is not annotation evidence.
Its claimed classes and earlier download timestamp must not be trusted. It was preserved unchanged.
The current Kaggle API listing declares CC0: Public Domain. No bundled license or upstream citation was provided.
This is publisher-declared licensing, not verification of ownership of every frame.
Archive digest, exact source URL, version, date and listing snapshot are in statistics.json and ../acquisition_v2/provenance.json.
''')
    write(REPORTS/'cctv_exam_monitor/AUDIT.md', '''# CCTV visual and annotation audit

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
''')
    candidate_text = '''# Phone dataset candidates and decision

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
'''
    write(REPORTS/'phone_dataset_candidates.md',candidate_text)
    inv_lines = ['# Source Class Inventory Final — acquisition audit', '',
        'Physical inventory only; final training rebuild is blocked. Percentage is share of source annotations.',
        'CCTV has 8,156 images but zero annotation-defined classes and therefore no class rows.', '']
    for row in inventory:
        ds, name = row['source_dataset'], row['source_class_name']
        if ds.startswith('phone_'):
            rule = decisions['dataset_decisions'][ds]['source_classes'][name]
            row['contact_sheet'] = f"reports/acquisition_v2/class_samples/{ds}_{row['source_class_id']}_1.jpg (pages 1–4)"
        else:
            rule = mapping['datasets'][ds]['source_classes'][name]
            row['contact_sheet'] = rule.get('visual_sample_reference')
        row['mapping_status'] = rule['status']
        row['canonical_target'] = rule.get('target')
        inv_lines += [f"## {ds} / {name}", '', f"- source_dataset: {ds}",f"- source_subset: {row['source_subset']}",
            f"- source_class_id: {row['source_class_id']}",f"- source_class_name: {name}",
            f"- annotation_count: {row['annotation_count']}",f"- image_count: {row['image_count']}",
            f"- percentage: {row['percentage']:.4f}%",f"- contact_sheet: {row['contact_sheet']}",
            f"- mapping_status: {rule['status']}",f"- canonical_target: {rule.get('target')}", '']
    write(REPORTS/'SOURCE_CLASS_INVENTORY_FINAL.md','\n'.join(inv_lines))
    write(AUDIT/'source_inventory_final.json',json.dumps(inventory,indent=2))
    exact_types = Counter(' + '.join(sorted({r['source'] for r in c})) for c in duplicates['exact_clusters'])
    near_phone_scb = sum(any(r['source'].startswith('phone_') for r in c) and any(r['source'].startswith('scb') for r in c)
                         for c in duplicates['perceptual_candidate_clusters'])
    duplicate_summary = {k:v for k,v in duplicates.items() if not isinstance(v,list)}
    final = ['# Acquisition and Audit Final', '', 'Audit date: 2026-10-02 (Asia/Bangkok).', '',
        '**READY FOR FULL STAGE 1 TRAINING: NO**', '',
        'Acquisition/audit work completed as far as source files permit. Final dataset rebuild and sanity training were NOT completed.', '',
        '## 1–6. CCTV acquisition, identity, classes, statistics, visuals and mappings', '',
        'Downloaded the known public Kaggle archive (627,054,407 bytes), preserved original ZIP and extracted JPEGs.',
        '8,156 images; 8,143 unique SHA256 images; 0 label files; 0 physically defined classes. Every image is 640x640.',
        'Archive splits: train 7,734; valid 202; test 220. These are not approved target-domain holdouts.',
        '36 sample frames reviewed: high-angle multi-student desk/classroom and computer-lab scenes; useful visual domain.',
        'No source YAML exists to print. No class-specific annotation audit can pass without annotations.',
        'Correct Posture, Stand and movement names in historical metadata are not verified local classes. No CCTV mappings approved.',
        'See cctv_exam_monitor/IDENTITY.md, AUDIT.md, statistics.json and class_samples/.', '',
        '## 7–12. Phone source, physical labels, visual audit, licensing and decision', '',
        'Downloaded Student Behaviour Detection v5 (5,661 images, 5,661 label files, 216,098 boxes).',
        'Physical class 0 Using_phone has 17,799 boxes; class 5 phone has 6,942 boxes.',
        'Reviewed 48 samples per phone-related class. Boxes mark devices/hands, not consistent student behavior boxes;',
        'phone also labels devices resting on desks. Do not relabel these as person-level use_phone.',
        'Option B selected: existing COCO phone association plus temporal/posture evidence. No new inference architecture implemented.',
        'Local export/upstream declare CC BY 4.0, but mirror Unknown and possible SCB rights conflict require manual verification.',
        'See phone_dataset_candidates.md and acquisition_v2/provenance.json for candidates, access limits, citation/version/date/license.', '',
        '## 13–18. Source inventory, proposed taxonomy and excluded mappings', '',
        'SOURCE_CLASS_INVENTORY_FINAL.md lists only annotation classes physically found. Source semantics are retained.',
        'Supported behavior set: normal, head_down, turn_head, discuss. Proposed compact IDs after rebuild: 0,1,2,3 respectively.',
        'Desired fallback set after clean CCTV Stand labels arrive: normal, head_down, turn_head, discuss, stand.',
        'Neither V1-A nor V1-B can be finalized today: stand has no verified labels; use_phone is deliberately omitted; lean not enabled.',
        'Existing verified mapping decisions: read/write -> normal; BowHead -> head_down; TurnHead -> turn_head; discuss -> discuss.',
        'Existing hand-raising remains ignored. New phone candidate phone/book/hand-raising ignored;',
        'Using_phone rejected as-is for behavior supervision; other new classes remain needs_review and excluded.',
        'Subjective single-frame cheating remains prohibited. No such class was found in these downloaded annotations.',
        'configs/acquisition_v2_decisions.yaml records the new decisions. configs/dataset_mapping.yaml and processed files remain',
        'the legacy matched pair, NOT a final V2 mapping. Changing IDs before a complete rebuild would break their consistency.', '',
        'Raw SCB hand-raising/read/write validation reports 12 annotation issues; raw counts above include parsed problematic boxes.',
        'The new phone candidate reports zero issues under the existing validator. Strict physical checks pass on legacy processed boxes.',
        'Issue records remain in acquisition_v2/source_statistics.json; no original labels were changed.', '',
        '## 19–23. Global duplicates, fusion, grouping and leakage', '',
        'Fresh physical SHA256 scan covers all five source roots, including the excluded new data. No corrupt images found.',
        '```json',json.dumps(duplicate_summary,indent=2),'```',
        'Exact cluster source combinations:', '```json',json.dumps(exact_types,indent=2),'```',
        f'{near_phone_scb} same-perceptual-hash clusters span phone and SCB sources despite differing bytes.',
        'Perceptual candidates use a 64-bit DCT hash at distance zero, not an exhaustive near-duplicate search or automatic equivalence proof.',
        'No new annotations fused or discarded. Global cluster membership and all source paths preserved in acquisition_v2/global_duplicates.json.',
        'Future fusion must deduplicate compatible boxes with provenance lists and quarantine overlapping contradictory labels.',
        'Existing builder merges rounded class/box keys but does not explicitly quarantine class conflicts and keeps only one',
        'source entry for identical boxes; do not assume it satisfies the stronger V2 fusion requirements.',
        'CCTV needs source camera/video/session metadata. Roboflow hash suffixes and numeric filenames are not verified sessions.',
        'Keep exact clusters, reviewed near duplicates and all augmented versions in one recording-group component.',
        'Reserve entire CCTV recordings for validation/test; do not trust archive split folders as independent scenes.',
        'Legacy processed exact duplicate leakage: 0; shared group-ID leakage: 0. Actual unseen-sequence independence is NOT verified.', '',
        '## 24–27. Split/class/source distributions and target-domain coverage', '',
        'No final V2 splits exist. The following are measured LEGACY counts, not a new final dataset.',
        'Source counts overlap for fused frames and must not be summed as image totals.',
        '```json',json.dumps(preflight['split_statistics'],indent=2),'```',
        'CCTV and phone contributions are zero in every processed split. Normal dominates; head_down is smallest overall',
        'and especially sparse in validation. Missing stand/use_phone/lean must not retain empty output slots in a final model.', '',
        '## 28. Sampling strategy', '',
        'No final numerical sampling weights approved: labeled CCTV unique frames and recording counts are zero.',
        'Do not reuse 40/30/20/10. Weight candidates only after reviewed grouped splits exist, using duplicate-adjusted',
        'frames, class instances and camera/session counts; cap per-session repetition. Excluded sources receive no samples.',
        'The 8,156 unlabeled CCTV images cannot justify a supervised sampling weight or be treated as empty-label negatives.', '',
        '## 29. Stage 1 configuration', '',
        'Hardware defaults already match: yolo26m.pt, imgsz 768, batch 8, device 0, AMP true, epochs 80, patience 15,',
        'fliplr/flipud 0, close_mosaic 10. Conservative augmentations retained. Added a blocked-state warning and corrected batch comment.',
        '80 epochs is a future configuration value, NOT permission or evidence of a training run.', '',
        '## 30–34. Sanity, CUDA, losses, validation and tests', '',
        'Final one-epoch training: NOT RUN because labeled CCTV, final taxonomy, rebuilt splits and preflight prerequisites failed.',
        'Epoch time, training VRAM peak, box/cls/dfl losses, validation result, best.pt and last.pt: NOT AVAILABLE for V2.',
        'No checkpoint resumed, overwritten or reused as new evidence; no Stage 1 or Stage 2 training launched.',
        'GPU check passed: CUDA True, NVIDIA GeForce RTX 5070, 11.94 GB, cuda:0.',
        'Separate YOLO26m inference check passed: predictor cuda:0 and result tensors cuda:0; 338 MB reserved inference peak',
        '(this is NOT training calibration). Wrapper model parameters can remain on CPU while predictor backend uses CUDA.',
        'Initial bare pytest used system Anaconda and failed collection (cv2 missing). Repository .venv: 38 tests passed before changes.',
        'Final .venv test suite: 42 passed, 1 existing FastAPI/Starlette deprecation warning. JUnit: acquisition_v2/tests.xml.', '',
        '## 35. Blockers and exact next action', '',
        'Supply the matching labeled CCTV export plus class definitions and original camera/video/session grouping metadata.',
        'Source page: https://www.kaggle.com/datasets/cctvdataset/cctv-exam-monitor-dataset . Version 1 is image-only;',
        'another download of it is unnecessary. Request the annotation export from its publisher or provide reviewed custom annotations.',
        'The credential-free command already used was:',
        '```powershell',
        r'.\.venv\Scripts\python.exe scripts/acquire_public_archive.py https://www.kaggle.com/api/v1/datasets/download/cctvdataset/cctv-exam-monitor-dataset datasets/raw/cctv_exam_monitor',
        '```',
        'No credentials needed for that public archive; Kaggle CLI was not used. No account credentials stored in code.',
        'Additional work after labels arrive: CCTV class audit; final compact mapping; conflict-aware fusion with full provenance;',
        'real recording-group split reconstruction; target-domain held-out checks; full preflight; exactly one fresh sanity epoch.',
        'Production licensing also requires review: SCB author repository restricts commercial use; mirrored CC licenses may not resolve that.', '',
        '## 36. Readiness', '', '**READY FOR FULL STAGE 1 TRAINING: NO**', '',
        'Stopped without training. Existing application architecture, inference, tracking, temporal engine and API were not redesigned.']
    write(REPORTS/'ACQUISITION_AND_AUDIT_FINAL.md','\n'.join(final))
    print('Wrote CCTV identity/audit/statistics, phone candidates, final physical inventory and final acquisition report.')


if __name__ == '__main__':
    main()
