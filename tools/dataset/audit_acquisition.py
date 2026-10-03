"""Physical-only acquisition audit. Never trains or changes raw annotations."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.validate_dataset import validate_annotation_file

SOURCES = {
    "scb_discuss": "scb_dataset3/SCB5-Discuss-2024-9-17",
    "scb_handrise_read_write": "scb_dataset3/SCB5-Handrise-Read-write-2024-9-17",
    "scbehavior_bow_turn": "scbehavior/SCB_BowTurnHead_20250509",
    "cctv_exam_monitor": "cctv_exam_monitor/CCTV-Exam -Monitor -Dataset",
    "phone_candidate_student_behaviour": "phone_candidate_student_behaviour",
}


def measure(item):
    source, path = item
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    img = cv2.imread(str(path))
    if img is None:
        return dict(source=source, path=str(path.relative_to(ROOT)), sha256=digest, corrupt=True)
    gray = cv2.cvtColor(cv2.resize(img, (32, 32)), cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(gray.astype(np.float32))[:8, :8].flatten()
    bits = dct > np.median(dct[1:])
    phash = sum(int(bit) << i for i, bit in enumerate(bits))
    return dict(source=source, path=str(path.relative_to(ROOT)), sha256=digest,
                phash=f"{phash:016x}", width=img.shape[1], height=img.shape[0], corrupt=False)


def sheet(samples, title, out):
    """12 full-frame examples per page with complete path and box metadata."""
    out.parent.mkdir(parents=True, exist_ok=True)
    for page in range(0, len(samples), 12):
        selected = samples[page:page + 12]
        canvas = np.full((60 + 310 * ((len(selected) + 2) // 3), 1800, 3), 25, np.uint8)
        cv2.putText(canvas, title, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, .65, (240, 240, 240), 1)
        for i, (path, box) in enumerate(selected):
            img = cv2.imread(str(path))
            if img is None:
                continue
            h, w = img.shape[:2]
            if box is not None:
                xc, yc, bw, bh = box
                cv2.rectangle(img, (int((xc-bw/2)*w), int((yc-bh/2)*h)),
                              (int((xc+bw/2)*w), int((yc+bh/2)*h)), (0,255,0), 2)
            tile = np.full((310, 600, 3), 25, np.uint8)
            ratio = min(600/w, 230/h)
            preview = cv2.resize(img, (int(w*ratio), int(h*ratio)))
            tile[:preview.shape[0], :preview.shape[1]] = preview
            metadata = [path.name, f"size={w}x{h} bbox={box if box else 'UNAVAILABLE (unlabeled)'}", str(path.parent.relative_to(ROOT))]
            for row, text in enumerate(metadata):
                cv2.putText(tile, text, (5, 250+row*20), cv2.FONT_HERSHEY_SIMPLEX, .35, (235,235,235), 1)
            y, x = 60+(i//3)*310, (i%3)*600
            canvas[y:y+310,x:x+600] = tile
        cv2.imwrite(str(out.with_name(out.stem + f"_{page//12+1}.jpg")), canvas)


def main():
    report = ROOT / "reports/acquisition_v2"
    report.mkdir(parents=True, exist_ok=True)
    all_images = []
    inventory = []
    stats = {}
    for name, rel in SOURCES.items():
        root = ROOT / "datasets/raw" / rel
        images = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        all_images.extend((name, p) for p in images)
        yaml_paths = sorted(root.rglob("*.yaml"))
        names = []
        if yaml_paths:
            content = yaml_paths[0].read_text(encoding="utf-8")
            print(f"ACTUAL YAML {yaml_paths[0]}:\n{content}", flush=True)
            names = yaml.safe_load(content).get("names", [])
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names)]
        counts, class_images = Counter(), defaultdict(set)
        samples = defaultdict(list)
        errors = []
        by_stem = defaultdict(list)
        for p in images:
            by_stem[p.stem].append(p)
        labels = []
        for label in sorted(root.rglob("*.txt")):
            if label.stem not in by_stem:
                continue
            labels.append(label)
            candidates = by_stem[label.stem]
            expected = Path(str(label).replace("\\labels\\", "\\images\\"))
            image = next((p for p in candidates if p.with_suffix("") == expected.with_suffix("")), None)
            if image is None and len(candidates) == 1:
                image = candidates[0]
            if image is None:
                errors.append(dict(label=str(label), reason="ambiguous image pairing"))
                continue
            annots, issues = validate_annotation_file(label)
            errors.extend(dict(label=str(label.relative_to(ROOT)), reason=str(e)) for e in issues)
            for cid, *box in annots:
                counts[cid] += 1
                class_images[cid].add(str(image))
                samples[cid].append((image, box))
        for cid in sorted(counts):
            cname = names[cid] if 0 <= cid < len(names) else f"UNKNOWN_ID_{cid}"
            target = report / "class_samples" / f"{name}_{cid}.jpg"
            selected = random.Random(42).sample(samples[cid], min(48, len(samples[cid])))
            if name.startswith("phone_"):
                sheet(selected, f"{name} | source class {cid}: {cname}", target)
            inventory.append(dict(source_dataset=name, source_subset=rel, source_class_id=cid,
                source_class_name=cname, annotation_count=counts[cid], image_count=len(class_images[cid]),
                percentage=100*counts[cid]/sum(counts.values()),
                contact_sheet=str(target.relative_to(ROOT)) if name.startswith("phone_") else "reports/class_samples_verified/",
                mapping_status="needs_review"))
        stats[name] = dict(root=str(root), images=len(images), labels=len(labels),
            classes=names, annotations=sum(counts.values()), annotation_counts=dict(counts),
            issues=errors, original_splits=dict(Counter(next((s for s in p.parts if s in {'train','val','valid','test'}), 'unknown') for p in images)))
        if name == "cctv_exam_monitor":
            sheet([(p,None) for p in random.Random(42).sample(images,min(36,len(images)))],
                  "CCTV archive: unlabeled visual domain samples (NOT class verification)",
                  ROOT / "reports/cctv_exam_monitor/class_samples/unlabeled_domain.jpg")
        print(name, len(images), "images", len(labels), "labels", dict(counts), flush=True)
    (report / "source_statistics.json").write_text(json.dumps(stats,indent=2), encoding="utf-8")
    (report / "source_inventory.json").write_text(json.dumps(inventory,indent=2), encoding="utf-8")
    print("Global SHA256 + perceptual image scan",len(all_images),flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        measured = list(pool.map(measure, all_images))
    (report / "image_hashes.json").write_text(json.dumps(measured,indent=2),encoding="utf-8")
    exact, perceptual = defaultdict(list), defaultdict(list)
    for row in measured:
        exact[row['sha256']].append(row)
        if not row['corrupt']:
            perceptual[row['phash']].append(row)
    clusters = [v for v in exact.values() if len(v)>1]
    # Same 64-bit perceptual hash: useful conservative near-duplicate candidate screen.
    near = [v for v in perceptual.values() if len({x['sha256'] for x in v})>1]
    result = dict(images=len(measured), corrupt_images=sum(r['corrupt'] for r in measured),
        unique_sha256=len(exact), exact_duplicate_clusters=len(clusters),
        duplicate_excess=sum(len(v)-1 for v in clusters),
        cross_source_exact_clusters=sum(len({x['source'] for x in v})>1 for v in clusters),
        same_phash_different_sha256_clusters=len(near),
        perceptual_method="32x32 grayscale DCT, 64-bit median hash; distance=0 candidate screen, not exhaustive",
        exact_clusters=clusters, perceptual_candidate_clusters=near)
    (report / "global_duplicates.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print({k:v for k,v in result.items() if not isinstance(v,list)},flush=True)


if __name__ == "__main__":
    main()
