"""Read-only V2 gate; an old SCB-only dataset cannot pass target-domain readiness."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]


def supported_classes(mapping, inventory):
    """Only approved mappings with physical annotations can support a taxonomy."""
    supported = set()
    for row in inventory:
        rule = mapping.get('datasets', {}).get(row['source_dataset'], {}).get('source_classes', {}).get(row['source_class_name'], {})
        if row['annotation_count'] > 0 and rule.get('status') == 'approved' and rule.get('target'):
            supported.add(rule['target'])
    return supported


def strict_box(box):
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        return False
    x, y, w, h = box
    return w > 0 and h > 0 and min(x-w/2, y-h/2) >= -1e-6 and max(x+w/2, y+h/2) <= 1+1e-6


def main():
    root = ROOT / 'datasets/processed'
    report = ROOT / 'reports/acquisition_v2'
    stats = json.loads((report / 'source_statistics.json').read_text())
    inventory = json.loads((report / 'source_inventory.json').read_text())
    mapping = yaml.safe_load((ROOT / 'configs/dataset_mapping.yaml').read_text())
    definition = yaml.safe_load((root / 'dataset.yaml').read_text())
    names = definition['names']
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    checks = dict(dataset_nonempty=True, all_classes_have_verified_source=set(names) <= supported_classes(mapping,inventory),
        canonical_matches_mapping=names == mapping['canonical_classes'], no_invalid_bbox=True,
        image_label_pairing=True, no_missing_manifest_images=True, labels_match_manifest=True,
        approved_only=True, physical_source_classes_only=True, source_provenance_preserved=True,
        no_duplicate_leakage=True, no_group_leakage=True, train_has_every_intended_class=True,
        cctv_annotations_exist=stats['cctv_exam_monitor']['labels'] > 0,
        cctv_in_validation=False, cctv_in_test=False, final_v2_rebuild_complete=False,
        real_sequence_grouping_verified=False)
    counts, hashes, groups = {}, defaultdict(set), defaultdict(set)
    physical = {(r['source_dataset'],r['source_class_name']) for r in inventory}
    for split in ['train','val','test']:
        manifest = json.loads((root / 'manifests' / f'{split}_manifest.json').read_text())
        images = {p.stem for p in (root / 'images' / split).glob('*') if p.is_file()}
        labels = {p.stem for p in (root / 'labels' / split).glob('*.txt')}
        checks['dataset_nonempty'] &= bool(manifest)
        checks['image_label_pairing'] &= images == labels
        boxes, class_images, sources = Counter(), Counter(), Counter()
        for row in manifest:
            groups[split].add(row['group_id'])
            for source in row.get('source_datasets',[]):
                sources[source] += 1
            p = ROOT / row['image_path']
            if p.exists():
                hashes[split].add(hashlib.sha256(p.read_bytes()).hexdigest())
            else:
                checks['no_missing_manifest_images'] = False
            physical_labels = []
            label = ROOT / row['label_path']
            try:
                for line in label.read_text().splitlines():
                    tokens = line.split()
                    if len(tokens) != 5:
                        checks['no_invalid_bbox'] = False
                        continue
                    cid, box = int(tokens[0]), [float(v) for v in tokens[1:]]
                    checks['no_invalid_bbox'] &= 0 <= cid < len(names) and strict_box(box)
                    physical_labels.append((cid, tuple(round(v,6) for v in box)))
            except (OSError,ValueError):
                checks['image_label_pairing'] = False
            expected = []
            seen = set()
            for a in row['annotations']:
                name = a['canonical_class_name']
                boxes[name] += 1
                seen.add(name)
                checks['approved_only'] &= a.get('mapping_status') == 'approved'
                checks['source_provenance_preserved'] &= all(k in a for k in ['source_class_id','source_class_name','source_dataset','source_image','canonical_class_name'])
                checks['physical_source_classes_only'] &= (a.get('source_dataset'), a.get('source_class_name')) in physical
                expected.append((a['canonical_class_id'],tuple(round(v,6) for v in a['bbox'])))
            checks['labels_match_manifest'] &= sorted(physical_labels) == sorted(expected)
            class_images.update(seen)
        counts[split] = dict(images=len(manifest), boxes=sum(boxes.values()),class_boxes=dict(boxes),class_images=dict(class_images),sources=dict(sources),cctv_images=sources['cctv_exam_monitor'],phone_images=sources['phone_candidate_student_behaviour'])
    duplicate_leakage, group_leakage = {}, {}
    for left,right in [('train','val'),('train','test'),('val','test')]:
        duplicate_leakage[f'{left}_{right}'] = len(hashes[left]&hashes[right])
        group_leakage[f'{left}_{right}'] = len(groups[left]&groups[right])
    checks['no_duplicate_leakage'] = not any(duplicate_leakage.values())
    checks['no_group_leakage'] = not any(group_leakage.values())
    checks['train_has_every_intended_class'] = set(names) <= set(counts['train']['class_boxes'])
    checks['cctv_in_validation'] = counts['val']['cctv_images'] > 0
    checks['cctv_in_test'] = counts['test']['cctv_images'] > 0
    result = dict(status='PASS' if all(checks.values()) else 'FAIL',dataset_scope='preserved legacy dataset; not a rebuilt V2 candidate',checks=checks,split_statistics=counts,duplicate_leakage=duplicate_leakage,group_id_leakage=group_leakage)
    (report / 'preflight_v2.json').write_text(json.dumps(result,indent=2))
    lines = ['# Final Dataset Preflight V2', '', '**FAIL — NOT READY. Final rebuild blocked by missing CCTV annotations.**', '',
        'This is a read-only check of the preserved legacy dataset, not approval of a final V2 dataset.',
        'Zero shared heuristic group IDs is not proof of unseen recording sessions.', '', '## Checks', '']
    lines.extend(f"- {k}: {'PASS' if v else 'FAIL / NOT VERIFIED'}" for k,v in checks.items())
    lines += ['', '## Measured legacy split statistics', '', '```json', json.dumps(counts,indent=2), '```', '',
        f'Exact cross-split SHA256 leakage: {duplicate_leakage}', f'Cross-split group-ID leakage: {group_leakage}', '',
        'No sanity epoch was launched. Do not use an older checkpoint or successful older preflight as V2 evidence.']
    (ROOT / 'reports/FINAL_DATASET_PREFLIGHT_V2.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    return all(checks.values())


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
