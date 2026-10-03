import json
from collections import Counter

manifest_path = 'datasets/v4_crop/manifest.jsonl'
records = [json.loads(line) for line in open(manifest_path, 'r', encoding='utf-8')]

print('Total records:', len(records))

# Splits
split_counts = Counter(r['split'] for r in records)
print('\n--- Splits ---')
for s, c in split_counts.most_common():
    print(f'  {s:22}: {c:6d} ({c/len(records)*100:.1f}%)')

# Classes
class_counts = Counter(r['ontology_label'] for r in records)
print('\n--- Ontology Classes ---')
for cl, c in class_counts.most_common():
    print(f'  {cl:25}: {c:6d} ({c/len(records)*100:.1f}%)')

# Sources
source_counts = Counter(r['source_dataset'] for r in records)
print('\n--- Sources ---')
for src, c in source_counts.most_common():
    print(f'  {src:25}: {c:6d} ({c/len(records)*100:.1f}%)')

# Scale Buckets
scale_counts = Counter(r['scale_bucket'] for r in records)
print('\n--- Scale Buckets ---')
for sc, c in scale_counts.most_common():
    print(f'  {sc:25}: {c:6d} ({c/len(records)*100:.1f}%)')

# Viewpoints
vp_counts = Counter(r['viewpoint'] for r in records)
print('\n--- Viewpoints ---')
for vp, c in vp_counts.most_common():
    print(f'  {vp:25}: {c:6d} ({c/len(records)*100:.1f}%)')

# Class x Scale
print('\n--- Class x Scale ---')
for cl in sorted(class_counts.keys()):
    sub = [r for r in records if r['ontology_label'] == cl]
    scs = Counter(r['scale_bucket'] for r in sub)
    print(f'  {cl:25}: LARGE={scs["PERSON_LARGE"]:4d}, MED={scs["PERSON_MEDIUM"]:4d}, SMALL={scs["PERSON_SMALL"]:4d}, VERY_SMALL={scs["PERSON_VERY_SMALL"]:4d}')

# Source x Scale
print('\n--- Source x Scale ---')
for src in sorted(source_counts.keys()):
    sub = [r for r in records if r['source_dataset'] == src]
    scs = Counter(r['scale_bucket'] for r in sub)
    print(f'  {src:25}: LARGE={scs["PERSON_LARGE"]:4d}, MED={scs["PERSON_MEDIUM"]:4d}, SMALL={scs["PERSON_SMALL"]:4d}, VERY_SMALL={scs["PERSON_VERY_SMALL"]:4d}')

# Viewpoint x Scale
print('\n--- Viewpoint x Scale ---')
for vp in sorted(vp_counts.keys()):
    sub = [r for r in records if r['viewpoint'] == vp]
    scs = Counter(r['scale_bucket'] for r in sub)
    print(f'  {vp:25}: LARGE={scs["PERSON_LARGE"]:4d}, MED={scs["PERSON_MEDIUM"]:4d}, SMALL={scs["PERSON_SMALL"]:4d}, VERY_SMALL={scs["PERSON_VERY_SMALL"]:4d}')

# Quality flags
q_counts = Counter()
for r in records:
    for q in r['quality_flags']:
        q_counts[q] += 1
print('\n--- Quality Flags ---')
for q, c in q_counts.most_common():
    print(f'  {q:25}: {c:6d}')

# Split x Class
print('\n--- Split x Class ---')
for s in ['train', 'same_domain_val', 'cross_source_holdout', 'high_angle_holdout', 'temporal_holdout']:
    sub = [r for r in records if r['split'] == s]
    cls_cnt = Counter(r['ontology_label'] for r in sub)
    print(f'Split {s} ({len(sub)}):')
    for cl, c in cls_cnt.most_common():
        print(f'    {cl:25}: {c:5d}')
