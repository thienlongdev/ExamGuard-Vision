import json
from collections import Counter, defaultdict

m = json.load(open('datasets/processed_v3/manifest_v3.json'))
train = m['splits']['train']

grp_data = defaultdict(lambda: {'imgs': 0, 'boxes': Counter(), 'subsets': set()})
for it in train:
    gid = it['group_id']
    grp_data[gid]['imgs'] += 1
    for ann in it.get('annotations', []):
        grp_data[gid]['boxes'][ann['canonical_class_name']] += 1
        for s in ann.get('sources', []):
            grp_data[gid]['subsets'].add(s.get('source_subset', ''))

for gid, d in grp_data.items():
    is_oblique = any(k in s for s in d['subsets'] for k in ['Stand', 'Bow', 'Turn', 'Discuss'])
    d['viewpoint'] = 'cctv_oblique_high_angle' if is_oblique else 'frontal_classroom'

sorted_grps = sorted(grp_data.keys(), key=lambda g: (grp_data[g]['boxes']['head_down'], grp_data[g]['boxes']['turn_head']), reverse=True)
print(f"{'Group':<10} | {'Imgs':<5} | {'Viewpoint':<25} | {'HD':<5} | {'TH':<5} | {'Norm':<6} | {'Disc':<5} | {'Stand':<5} | Subsets")
print("-" * 105)
for g in sorted_grps[:45]:
    d = grp_data[g]
    if d['boxes']['head_down'] > 0 or d['boxes']['turn_head'] > 0:
        subsets_str = ", ".join(d['subsets'])
        print(f"{g:<10} | {d['imgs']:<5} | {d['viewpoint']:<25} | {d['boxes']['head_down']:<5} | {d['boxes']['turn_head']:<5} | {d['boxes']['normal']:<6} | {d['boxes']['discuss']:<5} | {d['boxes']['stand']:<5} | {subsets_str}")
