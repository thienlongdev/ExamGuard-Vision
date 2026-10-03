import json
from collections import defaultdict

manifest = [json.loads(line) for line in open('datasets/v4_crop/manifest.jsonl', encoding='utf-8')]

# 1. SHA256 duplicates
sha_map = defaultdict(list)
for r in manifest:
    sha_map[r['sha256']].append(r)

sha_dups = {k: v for k, v in sha_map.items() if len(v) > 1}
print('Exact SHA256 duplicate sets:', len(sha_dups))
for k, v in list(sha_dups.items())[:5]:
    sample_ids = [s['sample_id'] for s in v]
    print(f'  Hash {k[:16]}...: {len(v)} samples ({sample_ids})')

# Check if any SHA duplicate crosses splits
sha_cross_split = 0
for k, v in sha_dups.items():
    splits = set(r['split'] for r in v)
    if len(splits) > 1:
        sha_cross_split += 1
print('SHA duplicates crossing splits:', sha_cross_split)

# 2. Source Image / Clip crossing splits
source_splits = defaultdict(set)
for r in manifest:
    src = r['source_clip_id'] if r['source_clip_id'] else f"scb_img_{r['source_image_id']}"
    source_splits[src].add(r['split'])

cross_source_leaks = {k: v for k, v in source_splits.items() if len(v) > 1}
print('Source images / clips crossing splits:', len(cross_source_leaks))

# 3. Perceptual hash (dHash) collisions across splits
dhash_splits = defaultdict(set)
for r in manifest:
    dhash_splits[r['dhash']].add(r['split'])
dhash_cross_splits = {k: v for k, v in dhash_splits.items() if len(v) > 1 and 'train' in v}
print('Perceptual dHash collisions between train and eval splits:', len(dhash_cross_splits))
