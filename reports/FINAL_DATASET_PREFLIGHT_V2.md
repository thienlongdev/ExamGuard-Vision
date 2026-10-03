# Final Dataset Preflight V2

**FAIL — NOT READY. Final rebuild blocked by missing CCTV annotations.**

This is a read-only check of the preserved legacy dataset, not approval of a final V2 dataset.
Zero shared heuristic group IDs is not proof of unseen recording sessions.

## Checks

- dataset_nonempty: PASS
- all_classes_have_verified_source: FAIL / NOT VERIFIED
- canonical_matches_mapping: PASS
- no_invalid_bbox: PASS
- image_label_pairing: PASS
- no_missing_manifest_images: PASS
- labels_match_manifest: PASS
- approved_only: PASS
- physical_source_classes_only: PASS
- source_provenance_preserved: PASS
- no_duplicate_leakage: PASS
- no_group_leakage: PASS
- train_has_every_intended_class: FAIL / NOT VERIFIED
- cctv_annotations_exist: FAIL / NOT VERIFIED
- cctv_in_validation: FAIL / NOT VERIFIED
- cctv_in_test: FAIL / NOT VERIFIED
- final_v2_rebuild_complete: FAIL / NOT VERIFIED
- real_sequence_grouping_verified: FAIL / NOT VERIFIED

## Measured legacy split statistics

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

Exact cross-split SHA256 leakage: {'train_val': 0, 'train_test': 0, 'val_test': 0}
Cross-split group-ID leakage: {'train_val': 0, 'train_test': 0, 'val_test': 0}

No sanity epoch was launched. Do not use an older checkpoint or successful older preflight as V2 evidence.
