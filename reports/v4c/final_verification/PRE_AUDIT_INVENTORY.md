# V4C Final Verification: Pre-Audit Artifact Inventory

**Generated At**: 2026-10-03 03:35:39
**Status**: Complete Forensic Artifact Inventory

---

## 1. Executive Checkpoint & Baseline Inventory

| Directory / Resource | Artifact Path | Size (Bytes) | SHA-256 Hash | Modified Time |
| :--- | :--- | :---: | :--- | :---: |
| `models/trained` | `models/trained/stage1_best.pt` | 44,048,089 | `6d713808f0bc670e8bf06e01b006fac73d8d6e5db7130584cec1658b003ce98a` | 2026-10-02T20:44:39 |
| `models/trained` | `models/trained/stage1_5_best.pt` | 44,038,169 | `68690cf82715dc6dad2eb35a08711531170e935eb7dbe1c30fafabae341e2c2c` | 2026-10-02T22:12:59 |
| `models/trained` | `models/trained/v4_posture_best.pt` | 18,518,447 | `529a23f96ebec6051ad29279fba3cd5279e7aa8fe2e46292ab4bb9c9523ca180` | 2026-10-03T02:36:17 |
| `models/trained` | `models/trained/v4_headpose_yaw_best.pt` | 284,178,037 | `5d15eec5941cfc8d2468d09c27bff8eca2014e62bb12fc9a95c0c6239fbbca55` | 2026-10-03T02:49:56 |

---

## 2. Training Run Checkpoint & Metric Inventory (`runs/v4c/`)

| Experiment Directory | Key Artifact | Size (Bytes) | SHA-256 Hash |
| :--- | :--- | :---: | :--- |
| `runs\v4c\A1_resnet18_cbam_tight_person_crop_224` | `best_model.pt` | 135,363,045 | `75b1bb76ee703aef77d4f4a83144c95199c2ff1b28ee0d6d64c0f7b50a03226b` |
| `runs\v4c\A1_resnet18_cbam_tight_person_crop_224` | `metrics.json` | 12,696 | `a307ac4b541177ab69d41400d7fa8d2c3cda898ff566c2462a45e29e072cbb89` |
| `runs\v4c\A1_resnet18_cbam_tight_person_crop_224_confirmed` | `best_model.pt` | 135,363,045 | `49be87629bf312a7099b72caed7930d290950150c3537dce4ad9af2a474edc3f` |
| `runs\v4c\A1_resnet18_cbam_tight_person_crop_224_confirmed` | `last_checkpoint.pt` | 135,364,967 | `cf82acf888287521fc50c4cd2103a79941f7301ed52c7f7d661081e9e86b10d4` |
| `runs\v4c\A1_resnet18_cbam_tight_person_crop_224_confirmed` | `metrics.json` | 12,697 | `b19edce326cebc4f7869284a7a3ad47f53268c901c74c4d42d3bf30d7c4decdb` |
| `runs\v4c\A1_resnet18_cbam_tight_person_crop_224_confirmed` | `training_history.csv` | 1,302 | `f50404a71855b7a3e4ecd2d62aa4c1c13753324b2c2e0736178a2cfc261682a9` |
| `runs\v4c\A2_resnet18_cbam_context_person_crop_224` | `best_model.pt` | 135,363,045 | `21ab0a4fc716798941d473976f08ab086316a174de9a651af46a4b2197f12879` |
| `runs\v4c\A2_resnet18_cbam_context_person_crop_224` | `last_checkpoint.pt` | 135,364,967 | `1bbcc7467c44f1de2175330667ded711947eb26b728782aa7eb1fd7e5e34da57` |
| `runs\v4c\A2_resnet18_cbam_context_person_crop_224` | `metrics.json` | 12,711 | `2b1b66717c2d945b755faffd8a59a1140114f54a893e4100ae896a931e2900a0` |
| `runs\v4c\A2_resnet18_cbam_context_person_crop_224` | `training_history.csv` | 1,611 | `91cf3deccf19a47dc51e5cfff981a2a900e3e45296795fbb4e354b8e42eae68c` |
| `runs\v4c\B1_resnet50_cbam_tight_person_crop_224` | `best_model.pt` | 312,915,892 | `b5ea5910167eca5536d2af2822598b7e341a215cdf706d7b12018e27028d0d2d` |
| `runs\v4c\B1_resnet50_cbam_tight_person_crop_224` | `last_checkpoint.pt` | 312,920,769 | `f10a46b970a23dc46b2a8ee775175e0f69d58a079b091965c57fc4c52ef41186` |
| `runs\v4c\B1_resnet50_cbam_tight_person_crop_224` | `metrics.json` | 12,555 | `ac89a6c112685861a3c33da854af34c4353c37aff327c001c8ac9be876d0d435` |
| `runs\v4c\B1_resnet50_cbam_tight_person_crop_224` | `training_history.csv` | 1,091 | `0de03a6f801aeee6e28fd05c414a646530b594c88e96a81e7128b31e834d4c52` |
| `runs\v4c\B2_resnet50_cbam_context_person_crop_224` | `best_model.pt` | 312,915,892 | `54482fb78d5f33df2181f7e1ac7bcc5a96f1668d81aebafab0195955adbebb35` |
| `runs\v4c\B2_resnet50_cbam_context_person_crop_224` | `last_checkpoint.pt` | 312,920,769 | `80cfd8fed4c317f30cc146b7f355733a8c1389dd2453301291b70ea842de63af` |
| `runs\v4c\B2_resnet50_cbam_context_person_crop_224` | `metrics.json` | 12,478 | `286e919140191047ceacf374c204ed8b00018dd96a7a06d18cc9f596a0d168f8` |
| `runs\v4c\B2_resnet50_cbam_context_person_crop_224` | `training_history.csv` | 523 | `4276b4d056224bf3ad7b52c710916130a7ed405a02a48a2c5b6c0ea3d77c1c56` |
| `runs\v4c\C1_mobilenet_v3_small_tight_person_crop_224` | `best_model.pt` | 18,518,447 | `529a23f96ebec6051ad29279fba3cd5279e7aa8fe2e46292ab4bb9c9523ca180` |
| `runs\v4c\C1_mobilenet_v3_small_tight_person_crop_224` | `last_checkpoint.pt` | 18,521,699 | `250f816186f9450a874a5a41a0d535985a020896a2df676ab2637fc3ce4503a3` |
| `runs\v4c\C1_mobilenet_v3_small_tight_person_crop_224` | `metrics.json` | 12,717 | `c280fb9fd5fb53eef003e0c93de60f9fb323cd111b3f06e5ca260976dc5babe9` |
| `runs\v4c\C1_mobilenet_v3_small_tight_person_crop_224` | `training_history.csv` | 727 | `34d55233a91b69bd793936ec1b5993adff4d3ecfd2a618dd852ed2799e0fba5b` |
| `runs\v4c\C2_mobilenet_v3_small_context_person_crop_224` | `best_model.pt` | 18,518,447 | `958252ffb5895d056732075dfd950abf948319d8e35dfff48ea60a5f93938f45` |
| `runs\v4c\C2_mobilenet_v3_small_context_person_crop_224` | `last_checkpoint.pt` | 18,521,699 | `025ec9d6460c21fcce02eed4260db793526e06568d5c24e3a7245e526718acc6` |
| `runs\v4c\C2_mobilenet_v3_small_context_person_crop_224` | `metrics.json` | 12,707 | `e4adba1b93b5c53d76814197c8f6de28bbc8012e0806e4635d4d2c8aaf47f356` |
| `runs\v4c\C2_mobilenet_v3_small_context_person_crop_224` | `training_history.csv` | 727 | `3b046538eb7625ab0f15a9b082b584194dfbee9dc5104b9e2a92451644283fd5` |
| `runs\v4c\headpose_hopenet_yaw` | `best_model.pt` | 284,178,037 | `5d15eec5941cfc8d2468d09c27bff8eca2014e62bb12fc9a95c0c6239fbbca55` |
| `runs\v4c\headpose_hopenet_yaw` | `headpose_metrics.json` | 4,825 | `e1f43447a9eb670a988c9be2624d3c4d1ad1086c3ed1b54f199cfdbbc26aa6d0` |
| `runs\v4c\headpose_hopenet_yaw` | `last_checkpoint.pt` | 284,181,191 | `43816dc945bda88fa951b4027602e0869a9688402ec69e9de6363c1a15be27c9` |
| `runs\v4c\headpose_hopenet_yaw` | `training_history.csv` | 492 | `d9b75910dc9afe2fdeb0bb769adf8e8e0798730f9799871f364680026827698e` |
| `runs\v4c\headpose_resnet18_yaw` | `best_model.pt` | 134,264,773 | `bc31d46cfea007ebddfb6a8d5845641a32fd1cc018a831c4c8ae8b61e579a7d9` |
| `runs\v4c\headpose_resnet18_yaw` | `headpose_metrics.json` | 4,823 | `dc0fa0a97a40def537ff0a5b3c348744ab1b9b211842934a46d722614554a211` |
| `runs\v4c\headpose_resnet18_yaw` | `last_checkpoint.pt` | 134,265,447 | `aa328ca49f414fca1616624dade012daf71623ec3ab773651f9d7cce17ea9749` |
| `runs\v4c\headpose_resnet18_yaw` | `training_history.csv` | 701 | `91554f44c1b34f1da2096c825e4ef2861ec762021f9361f55a2966aa6739fdd3` |
| `runs\v4c\UB_mobilenet_v3_small_upper_body_crop_224` | `metrics.json` | 240 | `cfac082dfebb5b47333568cf7f8ed37d5f5158a2eb868cf99888ec94cf40718d` |
| `runs\v4c\UB_resnet18_cbam_upper_body_crop_224` | `metrics.json` | 240 | `cfac082dfebb5b47333568cf7f8ed37d5f5158a2eb868cf99888ec94cf40718d` |
| `runs\v4c\UB_resnet50_cbam_upper_body_crop_224` | `metrics.json` | 240 | `cfac082dfebb5b47333568cf7f8ed37d5f5158a2eb868cf99888ec94cf40718d` |

---

## 3. Inventory Totals by Directory

| Directory | Total Files | Total Size (MB) |
| :--- | :---: | :---: |
| `models/trained` | 4 | 372.68 MB |
| `runs/v4c` | 50 | 2974.40 MB |
| `reports/v4c` | 51 | 0.59 MB |
| `configs` | 16 | 0.04 MB |
| `src/models` | 16 | 0.10 MB |
| `tests` | 36 | 0.59 MB |
