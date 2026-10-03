# Dataset Split & Leakage Verification Report

**Leakage Status**: ZERO LEAKAGE (PASS)  
**Total Records**: 10,138  
**Total Unique Groups / Sequences**: 510  

## 1. Split Sizes

| Split | Images | Image Share | Groups | Group Share |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 7,096 | 70.0% | 363 | 71.2% |
| **Validation** | 1,520 | 15.0% | 85 | 16.7% |
| **Test** | 1,522 | 15.0% | 62 | 12.2% |

## 2. Canonical Class Distribution across Splits

| Canonical Class | Train Instances | Val Instances | Test Instances | Total |
| :--- | :---: | :---: | :---: | :---: |
| **discuss** | 12,785 | 2,734 | 3,326 | 18,845 |
| **head_down** | 3,881 | 394 | 687 | 4,962 |
| **turn_head** | 8,174 | 1,757 | 1,225 | 11,156 |

## 3. Leakage Analysis

- **Train / Val Shared Groups**: 0
- **Train / Test Shared Groups**: 0
- **Val / Test Shared Groups**: 0

All recording sessions / sequence groups are strictly partitioned. No intra-sequence frame leakage between train and validation/test.
