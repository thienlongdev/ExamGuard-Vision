# Multi-Cue Risk Aggregation & Independent Clustering Audit

## 1. Defect Description
In previous Stage 2 orchestration code, the call to `RiskAggregator.assess_event_risk()` was hardcoded to:
```python
active_cues_count = 1
independent_cues_count = 1
mean_reliability = posture_reliability
```
This hardcoding completely bypassed multi-cue fusion semantics. Even when a student exhibited simultaneous posture turn, head yaw deviation, and an associated phone, the risk aggregator evaluated the evidence as a single isolated cue.

---

## 2. Independent Cue Clustering Implementation
To prevent correlated double-counting while properly rewarding independent corroboration, the orchestration layer now groups active physical observations into conceptual independent clusters:

| Cluster Name | Contributing Physical Observations | Correlation Rationale |
| :--- | :--- | :--- |
| **`ORIENTATION_CLUSTER`** | - Posture `TURN_HEAD`<br>- Head-Pose yaw deviation ($> 28^\circ$) | Posture turn and head yaw both observe head angle. They increase confidence in orientation, but do **not** represent two independent behavioral phenomena. |
| **`PHONE_CLUSTER`** | - Phone spatial association<br>- Phone object detection | Independent physical object branch. |
| **`DISCUSSION_CLUSTER`** | - Macro behavior `discuss` | Independent social interaction cue. |
| **`STANDING_CLUSTER`** | - Macro behavior `stand` | Independent vertical posture cue. |
| **`POSTURE_REST_CLUSTER`** | - Posture `HEAD_DOWN`<br>- Posture resting cue | Head-down / sleep indicator. |

### Algorithmic Formulation
For each emitted `FusedEvent` and associated `PerTrackCueState`:
1. **Active Cues Count**: Sum of all physical sensors/models providing positive supporting evidence.
2. **Independent Cues Count**: Number of distinct clusters represented by the active supporting cues.
   $$\text{independent\_cues\_count} = \left| \bigcup_{c \in \text{Active Cues}} \text{Cluster}(c) \right|$$
3. **Mean Supporting Reliability**: Arithmetic mean of the reliability weights of active supporting cues.
4. **Competing Evidence Discount**: If `read_write_suppression_active` is True, mean supporting reliability is discounted by 50% ($\times 0.50$) to prevent false alarms during normal student exam writing.

---

## 3. Empirical Verification
The implementation is verified by regression tests in `tests/test_stage2_integrity.py::test_multi_cue_risk_independent_clustering`:

```python
# Test Case 1: Posture Turn + Yaw Deviation
# Active cues = 2 (posture + yaw), Independent clusters = 1 (ORIENTATION_CLUSTER)
cues_cnt, indep_cnt, rel = pipeline._compute_multi_cue_risk_inputs(ev, cue_state)
assert cues_cnt == 2
assert indep_cnt == 1

# Test Case 2: Posture Turn + Phone Association
# Active cues = 2, Independent clusters = 2 (ORIENTATION_CLUSTER + PHONE_CLUSTER)
cues_cnt, indep_cnt, rel = pipeline._compute_multi_cue_risk_inputs(ev_phone, cue_state_phone)
assert cues_cnt == 2
assert indep_cnt == 2
```

---

## 4. Status Flag
- `MULTI_CUE_RISK_INTEGRATION_FIXED = YES`
