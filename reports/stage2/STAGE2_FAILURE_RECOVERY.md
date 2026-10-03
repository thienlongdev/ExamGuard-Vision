# Stage 2 Failure Recovery & Graceful Degradation Report

## 1. Principles of Error Isolation
In high-stakes examination environments, a transient perception failure, network glitch, or corrupted video frame must **never crash the orchestration engine**.
Stage 2 implements strict component-level fault isolation:

```
[Corrupted Frame] --------> Safe Frame Skip (Logged, Pipeline Stays Active)
[Headpose Failure] -------> Graceful Capability Gating (YAW_UNAVAILABLE, Posture Continues)
[Phone Occlusion] --------> AMBIGUOUS_ASSOCIATION (No False Positive Event Triggered)
[Evidence Disk Full] -----> Memory Ring Buffer Retains State (Warning Logged, Non-Fatal)
[Stream EOF / Drop] ------> Safe Event Finalization (Active Events Closed, Evidence Flushed)
```

---

## 2. Failure Scenarios & Validated Recovery Behaviors

### 2.1 Corrupted / Empty Video Frames
- **Failure Mode**: Network packet loss on RTSP or corrupted video container produces an un-decodable frame (`None` or 0-byte buffer).
- **Handling**: `process_frame()` verifies `frame is not None and frame.size > 0`. If invalid, the frame is skipped, a warning is logged, and previous track continuity is maintained.
- **Test**: `tests/test_stage2_failure_recovery.py::test_corrupted_frame_skipping` (passed).

### 2.2 Sub-Resolution / Unresolvable Head Crops
- **Failure Mode**: Student turns completely away from the camera or is seated in the far back row, resulting in a head bounding box smaller than $20\times 20$ pixels.
- **Handling**: The crop scheduler inspects crop dimensions before inference. If $\min(w, h) < 20$, headpose inference is bypassed.
  - Headpose Status: `FACE_UNRESOLVABLE`
  - Yaw Deg: `None`
  - Reliability: `0.0`
  - **Critical Rule Enforced**: The system NEVER substitutes `yaw = 0.0`. Posture inference continues unaffected.
- **Test**: `tests/test_stage2_failure_recovery.py::test_headpose_crop_too_small_handled_gracefully` (passed).

### 2.3 Ambiguous Phone Association
- **Failure Mode**: A mobile phone is detected on an exam desk equidistant between two adjacent student bounding boxes.
- **Handling**: `PhoneAssociator` evaluates geometric distance and containment:
  - If a phone overlaps multiple student boxes or the difference in distance between the two nearest tracks is below the threshold, association status is set to `AMBIGUOUS_ASSOCIATION`.
  - Only `ASSOCIATED` status can contribute positive phone evidence. Ambiguous phones are ignored by the V4D event engine.
- **Test**: `tests/test_stage2_pipeline_orchestration.py::test_phone_ambiguity_handling` (passed).

### 2.4 Video Stream Disconnection & Stream EOF
- **Failure Mode**: RTSP camera stream drops or offline video file reaches End-Of-File (EOF).
- **Handling**:
  - `VideoSource.read_frame()` returns `None`.
  - The pipeline loop breaks cleanly.
  - All currently `OPEN` events are finalized with status `CLOSED`, recording the final timestamp and writing closing evidence metadata.
- **Test**: `tests/test_stage2_failure_recovery.py::test_stream_eof_closes_active_events` (passed).

### 2.5 Evidence Storage Path Creation & Handling
- **Failure Mode**: The configured evidence snapshot directory does not pre-exist or write permissions are restricted.
- **Handling**: `EvidenceManager` automatically creates parent directories via `os.makedirs(..., exist_ok=True)`. If a write fails, the error is logged as a non-fatal warning, preserving pipeline operation.
- **Test**: `tests/test_stage2_evidence_lifecycle.py::test_evidence_directory_autocreation` (passed).

---

## 3. Recovery Verification Matrix

| Injected Failure Condition | Tested Module | Pipeline Survival | Event Engine State | Test Suite Status |
| :--- | :--- | :--- | :--- | :--- |
| Empty / None Frame Input | `Stage2Pipeline` | Survived (Skipped) | Unchanged | PASSED |
| Tiny Head Crop ($12\times 15\text{ px}$) | `CropScheduler` | Survived (Bypassed) | `YAW_UNAVAILABLE` | PASSED |
| Equidistant Dual Student Phone | `PhoneAssociator` | Survived | `AMBIGUOUS` (Ignored) | PASSED |
| Non-existent Evidence Dir | `EvidenceManager` | Survived (Auto-created) | Snapshots Saved | PASSED |
| Stream Sudden EOF | `Stage2Pipeline` | Clean Exit | All Events Closed | PASSED |

---

## 4. Verdict
- **FAILURE_RECOVERY_READY**: **YES**.
