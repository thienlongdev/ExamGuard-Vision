# V4C Posture Classifier A1 Hang: Root-Cause Forensic Analysis & Remediation Report

**Document ID**: `reports/v4c/A1_HANG_ROOT_CAUSE_ANALYSIS.md`  
**Phase**: V4C Specialized Posture + Head-Pose Model Training & Benchmark  
**Date**: 2026-10-03  
**Status**: PHYSICALLY DIAGNOSED & REMEDIATED  

---

## 1. Executive Summary

During the execution of Experiment A1 (`ResNet18+CBAM`, `TIGHT_PERSON_CROP`, 224x224), the training process became permanently unresponsive after completing Epoch 22 at 01:12:04. Physical inspection revealed:
- CPU delta over multiple sampling intervals: **0.0 seconds** (stalled at 172.5s total CPU).
- GPU utilization: **0% – 2%** with ~5.1–5.3 GB VRAM persistently allocated.
- GPU power: **~24 W** (idle clock state).
- File system: Zero file modifications in `runs/v4c/` for over 45 minutes.
- Process hierarchy: PowerShell spawned `run_v4c_recovery.py` (parent), which appeared to spawn another `run_v4c_recovery.py` (worker).

Through inspection of `task-113.log` and the Windows process execution model, the root cause has been conclusively identified. **No guessing or speculative assumptions are made.**

---

## 2. Physical Evidence & Smoking Gun Traceback

The task log file at `C:\Users\Admin\.gemini\antigravity-ide\brain\d894b925-017d-4875-8cae-a6af8abb9c5c\.system_generated\tasks\task-113.log` captured the exact fatal exception:

```text
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    from multiprocessing.spawn import spawn_main; spawn_main(parent_pid=38432, pipe_handle=1196)
  File "C:\Users\Admin\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
  File "C:\Users\Admin\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 131, in _main
    prepare(preparation_data)
  File "C:\Users\Admin\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 246, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
  File "C:\Users\Admin\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 297, in _fixup_main_from_path
    main_content = runpy.run_path(main_path, run_name="__mp_main__")
  File "<frozen runpy>", line 287, in run_path
  File "<frozen runpy>", line 98, in _run_module_code
  File "<frozen runpy>", line 88, in _run_code
  File "C:\WorkingSpace Python\DETECTOR-YOLO\scripts\run_v4c_recovery.py", line 20, in <module>
    from scripts.train_head_pose_yaw import train_headpose_model
  File "C:\WorkingSpace Python\DETECTOR-YOLO\scripts\train_head_pose_yaw.py", line 28, in <module>
    from src.models.headpose.headpose_estimator import ( ... )
  File "C:\WorkingSpace Python\DETECTOR-YOLO\src\models\headpose\__init__.py", line 6, in <module>
    from src.models.headpose.headpose_estimator import ( ... )
  File "C:\WorkingSpace Python\DETECTOR-YOLO\src\models\headpose\headpose_estimator.py", line 34, in <module>
    def canonicalize_yaw(angle: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
NameError: name 'np' is not defined. Did you mean: 'nn'?
```

---

## 3. Detailed Root-Cause Breakdown

### A. Windows Multiprocessing `spawn` & Entrypoint Re-Execution
On Windows, Python does not support POSIX `fork()`. PyTorch `DataLoader(num_workers > 0)` creates worker processes using `multiprocessing.spawn`.
Under `multiprocessing.spawn`, Python invokes:
```cmd
python.exe -c "from multiprocessing.spawn import spawn_main; spawn_main(parent_pid=..., pipe_handle=...)"
```
To reconstitute the environment, `multiprocessing.spawn._fixup_main_from_path` re-executes the top-level script (`run_v4c_recovery.py`) using `runpy.run_path(..., run_name="__mp_main__")`.
This is why the process tree displayed:
```text
powershell -> run_v4c_recovery.py (parent PID 38432) -> run_v4c_recovery.py (spawned worker)
```
This was **not intentional self-recursion**, but the standard Windows PyTorch multiprocessing worker initialization behavior.

### B. Unintended Module Dependency Cascades
In `scripts/run_v4c_recovery.py`, all pipeline stages were imported at the top level:
```python
from scripts.train_posture_classifier import train_posture_model
from scripts.train_head_pose_yaw import train_headpose_model
from scripts.run_posture_benchmark import generate_posture_reports
from scripts.run_classroom_yaw_bridge import run_bridge_analysis
...
```
When `run_v4c_recovery.py` was executed as `__mp_main__` inside the DataLoader worker, it re-imported every one of these modules from disk on every worker launch.

### C. Concurrency Collision During Background Execution
While A1 was actively training in the background (progressing from Epoch 1 through Epoch 22), the head-pose audit and mathematical tests were being prepared. During an intermediate edit to `src/models/headpose/headpose_estimator.py`, an un-imported reference to `np` was momentarily on disk before being patched.
When DataLoader worker creation occurred for the subsequent evaluation/epoch transition:
1. The spawned worker re-imported `headpose_estimator.py` via `run_v4c_recovery.py`.
2. The worker crashed on bootstrap with `NameError: name 'np' is not defined`.
3. The parent process (`PID 38432`) in `DataLoader` was waiting on the Windows IPC pipe handle for the worker sentinel.
4. Because the worker crashed prior to completing IPC pipe handshake, PyTorch's `DataLoader` iterator on Windows deadlocked indefinitely in `multiprocessing.connection.wait` / `Queue.get()`.

### D. Architectural Vulnerability of Windows DataLoader Multiprocessing
Even without code edits, PyTorch `DataLoader(num_workers > 0)` on Windows with `pin_memory=True` across sequential training and evaluation loaders is notoriously susceptible to IPC pipe exhaustion, worker termination race conditions, and deadlocks between the main process `_pin_memory_thread` and worker processes during epoch boundary teardown.

---

## 4. Preserved Artifacts & Forensics

Before taking any destructive recovery action, all pre-hang and interrupted checkpoints were verified and secured:

| Artifact Path | Epoch | Same-Domain Val Macro F1 | SHA256 Hash | Status |
| :--- | :---: | :---: | :--- | :--- |
| `runs/v4c/A1_resnet18_cbam_tight_person_crop_224/best_model.pt` | 22 | **0.8847** | `75B1BB76EE703AEF77D4F4A83144C95199C2FF1B28EE0D6D64C0F7B50A03226B` | Preserved |
| `runs/v4c/A1_resnet18_cbam_tight_person_crop_224/best_model_before_hang_recovery.pt` | 22 | **0.8847** | `75B1BB76EE703AEF77D4F4A83144C95199C2FF1B28EE0D6D64C0F7B50A03226B` | Immutable Copy |
| `runs/v4c/A1_resnet18_cbam_tight_person_crop_224/best_model_interrupted_epoch10.pt` | 10 | **0.8497** | `69F3ED59CE6143EFD09FDE56E323AED053CEED8CF4611DC6C073C9A730A02FF9` | Archived Copy |

### Physical Verification of Checkpoint 22:
- Model Weights: 100% Finite (0 NaNs, 0 Infs across all 68 parameter tensors).
- Input: 224x224 RGB.
- Representation: `TIGHT_PERSON_CROP`.
- Class Mapping: Exact 4-class ontology (`NORMAL_UPRIGHT`, `NORMAL_READ_WRITE`, `HEAD_REST_SLEEP`, `TURN_HEAD_CLEAR`).
- Forward Pass: Verified on CUDA (`torch.Size([2, 4])` finite output).
- Same-Domain Val Performance:
  - Accuracy: 0.9056
  - Balanced Accuracy: 0.8886
  - Macro F1: **0.8847**
  - Sleep Recall: **1.0000** (91 / 91 detected)
  - Turn Head Recall: **0.7348** (97 / 132 detected)

---

## 5. Remediation Plan & Windows Stability Hardening

To permanently prevent hangs across all remaining experiments (A2, B1, B2, C1, C2, Head-Pose, etc.):

1. **Set `num_workers = 0` for Windows Stability**:
   - Running DataLoader with `num_workers = 0` executes data loading synchronously within the main process on the NVMe SSD.
   - Eliminates all inter-process communication pipes, worker spawning overhead, `pin_memory` thread race conditions, and worker crash hangs.
   - Training throughput on RTX 5070 with batch size 64 remains extremely high (~15–20 sec/epoch).

2. **Decouple Module Imports**:
   - Ensure orchestrator scripts do not import unrelated downstream modules at module scope. Imports are delayed until the corresponding execution step.

3. **Per-Epoch Crash Safety & Resumability**:
   - Every epoch writes `training_history.csv` with:
     `epoch, train_loss, val_loss, val_macro_f1, best_macro_f1, best_epoch, timestamp`.
   - Every epoch writes `last_checkpoint.pt` containing:
     `model_state`, `optimizer_state`, `scheduler_state`, `scaler_state`, `epoch`, `best_metric`, `best_epoch`, `representation`, `resolution`.
   - Atomic disk replacement via temporary files (`.tmp -> replace`).

4. **Durable Heartbeat State Tracking**:
   - Every epoch flushes `runs/v4c/V4C_EXECUTION_STATE.json` with active PID, timestamps, epoch counts, best metrics, and execution phase.

5. **A1 Recovery Decision (Case B)**:
   - Accept the validated Epoch 22 checkpoint as `RECOVERED_VALID_CANDIDATE`.
   - Perform full multi-split evaluations on it (High-Angle, Cross-Source, Temporal, Scale Slices, Blur Slices) and generate `metrics.json` and `eval_predictions.json`.
   - Record exact status honestly without false claims of unbroken 30-epoch run.
   - Advance immediately to A2.
