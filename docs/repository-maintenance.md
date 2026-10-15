# Repository maintenance and retention

This cleanup was audited against `733c0724a46e69b149d1c10673d545fda1635ee2` on 16 October 2026. The baseline had 364 tracked files. Decisions checked Python imports (including dynamic-loader sites), configuration/model paths, CLI and Windows launcher references, test fixtures, dashboard templates/static imports, and documentation links. The complete path/import/reference inventory is retained locally with the maintenance verification record.

## Dependency-aware inventory

| Area | Role and retention decision |
|---|---|
| Root `.bat` wrappers, `scripts/windows/`, `run_examguard.py` | Supported setup/start/stop/source selection chain; retained. `.venv` and `.runtime` are generated separately on each machine. |
| `src/orchestration/` | Adaptive Runtime -> existing live orchestrator -> Stage2Pipeline -> model registry, source, tracker, association, fusion, evidence and session lifecycle. All runtime modules retained. |
| `src/detection/`, `tracking/`, `models/`, `fusion/`, `behavior/` | Detector/model adapters, lazy Ultralytics ByteTrack import, posture/headpose, temporal rules and event semantics. Retained without threshold, model-byte, scoring or tracking changes. |
| `src/api/`, `security/`, `persistence/`, `evidence/`, `video/`, `pilot/`, `analysis/` | API/WebSocket routes, HTML/static assets, auth/CSRF, encryption, storage, camera/video/RTSP sources, reference and legacy interfaces. Retained, including alternatives not on the default Auto path. |
| `configs/` | Runtime defaults, ASUS/pilot reference profiles, checkpoint manifest, mappings, ontology and training/split settings. Retained; an unreferenced generated packaging audit alone is untracked. ASUS deployment/environment manifests are historical references, not Auto configuration. |
| `models/` | Seven canonical checkpoints plus identity manifest. Four active runtime models, two fallback candidates, and one historical baseline remain. Two large files stay in Git LFS. |
| `tests/` | All existing tests, synthetic video, conflict index and dataset identity/statistics metadata retained. Added regression tests for secret-audit root/index/redaction behavior. |
| `tools/dataset/`, `research/`, `benchmark/`, `validation/` | Acquisition, annotation checks, research provenance, benchmarks and hardware/session validation retained even when they need external assets. Their prerequisites are documented, not silently removed. |
| `docs/` | Technical/security/evidence documents and academic results retained. README now gives the current Vietnamese setup path; detailed historical results stay in their original reports. |
| Local datasets and runtime outputs | No `datasets/` corpus exists in this checkout. No dataset or annotation was deleted/downloaded/relabelled. Ignored runtime/evidence/storage/backups/caches remain untouched. |

Normal inference uses the distributed checkpoints and runtime configurations, not a private dataset or the external `training.*` research package. See [testing prerequisites](testing.md) and [research lineage](research.md).

## Exact paths removed from tracking

**Every file below is preserved byte-for-byte on the owner PC.** Only Git tracking changed; future physical summaries are written to ignored `reports/validation/`. Their generators and the academic Markdown reports remain tracked. Existing Git history is preserved.

| Path | Why it is local-only |
|---|---|
| `tools/validation/canonical_physical_single_camera_benchmark.json` | Generated hardware benchmark output; only referenced by its writer. |
| `tools/validation/final_60m_soak_summary.json` | Generated real session/room/telemetry output; only referenced by its writer. |
| `tools/validation/physical_normal_baseline_results.json` | Generated physical participant/session summary; only referenced by its writer. |
| `tools/validation/physical_session_ab_workflow_results.json` | Generated one-run workflow checklist; only referenced by its writer. |
| `tools/validation/scale_and_dense_room_certification_results.json` | Generated benchmark/certification output; only referenced by its writer. |
| `configs/deployment/repository_packaging_audit.json` | Unreferenced generated packaging snapshot with outdated byte counts; canonical paths/hashes remain in `runtime_checkpoints.json` and the model integrity document. |
| `tools/dev/scratch_check_scb_sizes.py` | Unreferenced one-off remote archive-size probe; canonical dataset acquisition/audit tools remain. |
| `tools/dev/scratch_find_clean_crops.py` | Unreferenced temporary composite-scene crop-selection probe; does not define annotations or splits. |
| `tools/dev/scratch_inspect_kaggle.py` | Unreferenced one-off redirect inspection; canonical SCB acquisition remains. |
| `tools/dev/scratch_inspect_kaggle_zip.py` | Unreferenced ad hoc ZIP/YAML inspection; original annotations, mappings and acquisition/audit tools remain. |
| `tools/dev/scratch_inspect_remote_zips.py` | Unreferenced remote archive-listing experiment with fixed candidate URLs. |
| `tools/dev/scratch_inspect_roboflow.py` | Unreferenced HTML scraping experiment, not an operational dataset acquisition mechanism. |

No entry was removed solely because it was not directly imported by YOLO. The six probes have no import, CLI/documentation, configuration, fixture, or asset-path consumers in the baseline tree. They are not training datasets, loaders, or split definitions.

## Narrow code and environment changes

- Removed unused imports from five launcher/QA scripts after checking for external consumers; runtime statements and public callable signatures stay intact.
- Fixed `tools/dev/audit_repo_secrets.py`: resolve the repository root instead of `tools/`, inventory Git text rather than walking private runtime folders, support exact staged blobs, redact values, and return failure on findings. Regression tests exercise root selection, staged versus unstaged content, ignored data, and deleted paths.
- Redirected the five physical summary writers into `reports/validation/`, creating the output directory as needed. No physical script was executed during cleanup.
- Replaced the stale workstation `requirements.txt` freeze with a forwarding include. `requirements-runtime.txt` records the installed, tested direct dependency versions, including the previously implicit `requests` import used by the live orchestrator. PyTorch/torchvision CPU/CUDA selection remains in setup; no new runtime capability or dependency library was introduced.
- Replaced the mixed historical/current README, refreshed testing guidance, linked real upstream sources, qualified unsupported research claims, and removed session identifiers from two historical Markdown reports without changing their measurements.

Source cleanup does not prove first-run GUI behavior or accuracy on another computer. Verification must distinguish isolated dependency installation/software tests from physical camera/laptop acceptance; see [testing](testing.md).
