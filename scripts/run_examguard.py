"""Default Auto launcher. Reuses the existing authenticated, session-bound live application."""
import argparse
from collections import deque
import logging
import os
from pathlib import Path
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.orchestration.adaptive_runtime import LOCAL, discover_cameras, prepare_auto, save_json, validate_model_execution, select_cadence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--reconfigure", action="store_true")
    parser.add_argument("--config", help="Advanced override; ASUS reference launcher remains available separately")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    LOCAL.mkdir(exist_ok=True)
    import msvcrt
    lock = (LOCAL / "runtime.lock").open("a+b")
    lock.seek(0)
    if not lock.read(1):
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit("ExamGuard is already initializing or running.")
    instance = uuid.uuid4().hex
    state = {"state": "INITIALIZING", "pid": os.getpid(), "instance": instance, "reason": "Checking hardware and camera"}
    status_path = LOCAL / "status.json"
    save_json(status_path, state)
    logs = LOCAL / "logs"
    logs.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(logs / "examguard-runtime.log", encoding="utf-8")],
                        format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    logger = logging.getLogger("examguard.auto")
    orch = None
    stop_file = LOCAL / "stop.signal"
    stop = threading.Event()
    def watch_stop():
        while not stop.wait(0.25):
            if stop_file.exists():
                stop.set()
                if orch is not None:
                    orch.stop_event.set()
    threading.Thread(target=watch_stop, daemon=True).start()
    try:
        from scripts.run_local_live_validation import LiveValidationOrchestrator, check_port_available
        if not check_port_available("127.0.0.1", args.port):
            raise RuntimeError(f"Port {args.port} occupied by another application; it was left untouched.")
        config_path, settings = prepare_auto(args.camera_index, args.reconfigure)
        logger.info("Selected inference device=%s; reason=%s; camera=%s; mode=%s", settings["hardware"]["device"],
                    settings["hardware"].get("fallback_reason"), settings["camera"]["name"], settings["mode"])
        if stop.is_set():
            return
        orch = LiveValidationOrchestrator(camera_index=settings["camera"]["index"], port=args.port,
                                          headless=True, config_path=str(args.config or config_path), session_bound_capture=True)
        if not orch.setup_pipeline():
            raise RuntimeError("Required AI pipeline initialization failed.")
        # Check identity before every session open, including after a USB unplug/replug.
        selected = settings["camera"]
        if orch.webcam_source.source_type == "webcam":
            orch.webcam_source.identity_check = lambda: any(c["id"] == selected["id"] and c["index"] == selected["index"] for c in discover_cameras())
        model_results = validate_model_execution(orch.pipeline.registry)
        if not settings["reused_verified_settings"]:
            settings["cadence"] = select_cadence(model_results, settings["hardware"], settings["mode"].get("measured_capture_fps") or settings["mode"]["fps"])
            import yaml
            cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
            cfg["cadence"] = settings["cadence"]
            cfg["performance"]["auto_ai_target_fps"] = settings["cadence"]["detector_hz"]
            Path(config_path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            scheduler = orch.pipeline.crop_scheduler
            scheduler.posture_hz = settings["cadence"]["posture_hz"]
            scheduler.headpose_hz = settings["cadence"]["headpose_hz"]
            scheduler.posture_interval = 1.0 / scheduler.posture_hz
            scheduler.headpose_interval = 1.0 / scheduler.headpose_hz
            orch.pipeline.macro_hz = settings["cadence"]["macro_hz"]
            orch.pipeline.macro_interval = 1.0 / orch.pipeline.macro_hz
            orch.pipeline.config["performance"]["auto_ai_target_fps"] = settings["cadence"]["detector_hz"]
        settings.update(verified=True, model_execution=model_results)
        save_json(LOCAL / "auto-settings.json", settings)
        state.update(state="DEGRADED" if settings["hardware"]["fallback_reason"] else "READY",
                     reason=settings["hardware"]["fallback_reason"], device=str(orch.pipeline.registry.device), models_verified=True,
                     detector_image_size=int(orch.pipeline.registry.detector.image_size),
                     loaded_config=str(Path(args.config or config_path).resolve()),
                     monitoring="IDLE_LOGIN_OR_START_SESSION_REQUIRED")
        orch.pipeline.runtime_status = state
        if stop.is_set():
            return
        if not orch.start_backend():
            raise RuntimeError("Local dashboard failed to initialize.")
        import psutil
        (LOCAL / "examguard.pid").write_text(str(os.getpid()), encoding="utf-8")
        save_json(LOCAL / "examguard-launch.json", dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
                  project_root=str(ROOT), command=str(ROOT / "scripts/run_examguard.py"), instance=instance, port=args.port))
        last_write = [0.0]
        warm_times = deque(maxlen=600)
        samples = deque(maxlen=600)
        def update(result):
            now = time.time()
            warm_times.append(now)
            # Includes decoded-frame age in the camera's latest-frame slot, not just the AI queue.
            samples.append(max(0.0, (now-result.timestamp_sec)*1000))
            if now - last_write[0] < 2:
                return
            last_write[0] = now
            import numpy as np
            import torch
            source = orch.webcam_source
            state.update(monitoring="ACTIVE", camera=dict(name=selected["name"], resolution=[source.width, source.height],
                         fourcc=getattr(source, "actual_fourcc", None), capture_fps=getattr(source, "observed_capture_fps", None), skipped_frames=getattr(source, "skipped_frames", 0)),
                         frames_processed=orch.total_frames_processed,
                         ai_fps=(len(warm_times)-1)/(warm_times[-1]-warm_times[0]) if len(warm_times)>1 else None,
                         latency_p50_ms=float(np.percentile(samples, 50)), latency_p95_ms=float(np.percentile(samples, 95)),
                         queue_depth=orch.pipeline.ingestion_queue.qsize,
                         skipped_frames=orch.pipeline.dropped_frames_count,
                         rss_mb=round(psutil.Process().memory_info().rss/2**20, 1),
                         peak_gpu_mb=round(torch.cuda.max_memory_allocated()/2**20, 1) if str(orch.pipeline.registry.device).startswith("cuda") else 0,
                         active_tracks=len(result.tracks), updated_at=now)
            save_json(status_path, state)
            if orch.total_frames_processed % 120 < 60 and len(warm_times) >= 120:
                # Reuse steady-state observations at the NEXT launch; no in-session retuning.
                settings["observed_runtime"] = {k: state[k] for k in ("ai_fps", "latency_p95_ms", "queue_depth", "rss_mb", "peak_gpu_mb")}
                save_json(LOCAL / "auto-settings.json", settings)
        save_json(status_path, state)
        def status_worker():
            while not stop.wait(2):
                if orch.capture_controller and not orch.capture_controller.is_active():
                    state.update(monitoring="IDLE_LOGIN_OR_START_SESSION_REQUIRED")
                    save_json(status_path, state)
        threading.Thread(target=status_worker, daemon=True).start()
        orch.run_live_loop(on_frame_fn=update)
    except Exception as exc:
        state.update(state="BLOCKED", reason=str(exc))
        logger.exception("ExamGuard runtime blocked")
        save_json(status_path, state)
        return 1
    finally:
        stop.set()
        if orch is not None:
            orch.shutdown()
        if state["state"] != "BLOCKED":
            state.update(state="STOPPED", monitoring="STOPPED")
        save_json(status_path, state)
        for filename in ("examguard.pid", "examguard-launch.json", "stop.signal"):
            (LOCAL / filename).unlink(missing_ok=True)
        lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
