"""Optional first-run source chooser. Ordinary Auto startup requires no source editing."""
from pathlib import Path
import sys
from urllib.parse import quote, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import tkinter as tk
    from tkinter import filedialog, simpledialog, messagebox
    from src.orchestration.adaptive_runtime import LOCAL, choose_camera, discover_cameras, save_json
    from src.video.factory import create_video_source
    from src.video.source_settings import protect_rtsp_url
    import msvcrt
    LOCAL.mkdir(exist_ok=True)
    lock = (LOCAL / "runtime.lock").open("a+b")
    lock.seek(0)
    if not lock.read(1):
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit("Stop ExamGuard before changing the physical/video source.")
    root = tk.Tk()
    root.withdraw()
    try:
        kind = simpledialog.askstring("ExamGuard source", "Source: webcam, video_file, or rtsp", initialvalue="webcam", parent=root)
        if kind is None:
            return
        kind = kind.strip().lower()
        if kind == "webcam":
            selected = choose_camera(discover_cameras())
            settings = {"type": kind, "camera": selected}
        elif kind == "video_file":
            path = filedialog.askopenfilename(title="Choose authorized replay video", parent=root)
            if not path:
                return
            settings = {"type": kind, "path": str(Path(path).resolve()), "name": Path(path).name}
        elif kind == "rtsp":
            url = simpledialog.askstring("Authorized RTSP", "RTSP address (no network scan):", parent=root)
            if not url:
                return
            parts = urlsplit(url)
            if parts.scheme.lower() not in ("rtsp", "rtsps") or not parts.hostname:
                raise ValueError("A valid rtsp:// or rtsps:// camera address is required.")
            username = simpledialog.askstring("Camera credentials", "Username (leave empty if not required):", parent=root)
            if username:
                password = simpledialog.askstring("Camera credentials", "Password:", show="*", parent=root)
                if password is None:
                    return
                host = parts.netloc.rsplit("@", 1)[-1]
                url = urlunsplit((parts.scheme, f"{quote(username, safe='')}:{quote(password, safe='')}@{host}", parts.path, parts.query, parts.fragment))
            settings = {"type": kind, "path": protect_rtsp_url(url), "name": "Authorized IP camera"}
        else:
            raise ValueError("Choose webcam, video_file or rtsp.")
        if kind != "webcam":
            source = create_video_source(kind, settings["path"], source_id="selected_source", extra_config={"reconnect_attempts": 1})
            try:
                if not source.open() or source.read() is None:
                    raise RuntimeError("Cannot acquire a frame. Check the file/address, camera permission or credentials.")
                settings["mode"] = dict(width=source.width, height=source.height, fps=source.fps, fourcc=None,
                                        measured_capture_fps=None, backend=kind)
            finally:
                source.release()
        save_json(LOCAL / "source-settings.json", settings)
        messagebox.showinfo("ExamGuard", "Source saved. Start ExamGuard Vision to use it. Authentication and session start remain required.", parent=root)
    except Exception as exc:
        messagebox.showerror("ExamGuard source setup", str(exc), parent=root)
    finally:
        root.destroy()
        lock.close()


if __name__ == "__main__":
    main()
