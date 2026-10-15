"""Launch the ExamGuard server without inheriting launcher/test pipe handles."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reconfigure", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    logs = root / ".runtime" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(root / "scripts" / "run_examguard.py"), "--port", str(args.port)]
    if args.reconfigure:
        command.append("--reconfigure")

    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
    with (logs / "startup.stdout.log").open("ab", buffering=0) as stdout_log, (
        logs / "startup.stderr.log"
    ).open("ab", buffering=0) as stderr_log:
        process = subprocess.Popen(
            command,
            cwd=str(root),
            stdin=subprocess.DEVNULL,
            stdout=stdout_log,
            stderr=stderr_log,
            close_fds=True,
            creationflags=flags,
        )
    print(process.pid)


if __name__ == "__main__":
    main()
