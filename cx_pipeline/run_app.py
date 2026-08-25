from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from cx_pipeline.app.db import init_db
from cx_pipeline.app.shutdown import mark_interrupted


ROOT = Path(__file__).resolve().parent.parent


def _start(name: str, args: list[str]) -> subprocess.Popen[bytes]:
    print(f"Starting {name}: {' '.join(args)}", flush=True)
    return subprocess.Popen(args, cwd=ROOT, start_new_session=True)


def _kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def main() -> None:
    init_db()
    children = [
        _start(
            "api",
            [
                sys.executable,
                "-m",
                "uvicorn",
                "cx_pipeline.app.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8088",
            ],
        ),
        _start(
            "dashboard",
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "cx_pipeline/app/dashboard.py",
                "--server.port",
                "8502",
                "--server.address",
                "127.0.0.1",
            ],
        ),
    ]

    def kill_switch(signum: int, _: object) -> None:
        print(f"\nSignal {signum} received. Killing API and dashboard immediately.", flush=True)
        try:
            mark_interrupted(reason=f"launcher signal {signum} kill switch")
        except Exception as exc:  # noqa: BLE001
            print(f"Kill switch DB cleanup failed: {exc}", file=sys.stderr, flush=True)
        for child in children:
            _kill_process_tree(child)
        os._exit(130 if signum == signal.SIGINT else 143)

    signal.signal(signal.SIGINT, kill_switch)
    signal.signal(signal.SIGTERM, kill_switch)

    print("API: http://127.0.0.1:8088")
    print("Dashboard: http://127.0.0.1:8502")
    print("Press Ctrl-C to kill all app processes immediately.", flush=True)

    while True:
        for child in children:
            if child.poll() is not None:
                kill_switch(signal.SIGTERM, None)
        time.sleep(1)


if __name__ == "__main__":
    main()
