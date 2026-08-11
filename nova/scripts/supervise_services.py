#!/usr/bin/env python3
"""Keep the two NOVA web services alive inside a Galaxy interactive job."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass
class Service:
    name: str
    command: tuple[str, ...]
    process: subprocess.Popen[bytes] | None = None
    restarts: int = 0

    def start(self) -> None:
        self.process = subprocess.Popen(self.command, start_new_session=True)
        print(
            f"[radar-pd-nova] started {self.name} pid={self.process.pid}",
            flush=True,
        )

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=10)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=5)


SERVICES = [
    Service("trame", ("/usr/local/bin/run_trame.sh",)),
    Service("nginx", ("/usr/local/bin/run_nginx.sh",)),
]
STOP_REQUESTED = False


def request_stop(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"[radar-pd-nova] received signal {signum}; stopping", flush=True)


def main() -> int:
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    for service in SERVICES:
        service.start()

    try:
        while not STOP_REQUESTED:
            for service in SERVICES:
                process = service.process
                if process is None:
                    service.start()
                    continue
                return_code = process.poll()
                if return_code is None:
                    continue
                service.restarts += 1
                delay = min(30, 2 ** min(service.restarts - 1, 5))
                print(
                    f"[radar-pd-nova] {service.name} exited rc={return_code}; "
                    f"restart={service.restarts} in {delay}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
                if not STOP_REQUESTED:
                    service.start()
            time.sleep(1)
    finally:
        for service in reversed(SERVICES):
            service.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
