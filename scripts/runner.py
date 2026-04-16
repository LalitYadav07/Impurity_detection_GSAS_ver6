"""Pipeline runner helper used by `app.py`.

Provides blocking and non-blocking pipeline execution helpers with
centralized logging setup.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import psutil

try:
    from logging_config import configure_logging
except Exception:
    try:
        from scripts.logging_config import configure_logging
    except Exception:
        configure_logging = None

if configure_logging:
    try:
        configure_logging()
    except Exception:
        logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def stop_process_tree(process: subprocess.Popen, timeout: float = 5.0) -> None:
    """Terminate a pipeline process and any child workers it spawned."""
    if process is None:
        return
    if process.poll() is not None:
        return

    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
    except Exception:
        children = []

    for child in children:
        with suppress(Exception):
            child.terminate()

    with suppress(Exception):
        process.terminate()

    gone = []
    alive = []
    if children:
        with suppress(Exception):
            gone, alive = psutil.wait_procs(children, timeout=timeout)

    for child in alive:
        with suppress(Exception):
            child.kill()

    try:
        process.wait(timeout=timeout)
    except Exception:
        with suppress(Exception):
            process.kill()


class PipelineRunner:
    def __init__(self, project_root: str, use_pixi: bool = True):
        self.project_root = Path(project_root)
        self.use_pixi = use_pixi

    def _get_execution_context(self) -> Tuple[list[str], str]:
        if self.use_pixi:
            return ["pixi", "run", "python"], str(self.project_root)
        return [sys.executable], str(self.project_root)

    def run(self, config_path: str, dataset_name: str) -> Generator[str, None, None]:
        prefix, cwd = self._get_execution_context()
        cmd = prefix + [
            str(self.project_root / "scripts" / "gsas_complete_pipeline_nomain.py"),
            "--config",
            str(config_path),
            "--dataset",
            dataset_name,
        ]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        scripts_path = str((self.project_root / "scripts").resolve())
        current_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [scripts_path, current_pp]))

        try:
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
        except OSError:
            logger.exception("pipeline_spawn_failed", extra={"cmd": cmd})
            yield "\n[ERROR] Pipeline failed to start\n"
            return

        assert process.stdout is not None
        try:
            for line in process.stdout:
                yield line
        finally:
            with suppress(Exception):
                process.stdout.close()

        process.wait()
        if process.returncode != 0:
            logger.error("pipeline_exit_nonzero", extra={"returncode": process.returncode})
            yield f"\n[ERROR] Pipeline failed with exit code {process.returncode}\n"
        else:
            logger.info("pipeline_exit_success")
            yield "\n[INFO] Pipeline finished successfully\n"

    def start_non_blocking(self, config_path: str, dataset_name: str, log_path: Optional[str] = None):
        prefix, cwd = self._get_execution_context()
        cmd = prefix + [
            str(self.project_root / "scripts" / "gsas_complete_pipeline_nomain.py"),
            "--config",
            str(config_path),
            "--dataset",
            dataset_name,
        ]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        scripts_path = str((self.project_root / "scripts").resolve())
        current_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [scripts_path, current_pp]))

        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        output_queue: queue.Queue[str] = queue.Queue()

        def _reader(out, q_, path=None):
            log_file = None
            if path:
                try:
                    log_file = open(path, "a", encoding="utf-8")
                except Exception:
                    logger.exception("log_open_failed", extra={"path": path})
                    log_file = None
            try:
                if out is None:
                    return
                for line in iter(out.readline, ""):
                    q_.put(line)
                    if log_file:
                        try:
                            log_file.write(line)
                            log_file.flush()
                        except Exception:
                            logger.exception("log_write_failed")
                            log_file.close()
                            log_file = None
            finally:
                if log_file:
                    with suppress(Exception):
                        log_file.close()
                with suppress(Exception):
                    out.close()

        thread = threading.Thread(target=_reader, args=(process.stdout, output_queue, log_path), daemon=True)
        thread.start()
        return process, output_queue


def watch_events(event_file: str) -> Generator[Dict[str, Any], None, None]:
    if not os.path.exists(event_file):
        for _ in range(20):
            if os.path.exists(event_file):
                break
            time.sleep(0.5)
        else:
            return

    with open(event_file, "r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                time.sleep(0.1)
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.debug("malformed_event_line", extra={"preview": line[:200]})
                continue
