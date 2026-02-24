"""Compatibility runner shim used by `app.py`.

Provides a small, well-tested PipelineRunner that imports the centralized
logging helper so UI and CLI runs share configuration.
"""
from __future__ import annotations
import subprocess
import os
import sys
import threading
import queue
import time
import json
import logging
from contextlib import suppress
from pathlib import Path
from typing import Generator, Dict, Any, Tuple

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


class PipelineRunner:
    def __init__(self, project_root: str, use_pixi: bool = True):
        self.project_root = Path(project_root)
        self.use_pixi = use_pixi

    def _get_execution_context(self) -> Tuple[list, str]:
        if self.use_pixi:
            return ["pixi", "run", "python"], str(self.project_root)
        return [sys.executable], str(self.project_root)

    def run(self, config_path: str, dataset_name: str) -> Generator[str, None, None]:
        prefix, cwd = self._get_execution_context()
        cmd = prefix + [str(self.project_root / "scripts" / "gsas_complete_pipeline_nomain.py"), "--config", str(config_path), "--dataset", dataset_name]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        scripts_path = str((self.project_root / "scripts").resolve())
        current_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [scripts_path, current_pp]))

        try:
            p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
        except OSError:
            logger.exception("pipeline_spawn_failed", extra={"cmd": cmd})
            yield "\n[ERROR] Pipeline failed to start\n"
            return

        assert p.stdout is not None
        try:
            for line in p.stdout:
                yield line
        finally:
            with suppress(Exception):
                p.stdout.close()

        p.wait()
        if p.returncode != 0:
            logger.error("pipeline_exit_nonzero", extra={"returncode": p.returncode})
            yield f"\n[ERROR] Pipeline failed with exit code {p.returncode}\n"
        else:
            logger.info("pipeline_exit_success")
            yield "\n[INFO] Pipeline finished successfully\n"
