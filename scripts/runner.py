import subprocess
import os
import sys
import threading
import queue
import time
import json
from pathlib import Path
from typing import Generator, Optional, Dict, Any

class PipelineRunner:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.pixi_dir = self.project_root / "GSAS-II" / "pixi"
        
    def run(self, config_path: str, dataset_name: str) -> Generator[str, None, None]:
        """
        Runs the pipeline and yields log lines.
        """
        cmd = [
            "pixi", "run", "python", 
            str(self.project_root / "scripts" / "gsas_complete_pipeline_nomain.py"),
            "--config", str(config_path),
            "--dataset", dataset_name
        ]
        
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        # Start the process in the background
        process = subprocess.Popen(
            cmd,
            cwd=str(self.pixi_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            env=env
        )
        
        for line in process.stdout:
            yield line
            
        process.wait()
        if process.returncode != 0:
            yield f"\n[ERROR] Pipeline failed with exit code {process.returncode}\n"
        else:
            yield "\n[INFO] Pipeline finished successfully\n"

    def start_non_blocking(self, config_path: str, dataset_name: str, log_path: str = None) -> tuple[subprocess.Popen, queue.Queue]:
        """
        Starts the pipeline in the background and returns the process and a queue 
        that will be populated with stdout lines. Optionally mirrors logs to log_path.
        """
        cmd = [
            "pixi", "run", "python", 
            str(self.project_root / "scripts" / "gsas_complete_pipeline_nomain.py"),
            "--config", str(config_path),
            "--dataset", dataset_name
        ]
        
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        # Start the process
        process = subprocess.Popen(
            cmd,
            cwd=str(self.pixi_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1, # Line buffered
            env=env
        )
        
        q = queue.Queue()
        
        def enqueue_output(out, queue, log_file_path=None):
            f = None
            if log_file_path:
                try:
                    f = open(log_file_path, "a", encoding="utf-8")
                except Exception as e:
                    print(f"[ERROR] Could not open log file {log_file_path}: {e}")
            
            for line in iter(out.readline, ''):
                queue.put(line)
                if f:
                    try:
                        f.write(line)
                        f.flush()
                    except:
                        pass
            
            if f:
                f.close()
            out.close()
            
        t = threading.Thread(target=enqueue_output, args=(process.stdout, q, log_path))
        t.daemon = True # Thread dies with the program
        t.start()
        
        return process, q

def watch_events(event_file: str) -> Generator[Dict, None, None]:
    """
    Watches a jsonl event file and yields new events.
    """
    if not os.path.exists(event_file):
        # Wait up to 10 seconds for file to appear
        for _ in range(20):
            if os.path.exists(event_file):
                break
            time.sleep(0.5)
        else:
            return

    with open(event_file, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
