"""Persistent, idempotent dispatcher for approved per-scan continuation jobs.

The local runner invokes the same CLI that an NDIP Galaxy wrapper can invoke.
It never calls the model, infers a new recipe, or modifies incoming GPX files.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

from .compiler import compile_script
from .contracts import plan_hash
from .execution import write_json


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific_environment(gsasii_root=None):
    """The scientific worker never inherits any model credential."""
    allowed = {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT", "WINDIR",
               "RADAR_GPX_DOCKER_WORKDIR", "RADAR_GPX_DOCKER_IMAGE"}
    env = {k: v for k, v in os.environ.items() if k in allowed}
    env.update(PYTHONPATH=str(Path(__file__).resolve().parents[1]), PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    if gsasii_root:
        env["GSASII_ROOT"] = str(gsasii_root)
    return env


class RecipeQueue:
    def __init__(self, root, *, python, gsasii_root=None, timeout=300):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.python, self.gsasii_root, self.timeout = str(python), gsasii_root, timeout
        self.manifest = self.root / "jobs.json"
        self.jobs = json.loads(self.manifest.read_text()) if self.manifest.exists() else {}
        for job in self.jobs.values():
            if job["status"] == "running":
                job.update(status="interrupted", error="Runner restarted; explicitly retry this job")
        self._save()

    def _save(self):
        write_json(self.manifest, self.jobs)

    def enqueue(self, source, plan, approval, *, scan_id=None):
        source = Path(source).resolve()
        source_hash = file_hash(source)
        # NDIP supplies a stable run/bank/revision or HDA identity. A local
        # path is an explicit fallback, so identical bytes in two different
        # acquisitions are not silently collapsed into one scan.
        scan_id = str(scan_id or source)
        uid = hashlib.sha256((scan_id + "\0" + source_hash + plan_hash(plan)).encode()).hexdigest()[:24]
        if uid in self.jobs:
            job = self.jobs[uid]
            if job["status"] == "completed" and file_hash(job["output"]) != job["output_sha256"]:
                raise ValueError("A completed advanced GPX was modified; review its active revision")
            return job
        directory = self.root / uid
        directory.mkdir()
        # Compiling also verifies that the approved plan has not been edited.
        script = directory / "refine.py"
        compile_script(plan, approval, script)
        write_json(directory / "plan.json", plan)
        write_json(directory / "approval.json", approval)
        self.jobs[uid] = {"id": uid, "status": "queued", "scan_id": scan_id, "source": str(source), "source_sha256": source_hash,
                          "plan_sha256": plan_hash(plan), "script": str(script),
                          "queued_utc": datetime.now(timezone.utc).isoformat(), "attempts": 0}
        self._save()
        return self.jobs[uid]

    def run_next(self, uid=None):
        job = self.jobs.get(uid) if uid else next((j for j in self.jobs.values() if j["status"] == "queued"), None)
        if job is None:
            return None
        if job["status"] != "queued":
            raise ValueError("Only queued jobs may execute; retries require an explicit new attempt")
        if file_hash(job["source"]) != job["source_sha256"]:
            job.update(status="failed", error="Starting GPX changed after submission")
            self._save(); return job
        job.update(status="running", attempts=job["attempts"] + 1)
        self._save()
        attempt = self.root / job["id"] / ("attempt-" + str(job["attempts"]) + "-" + uuid.uuid4().hex[:8])
        attempt.mkdir()
        output = attempt / "polished.gpx"
        env = scientific_environment(self.gsasii_root)
        command = [self.python, job["script"], "--source-gpx", job["source"], "--output-gpx", str(output)]
        job["command"] = command
        try:
            with (attempt / "stdout.log").open("w") as stdout, (attempt / "stderr.log").open("w") as stderr:
                process = subprocess.run(command, cwd=attempt, env=env, stdout=stdout, stderr=stderr, timeout=self.timeout)
            receipt_path = output.with_suffix(".execution.json")
            receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
            if receipt.get("source_sha256") != job["source_sha256"]:
                raise ValueError("The executed source does not match the queued immutable GPX")
            if process.returncode or receipt.get("status") != "completed_needs_scientific_review":
                raise ValueError(receipt.get("error") or "Advanced refinement failed; inspect captured logs")
            job.update(status="completed", scientific_status="needs_review", output=str(output),
                       output_sha256=file_hash(output), execution=str(receipt_path))
        except Exception as exc:
            job.update(status="failed", error=str(exc), error_type=type(exc).__name__)
        job["attempt_directory"] = str(attempt)
        job["finished_utc"] = datetime.now(timezone.utc).isoformat()
        self._save()
        return job

    def retry(self, uid):
        job = self.jobs[uid]
        if job["status"] not in {"failed", "interrupted"}:
            raise ValueError("Only failed/interrupted jobs can be retried")
        job.update(status="queued")
        self._save()


class StableArrivals:
    """Only complete, unchanged GPX arrivals qualify, including out-of-order scans."""
    def __init__(self):
        self.seen = {}

    def poll(self, folder):
        ready = []
        for path in sorted(Path(folder).glob("*.gpx")):
            if not path.is_file() or path.is_symlink():
                continue
            stat = path.stat()
            stamp = (stat.st_size, stat.st_mtime_ns)
            if stamp[0] and self.seen.get(str(path)) == stamp:
                ready.append(path)
            self.seen[str(path)] = stamp
        return ready
