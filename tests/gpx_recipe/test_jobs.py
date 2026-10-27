import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from gpx_recipe.jobs import RecipeQueue, StableArrivals
from gpx_recipe.contracts import plan_hash


def recipe():
    p = {"schema": "radar-gpx-recipe/v1", "model_signature": "a", "request": "test", "assumptions": [], "questions": [], "unsupported": [],
         "stages": [{"id": "background", "title": "Background", "cycles": 4, "free": [{"kind": "background", "histogram": "h0"}], "constraints": []}]}
    return p, {"approved_plan_sha256": plan_hash(p), "model_signature": "a"}


def test_duplicate_arrivals_and_distinct_scan_identity(tmp_path):
    source = tmp_path / "one.gpx"; source.write_bytes(b"same product bytes")
    p, a = recipe(); q = RecipeQueue(tmp_path / "queue", python=sys.executable)
    first = q.enqueue(source, p, a, scan_id="run1-bank1")
    assert q.enqueue(source, p, a, scan_id="run1-bank1")["id"] == first["id"]
    assert q.enqueue(source, p, a, scan_id="run2-bank1")["id"] != first["id"]
    assert len(q.jobs) == 2


def test_changed_queued_source_is_not_executed(tmp_path, monkeypatch):
    source = tmp_path / "one.gpx"; source.write_bytes(b"old")
    p, a = recipe(); q = RecipeQueue(tmp_path / "queue", python=sys.executable)
    j = q.enqueue(source, p, a)
    source.write_bytes(b"new")
    monkeypatch.setattr("gpx_recipe.jobs.subprocess.run", lambda *a, **k: pytest.fail("Changed input was executed"))
    assert q.run_next(j["id"])["status"] == "failed"


def test_restart_does_not_silently_repeat_running_job(tmp_path):
    source = tmp_path / "one.gpx"; source.write_bytes(b"GPX")
    p, a = recipe(); q = RecipeQueue(tmp_path / "queue", python=sys.executable)
    job = q.enqueue(source, p, a); job["status"] = "running"; q._save()
    restarted = RecipeQueue(tmp_path / "queue", python=sys.executable)
    assert restarted.jobs[job["id"]]["status"] == "interrupted"
    restarted.retry(job["id"])
    assert restarted.jobs[job["id"]]["status"] == "queued"


def test_arrival_waits_for_stable_complete_file(tmp_path):
    arrivals = StableArrivals()
    path = tmp_path / "scan.gpx"; path.write_bytes(b"initial")
    (tmp_path / "scan.gpx.partial").write_bytes(b"unfinished")
    assert arrivals.poll(tmp_path) == []
    path.write_bytes(b"growing data")
    assert arrivals.poll(tmp_path) == []
    assert arrivals.poll(tmp_path) == [path]
    (tmp_path / "earlier.gpx").write_bytes(b"late arrival")
    assert arrivals.poll(tmp_path) == [path]
    assert len(arrivals.poll(tmp_path)) == 2
