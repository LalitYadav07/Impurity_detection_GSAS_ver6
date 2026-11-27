"""Bounded, content-free measurements of a single result-loading transaction.

These are client wall-clock measurements. Concurrent stages overlap, and a
server-side presentation commit does not measure the browser's first paint.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
import uuid


_active: ContextVar[tuple | None] = ContextVar("radar_result_loading", default=None)
_subject: ContextVar[str] = ContextVar("radar_result_loading_subject", default="")
_LABELS = {
    "canonical_wait": "Wait for another result read",
    "transfer_wait": "Wait for transfer capacity",
    "job_metadata": "Read job identity",
    "associations": "Resolve output associations",
    "collection_metadata": "Read collection identities",
    "dataset_metadata": "Read file identity",
    "response_headers": "Wait for file response",
    "response_body": "Transfer file contents",
    "archive_extract": "Unpack complete results",
    "observation_check": "Verify observation and masks",
    "preview_parse": "Read and rank preview metadata",
    "preview_plot": "Prepare preview fit",
    "cache_publish": "Commit complete local cache",
    "export": "Result delivery",
    "view_prepare": "Prepare result view",
    "view_transport": "Queue plot for browser",
    "shared_transfer_wait": "Wait for the same file already being read",
    "shared_transfer_copy": "Reuse verified file bytes",
}


class ResultLoadTrace:
    """Thread-safe timing receipt; never contains file contents or error text."""

    MAX_EVENTS = 192

    def __init__(self, job_id: str, *, clock=time.monotonic):
        self.identifier = uuid.uuid4().hex
        self.job_id = str(job_id)
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.clock = clock
        self.started = clock()
        self.lock = threading.RLock()
        self.events: list[dict] = []
        self.active: dict[int, dict] = {}
        self.aggregates: dict[tuple, dict] = {}
        self.serial = 0
        self.milestones: dict[str, float] = {}
        self.dropped = 0
        self.transfers = None

    @contextmanager
    def activate(self, branch: str):
        token = _active.set((self, branch))
        try:
            yield self
        finally:
            _active.reset(token)

    def mark(self, name: str):
        with self.lock:
            self.milestones.setdefault(name, max(0., self.clock() - self.started))

    @contextmanager
    def measure(self, stage: str, *, branch="", subject=""):
        event = dict(stage=stage, branch=branch, subject=subject,
                     start_seconds=max(0., self.clock() - self.started),
                     duration_seconds=None, status="running", bytes=0)
        with self.lock:
            self.serial += 1
            serial = self.serial
            self.active[serial] = event
            if len(self.events) < self.MAX_EVENTS:
                self.events.append(event)
            else:
                self.dropped += 1
        try:
            yield event
        except BaseException as exc:
            event["error_type"] = type(exc).__name__
            event["status"] = "failed"
            raise
        finally:
            with self.lock:
                event["duration_seconds"] = max(0., self.clock() - self.started - event["start_seconds"])
                if event["status"] == "running":
                    event["status"] = "complete"
                self.active.pop(serial)
                key = (branch, stage)
                total = self.aggregates.setdefault(key, dict(branch=branch, stage=stage,
                    duration_seconds=0., count=0, bytes=0, active=0, failed=0))
                total["duration_seconds"] += event["duration_seconds"]
                total["count"] += 1
                total["bytes"] += event["bytes"]
                total["failed"] += event["status"] == "failed"

    def snapshot(self) -> dict:
        with self.lock:
            now = max(0., self.clock() - self.started)
            events = [dict(event) for event in self.events]
            for event in events:
                if event["duration_seconds"] is None:
                    event["duration_seconds"] = max(0., now - event["start_seconds"])
            aggregates = {key: dict(value) for key,value in self.aggregates.items()}
            for event in self.active.values():
                key = (event["branch"], event["stage"])
                total = aggregates.setdefault(key, dict(branch=event["branch"], stage=event["stage"],
                    duration_seconds=0., count=0, bytes=0, active=0, failed=0))
                total["duration_seconds"] += max(0., now - event["start_seconds"])
                total["count"] += 1
                total["bytes"] += event["bytes"]
                total["active"] += 1
            return dict(schema="radar-result-loading/v1", trace_id=self.identifier,
                        job_id=self.job_id, started_utc=self.started_utc,
                        elapsed_seconds=now, milestones=dict(self.milestones),
                        events=events, aggregates=list(aggregates.values()), dropped_events=self.dropped,
                        timing_note="Client wall time; concurrent stages overlap. Browser paint is not measured.")

    def rows(self) -> list[dict]:
        return [dict(row, branch={"complete":"Complete files", "preview":"Fit preview",
                                 "display":"Presentation"}.get(row["branch"],row["branch"].title()),
                     stage=_LABELS.get(row["stage"],row["stage"]),
                     seconds=f"{row['duration_seconds']:.3f}",
                     transferred=f"{row['bytes'] / 1024:.1f} KiB" if row["bytes"] else "—",
                     status="Running" if row["active"] else "Failed" if row["failed"] else "Complete")
                for row in self.snapshot()["aggregates"]]

    def profile(self) -> dict:
        """Keep recoverable records compact; the local receipt has the timeline."""
        return {key:value for key,value in self.snapshot().items() if key != "events"}

    def persist(self, output_root: Path):
        """Best-effort diagnostics never affect scientific cache readiness."""
        temporary = None
        try:
            root = Path(output_root) / "result-loading"
            root.mkdir(parents=True, exist_ok=True)
            target = root / (self.identifier + ".json")
            temporary = target.with_suffix(".partial")
            temporary.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
            temporary.replace(target)
        except (OSError, TypeError, ValueError):
            pass
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


@contextmanager
def measure(stage: str, subject: str = ""):
    active = _active.get()
    if active is None:
        yield {"bytes": 0}
        return
    trace, branch = active
    with trace.measure(stage, branch=branch, subject=subject or _subject.get()) as event:
        yield event


@contextmanager
def subject(name: str):
    token = _subject.set(name)
    try:
        yield
    finally:
        _subject.reset(token)


@contextmanager
def transfer_slot(semaphore):
    with measure("transfer_wait"):
        semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


def submit_context(pool, callback, *args):
    """Propagate the transaction to each independent transfer worker."""
    return pool.submit(copy_context().run, callback, *args)


def shared_transfers():
    active = _active.get()
    return (active[0].transfers, _subject.get()) if active else (None, '')
