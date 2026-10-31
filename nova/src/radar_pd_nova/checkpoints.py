"""One background writer for immutable, coalesced recovery snapshots."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any
from .checkpoint_uploads import CheckpointUploadPending


class CoalescedCheckpoint:
    """Keep only the newest waiting snapshot while one upload is in flight.

    Capture runs on the caller's event loop, between controller mutations.
    Only the detached snapshot goes to a worker thread. Final draining keeps
    stop/restart recovery explicit, while upload latency never blocks admission.
    """

    def __init__(self, capture: Callable[[], Any], write: Callable[[Any], Any]):
        self.capture, self.write = capture, write
        self.pending: Any = None
        self.task: asyncio.Task | None = None
        self.error: Exception | None = None

    def offer(self) -> None:
        self.pending = self.capture()
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._run(), name="radar-watch-checkpoint")

    async def _run(self) -> None:
        while self.pending is not None:
            snapshot, self.pending = self.pending, None
            try:
                await asyncio.to_thread(self.write, snapshot)
                self.error = None
            except CheckpointUploadPending as exc:
                # Leave the newest waiting snapshot available for an explicit
                # later offer, rather than stacking bounded waits during drain.
                self.error = exc
                return
            except Exception as exc:
                # The controller already retries transient transport failures.
                # A subsequent offer retries the latest state, without a busy
                # retry loop or publishing an older state after a newer one.
                self.error = exc

    async def drain(self) -> None:
        self.offer()
        if self.task is not None:
            await self.task
        if self.error is not None:
            raise self.error
