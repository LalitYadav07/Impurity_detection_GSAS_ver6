import asyncio
import threading

import pytest

from radar_pd_nova.checkpoints import CoalescedCheckpoint


def test_upload_coalesces_newer_states_without_blocking_work_or_reordering():
    entered, release = threading.Event(), threading.Event()
    current = [0]
    saved = []
    def write(snapshot):
        if snapshot == 0:
            entered.set()
            assert release.wait(5)
        saved.append(snapshot)
    async def run():
        worker = CoalescedCheckpoint(lambda: current[0], write)
        worker.offer()
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            for n in range(1, 11):
                current[0] = n
                worker.offer()
            assert saved == []
        finally:
            release.set()
        await worker.drain()
        assert saved == [0, 10]
    asyncio.run(run())


def test_failed_upload_is_visible_and_retry_uses_latest_state():
    current, saved = [0], []
    def write(snapshot):
        if snapshot == 0:
            raise RuntimeError('temporary upload failure')
        saved.append(snapshot)
    async def run():
        worker = CoalescedCheckpoint(lambda: current[0], write)
        worker.offer()
        await worker.task
        assert isinstance(worker.error, RuntimeError)
        current[0] = 2
        await worker.drain()
        assert saved == [2] and worker.error is None
    asyncio.run(run())


def test_final_checkpoint_failure_is_not_reported_as_durable():
    def fail(_): raise RuntimeError('unavailable')
    async def run():
        worker = CoalescedCheckpoint(lambda: 1, fail)
        with pytest.raises(RuntimeError, match='unavailable'):
            await worker.drain()
    asyncio.run(run())
