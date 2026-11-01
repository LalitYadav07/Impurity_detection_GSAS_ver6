"""Bounded native process lifecycle. Each launch owns a fresh process group."""
import asyncio,os,signal

async def stop_process(process,grace=3):
    if process is None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:await asyncio.wait_for(process.wait(),timeout=grace)
    except asyncio.TimeoutError:pass
    # Descendants can remain after their parent exits or ignores SIGTERM.
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    await process.wait()

async def wait_process(process,timeout):
    try:return await asyncio.wait_for(process.wait(),timeout=timeout)
    except (asyncio.TimeoutError,asyncio.CancelledError):
        await stop_process(process)
        raise

async def spawn_process(*args,**kwargs):
    task=asyncio.create_task(asyncio.create_subprocess_exec(*args,**kwargs,start_new_session=True))
    try:return await asyncio.shield(task)
    except asyncio.CancelledError:
        process=await task
        await stop_process(process)
        raise
