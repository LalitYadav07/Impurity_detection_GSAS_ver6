"""Bounded scientific subprocesses with cleanup of their own process group."""
import os
import signal
import subprocess
import time


class ScientificCancelled(subprocess.SubprocessError):pass


def run_scientific(command, *, timeout, cancel_event=None, **kwargs):
    process=subprocess.Popen(command,start_new_session=os.name=='posix',**kwargs)
    try:
        if cancel_event is None:process.wait(timeout=timeout)
        else:
            deadline=time.monotonic()+timeout
            while process.poll() is None:
                if cancel_event.is_set():raise ScientificCancelled('Scientific operation cancelled')
                remaining=deadline-time.monotonic()
                if remaining<=0:raise subprocess.TimeoutExpired(command,timeout)
                try:process.wait(timeout=min(.1,remaining))
                except subprocess.TimeoutExpired:pass
    except (subprocess.TimeoutExpired,KeyboardInterrupt,ScientificCancelled):
        if os.name=='posix':
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        else:process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        finally:
            if os.name=='posix':
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        raise
    return subprocess.CompletedProcess(command,process.returncode)
