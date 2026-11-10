"""Bounded scientific subprocesses with cleanup of their own process group."""
import os
import signal
import subprocess
import time
import threading
from contextlib import contextmanager


class ScientificCancelled(subprocess.SubprocessError):pass


@contextmanager
def cancellation_signals(event):
    """Relay termination to owned nested workers before this process exits."""
    previous={}
    if threading.current_thread() is threading.main_thread():
        for name in (signal.SIGTERM,signal.SIGINT):
            previous[name]=signal.getsignal(name)
            signal.signal(name,lambda *_:event.set())
    try:
        yield
    finally:
        for name,handler in previous.items():signal.signal(name,handler)


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
