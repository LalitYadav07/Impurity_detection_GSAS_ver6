"""Bounded scientific subprocesses with cleanup of their own process group."""
import os
import signal
import subprocess


def run_scientific(command, *, timeout, **kwargs):
    process=subprocess.Popen(command,start_new_session=os.name=='posix',**kwargs)
    try:
        process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired,KeyboardInterrupt):
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
