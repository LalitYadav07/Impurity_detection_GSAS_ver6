"""Local launch monitor: fixed command, bounded native process, durable receipt."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ..execution import write_json


def cleanup(process):
    try:
        if os.name=='posix':os.killpg(process.pid,signal.SIGTERM)
        else:process.terminate()
    except ProcessLookupError:pass
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:process.kill();process.wait()
    finally:
        if os.name=='posix':
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass


def main(directory,token):
    root=Path(directory).resolve();ticket=json.loads((root/'launch.json').read_text());job=json.loads((root/'job.json').read_text())
    if ticket['token']!=token or ticket['key']!=job['id']:raise ValueError('Launch identity changed')
    write_json(root/'launch-ready.json',dict(key=job['id'],token=token,pid=os.getpid()))
    cancelled=False
    def stop(*_):
        nonlocal cancelled
        cancelled=True
    # A signal only sets intent. Raising during Popen construction can lose
    # the newly created process before its handle is assigned and orphan it.
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    response=dict(schema='radar-symmetry-trial/v1',id=job['id'],protocol_sha256=job['protocol_sha256'],llm_calls=0)
    try:
        from ..jobs import scientific_environment
        env={**scientific_environment(),**ticket['environment']}
        env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',RADAR_GPX_DOCKER_WORKDIR=str(root))
        with (root/'native.log').open('w') as log:
            process=subprocess.Popen(ticket['command']+[str(root/'worker.py')],env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name=='posix')
            deadline=time.monotonic()+job['protocol']['policy']['timeout_seconds']
            while process.poll() is None and not cancelled and time.monotonic()<deadline:time.sleep(.04)
            if cancelled:
                cleanup(process)
                if not (root/'result.json').exists():write_json(root/'cancelled.json',dict(id=job['id'],cleanup_completed=True))
                return
            if process.poll() is None:
                cleanup(process);response.update(status='timed_out',reason='Declared trial walltime exhausted')
            else:response.update(status='technical_failure',reason='Native process exited without a terminal receipt',return_code=process.returncode)
    except Exception as exc:response.update(status='technical_failure',reason=str(exc))
    if not (root/'result.json').exists():write_json(root/'result.json',response)


if __name__=='__main__':main(*sys.argv[1:])
