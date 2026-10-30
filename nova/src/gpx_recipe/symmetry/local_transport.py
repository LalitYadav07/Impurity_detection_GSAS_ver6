"""Local adapter with durable launch tickets and conservative process recovery."""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid


class LocalTransport:
    def __init__(self,command,*,environment=None):
        if not command or any(not isinstance(v,str) or not v for v in command):raise ValueError('Configure a scientific Python command')
        self.command=list(command);self.environment=dict(environment or {})
        if set(self.environment)-{'GSASII_ROOT','RADAR_GPX_DOCKER_IMAGE','PYTHONPATH'}:raise ValueError('Only scientific runtime settings belong in a launch ticket')

    def submit(self,key,directory):
        directory=Path(directory);ticket=directory/'launch.json'
        if ticket.exists():
            recovered=self.recover(key,directory)
            if recovered['state']=='running':return recovered['handle']
            raise ValueError('A launch ticket already exists; recover it rather than launching again')
        token=uuid.uuid4().hex
        body=dict(key=key,token=token,command=self.command,environment=self.environment)
        with ticket.open('x') as f:json.dump(body,f);f.flush();os.fsync(f.fileno())
        # The monitor persists its own PID before starting the native child.
        with (directory/'monitor.log').open('ab') as log:
            from ..jobs import scientific_environment
            process=subprocess.Popen([sys.executable,'-m','gpx_recipe.symmetry.launch',str(directory),token],env=scientific_environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        return dict(key=key,token=token,pid=process.pid)

    @staticmethod
    def alive(handle):
        try:
            result=subprocess.run(['ps','-p',str(int(handle['pid'])),'-o','args='],capture_output=True,text=True,timeout=3)
            return result.returncode==0 and handle['token'] in result.stdout and 'gpx_recipe.symmetry.launch' in result.stdout
        except (OSError,subprocess.SubprocessError,KeyError,ValueError):return False

    def recover(self,key,directory):
        directory=Path(directory)
        if (directory/'result.json').exists():return dict(state='finished')
        if (directory/'cancelled.json').exists():return dict(state='cancelled')
        if not (directory/'launch.json').exists():return dict(state='absent')
        ticket=json.loads((directory/'launch.json').read_text())
        ready=directory/'launch-ready.json'
        if ticket['key']!=key:return dict(state='unknown',reason='Launch key differs')
        if not ready.exists():return dict(state='unknown',reason='Launch ticket exists but process acknowledgment is not yet available')
        handle=json.loads(ready.read_text())
        if handle.get('token')!=ticket['token']:return dict(state='unknown',reason='Process generation differs')
        return dict(state='running',handle=handle) if self.alive(handle) else dict(state='unknown',reason='Native monitor disappeared without a terminal receipt')

    def poll(self,handle,directory):
        if (Path(directory)/'result.json').exists():return dict(state='finished')
        if self.alive(handle):return dict(state='running',handle=handle)
        return self.recover(handle['key'],directory)

    def cancel(self,handle,directory):
        if not self.alive(handle):return False
        try:os.kill(int(handle['pid']),signal.SIGTERM)
        except ProcessLookupError:return False
        # Cancellation is confirmed on a later poll, after process-group cleanup.
        return False
