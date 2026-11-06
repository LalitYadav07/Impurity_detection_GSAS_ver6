"""Real process cleanup; no diffraction fit is simulated by this control."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
import time
import uuid

import pytest

from gpx_recipe.process import run_scientific,ScientificCancelled
from gpx_recipe.symmetry.compiler import native_environment
from test_preparation_batch import plan


@pytest.mark.skipif(os.name!='posix',reason='POSIX process-group cancellation control')
def test_cancelling_batch_reaps_its_owned_native_process_group(tmp_path):
    request=plan(tmp_path,count=2);request['workers']=1
    message=tmp_path/'request.json';message.write_text(json.dumps(request))
    child_pid=tmp_path/'child.pid';stop_child=tmp_path/'stop-owned-child';token='owned-preparation-test-'+uuid.uuid4().hex
    script=tmp_path/'owned_batch.py'
    script.write_text(textwrap.dedent('''
        import json,sys,threading
        from pathlib import Path
        from gpx_recipe.preparation.batch import execute
        from gpx_recipe.process import cancellation_signals,run_scientific,ScientificCancelled
        cancel=threading.Event()
        def runner(task,request,scratch,**kwargs):
            child="import os,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); deadline=time.monotonic()+60\\nwhile time.monotonic()<deadline and not Path(sys.argv[2]).exists(): time.sleep(.02)"
            try:
                run_scientific([sys.executable,'-c',child,sys.argv[2],sys.argv[5],sys.argv[3]],
                    timeout=kwargs['timeout'],cancel_event=cancel)
            except ScientificCancelled:
                return dict(state='cancelled',reason='Owned native child stopped')
            return dict(state='failed',reason='Unexpected natural completion')
        with cancellation_signals(cancel):
            result=execute(json.loads(Path(sys.argv[1]).read_text()),cancel=cancel,runner=runner)
        Path(sys.argv[4]).write_text(json.dumps(result))
    '''))
    stopped=threading.Event();log=tmp_path/'outer.log';result=tmp_path/'result.json';pid=None
    def outer():
        with log.open('w') as stream:
            return run_scientific([sys.executable,str(script),str(message),str(child_pid),token,str(result),str(stop_child)],
                timeout=40,cancel_event=stopped,env=native_environment(),stdout=stream,stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(outer)
        try:
            deadline=time.monotonic()+10
            while not child_pid.exists():
                if future.done():future.result();pytest.fail('Batch stopped before its child started')
                if time.monotonic()>deadline:pytest.fail('Owned child did not start: '+log.read_text())
                time.sleep(.02)
            pid=int(child_pid.read_text());stopped.set()
            with pytest.raises(ScientificCancelled):future.result(timeout=15)
            with pytest.raises(ProcessLookupError):os.kill(pid,0)
            receipt=json.loads(result.read_text())
            assert [row['state'] for row in receipt['entries']]==['cancelled','pending']
            assert receipt['counts']==dict(prepared=0,failed=0,cancelled=1,pending=1)
        finally:
            stopped.set()
            # A cooperative, test-owned fallback avoids inspecting or signalling
            # unrelated host processes if an assertion fails during cleanup.
            stop_child.write_text('Stop this test-owned child only')
