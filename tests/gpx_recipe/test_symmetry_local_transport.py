import json
import os
from pathlib import Path
import sys
import time
import pytest

from gpx_recipe.symmetry.local_transport import LocalTransport


def until(predicate,seconds=6):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if predicate():return
        time.sleep(.03)
    raise AssertionError('Local monitor did not reach its expected bounded state')


def prepare(root,timeout=30):
    root.mkdir();key='a'*24
    (root/'job.json').write_text(json.dumps(dict(id=key,protocol_sha256='b'*64,protocol=dict(policy=dict(timeout_seconds=timeout)))))
    (root/'worker.py').write_text("import json,os,time\nfrom pathlib import Path\nPath('child.partial').write_text(json.dumps(dict(pid=os.getpid(),secret=os.getenv('TEST_MODEL_TOKEN'))));Path('child.partial').replace('child.json')\ntime.sleep(30)\n")
    return key


def test_real_process_reconnect_duplicate_prevention_and_cancellation_cleanup(tmp_path,monkeypatch):
    root=tmp_path/'trial';key=prepare(root);monkeypatch.setenv('TEST_MODEL_TOKEN','private-test-placeholder')
    transport=LocalTransport([sys.executable]);handle=transport.submit(key,root)
    try:
        until(lambda:(root/'child.json').exists())
        child=json.loads((root/'child.json').read_text());assert child['secret'] is None
        recovered=LocalTransport([sys.executable]).recover(key,root)
        assert recovered['state']=='running';assert recovered['handle']==handle
        assert transport.submit(key,root)==handle
        assert transport.cancel(handle,root) is False
        until(lambda:(root/'cancelled.json').exists())
        assert transport.recover(key,root)['state']=='cancelled'
        with pytest.raises(ProcessLookupError):os.kill(child['pid'],0)
    finally:
        if transport.alive(handle):os.kill(handle['pid'],15)


def test_real_native_deadline_produces_a_terminal_timeout(tmp_path):
    root=tmp_path/'trial';key=prepare(root,timeout=1);transport=LocalTransport([sys.executable]);handle=transport.submit(key,root)
    try:
        until(lambda:(root/'result.json').exists())
        result=json.loads((root/'result.json').read_text());assert result['id']==key;assert result['status']=='timed_out'
        assert result['llm_calls']==0
    finally:
        if transport.alive(handle):os.kill(handle['pid'],15)


def test_launch_ticket_refuses_model_credentials():
    with pytest.raises(ValueError,match='scientific'):LocalTransport([sys.executable],environment={'API_KEY':'test-placeholder'})
