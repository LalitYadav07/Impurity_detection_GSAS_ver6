"""Native representatives own fresh processes; one budget covers both levels."""
import json
import os
from types import SimpleNamespace

import pytest

from gpx_recipe.symmetry import grouped_worker


def test_parallel_native_receipts_come_from_distinct_fresh_processes(tmp_path):
    # Deliberately invalid job inputs exercise worker failure receipts without
    # requiring or pretending to run scientific fits. Real numerical parity
    # is covered by the native controls and the measured private-data pilot.
    tasks=[]
    for i in range(4):
        path=tmp_path/str(i);path.mkdir();(path/'job.json').write_text('{}')
        tasks.append((str(i),path,{}))
    results={}
    grouped_worker.verify_representatives(tasks,workers=2,cancelled=lambda:False,
        completed=lambda uid,result:results.update({uid:result}))
    assert set(results)=={'0','1','2','3'}
    pids={result['resources']['process_id'] for result in results.values()}
    assert len(pids)==4 and os.getpid() not in pids
    assert all(result['status']=='technical_failure' for result in results.values())
    assert all(json.loads((path/'result.json').read_text())==results[uid] for uid,path,_ in tasks)


def test_cancelled_portfolio_does_not_launch_representatives(monkeypatch):
    monkeypatch.setattr(grouped_worker,'execute',lambda *a,**k:pytest.fail('Cancelled work launched'))
    results=[]
    grouped_worker.verify_representatives([('id','unused',{})],workers=1,
        cancelled=lambda:True,completed=lambda *a:results.append(a))
    assert results==[]


def test_serial_receipts_keep_the_same_native_path(monkeypatch):
    calls=[]
    monkeypatch.setattr(grouped_worker,'execute',lambda p,**k:calls.append((p,k)) or {'status':'valid'})
    results=[]
    grouped_worker.verify_representatives([('a','one',{'fit':'screened'})],workers=1,
        cancelled=lambda:False,completed=lambda *a:results.append(a))
    assert calls==[('one',{'screened':{'fit':'screened'}})]
    assert results==[('a',{'status':'valid'})]


def test_dead_native_child_has_a_terminal_failure_instead_of_a_lost_future(tmp_path,monkeypatch):
    class Crashed:
        exitcode=17
        def __init__(self,**kwargs):pass
        def start(self):pass
        def is_alive(self):return False
        def join(self,*a,**k):pass
        def close(self):pass
    monkeypatch.setattr(grouped_worker.multiprocessing,'get_context',
        lambda method:SimpleNamespace(Process=Crashed))
    path=tmp_path/'trial';path.mkdir();(path/'job.json').write_text('{"protocol_sha256":"approved"}')
    results=[]
    grouped_worker.verify_representatives([('trial',path,{})],workers=2,
        cancelled=lambda:False,completed=lambda *a:results.append(a))
    assert results[0][1]['status']=='technical_failure'
    assert results[0][1]['protocol_sha256']=='approved'
    assert '17' in results[0][1]['reason']


def test_parallel_cancel_terminates_only_owned_children_without_starting_more(tmp_path,monkeypatch):
    processes=[]
    class Active:
        def __init__(self,**kwargs):self.stopped=False;self.closed=False
        def start(self):processes.append(self)
        def is_alive(self):return not self.stopped
        def terminate(self):self.stopped=True
        def join(self,*a,**k):pass
        def close(self):self.closed=True
    monkeypatch.setattr(grouped_worker.multiprocessing,'get_context',
        lambda method:SimpleNamespace(Process=Active))
    results=[]
    tasks=[(str(i),tmp_path/str(i),{}) for i in range(10)]
    grouped_worker.verify_representatives(tasks,workers=2,
        cancelled=lambda:len(processes)>=2,completed=lambda *a:results.append(a))
    assert len(processes)==2 and not results
    assert all(p.stopped and p.closed for p in processes)
