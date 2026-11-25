"""Full simulation budgets and allocation guards, without native diffraction."""
import json
from pathlib import Path
import threading

import pytest

from gpx_recipe.execution import write_json
from gpx_recipe.preparation import batch
from gpx_recipe.symmetry.native_protocol import validate_native_payload,validate_native_timeout,native_timeout_limit
from gpx_recipe.symmetry.resources import allocation
from test_preparation_native_size import example_plan


def request_for(tmp_path,monkeypatch,*,count=3,workers=1):
    request,_=example_plan(tmp_path)
    path=Path(request['plan']);plan=json.loads(path.read_text());plan['tasks']=plan['tasks'][:count]
    write_json(path,plan);request.update(seconds=batch.EXAMPLES_SECONDS,workers=workers)
    monkeypatch.setattr(batch,'allocation',lambda requested,**kwargs:
        allocation(requested,cgroup=tmp_path/'unreported-cgroup',**kwargs))
    # Receipt verification has separate real-array tests. These tests isolate
    # scheduling and never present a fake curve as native scientific evidence.
    monkeypatch.setattr(batch,'verify_result',lambda *args,**kwargs:[])
    return request


def prepared():return dict(state='prepared',reason='',result={})


def test_late_admission_gets_full_budget_and_remaining_unit_stays_pending(tmp_path,monkeypatch):
    request=request_for(tmp_path,monkeypatch);now=[0.];timeouts=[];events=[]
    def runner(task,payload,scratch,**kwargs):
        timeouts.append(kwargs['timeout']);events.append(kwargs['cancel'])
        now[0]=59. if len(timeouts)==1 else 100.
        return prepared()
    result=batch.execute(request,examples=True,runner=runner,clock=lambda:now[0],
        environ={'GALAXY_SLOTS':'1'})
    assert timeouts==[3600.,3600.]
    assert not any(event.is_set() for event in events)
    assert [r['state'] for r in result['entries']]==['prepared','prepared','pending']
    assert result['entries'][-1]['reason']=='batch_time_budget'
    assert result['entries'][-1]['attempt']==1
    assert result['allocation']['deadline_mode']=='complete_units'
    assert result['allocation']['peak_admitted_tasks']==1


def test_admission_closure_does_not_cancel_already_running_full_unit(tmp_path,monkeypatch):
    request=request_for(tmp_path,monkeypatch,workers=2);now=[0.]
    second_started=threading.Event();admission_closed=threading.Event();calls=[]
    def runner(task,payload,scratch,**kwargs):
        calls.append(kwargs['timeout'])
        if task['entry']=='0'*24:
            assert second_started.wait(3);now[0]=61.;admission_closed.set()
        else:
            second_started.set();assert admission_closed.wait(3)
            assert not kwargs['cancel'].is_set();now[0]=100.
        return prepared()
    result=batch.execute(request,examples=True,runner=runner,clock=lambda:now[0],
        environ={'GALAXY_SLOTS':'2'})
    assert calls==[3600.,3600.]
    assert result['counts']==dict(prepared=2,failed=0,cancelled=0,pending=1)
    assert result['allocation']['peak_admitted_tasks']==2


def test_full_unit_timeout_is_terminal_even_after_an_earlier_unit(tmp_path,monkeypatch):
    request=request_for(tmp_path,monkeypatch,count=2);now=[0.];calls=[]
    def runner(task,*args,**kwargs):
        calls.append(kwargs['timeout'])
        if len(calls)==1:now[0]=10.;return prepared()
        now[0]=3610.
        return dict(state='interrupted',reason=batch.DEADLINE_INTERRUPTED,seconds=3600.)
    result=batch.execute(request,examples=True,runner=runner,clock=lambda:now[0],
        environ={'GALAXY_SLOTS':'1'})
    assert result['counts']==dict(prepared=1,failed=1,cancelled=0,pending=0)
    assert result['entries'][1]['reason']=='Native preparation exceeded its full unit budget'


@pytest.mark.parametrize('mutation',['ordinary','training','too_many'])
def test_new_operation_rejects_other_kinds_or_more_than_four_units(tmp_path,monkeypatch,mutation):
    request=request_for(tmp_path,monkeypatch,count=4);path=Path(request['plan']);plan=json.loads(path.read_text())
    if mutation=='too_many':
        extra=json.loads(json.dumps(plan['tasks'][0]));extra['entry']='a'*24
        extra['request']['destination']='tasks/'+extra['entry']+'/attempt-1/compiled';plan['tasks'].append(extra)
    else:plan['tasks'][0]['kind']='reference' if mutation=='ordinary' else 'structural-training'
    write_json(path,plan)
    with pytest.raises(ValueError):batch.execute(request,examples=True,runner=lambda *a,**k:pytest.fail('No invalid work'))


def test_new_deadline_is_explicit_and_does_not_widen_ordinary_batches(tmp_path,monkeypatch):
    request=request_for(tmp_path,monkeypatch,count=1)
    validate_native_payload(batch.EXAMPLES_OPERATION,request)
    assert native_timeout_limit(batch.EXAMPLES_OPERATION)==3690
    assert validate_native_timeout(batch.EXAMPLES_OPERATION,3690)==3690
    for value in (600,3660,3689,3691):
        with pytest.raises(ValueError):validate_native_timeout(batch.EXAMPLES_OPERATION,value)
    assert native_timeout_limit('preparation-batch')==600
    with pytest.raises(ValueError):validate_native_timeout('preparation-batch',601)
    with pytest.raises(ValueError):batch.validate_request(request)
    request['seconds']=480
    assert batch.validate_request(request)['seconds']==480
    with pytest.raises(ValueError):batch.validate_examples_request(request)


def test_worker_entrypoint_uses_scoped_cancellation_for_new_operation(tmp_path,monkeypatch):
    import signal,sys
    from gpx_recipe.preparation import worker
    source=tmp_path/'request.json';source.write_text('{}');target=tmp_path/'result.json';seen=[]
    before=signal.getsignal(signal.SIGTERM)
    def execute(request,*,cancel,examples):
        assert examples and not cancel.is_set()
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
        assert cancel.is_set();seen.append(True);return {'cancelled':True}
    monkeypatch.setattr(batch,'execute',execute)
    monkeypatch.setattr(sys,'argv',['worker','examples-batch',str(source),str(target)])
    worker.main()
    assert seen and json.loads(target.read_text())=={'cancelled':True}
    assert signal.getsignal(signal.SIGTERM)==before
