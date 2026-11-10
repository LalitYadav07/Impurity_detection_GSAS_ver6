"""Failed native work stays failed, complete and actionable without raw logs."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from gpx_recipe.preparation import failures,worker,batch
from test_preparation_batch import plan,prepared


@pytest.mark.parametrize('error,code',[
    (failures.PreparationScopeError('parent_cell_domain_conflict'),'parent_cell_domain_conflict'),
    (ValueError('private file or untrusted text'),'native_validation_failed'),
    (RuntimeError('private file or untrusted text'),'native_execution_failed')])
def test_worker_diagnostic_is_static_and_bound_to_exact_kind_request_and_code(tmp_path,error,code):
    path=tmp_path/'failure.json';request={'source':'captured.cif','destination':'owned-attempt'}
    assert failures.write(path,'structural-examples',request,error)
    assert failures.read(path,'structural-examples',request)==failures.REASONS[code]
    assert 'private file' not in path.read_text()
    assert failures.read(path,'structural',request) is None
    assert failures.read(path,'structural-examples',dict(request,destination='next-attempt')) is None
    value=json.loads(path.read_text());value['implementation_sha256']='f'*64;path.write_text(json.dumps(value))
    assert failures.read(path,'structural-examples',request) is None


@pytest.mark.parametrize('change',['message','code','extra','oversize','symlink'])
def test_invalid_diagnostics_cannot_replace_a_queue_reason(tmp_path,change):
    path=tmp_path/'failure.json';request={}
    failures.write(path,'reference',request,ValueError('sensitive'))
    value=json.loads(path.read_text())
    if change=='message':value['reason']='arbitrary uploaded message'
    elif change=='code':value['code']={'not':'a code'}
    elif change=='extra':value['scientific_result']=True
    elif change=='oversize':path.write_bytes(b'x'*(failures.MAX_BYTES+1))
    elif change=='symlink':
        target=tmp_path/'elsewhere';path.rename(target);path.symlink_to(target)
    if change in {'message','code','extra'}:path.write_text(json.dumps(value))
    assert failures.read(path,'reference',request) is None


def test_failure_reporting_never_overwrites_a_published_result_or_a_symlink(tmp_path):
    existing=tmp_path/'result.json';existing.write_bytes(b'preserved result')
    assert failures.write(existing,'reference',{},RuntimeError()) is False
    link=tmp_path/'link.json';link.symlink_to(existing)
    assert failures.write(link,'reference',{},RuntimeError()) is False
    assert existing.read_bytes()==b'preserved result'


@pytest.mark.parametrize('kind',['reference','batch'])
def test_main_retains_failure_exit_and_writes_only_a_non_scientific_diagnostic(tmp_path,monkeypatch,kind):
    request=tmp_path/'request.json';request.write_text('{}');output=tmp_path/'output.json'
    error=failures.PreparationScopeError('parent_cell_domain_conflict')
    def reject(*args,**kwargs):raise error
    monkeypatch.setattr(worker,'execute',reject);monkeypatch.setattr(batch,'execute',reject)
    monkeypatch.setattr(sys,'argv',['preparation-worker',kind,str(request),str(output)])
    with pytest.raises(failures.PreparationScopeError) as caught:worker.main()
    assert caught.value is error
    assert failures.read(output,kind,{})==failures.REASONS[error.code]
    assert 'artifacts' not in json.loads(output.read_text())


def test_diagnostic_io_failure_does_not_mask_the_original_native_error(tmp_path,monkeypatch):
    request=tmp_path/'request.json';request.write_text('{}')
    error=ValueError('original check')
    def reject(*args,**kwargs):raise error
    def broken(*args,**kwargs):raise RuntimeError('diagnostic failure')
    monkeypatch.setattr(worker,'execute',reject);monkeypatch.setattr(failures,'write',broken)
    monkeypatch.setattr(sys,'argv',['preparation-worker','reference',str(request),str(tmp_path/'output.json')])
    with pytest.raises(ValueError) as caught:worker.main()
    assert caught.value is error


def test_one_failed_member_keeps_its_reason_while_successful_siblings_publish(tmp_path,monkeypatch):
    request=plan(tmp_path,count=2)
    def native(command,**kwargs):
        captured=json.loads(Path(command[-2]).read_text())
        failures.write(command[-1],'reference',captured,ValueError('sensitive native detail'))
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(batch,'run_scientific',native)
    def runner(task,payload,scratch,**kwargs):
        if task['entry'].endswith('0'):return batch.run_task(task,payload,scratch,**kwargs)
        return prepared(task,payload,scratch,**kwargs)
    result=batch.execute(request,runner=runner,environ={'GALAXY_SLOTS':'2'})
    assert result['status']=='complete' and not result['all_prepared']
    assert result['counts']==dict(prepared=1,failed=1,cancelled=0,pending=0)
    assert result['entries'][0]['reason']==failures.REASONS['native_validation_failed']
    assert result['entries'][1]['state']=='prepared'
    assert all('response.json' not in name for name in result['artifacts'])
