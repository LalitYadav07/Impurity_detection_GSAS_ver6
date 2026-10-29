import copy
import json
from pathlib import Path
import threading
import numpy as np
import pytest
from test_symmetry_contracts import make_protocol
from gpx_recipe.symmetry.k_search import KSearch,NativeRunner,SearchStopped,representative_starts
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.symmetry.contracts import digest


class Native:
    def __init__(self,*,null=False,fail=False):self.calls=[];self.null=null;self.fail=fail
    def call(self,action,request,directory,**kwargs):
        self.calls.append(action);root=Path(directory);x=np.linspace(0,10,1000)
        a=np.exp(-((x-3)/.08)**2);b=np.exp(-((x-7)/.08)**2)
        if action=='prepare':
            context=dict(sha256=request['source_sha256'],phases=[dict(name='nuclear',setting_sha256='a'*64,histograms=['PWDR sample'],atoms=[dict(label='Mn1',type='Mn',occupancy=1.)],operations=[dict(rotation=np.eye(3).tolist(),translation=[0,0,0])])],
                histograms=[dict(name='PWDR sample',instrument={'Type':['PNC','PNC',False]})],runtime=dict(magnetic_function_sha256='b'*64,native_stack_sha256='d'*64))
            path=root/'baseline.npz';np.savez(path,x=x,q=x+1,observed=x*0 if self.null else 5*b,nuclear=x*0,weights=x*0+1,mask=x>0)
            return dict(source_sha256=request['source_sha256'],input_unchanged=True,context=context,profile=path.name,profile_sha256=file_sha(path),points={'candidates':[dict(k=['0','0','0'],labels=['GAMMA']),dict(k=['0','0','1/2'],labels=['Z'])]})
        if action=='inventory':
            return dict(source_sha256=request['source_sha256'],families=[dict(k=c['k'],discovered=1,eligible=[dict(candidate_id=c['candidates'][0]['id'],axis='x',family='fixture',operation_count=4,cell_multiplier=1.)],dispositions=[]) for c in request['catalogs']])
        if action=='portfolio':
            results=[]
            for relative in request['directories']:
                directory=root/relative
                results.append(dict(directory=relative,response=self.call('template',{'directory':str(directory)},directory)))
            return dict(results=results,pending=[])
        job=json.loads((root/'job.json').read_text());path=root/'magnetic.npz'
        np.savez(path,intensity=a if job['protocol']['k']==['0','0','0'] else b)
        return dict(status='simulated',id=job['id'],profile=path.name,profile_sha256=file_sha(path),
            source_sha256=job['protocol']['source']['sha256'],protocol_sha256=job['protocol_sha256'],initial_maximum_moment=2.,no_refinement=True,audit={'passed':not self.fail},periodicity={'passed':True})


def setup(tmp_path,*,null=False,fail=False,unavailable=False):
    protocol=make_protocol(tmp_path/'input');source=tmp_path/'input/source.gpx'
    catalog=json.loads((tmp_path/'input/catalog.json').read_text())
    calls=[]
    def discover(parent,k,**kwargs):
        calls.append(k)
        if unavailable and k==['0','0','1/2']:raise ValueError('Service unavailable')
        return dict(copy.deepcopy(catalog),k=k)
    native=Native(null=null,fail=fail);search=KSearch(tmp_path/'cache',native,discover)
    values=dict(parent='nuclear',histogram='PWDR sample',sites=protocol['sites'],directions=['x'],relative_signs=False,budget_seconds=30)
    return search,native,calls,dict(source_path=str(source),source_sha256=file_sha(source)),values


@pytest.mark.parametrize('portfolio',[False,True])
def test_full_survey_preserves_null_and_prefers_nonzero_family(tmp_path,portfolio):
    search,native,calls,source,values=setup(tmp_path)
    native.supports_portfolio=portfolio
    result=search.run(source,tmp_path/'search',**values)
    assert result['complete'] and result['status']=='shortlist'
    assert result['families'][0]['k']==['0','0','1/2']
    assert len(result['templates'])==2 and result['source_unchanged'] and result['llm_calls']==0
    assert all(f['simulated']==1 for f in result['families'])
    assert all('api_key' not in p.read_text() for p in (tmp_path/'cache').rglob('job.json'))


def test_no_signal_does_not_become_a_k_assignment(tmp_path):
    search,_,_,source,values=setup(tmp_path,null=True)
    result=search.run(source,tmp_path/'search',**values)
    assert result['complete'] and result['status']=='no_support'


def test_unavailable_catalog_remains_visible_without_zero_score(tmp_path):
    search,_,_,source,values=setup(tmp_path,unavailable=True)
    result=search.run(source,tmp_path/'search',**values)
    missing=next(f for f in result['families'] if f['k']==['0','0','1/2'])
    assert not result['complete'] and result['status']=='incomplete'
    assert 'score' not in missing and missing['discovery']=='unavailable'


def test_failed_native_effect_check_is_not_scored(tmp_path):
    search,_,_,source,values=setup(tmp_path,fail=True)
    result=search.run(source,tmp_path/'search',**values)
    assert not result['templates'] and result['status']=='incomplete'
    assert all(f['simulation_failures'] for f in result['families'])


def test_unchanged_source_reuses_native_templates_and_catalogs(tmp_path):
    search,native,calls,source,values=setup(tmp_path)
    first=search.run(source,tmp_path/'first',**values)
    native.calls.clear();calls.clear()
    second=search.run(source,tmp_path/'second',**values)
    assert native.calls==['prepare'] and not calls
    assert first['ranking']['families']==second['ranking']['families']
    assert all(t['cached'] for t in second['templates'].values())


def test_changed_gpx_revision_cannot_reuse_old_profiles(tmp_path):
    search,native,_,source,values=setup(tmp_path)
    search.run(source,tmp_path/'first',**values);native.calls.clear()
    Path(source['source_path']).write_bytes(b'new observation revision');source['source_sha256']=file_sha(source['source_path'])
    result=search.run(source,tmp_path/'second',**values)
    assert native.calls.count('template')==2
    assert not any(t['cached'] for t in result['templates'].values())


def test_cached_artifact_tampering_is_reported_without_reexecution(tmp_path):
    search,native,_,source,values=setup(tmp_path)
    first=search.run(source,tmp_path/'first',**values);native.calls.clear()
    for t in first['templates'].values():Path(t['path']).write_bytes(b'changed')
    result=search.run(source,tmp_path/'second',**values)
    assert native.calls==['prepare']
    assert result['status']=='incomplete' and not result['templates']


def test_cancelled_search_never_starts_scientific_process(tmp_path):
    search,native,_,source,values=setup(tmp_path);cancel=threading.Event();cancel.set()
    with pytest.raises(SearchStopped):search.run(source,tmp_path/'search',cancel=cancel,**values)
    assert not native.calls


def test_changed_source_refused_before_native_loading(tmp_path):
    search,native,_,source,values=setup(tmp_path);source['source_sha256']='0'*64
    with pytest.raises(ValueError,match='changed'):search.run(source,tmp_path/'search',**values)
    assert not native.calls


def test_representatives_cover_multiple_families_before_extra_starts():
    rows=[dict(candidate_id=f'{f}-{i}',axis='x',family=f,operation_count=4,cell_multiplier=1) for f in ['A','B','C'] for i in range(10)]
    chosen=representative_starts(rows,[dict(label='Mn1'),dict(label='Mn2')],limit=6)
    assert [r['family'] for r in chosen[:3]]==['A','B','C']
    assert len(chosen)==6


def test_representative_budget_balances_axes_and_relative_signs():
    rows=[dict(candidate_id=f,axis=axis,family=f,operation_count=4,cell_multiplier=1) for f in ['A','B','C','D','E','F'] for axis in 'xyz']
    chosen=representative_starts(rows,[dict(label='Mn1'),dict(label='Mn2')],limit=6)
    assert set(r['axis'] for r in chosen)==set('xyz')
    assert {tuple(r['site_signs'].values()) for r in chosen}=={(1,1),(1,-1)}


def test_corrupt_inventory_is_never_used_to_select_templates(tmp_path):
    search,native,_,source,values=setup(tmp_path)
    search.run(source,tmp_path/'first',**values);native.calls.clear()
    for path in (tmp_path/'cache/inventories').glob('*.json'):
        saved=json.loads(path.read_text());saved['inventory']['families'][0]['eligible'].clear();path.write_text(json.dumps(saved))
    result=search.run(source,tmp_path/'second',**values)
    assert result['status']=='incomplete' and not result['templates'] and result['errors']


def test_model_or_service_credentials_are_not_native_worker_environment():
    with pytest.raises(ValueError,match='scientific runtime'):NativeRunner(['python'],environment={'BCS_API_KEY':'not-a-real-key'})


def test_longer_survey_budget_applies_to_native_inventory(tmp_path):
    search,native,_,source,values=setup(tmp_path)
    original=native.call;timeouts=[]
    def call(action,request,directory,**kwargs):
        if action=='inventory':timeouts.append(kwargs['timeout'])
        return original(action,request,directory,**kwargs)
    native.call=call;values['budget_seconds']=120
    result=search.run(source,tmp_path/'search',**values)
    assert result['complete'] and len(timeouts)==1 and 60<timeouts[0]<=120


@pytest.mark.parametrize('portfolio',[False,True])
def test_wrong_periodicity_is_an_unscored_scientific_disposition(tmp_path,portfolio):
    search,native,_,source,values=setup(tmp_path)
    native.supports_portfolio=portfolio;original=native.call
    def call(action,request,directory,**kwargs):
        if action=='template':return dict(status='k_periodicity_mismatch',reason='The seed carries another translation character')
        return original(action,request,directory,**kwargs)
    native.call=call
    result=search.run(source,tmp_path/'search',**values)
    assert result['complete'] and result['status']=='no_support' and not result['templates']
    assert all(f['status']=='outside_supported_scope' and len(f['simulation_rejections'])==1 for f in result['families'])
    assert not result['errors'] and not any(f.get('simulation_failures') for f in result['families'])


def test_real_subprocess_cancel_kills_its_process_group(tmp_path):
    import os,subprocess,sys,time
    from gpx_recipe.process import run_scientific,ScientificCancelled
    if os.name!='posix':pytest.skip('POSIX process-group cleanup')
    marker=tmp_path/'child-was-not-killed';ready=tmp_path/'ready'
    child=f"import pathlib,time;time.sleep(.8);pathlib.Path({str(marker)!r}).write_text('unexpected')"
    parent=f"import subprocess,sys,pathlib,time;subprocess.Popen([sys.executable,'-c',{child!r}]);pathlib.Path({str(ready)!r}).touch();time.sleep(10)"
    event=threading.Event();timer=threading.Timer(.2,event.set);timer.start()
    with pytest.raises(ScientificCancelled):run_scientific([sys.executable,'-c',parent],timeout=10,cancel_event=event,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    timer.join();time.sleep(.85)
    assert ready.exists() and not marker.exists()
