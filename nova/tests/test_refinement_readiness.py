"""Boundary and failure-path checks for the deployment candidate."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from gpx_recipe import guided, worker, compiler
from gpx_recipe.execution import write_json
from radar_pd_nova.guided_refinement_controller import GuidedRefinementMixin
from radar_pd_nova.refinement_controller import RefinementMixin,refinement_status_message
from radar_pd_nova.refinement_runtime import verify_result_files
from radar_pd_nova.refinement_sources import RefinementScope, file_sha
from test_refinement_integration import create, source, approved


def test_repeated_approval_binds_the_current_draft_to_the_existing_recipe(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);scope=RefinementScope('exp')
    first=approved(w,scope)
    second=approved(w,scope)
    assert second['id']==first['id'] and len(w.state['recipes'])==1
    assert w.state['drafts']['exp']['approved_recipe']==second['id']
    assert w.start(second['id'],second['selection_token'])


def test_start_after_pause_resumes_captured_scans_without_repeating_completed_work(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1'),source('2')]);recipe=approved(w,RefinementScope('exp'))
    w.start(recipe['id'],recipe['selection_token']);w.run_next();w.pause('exp')
    assert sorted(j['status'] for j in w.state['jobs'].values())==['completed','paused']
    w.start(recipe['id'],recipe['selection_token'])
    assert w.run_next()['status']=='completed' and w.run_next() is None
    assert len([c for c in rt.calls if c[0]=='run'])==2


def test_discarding_a_draft_while_preflight_finishes_does_not_resurrect_it(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);original=rt.call
    def call(action,*a,**k):
        result=original(action,*a,**k)
        if action=='preview':w.discard_draft('exp')
        return result
    rt.call=call
    assert w.prepare('exp','1','background') is None
    assert 'exp' not in w.state['drafts']


def test_remote_restore_releases_interrupted_busy_drafts(tmp_path):
    w,rt=create(tmp_path)
    saved=copy.deepcopy(w.state);saved['drafts']['exp']={'status':'busy'}
    rt.recover_workspace=lambda:saved
    assert w.restore_remote()
    assert w.state['drafts']['exp']['status']=='interrupted'


def test_phase_edit_uses_a_dispatch_key_accepted_by_the_galaxy_boundary(tmp_path):
    from gpx_recipe.model_edits import SCHEMA
    w,rt=create(tmp_path);w.observe([source('1')]);ctx=w.inspect('1')['context']
    ctx['phases'].append(dict(key='p1',name='unwanted',histograms=['h'],atoms=[]))
    w.state['sources']['1']['context']=ctx
    original=rt.call
    def call(action,src,directory,**kwargs):
        if action!='edit':return original(action,src,directory,**kwargs)
        assert re.fullmatch('[a-f0-9]{24}',kwargs['dispatch_key'])
        directory=Path(directory);directory.mkdir(parents=True)
        target=directory/'verified_model.gpx';target.write_bytes(b'corrected')
        changed=copy.deepcopy(ctx);changed['source']['sha256']=file_sha(target)
        return dict(status='model_edited',final_gpx=target.name,output_sha256=file_sha(target),context=changed)
    rt.call=call
    edit=w.edit_source('1',dict(schema=SCHEMA,operation='remove',phase='p1',reason='Wrong phase',
        model_signature=ctx['model_signature'],source_sha256=ctx['source']['sha256']))
    assert edit['model_parent_id']=='1'


def test_galaxy_phase_edit_passes_the_corrected_dataset_to_its_next_preflight(tmp_path):
    import zipfile
    from gpx_recipe.model_edits import SCHEMA
    from radar_pd_nova.models import UtilityActionRecord,RunStatus
    from test_refinement_transport import runtime
    w,_=create(tmp_path/'workspace');rt,service,original=runtime(tmp_path)
    item=source('1');item.update(original,id='1')
    w.observe([item]);ctx={'source':{'sha256':original['source_sha256']},'model_signature':'initial',
        'phases':[dict(key='p0',name='Host',histograms=['h'],atoms=[]),dict(key='p1',name='Unwanted',histograms=['h'],atoms=[])]}
    w.state['sources']['1']['context']=ctx;w.runtime=rt
    from gpx_recipe.symmetry.compiler import package_signature
    w.state['sources']['1']['inspection_revision']=package_signature()
    corrected=copy.deepcopy(ctx);corrected['phases'].pop();corrected['model_signature']='corrected'
    contents={'verified_model.gpx':b'corrected native bytes','baseline.view.json':b'{}'}
    corrected['source']['sha256']=hashlib.sha256(contents['verified_model.gpx']).hexdigest()
    response=dict(schema='radar-refinement-worker/v1',action='edit',status='model_edited',
        source_sha256=original['source_sha256'],context=corrected,final_gpx='verified_model.gpx',
        output_sha256=corrected['source']['sha256'],view='baseline.view.json',
        files={name:hashlib.sha256(data).hexdigest() for name,data in contents.items()})
    with zipfile.ZipFile(service.archive,'w') as archive:
        archive.writestr('response.json',json.dumps(response))
        for name,data in contents.items():archive.writestr(name,data)
    service.job=lambda:UtilityActionRecord(uid='edit',tool_id=rt.tool_id,name='native',galaxy_job_id='edit-job',
        status=RunStatus.OK,outputs={'result_bundle':'bundle','stage_projects':'stages'})
    service.collection_elements=lambda key:[dict(name='model',id='corrected-dataset')]
    edited=w.edit_source('1',dict(schema=SCHEMA,operation='remove',phase='p1',reason='Wrong phase',
        model_signature='initial',source_sha256=original['source_sha256']))
    assert edited['dataset_id']=='corrected-dataset' and 'record_uid' not in edited
    response=dict(schema='radar-refinement-worker/v1',action='preview',status='preview_passed',
        source_sha256=edited['source_sha256'],files={})
    with zipfile.ZipFile(service.archive,'w') as archive:archive.writestr('response.json',json.dumps(response))
    rt.call('preview',edited,tmp_path/'next-preflight',dispatch_key='e'*24)
    assert service.submissions[-1]['inputs']['source_gpx']=={'dataset_id':'corrected-dataset'}
    assert service.requests[-1]['expected_source_sha256']==edited['source_sha256']
    assert Path(original['source_path']).read_bytes()==b'actual published bytes'


def test_luna_answer_can_be_recorded_and_accepted_in_a_fresh_workspace(tmp_path,monkeypatch):
    from gpx_recipe import providers
    w,rt=create(tmp_path);w.observe([source('1')]);ctx=w.inspect('1')['context']
    class State(SimpleNamespace):
        def flush(self):pass
    state=State(rf_source='1',rf_experiment='exp',rf_busy=False,rf_error='')
    app=GuidedRefinementMixin();app.server=SimpleNamespace(state=state);app._refinement_workspace=w
    app._init_guided_refinement();app._guided_state=guided.start(ctx);app._guided_source_id='1'
    app._refinement_sync=lambda:None
    tasks=[];app._refinement_schedule=lambda coroutine,name:tasks.append(asyncio.create_task(coroutine))
    class Client:
        def complete_structured(self,*a):return dict(status='answer',choices=['mixture'],number=None,text=None,evidence='phase mixture',unhandled=[])
    monkeypatch.setattr(providers,'configure_provider',lambda *a:None)
    monkeypatch.setattr(providers,'IllumineLuna',Client)
    token=guided.question_token(ctx,app._guided_state)
    async def run():
        app.interpret_guided_answer(dict(token=token,text='phase mixture'))
        await asyncio.gather(*tasks)
    asyncio.run(run())
    assert not state.rf_error and state.rf_guide_can_accept
    assert len(list((w.root/'guided_calls').glob('*.json')))==1
    app.accept_guided_answer(dict(token=token,text='phase mixture'))
    assert app._guided_state['answers']['recipe']=='mixture'
    assert w.state['guided_setups']['1']['answers']['recipe']=='mixture'


@pytest.mark.parametrize('field',['view','baseline_view','final_view','execution'])
def test_referenced_views_cannot_bypass_the_verified_manifest(tmp_path,field):
    outside=tmp_path/'outside.json';outside.write_text('{}')
    directory=tmp_path/'job';directory.mkdir()
    with pytest.raises(ValueError):verify_result_files(directory,{field:'../outside.json','files':{}})


def test_wrong_worker_action_and_partial_final_results_are_rejected(tmp_path):
    with pytest.raises(ValueError,match='requested operation'):
        verify_result_files(tmp_path,dict(schema='radar-refinement-worker/v1',action='inspect',status='inspected',files={}),action='run')
    with pytest.raises(ValueError,match='incomplete refinement'):
        verify_result_files(tmp_path,dict(status='needs_review',final_view='final.json',files={}))


@pytest.mark.parametrize('failure',['stage','final','trends'])
def test_export_failure_never_publishes_a_final_or_unreadable_stage(tmp_path,monkeypatch,failure):
    original=tmp_path/'source.gpx';original.write_bytes(b'original')
    directory=tmp_path/'worker'
    def compiled(plan,approval,target):
        code="""from pathlib import Path
import json,sys
p=Path(sys.argv[0]).parent
(p/'polished.gpx').write_bytes(b'refined')
(p/'polished_first.gpx').write_bytes(b'first')
(p/'polished_second.gpx').write_bytes(b'second')
(p/'polished.execution.json').write_text(json.dumps({'status':'completed_needs_scientific_review','stages':[
 {'id':'first','title':'First','status':'completed'}, {'id':'second','title':'Second','status':'completed'}]}))
"""
        target.write_text(code);return code
    def export(src,target):
        target=Path(target)
        if (failure=='stage' and target.name=='second.view.json' or failure=='final' and target.name=='final.view.json'):
            raise ValueError('Injected export failure')
        value=dict(histograms=[],parameters=[]);write_json(target,value);return value
    monkeypatch.setattr(compiler,'compile_script',compiled);monkeypatch.setattr(worker,'write_project_view',export)
    if failure=='trends':monkeypatch.setattr(worker,'trend_view',lambda value:(_ for _ in ()).throw(ValueError('Injected trend failure')))
    result=worker.work('run',original,directory,plan={'stages':[{'id':'first','title':'First'},{'id':'second','title':'Second'}]},approval={})
    assert result['status']=='needs_review' and not result.get('final_gpx') and not result.get('final_view')
    expected={'verified_first.gpx'} if failure=='stage' else {'verified_first.gpx','verified_second.gpx'}
    assert {p.name for p in directory.glob('verified_*.gpx')}==expected
    verify_result_files(directory,result,action='run')
    assert original.read_bytes()==b'original'


def test_existing_worker_receipts_are_never_republished_as_a_new_attempt(tmp_path):
    src=tmp_path/'source.gpx';src.write_bytes(b'original');directory=tmp_path/'job';directory.mkdir()
    receipt=directory/'response.json';receipt.write_text('{"previous":true}')
    with pytest.raises(ValueError,match='fresh attempt'):
        worker.work('inspect',src,directory)
    assert json.loads(receipt.read_text())=={'previous':True}


def test_atomic_json_writes_support_new_paths_and_concurrent_complete_documents(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    target=tmp_path/'new'/'nested'/'record.json'
    values=[dict(index=i,payload=str(i)*2000) for i in range(30)]
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda value:write_json(target,value),values))
    assert json.loads(target.read_text()) in values
    assert not list(target.parent.glob('*.partial'))


def test_native_failure_labels_use_native_ids_not_alphabetical_phase_order():
    baseline=dict(phases=[dict(native_id=7,name='First alphabetically',atoms=[dict(native_id=3,label='site-a')]),
                          dict(native_id=0,name='Last alphabetically',atoms=[dict(native_id=1,label='site-b')])])
    result=dict(status='needs_review',stages=[],error="Native dependent parameters violated physical/requested bounds: [('0::AUiso:1', -0.01)]")
    assert 'Last alphabetically · site-b' in refinement_status_message(result,baseline)
    assert refinement_status_message(result,{'phases':[]})==result['error']


def test_corrected_project_can_open_its_verified_galaxy_result_without_a_parent_record():
    app=RefinementMixin();app.server=SimpleNamespace(state=SimpleNamespace(
        rf_selected_job='j',rf_stage='final',gsasii_session_status='idle',flush=lambda:None))
    app._refinement_workspace=SimpleNamespace(state={'jobs':{'j':dict(source={'scan_label':'Corrected model'},result={'final_dataset_id':'verified-dataset'})}})
    calls=[];tasks=[]
    async def submit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace()
    app._submit_utility_action=submit
    app._schedule_utility=lambda coroutine,name:tasks.append(coroutine)
    app.open_refinement_in_gsasii()
    asyncio.run(tasks[0])
    assert calls[0]['associated_run_uid'] is None
    assert calls[0]['inputs']['project_source|gpx_project']=={'dataset_id':'verified-dataset'}


def test_reusing_a_guide_replaces_an_older_partial_form_for_the_same_source(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);ctx=w.inspect('1')['context']
    saved=guided.start(ctx)
    for key,value in [('recipe','mixture'),('model_review','reviewed'),('cell_phases',[])]:saved=guided.answer(ctx,saved,key,value)
    p=guided.build_plan(ctx,saved);draft=w.prepare('exp','1',p['request'],guided=saved)
    scope=RefinementScope('exp');_,token=w.selection(scope)
    recipe=w.approve('exp',draft['generation'],scope,existing=True,future=False,selection_token=token,approver='test')
    class State(SimpleNamespace):
        def flush(self):pass
    s=State(rf_source='1',rf_experiment='exp',rf_busy=False,rf_saved_recipe=recipe['id'])
    app=RefinementMixin();app.server=SimpleNamespace(state=s);app._refinement_workspace=w;app._init_guided_refinement()
    app._guided_source_id='1';app._guided_binding=('1',ctx['source']['sha256'])
    app._guided_state=guided.start(ctx)  # An unfinished different form is still open.
    tasks=[];app._refinement_schedule=lambda coroutine,name:tasks.append(asyncio.create_task(coroutine))
    app._refinement_hold_enrollment=lambda:None;app._refinement_checkpoint=lambda:None
    app._refinement_sync=app._guided_source_changed
    async def run():
        app.reuse_refinement_recipe();await asyncio.gather(*tasks)
    asyncio.run(run())
    assert app._guided_state==saved and s.rf_guide_complete
    assert s.rf_guide_question is None
