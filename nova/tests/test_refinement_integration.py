"""Intent/scope/lifecycle contracts independent of the GPX scientific engine."""
import copy
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from radar_pd_nova.refinement_sources import (RefinementScope, source_from_record,select_published_project,published_project_path,file_sha)
from radar_pd_nova.refinement_controller import (published_refinement_source,refinement_rwp_summary,
    refinement_status_message,refinement_output_provenance,selected_refinement_dataset,
    recover_completed_refinement_records,refinement_experiment_options,RefinementMixin,
    refinement_parameter_warnings,refinement_result_title)
from radar_pd_nova.refinement_workspace import RefinementWorkspace
from radar_pd_nova.refinement_runtime import LocalRefinementRuntime,verify_result_files


def context(source):
    return dict(model_signature='same-model',source={'name':'source.gpx','sha256':source['source_sha256']},
        histograms=[dict(key='h0',name='h',background={'function':'chebyschev-1','coefficients':6})],
        phases=[dict(key='p0',name='oxide',histograms=['h'],atoms=[])])


def plan(ctx):
    return dict(schema='radar-gpx-recipe/v1',model_signature=ctx['model_signature'],request='background',assumptions=[],questions=[],unsupported=[],
        stages=[dict(id='bg',title='Background only',cycles=3,free=[dict(kind='background',histogram='h0')],constraints=[])])


class Runtime:
    kind='local'
    def __init__(self):self.calls=[];self.response_status='completed';self.callback=None
    def resolve(self,source):return copy.deepcopy(source)
    def call(self,action,source,directory,**kwargs):
        self.calls.append((action,source['id'],kwargs));Path(directory).mkdir(parents=True,exist_ok=True)
        if self.callback:self.callback()
        if action=='inspect':return dict(status='inspected',context=context(source),source_sha256=source['source_sha256'],directory=str(directory))
        if action=='preview':return dict(status='preview_passed',preview={'status':'preview_passed','model_signature':'same-model','source_sha256':source['source_sha256'],
            'plan_sha256':__import__('gpx_recipe.contracts',fromlist=['plan_hash']).plan_hash(kwargs['plan']),'stages':[]})
        return dict(status=self.response_status,directory=str(directory),error='Plan does not match the inspected GPX model' if self.response_status=='needs_review' else None)
    def save_workspace(self,manifest):return None
    def recover_workspace(self):return None


def source(uid,*,exp='exp',sample='id:0012',bank='2',checksum=None):
    return dict(id=uid,experiment=exp,scan_key='scan-'+uid,source_sha256=checksum or uid+'hash',source_path='/not-loaded.gpx',
        sample_key=sample,sample_label=sample,bank=bank,scan_label=uid,run_number=int(''.join(c for c in uid if c.isdigit()) or 1),record_uid=uid)


def create(tmp_path):
    runtime=Runtime();w=RefinementWorkspace(tmp_path,history_id='history',runtime=runtime,
        planner=lambda request,ctx,previous=None:dict(plan=plan(ctx),calls=[dict(label='test planner')]))
    return w,runtime


def test_unchanged_refinement_backups_coalesce_but_new_claims_and_failures_do_not(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    w,rt=create(tmp_path);entered=threading.Event();release=threading.Event();snapshots=[]
    def upload(document):
        snapshots.append(copy.deepcopy(document));entered.set();assert release.wait(5)
        return 'backup-'+str(len(snapshots))
    rt.save_workspace=upload
    with ThreadPoolExecutor(max_workers=2)as pool:
        first=pool.submit(w.checkpoint);assert entered.wait(5)
        # A timestamp-only save does not justify another Galaxy metadata job.
        with w.lock:w.save()
        second=pool.submit(w.checkpoint);release.set()
        assert first.result(timeout=5)==second.result(timeout=5)=='backup-1'
    assert len(snapshots)==1
    with w.lock:w.state['jobs']['claimed']=dict(status='running',remote_action=None);w.save()
    assert w.checkpoint()=='backup-2' and len(snapshots)==2
    def unavailable(document):raise RuntimeError('Backup rejected')
    rt.save_workspace=unavailable
    with w.lock:w.state['jobs']['claimed']['remote_action']=dict(galaxy_job_id='acknowledged');w.save()
    with pytest.raises(RuntimeError,match='Backup rejected'):w.checkpoint()
    rt.save_workspace=upload
    assert w.checkpoint()=='backup-3'
    assert snapshots[-1]['jobs']['claimed']['remote_action']['galaxy_job_id']=='acknowledged'


def test_planner_knows_multi_scan_enrollment_is_host_managed_without_changing_native_context(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);seen=[]
    def planner(request,context,previous=None):
        seen.append(copy.deepcopy(context));return dict(plan=plan(context),calls=[])
    w._planner=planner
    draft=w.prepare('exp','1','Apply to the three scans selected in NOVA')
    assert seen[0]['host_execution']['histogram_keys']==['h0']
    assert 'independently' in seen[0]['host_execution']['scan_enrollment']
    assert 'host_execution' not in draft['context']
    assert draft['context']==w.state['sources']['1']['context']
    assert draft['status']=='ready' and draft['preview']['model_signature']=='same-model'


def test_parallel_polishing_claims_distinct_scans_once_and_respects_limit(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    w,rt=create(tmp_path);rt.kind='galaxy'
    w.observe([source(str(i)) for i in range(1,5)])
    recipe=approved(w,RefinementScope('exp'));w.start(recipe['id'],recipe['selection_token'])
    w.set_parallel_jobs(2)
    release=threading.Event();entered=threading.Event();declined=threading.Event();lock=threading.Lock();active=0;peak=0;refusals=0;started=[]
    original=rt.call
    def call(action,src,directory,**kwargs):
        nonlocal active,peak
        if action!='run':return original(action,src,directory,**kwargs)
        with lock:
            active+=1;peak=max(peak,active);started.append(src['id'])
            if active==2:entered.set()
        try:
            assert release.wait(5)
            return original(action,src,directory,**kwargs)
        finally:
            with lock:active-=1
    rt.call=call
    def attempt():
        nonlocal refusals
        result=w.run_next()
        if result is None:
            with lock:
                refusals+=1
                if refusals==2:declined.set()
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(attempt) for _ in range(4)]
        assert entered.wait(5) and declined.wait(5)
        assert len(w._inflight)==2
        with pytest.raises(ValueError,match='active polishing'):w.set_parallel_jobs(4)
        release.set()
        results=[f.result(timeout=5) for f in futures]
    assert peak==2 and len(set(started))==2
    assert sum(r is not None for r in results)==2
    assert sum(j['status']=='queued' for j in w.state['jobs'].values())==2
    while w.run_next():pass
    assert sorted(started)==['1','2','3','4']
    assert all(j['attempts']==1 and j['status']=='completed' for j in w.state['jobs'].values())
    restored=RefinementWorkspace(w.root,history_id='history',runtime=rt)
    assert restored.parallel_jobs==2 and restored.run_next() is None


def test_pause_parallel_polishing_keeps_active_jobs_and_holds_unstarted_scans(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    w,rt=create(tmp_path);rt.kind='galaxy';w.observe([source(str(i))for i in range(1,5)])
    recipe=approved(w,RefinementScope('exp'),future=True);w.start(recipe['id'],recipe['selection_token']);w.set_parallel_jobs(2)
    entered=threading.Barrier(3);release=threading.Event();original=rt.call
    def call(action,src,directory,**kwargs):
        if action=='run':entered.wait(timeout=5);assert release.wait(5)
        return original(action,src,directory,**kwargs)
    rt.call=call
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(w.run_next)for _ in range(2)];entered.wait(timeout=5)
        w.pause('exp');w.observe([source('5')])
        assert sum(j['status']=='running'for j in w.state['jobs'].values())==2
        assert sum(j['status']=='paused'for j in w.state['jobs'].values())==2
        assert len(w.state['jobs'])==4 and w.run_next() is None
        release.set();assert all(f.result(timeout=5)['status']=='completed' for f in futures)
    assert not w.state['recipes'][recipe['id']]['armed'] and w.run_next() is None


@pytest.mark.parametrize('value',[True,0,3,8,'2'])
def test_polishing_concurrency_requires_a_bounded_integer(value,tmp_path):
    w,rt=create(tmp_path);rt.kind='galaxy'
    with pytest.raises(ValueError,match='Choose 1, 2 or 4'):w.set_parallel_jobs(value)
    assert w.parallel_jobs==1


def test_local_polishing_stays_serial_without_a_galaxy_allocation(tmp_path):
    w,rt=create(tmp_path)
    with pytest.raises(ValueError,match='Galaxy-managed'):w.set_parallel_jobs(2)
    assert w.parallel_jobs==1


def test_inspection_and_preview_recover_under_new_worker_revision_without_reusing_failed_export(tmp_path,monkeypatch):
    import gpx_recipe.symmetry.compiler as compiler
    w,rt=create(tmp_path);w.observe([source('1')])
    legacy=w.root/'inspections'/'1hash';legacy.mkdir(parents=True)
    (legacy/'response.json').write_text(json.dumps(dict(status='failed',error='Old missing-Rwp export failure')))
    monkeypatch.setattr(compiler,'package_signature',lambda:'a'*64)
    first=w.inspect('1');assert first['inspection_revision']=='a'*64
    first_call=next(row for row in rt.calls if row[0]=='inspect')
    w.observe([source('1')])
    w.inspect('1');assert sum(row[0]=='inspect'for row in rt.calls)==1
    first_preview=w.prepare('exp','1','background only')
    preview_key=next(row[2]['dispatch_key']for row in rt.calls if row[0]=='preview')
    monkeypatch.setattr(compiler,'package_signature',lambda:'b'*64)
    second=w.prepare('exp','1','background only')
    inspections=[row for row in rt.calls if row[0]=='inspect']
    previews=[row for row in rt.calls if row[0]=='preview']
    assert len(inspections)==2 and inspections[-1][2]['dispatch_key']!=first_call[2]['dispatch_key']
    assert previews[-1][2]['dispatch_key']!=preview_key
    assert second['source']['inspection_revision']=='b'*64
    assert (legacy/'response.json').is_file()


def test_parallel_polishing_pump_keeps_working_after_an_earlier_scan_finishes(tmp_path):
    import threading
    w,rt=create(tmp_path);rt.kind='galaxy';w.observe([source(str(i))for i in range(1,5)])
    recipe=approved(w,RefinementScope('exp'));w.start(recipe['id'],recipe['selection_token']);w.set_parallel_jobs(2)
    barrier=threading.Barrier(2);original=rt.call;seen=[]
    def call(action,src,directory,**kwargs):
        if action=='run':barrier.wait(timeout=5)
        return original(action,src,directory,**kwargs)
    rt.call=call
    class State(SimpleNamespace):
        def flush(self):pass
    class App(RefinementMixin):
        def _refinement_sync(self):pass
        def _refinement_checkpoint(self):pass
        async def _select_refinement_job(self,uid):seen.append(uid)
    app=App();app._refinement_workspace=w
    app.server=SimpleNamespace(state=State(rf_experiment='exp',rf_follow=True,rf_selected_job='',rf_error=''))
    asyncio.run(app._refinement_pump())
    assert len(seen)==4 and len(set(seen))==4 and not app.server.state.rf_error
    assert all(j['status']=='completed' and j['attempts']==1 for j in w.state['jobs'].values())
    assert not app.server.state.rf_queue_running


def test_slow_previous_refinement_load_cannot_overwrite_a_new_scan_selection(tmp_path):
    import threading
    entered=threading.Event();release=threading.Event()
    jobs={uid:dict(id=uid,source=dict(experiment='exp',scan_label=uid,sample_label='sample'),message=uid,
                  result=dict(directory=str(tmp_path),status='completed',stages=[],final_view='view.json',
                              baseline_view='baseline.json',files={}))for uid in ['first','second']}
    class State(SimpleNamespace):
        def flush(self):pass
    class App(RefinementMixin):
        def _read_refinement_view(self,job,name):
            if job['id']=='first':entered.set();assert release.wait(5)
            return dict(histograms=[dict(name=job['id']+' histogram')])
        def _refinement_plot_changed(self):pass
        def _refinement_trend_changed(self):pass
    app=App();app._refinement_workspace=SimpleNamespace(state=dict(jobs=jobs))
    app.server=SimpleNamespace(state=State(rf_experiment='exp',rf_selected_job=''))
    async def exercise():
        first=asyncio.create_task(app._select_refinement_job('first'))
        assert await asyncio.to_thread(entered.wait,5)
        await app._select_refinement_job('second');release.set();await first
    asyncio.run(exercise())
    assert app.server.state.rf_selected_job=='second'
    assert app.server.state.rf_histogram=='second histogram'
    assert app.server.state.rf_selected_title.startswith('second')


@pytest.mark.parametrize('mode,expected',[('models','models'),('recipe','results'),('results','results')])
def test_changing_experiment_does_not_leave_model_exploration_for_an_existing_recipe_job(mode,expected):
    state=SimpleNamespace(rf_experiment='new-experiment',rf_mode=mode)
    app=RefinementMixin();app.server=SimpleNamespace(state=state);app._refinement_internal=False
    app._refinement_workspace=SimpleNamespace(state=dict(drafts={},recipes={},jobs={'finished':dict(source=dict(experiment='new-experiment'))}))
    synced=[];app._refinement_sync=lambda:synced.append(True)
    app._refinement_experiment_changed()
    assert state.rf_mode==expected and synced==[True]


def approved(w,scope,*,future=False,existing=True):
    base=next(s for s in w.sources(scope.experiment) if scope.matches(s))
    draft=w.prepare(scope.experiment,base['id'],'background only')
    _,token=w.selection(scope,existing=existing,future=future)
    r=w.approve(scope.experiment,draft['generation'],scope,existing=existing,future=future,selection_token=token,approver='test approval')
    return r


def test_exact_sample_and_bank_scope_and_existing_backlog_are_snapshotted(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1'),source('2',sample='id:12'),source('3',bank='3'),source('4',exp='other')])
    scope=RefinementScope('exp','samples',samples=('id:0012',),banks=('2',));recipe=approved(w,scope)
    w.observe([source('5')])
    jobs=w.start(recipe['id'],recipe['selection_token'])
    assert [j['source']['id'] for j in jobs]==['1']
    w.run_next();assert w.run_next() is None
    assert w.public('exp')['jobs'][0]['status']=='completed'
    assert w.state['sources']['1']['source_sha256']=='1hash'


def test_model_usage_survives_review_and_workspace_recovery(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')])
    call=dict(label='Illumine Luna',provider='illumine-luna',model='gpt-6-luna',
              usage=dict(input_tokens=2000,input_tokens_details=dict(cache_write_tokens=1500,cached_tokens=100),
                         output_tokens=300,total_tokens=2300))
    w._planner=lambda request,ctx,previous=None:dict(plan=plan(ctx),calls=[call])
    draft=w.prepare('exp','1','background')
    assert draft['model_usage']['total_tokens']==2300
    assert draft['model_usage']['estimated_usd'] is not None
    restored=RefinementWorkspace(tmp_path,history_id='history',runtime=rt)
    assert restored.state['drafts']['exp']['model_usage']==draft['model_usage']


def test_failed_plan_retains_reported_token_usage(tmp_path):
    w,_=create(tmp_path);w.observe([source('1')])
    call=dict(provider='illumine-luna',model='gpt-6-luna',
              usage=dict(input_tokens=700,output_tokens=100,total_tokens=800))
    def failed(request,ctx,previous=None):
        error=ValueError('The model plan did not pass validation')
        error.planning_calls=[call]
        raise error
    w._planner=failed
    with pytest.raises(ValueError,match='did not pass'):
        w.prepare('exp','1','background')
    draft=w.state['drafts']['exp']
    assert draft['status']=='error' and draft['planning_calls']==1
    assert draft['model_usage']['total_tokens']==800


def test_future_only_enrolls_new_matching_arrivals_not_existing_or_other_experiments(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);scope=RefinementScope('exp','samples',samples=('id:0012',),banks=('2',));r=approved(w,scope,future=True,existing=False)
    assert not w.start(r['id'],r['selection_token'])
    w.observe([source('2'),source('3',sample='id:B'),source('4',bank='4'),source('5',exp='other')])
    assert [j['source']['id'] for j in w.state['jobs'].values()]==['2']
    w.observe([source('2')]);assert len(w.state['jobs'])==1


def test_same_source_revision_and_recipe_is_idempotent_but_distinct_acquisitions_are_not(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1',checksum='samebytes'),source('2',checksum='samebytes')]);r=approved(w,RefinementScope('exp'))
    w.start(r['id'],r['selection_token']);w.start(r['id'],r['selection_token']);assert len(w.state['jobs'])==2
    while w.run_next():pass
    w.start(r['id'],r['selection_token']);assert w.run_next() is None
    assert all(j['attempts']==1 for j in w.state['jobs'].values())


def test_approval_rejects_changed_scope_and_changed_draft_generation(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);s=RefinementScope('exp');d=w.prepare('exp','1','background');_,token=w.selection(s)
    w.observe([source('2')])
    with pytest.raises(ValueError,match='scan set changed'):w.approve('exp',d['generation'],s,existing=True,future=False,selection_token=token,approver='test')
    w.discard_draft('exp');_,token=w.selection(s)
    with pytest.raises(ValueError,match='draft changed'):w.approve('exp',d['generation'],s,existing=True,future=False,selection_token=token,approver='test')


def test_restart_holds_interrupted_local_work_and_never_silently_repeats(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);r=approved(w,RefinementScope('exp'));job=w.start(r['id'],r['selection_token'])[0]
    w.state['jobs'][job['id']].update(status='running',attempts=1);w.save()
    restored=RefinementWorkspace(tmp_path,history_id='history',runtime=rt)
    assert restored.state['jobs'][job['id']]['status']=='interrupted'
    assert restored.run_next() is None
    restored.retry(job['id']);restored.run_next();assert restored.state['jobs'][job['id']]['attempts']==2


def test_pause_prevents_new_enrollment_and_queued_execution(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);r=approved(w,RefinementScope('exp'),future=True);w.start(r['id'],r['selection_token']);w.pause('exp')
    w.observe([source('2')]);assert len(w.state['jobs'])==1 and w.run_next() is None
    w.resume_queue('exp');assert w.run_next()['status']=='completed'


def test_changed_models_need_a_recipe_review_not_an_automatic_llm_call(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);r=approved(w,RefinementScope('exp'));w.start(r['id'],r['selection_token']);rt.response_status='needs_review'
    assert w.run_next()['status']=='needs_recipe'
    with pytest.raises(ValueError,match='new recipe'):w.retry(next(iter(w.state['jobs'])))
    assert len([c for c in rt.calls if c[0]=='run'])==1


def test_saved_needs_review_result_can_recover_verified_intermediate_stages(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);recipe=approved(w,RefinementScope('exp'))
    job=w.start(recipe['id'],recipe['selection_token'])[0]
    w.state['jobs'][job['id']]['remote_action']={'galaxy_job_id':'existing'}
    rt.response_status='needs_review'
    recovered=w.recover_job_artifacts(job['id'])
    assert recovered['status']=='needs_review'
    assert w.state['jobs'][job['id']]['artifact_recoveries']==1


def test_published_source_prefers_explicit_publication_and_hashes_its_actual_bytes(tmp_path):
    raw=tmp_path/'Technical/model.gpx';raw.parent.mkdir();raw.write_bytes(b'internal')
    pub=tmp_path/'gpx/02_Main.gpx';pub.parent.mkdir();pub.write_bytes(b'enriched publication')
    project=dict(path='Technical/model.gpx',source_path='Technical/model.gpx',collection_path='gpx/02_Main.gpx',collection_name='02_Main.gpx',sha256=file_sha(raw),published_sha256=file_sha(pub),status='accepted',stage='main_phase_anchor')
    record=SimpleNamespace(uid='record',name='run',local_output_dir=tmp_path,output_dataset_ids={'gpx_projects':'collection'},inputs=None)
    payload=dict(summary={'observables':{'source_gpx_sha256':file_sha(raw)},'gpx_projects':[project]})
    item=source_from_record(record,payload,dict(instrument='PG3',ipts='IPTS-1',run_number=1,metadata={'sample_id':'0012'}),'history')
    assert item['source_path']==str(pub) and item['source_sha256']==file_sha(pub) and item['sample_key']=='id:0012'
    pub.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum changed'):source_from_record(record,payload,{},'history')


def test_ambiguous_or_stale_model_index_cannot_fall_back_to_arbitrary_project(tmp_path):
    rows=[dict(status='accepted',stage='final_refinement',sha256='a'),dict(status='accepted',stage='final_refinement',sha256='b')]
    with pytest.raises(ValueError,match='Several accepted'):select_published_project({'gpx_projects':rows},{})
    with pytest.raises(ValueError,match='does not identify'):select_published_project({'gpx_projects':rows,'observables':{'source_gpx_sha256':'missing'}},{})
    assert published_project_path(tmp_path,{'path':'../outside.gpx'}) is None


def test_identical_anchor_and_frozen_final_publications_are_one_model():
    anchor = dict(status='accepted', stage='main_phase_anchor', sha256='a'*64,
                  published_sha256='b'*64, collection_name='02_Main_phase_anchor.gpx')
    final = dict(anchor, stage='final_refinement', collection_name='03_Final_accepted_model.gpx')
    result = dict(observables={'source_gpx_sha256':'a'*64}, gpx_projects=[anchor, final])
    assert select_published_project(result, {}) == final
    # Internal source identity alone cannot establish publication identity.
    for replacement in (dict(anchor, published_sha256='c'*64),
                        {k:v for k,v in anchor.items() if k != 'published_sha256'}):
        with pytest.raises(ValueError, match='Several published GPXs'):
            select_published_project(dict(result, gpx_projects=[replacement, final]), {})


def test_identical_final_and_joint_cell_aliases_keep_canonical_source_identity(tmp_path):
    final = dict(status='accepted', stage='final_refinement', sha256='a'*64,
                 published_sha256='b'*64, collection_name='03_Final_accepted_model.gpx',
                 source_path='Technical/GSAS_Projects/seq_final_accepted_model.gpx')
    joint = dict(final, collection_name='03_Final_accepted_joint_cell_model.gpx',
                 source_path='Technical/GSAS_Projects/seq_final_joint_polished.gpx')
    record = SimpleNamespace(uid='live-handle', galaxy_job_id='job', name='IPTS-1_PG3_1',
                             local_output_dir=tmp_path,
                             output_dataset_ids={'gpx_projects':'collection'}, inputs=None)
    sources = []
    for projects in ([final, joint], [joint, final]):
        result = dict(observables={'source_gpx_sha256':'a'*64}, gpx_projects=projects)
        assert select_published_project(result, {}) == final
        sources.append(source_from_record(record, {'summary':result},
                       dict(instrument='PG3', ipts='IPTS-1', run_number=1), 'history'))
    assert sources[0]['id'] == sources[1]['id']
    assert sources[0]['collection_name'] == final['collection_name']
    for replacement in (dict(joint, published_sha256='c'*64),
                        {k:v for k,v in joint.items() if k != 'published_sha256'}):
        with pytest.raises(ValueError, match='Several published GPXs'):
            select_published_project(dict(result, gpx_projects=[final, replacement]), {})


def test_local_runtime_captures_immutable_original_and_holds_changed_input(tmp_path):
    original=tmp_path/'incoming.gpx';original.write_bytes(b'original')
    rt=LocalRefinementRuntime(tmp_path/'runtime',python='unused');s=dict(source_path=str(original),source_sha256=file_sha(original));capture=rt.resolve(s)
    original.write_bytes(b'revised')
    assert Path(capture['source_path']).read_bytes()==b'original'
    with pytest.raises(ValueError,match='changed'):rt.resolve(s)
    view=tmp_path/'view.json';view.write_text('{}');result={'files':{'view.json':file_sha(view)}};verify_result_files(tmp_path,result);view.write_text('{"tampered":true}')
    with pytest.raises(ValueError,match='changed'):verify_result_files(tmp_path,result)


def test_late_watch_metadata_enriches_same_source_identity_without_second_enrollment(tmp_path):
    pub=tmp_path/'gpx/final.gpx';pub.parent.mkdir();pub.write_bytes(b'published')
    p=dict(collection_path='gpx/final.gpx',collection_name='final.gpx',sha256='original',status='accepted',stage='main_phase_anchor')
    record=SimpleNamespace(uid='job',name='run',local_output_dir=tmp_path,output_dataset_ids={},inputs=SimpleNamespace(instrument='PG3',ipts='IPTS-3',run_number=123,bank=2))
    first=source_from_record(record,dict(summary={'gpx_projects':[p]}),None,'history')
    later=source_from_record(record,dict(summary={'gpx_projects':[p]}),dict(run_id='PG3_123_BANK2',metadata={'sample_id':'007','sample_name':'Oxide'}),'history')
    assert first['id']==later['id'] and first['scan_key']==later['scan_key']
    w,rt=create(tmp_path/'workspace');w.observe([first]);stamp=w.state['sources'][first['id']]['observed_utc'];w.observe([later])
    restored=RefinementWorkspace(w.root,history_id='history',runtime=rt)
    assert len(restored.state['sources'])==1
    assert restored.state['sources'][first['id']]['sample_key']=='id:007'
    assert restored.state['sources'][first['id']]['observed_utc']==stamp


def test_summary_only_refresh_preserves_reviewed_live_provenance_for_same_publication(tmp_path):
    w,_=create(tmp_path)
    reviewed=dict(source('1'),collection_id='collection',collection_name='final.gpx',internal_source_sha256='internal',
        published_sha256='published',live_watch=True,experiment_label='HB2A · IPTS-28749',
        sample_label='Reviewed Er specimen',metadata={'conditions':{'reviewed':True,'temperature':.28}})
    weak=dict(reviewed,live_watch=False,experiment='single-job',sample_key='scan:1',sample_label='Unassigned sample',metadata={})
    w.observe([reviewed]);w.observe([weak])
    recovered=w.state['sources']['1']
    assert recovered['experiment']=='exp' and recovered['sample_label']=='Reviewed Er specimen'
    assert recovered['metadata']==reviewed['metadata'] and recovered['live_watch']
    w.observe([dict(reviewed,metadata={'conditions':{'reviewed':True,'temperature':.29}})])
    assert w.state['sources']['1']['metadata']['conditions']['temperature']==.29
    # A different publication must not inherit the old sample/condition mapping.
    w.observe([dict(weak,collection_id='new-collection')])
    assert w.state['sources']['1']['experiment']=='single-job'


def test_acknowledged_live_and_recovered_records_have_the_same_publication_identity(tmp_path):
    project=dict(collection_name='final.gpx',sha256='original',published_sha256='published',status='accepted',stage='main_phase_anchor')
    def record(uid):
        return SimpleNamespace(uid=uid,galaxy_job_id='acknowledged-job',name='Live scan',local_output_dir=tmp_path,
            output_dataset_ids={'gpx_projects':'collection'},inputs=None)
    live=source_from_record(record('pending-token'),{'summary':{'gpx_projects':[project]}},None,'history')
    restored=source_from_record(record('acknowledged-job'),{'summary':{'gpx_projects':[project]}},None,'history')
    assert live['id']==restored['id']
    assert live['record_uid']=='pending-token' and restored['record_uid']=='acknowledged-job'
    assert live['galaxy_job_id']==restored['galaxy_job_id']=='acknowledged-job'


def test_recovered_live_ages_scans_share_experiment_sample_and_bank_from_publications(tmp_path):
    project=dict(collection_name='final.gpx',sha256='model',status='accepted',stage='final_refinement')
    def recovered(run):
        record=SimpleNamespace(uid=str(run),name=f'IPTS-2767_PG3_{run}',local_output_dir=tmp_path,
                               output_dataset_ids={'gpx_projects':f'collection-{run}'},inputs=None)
        summary=dict(gpx_projects=[project],acquisition=dict(instrument='PG3',run_number=run,sample_id='21213',
                    sample_name='MgO+Al2O3 temperature calibration'),provenance={'input':{'source_bank':2}})
        return source_from_record(record,{'summary':summary},None,'history')
    cold,warm=recovered(43106),recovered(43124)
    assert cold['experiment']==warm['experiment']
    assert cold['experiment_label']=='PG3 · IPTS-2767'
    assert cold['sample_key']==warm['sample_key']=='id:21213'
    assert cold['bank']==warm['bank']=='2'
    assert cold['run_number']==43106 and warm['run_number']==43124
    assert cold['source_path'] is None and cold['collection_id']=='collection-43106'
    placeholder=source_from_record(SimpleNamespace(uid='placeholder',name='IPTS-2767_PG3_43124',local_output_dir=tmp_path,
        output_dataset_ids={'gpx_projects':'collection'},inputs=None),
        {'summary':dict(gpx_projects=[project],acquisition=dict(instrument='PG3',run_number=43124,sample_id='21213',sample_name='MgO+Al2O3 temperature calibration'))},
        dict(sample_key='unassigned',sample_label='Unassigned sample',axis_overrides={'temperature':{'value':300,'unit':'C','reviewed_mapping':True}}),'history')
    assert placeholder['sample_label']=='21213 · MgO+Al2O3 temperature calibration'
    assert placeholder['axis_overrides']['temperature']['value']==300


def test_history_only_source_recovers_published_measured_temperature(tmp_path):
    from radar_pd_nova.scan_conditions import qualified_metadata
    window=dict(start='2025-09-13T00:11:00.721224667-04:00',end='2025-09-13T00:16:00.859446667-04:00')
    acquisition=dict(instrument='PG3',run_number=61020,sample_id='sample',sample_name='Oxide',
        acquisition_window=window,product_window_verified=True,channels=[dict(quantity='temperature',value=169.6,
        unit='K',minimum=169.5,maximum=169.7,role='measured',status='recorded',coverage_fraction=1.,
        statistic='time_weighted_step_hold',source='/entry/DASlogs/temperature/value',acquisition_window=window)])
    acquisition['$schema']='radar-pd-acquisition/v1'
    project=dict(collection_name='final.gpx',sha256='model',status='accepted',stage='final_refinement')
    record=SimpleNamespace(uid='record',name='IPTS-35915_PG3_61020',local_output_dir=tmp_path,
                           output_dataset_ids={'gpx_projects':'collection'},inputs=None)
    row=source_from_record(record,{'summary':dict(gpx_projects=[project],acquisition=acquisition)},None,'history')
    temperature=qualified_metadata(row['metadata'])['temperature']
    assert temperature['value']==169.6 and temperature['minimum']==169.5 and temperature['maximum']==169.7


def test_recovered_banked_run_names_rejoin_the_verified_ipts_without_mixing_conflicts(tmp_path):
    project=dict(collection_name='final.gpx',sha256='model',status='accepted',stage='final_refinement')
    def recovered(run,*,instrument='NOM',bank=2,metadata_run=None,suffix=''):
        record=SimpleNamespace(uid=str(run),name=f'IPTS-34537_NOM_{run}_BANK2_f7539ed36c037c34'+suffix,
            local_output_dir=tmp_path,output_dataset_ids={'gpx_projects':'collection-'+str(run)},inputs=None)
        summary=dict(gpx_projects=[project],acquisition=dict(instrument=instrument,run_number=metadata_run or run,
            sample_id='77003',sample_name='Iron oxide'),provenance={'input':{'source_bank':bank}})
        return source_from_record(record,{'summary':summary},None,'history')
    first,second=recovered(220653),recovered(220654)
    assert first['experiment']==second['experiment']
    assert first['experiment_label']=='NOM · IPTS-34537' and first['bank']=='2'
    for conflicting in [recovered(220654,instrument='PG3'),recovered(220654,bank=3),
                        recovered(220654,metadata_run=220650),recovered(220654,suffix='_unverified')]:
        assert conflicting['experiment']!=first['experiment']


def test_distinct_nomad_slices_of_one_parent_stay_distinct_after_recovery(tmp_path):
    project=dict(collection_name='final.gpx',sha256='model',status='accepted',stage='final_refinement')
    def recovered(product):
        record=SimpleNamespace(uid=product,name=f'IPTS-34537_NOM_208518_{product}_BANK2_f7539ed36c037c34',
            local_output_dir=tmp_path,output_dataset_ids={'gpx_projects':'collection-'+product},inputs=None)
        summary=dict(gpx_projects=[project],acquisition=dict(instrument='NOM',run_number=208518,
            sample_id='77003',sample_name='Iron oxide'),provenance={'input':{'source_bank':2,'product_id':product}})
        return source_from_record(record,{'summary':summary},None,'history')
    first,second=recovered('SLICE_100'),recovered('SLICE_103')
    assert first['experiment']==second['experiment'] and first['sample_key']==second['sample_key']
    assert first['experiment_label']=='NOM · IPTS-34537'
    assert first['scan_key']!=second['scan_key'] and first['id']!=second['id']
    assert first['run_id']=='NOM_208518_SLICE_100'
    assert second['metadata']['product_id']=='SLICE_103'


def test_refinement_recovers_completed_live_job_outputs_after_viewer_restart():
    queued=SimpleNamespace(uid='job-43124',name='IPTS-2767_PG3_43124',status=SimpleNamespace(value='queued'),output_dataset_ids={})
    complete=copy.deepcopy(queued)
    complete.status=SimpleNamespace(value='ok')
    complete.output_dataset_ids={'summary':'small-summary','gpx_projects':'published-projects'}
    calls=[]
    service=SimpleNamespace(refresh=lambda record: calls.append(record.uid) or complete)
    run=SimpleNamespace(galaxy_job_id='job-43124')
    controller=SimpleNamespace(state=SimpleNamespace(completed={'PG3_43124':run}),records={'PG3_43124':queued})
    assert recover_completed_refinement_records(service,controller)==[]
    assert controller.records['PG3_43124'] is complete
    assert calls==['job-43124']
    assert recover_completed_refinement_records(service,controller)==[]
    assert calls==['job-43124']


def test_recovered_publications_keep_older_refinement_results_browsable(tmp_path):
    w,_=create(tmp_path)
    w.observe([dict(source('43124',exp='PG3-IPTS-2767'),experiment_label='PG3 · IPTS-2767')])
    w.state['jobs']['old']=dict(source=dict(source('43124',exp='old-single-run'),experiment_label='IPTS-2767_PG3_43124'))
    labels=refinement_experiment_options(w)
    assert labels['PG3-IPTS-2767']=='PG3 · IPTS-2767'
    assert labels['old-single-run']=='Earlier refinement results · IPTS-2767_PG3_43124'


def test_live_watch_publication_replaces_earlier_test_job_in_recipe_scope(tmp_path):
    w,_=create(tmp_path)
    manual=dict(source('test43106',exp='ages'),live_watch=False)
    watched=dict(source('watch43106',exp='ages'),live_watch=True)
    w.observe([manual,watched])
    assert len(w.state['sources'])==2  # Both publications retain provenance.
    assert [row['id'] for row in w.sources('ages')]==['watch43106']
    scope=RefinementScope('ages')
    snapshot,_=w.selection(scope)
    assert [row['id'] for row in snapshot['baseline']]==['watch43106']


def test_switching_live_reduction_retires_stale_watch_without_erasing_provenance(tmp_path):
    w,_=create(tmp_path)
    old=dict(source('modern43106',exp='ages'),live_watch=True,
             collection_id='modern',collection_name='final.gpx',internal_source_sha256='modern-hash',
             metadata={'reduction_family':'modern'})
    new=dict(source('original43106',exp='ages'),live_watch=True,
             collection_id='original',collection_name='final.gpx',internal_source_sha256='original-hash',
             metadata={'reduction_family':'original'})
    w.observe([old],active_live_ids={old['id']})
    # A recovered summary must not keep the previous controller's watch active.
    recovered=dict(old,live_watch=False,metadata={})
    w.observe([recovered,new],active_live_ids={new['id']})
    assert [row['id'] for row in w.sources('ages')]==[new['id']]
    assert w.state['sources'][old['id']]['metadata']==old['metadata']
    assert not w.state['sources'][old['id']]['live_watch']
    snapshot,_=w.selection(RefinementScope('ages'))
    assert [row['id'] for row in snapshot['baseline']]==[new['id']]
    # Incremental materialization cannot reactivate the prior reduction.
    w.observe([dict(w.state['sources'][old['id']],context=context(old))])
    assert [row['id'] for row in w.sources('ages')]==[new['id']]
    restored=RefinementWorkspace(w.root,history_id='history',runtime=w.runtime)
    assert [row['id'] for row in restored.sources('ages')]==[new['id']]
    w.observe([],active_live_ids=set())
    assert not any(row.get('live_watch') for row in w.state['sources'].values())


def test_summary_refresh_retains_inspected_atom_context_for_immutable_collection(tmp_path):
    w,_=create(tmp_path)
    summary=dict(source('43124'),source_sha256=None,source_path=None,collection_id='collection-43124',
                 collection_name='accepted.gpx',internal_source_sha256='internal-hash')
    w.observe([summary])
    material=dict(summary,source_sha256='actual-published-hash',context=context(dict(source_sha256='actual-published-hash')))
    w.observe([material])
    _,approved_token=w.selection(RefinementScope('exp'))
    w.observe([summary])
    restored=w.state['sources'][summary['id']]
    assert restored['source_sha256']=='actual-published-hash'
    assert restored['context']['phases'][0]['name']=='oxide'
    assert w.selection(RefinementScope('exp'))[1]==approved_token


def test_refinement_catalog_reads_only_summary_until_source_gpx_is_selected(tmp_path):
    project=dict(collection_name='final.gpx',sha256='model',status='accepted',stage='final_refinement')
    summary=dict(gpx_projects=[project],acquisition=dict(instrument='PG3',run_number=43124,sample_id='21213'),
                 provenance={'input':{'source_bank':2}})
    class Service:
        def __init__(self):self.downloaded=[]
        def _download_dataset(self,dataset,target):
            self.downloaded.append(dataset);target.write_text(json.dumps(summary))
        def collect_results(self,record):raise AssertionError('Full result collection is too costly for a catalog')
    service=Service();workspace=SimpleNamespace(root=tmp_path,history_id='history')
    record=SimpleNamespace(uid='job',name='IPTS-2767_PG3_43124',local_output_dir=None,
                           output_dataset_ids={'summary':'summary-dataset','gpx_projects':'gpx-collection'},inputs=None)
    first=published_refinement_source(service,workspace,record,None,{})
    second=published_refinement_source(service,workspace,record,None,{})
    assert first['sample_key']=='id:21213' and first['experiment_label']=='PG3 · IPTS-2767'
    assert first['id']==second['id'] and service.downloaded==['summary-dataset']


def test_needs_review_explains_unphysical_uiso_and_keeps_last_verified_stage_visible():
    baseline={'phases':[{'name':'Al2O3','atoms':[{'label':'Al1'},{'label':'O1'}]}]}
    result={'status':'needs_review','error':("Native dependent parameters violated physical/requested bounds before save-time clipping: "
            "[('0::AUiso:1', -0.002485644520797271)]. Later stages were not executed"),
            'stages':[{'id':f'stage-{i}','gpx':f'verified_stage-{i}.gpx'} for i in range(1,4)],
            'stage_dataset_ids':{'stage-3':'saved-stage-three'}}
    message=refinement_status_message(result,baseline)
    assert 'Al2O3 · O1 Uiso = -0.0024856 Å²' in message
    assert '3 earlier verified stages' in message and 'no final GPX' in message
    assert '0::AUiso:1' not in message
    assert selected_refinement_dataset(result,'stage-3')=='saved-stage-three'
    assert 'dataset saved-stage-three' in refinement_output_provenance(result,'stage-3')
    assert selected_refinement_dataset(result,'final') is None


def test_rwp_summary_does_not_imply_improvement_when_fit_range_changes():
    comparable={'original_rwp':3.381,'polished_rwp':3.202,'rwp_comparable':True}
    assert '→' in refinement_rwp_summary(comparable)
    changed={**comparable,'polished_rwp':8.202,'rwp_comparable':False}
    label=refinement_rwp_summary(changed)
    assert 'different fit range' in label and '→' not in label


def test_completed_refinement_reports_verified_stage_count_and_final_gpx():
    message=refinement_status_message(dict(status='completed',final_gpx='final.gpx',stages=[{}, {}, {}, {}]),None)
    assert '4 verified stages completed' in message
    assert 'Final GPX ready for scientific review' in message


def test_switching_to_running_scan_clears_the_previous_scans_parameters_and_outputs():
    snapshots=[]
    class State(SimpleNamespace):
        def flush(self):snapshots.append(copy.deepcopy(vars(self)))
    class App(RefinementMixin):
        def _refinement_plot_changed(self):pass
        def _refinement_trend_changed(self):pass
    job=dict(id='next',source=dict(experiment='exp',scan_label='Run 20',sample_label='sample'),
             recipe_id='r3',recipes=['r3'],status='running',message='Refining the approved stages')
    app=App();app.server=SimpleNamespace(state=State(rf_experiment='exp',rf_selected_job='earlier',
        rf_parameters=[{'label':'earlier scan cell','value':5.1}],rf_downloads=[{'value':'earlier.gpx'}],
        rf_download='earlier.gpx',rf_has_output=True,rf_can_open_gsas=True,
        rf_rwp_summary='Earlier score',rf_fit_notice='Earlier observations'))
    app._refinement_workspace=SimpleNamespace(state={'jobs':{'next':job},'recipes':{'r3':{'version':3}}})
    asyncio.run(app._select_refinement_job('next'))
    assert snapshots and all(row['rf_selected_job']=='next' for row in snapshots)
    assert all(not row['rf_parameters'] and not row['rf_downloads'] and not row['rf_has_output']
               and not row['rf_can_open_gsas'] for row in snapshots)
    assert all(row['rf_selected_title']=='Run 20 · sample · Recipe 3' for row in snapshots)
    assert app.server.state.rf_selected_message=='Refining the approved stages'
    assert not app.server.state.rf_rwp_summary and not app.server.state.rf_fit_notice


def test_reused_result_heading_identifies_each_actual_recipe():
    job={'source':{'scan_label':'Run 20','sample_label':'sample'},'recipes':['first','reused']}
    assert refinement_result_title(job,{'first':{'version':1},'reused':{'version':4}}).endswith('Recipe 1, 4')
    assert refinement_result_title(job,{}).endswith('Recipe unavailable')


@pytest.mark.parametrize('selected,follow,experiment,expected',[
    ('current',False,'exp',['current']),
    ('earlier',False,'exp',[]),
    ('earlier',True,'exp',['current']),
    ('',False,'exp',['current']),
    ('current',True,'other',[]),
])
def test_running_scan_details_refresh_on_completion_without_following_other_scans(selected,follow,experiment,expected):
    class State(SimpleNamespace):
        def flush(self):pass
    completed=dict(id='current',source=dict(experiment='exp'),status='completed')
    queue=iter([completed,None]);seen=[]
    class App(RefinementMixin):
        def _refinement_sync(self):pass
        def _refinement_checkpoint(self):pass
        async def _select_refinement_job(self,uid):seen.append(uid)
    app=App();app._refinement_workspace=SimpleNamespace(run_next=lambda:next(queue))
    app.server=SimpleNamespace(state=State(rf_experiment=experiment,rf_follow=follow,rf_selected_job=selected,rf_error=''))
    asyncio.run(app._refinement_pump())
    assert seen==expected
    assert not app.server.state.rf_queue_running and not app.server.state.rf_error


def test_weak_uiso_warning_uses_actual_conditional_uncertainty_only():
    atom=dict(parameter='Uiso',label='Al2O3 · Al1 Uiso',value=.00340981,sigma=.00286251,status='refined')
    warnings=refinement_parameter_warnings(dict(parameters=[atom]))
    assert len(warnings)==1 and 'conditional 1σ' in warnings[0] and 'weakly constrained' in warnings[0]
    assert not refinement_parameter_warnings(dict(parameters=[{**atom,'value':.02}]))
    assert not refinement_parameter_warnings(dict(parameters=[{**atom,'status':'fixed'}]))
    assert not refinement_parameter_warnings(dict(parameters=[{**atom,'parameter':'z'}]))
    for sigma in [None,0,float('nan')]:
        assert 'no usable conditional uncertainty' in refinement_parameter_warnings(dict(parameters=[{**atom,'sigma':sigma}]))[0]


def test_approval_holds_representative_changed_after_preflight(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);d=w.prepare('exp','1','background')
    w.observe([source('1',checksum='changed')]);s=RefinementScope('exp');_,token=w.selection(s)
    with pytest.raises(ValueError,match='representative GPX changed'):w.approve('exp',d['generation'],s,existing=True,future=False,selection_token=token,approver='test')
    assert not w.state['recipes']


def test_future_arrivals_out_of_order_and_repeat_restoration_do_not_repeat_work(tmp_path):
    w,rt=create(tmp_path);w.observe([source('10')]);r=approved(w,RefinementScope('exp'),future=True);w.start(r['id'],r['selection_token'])
    w.observe([source('13'),source('11'),source('12')])
    while w.run_next():pass
    restored=RefinementWorkspace(w.root,history_id='history',runtime=rt)
    restored.observe([source('12'),source('13'),source('11')]);assert restored.run_next() is None
    assert len(restored.state['jobs'])==4 and all(j['attempts']==1 for j in restored.state['jobs'].values())
    assert len([c for c in rt.calls if c[0]=='preview'])==1


def test_a_recipe_revision_disarms_older_future_enrollment_but_keeps_old_outputs(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);r=approved(w,RefinementScope('exp'),future=True);w.start(r['id'],r['selection_token']);w.run_next()
    w.prepare('exp','1','revised background');w.observe([source('2')])
    assert not w.state['recipes'][r['id']]['armed']
    assert len(w.state['jobs'])==1 and next(iter(w.state['jobs'].values()))['status']=='completed'


def test_queued_copy_cannot_change_when_current_catalog_revision_changes(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);r=approved(w,RefinementScope('exp'));jobs=w.start(r['id'],r['selection_token'])
    w.observe([source('1',checksum='replacement')]);w.run_next()
    assert rt.calls[-1][1]=='1'
    assert w.state['jobs'][jobs[0]['id']]['source']['source_sha256']=='1hash'
    assert w.state['sources']['1']['source_sha256']=='replacement'


def test_remote_job_recovers_existing_dispatch_instead_of_creating_new_attempt_identity(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);r=approved(w,RefinementScope('exp'));job=w.start(r['id'],r['selection_token'])[0]
    w.state['jobs'][job['id']].update(status='running',attempts=1,remote_action={'uid':'existing-Galaxy-job'});w.save()
    restored=RefinementWorkspace(w.root,history_id='history',runtime=rt);assert restored.state['jobs'][job['id']]['status']=='recovering'
    restored.run_next();assert rt.calls[-1][2]['resume']=={'uid':'existing-Galaxy-job'}
    assert rt.calls[-1][2]['dispatch_key']==job['id']


def test_result_archive_cannot_reference_files_outside_its_workspace(tmp_path):
    secret=tmp_path/'outside.json';secret.write_text('{}');directory=tmp_path/'job';directory.mkdir()
    with pytest.raises(ValueError,match='changed or is missing'):verify_result_files(directory,{'files':{'../outside.json':file_sha(secret)}})


def test_partial_native_failure_exposes_only_completed_checkpoint_never_final(tmp_path,monkeypatch):
    from gpx_recipe import worker,compiler
    source=tmp_path/'source.gpx';source.write_bytes(b'original')
    directory=tmp_path/'worker'
    def compile_partial(plan,approval,target):
        code="""from pathlib import Path
import json
p=Path(__file__).resolve().parent if '__file__' in globals() else Path(__import__('sys').argv[0]).parent
(p/'polished.gpx').write_bytes(b'partial failed stage')
(p/'polished_first.gpx').write_bytes(b'completed stage')
(p/'polished.execution.json').write_text(json.dumps({'stages':[{'id':'first','title':'Completed','status':'completed'},{'id':'second','title':'Failed','status':'failed'}]}))
raise ValueError('Scientific bound stop')
"""
        target.write_text(code);return code
    def export(source,target):target.write_text('{}');return {}
    monkeypatch.setattr(compiler,'compile_script',compile_partial);monkeypatch.setattr(worker,'write_project_view',export)
    result=worker.work('run',source,directory,plan={'stages':[]},approval={})
    assert result['status']=='needs_review'
    assert [s['id'] for s in result['stages']]==['first']
    assert not result.get('final_gpx') and not (directory/'verified_final.gpx').exists()
    assert (directory/'verified_first.gpx').read_bytes()==b'completed stage'
    assert not any('stopped' in name for name in result['files'])
    assert source.read_bytes()==b'original'


def test_integrated_ui_has_one_workspace_and_reuses_existing_scientific_tabs():
    from radar_pd_nova.app import RadarPdNovaApp
    app=RadarPdNovaApp();template=app.layout.html
    assert template.count('<section class="rf-workspace"')==1
    assert 'Transition tracking' in template and 'Refinement' in template
    assert "'rf-scan ' +" in template and ':class=""rf-scan' not in template
    assert 'AZURE_ILLUMINE_API_KEY' not in template and 'REFINEMENT_KEY_FILE' not in template
    assert app.server.state.rf_future is False


def test_plan_click_uses_visible_request_when_text_update_is_still_pending():
    class State(SimpleNamespace):
        def flush(self): pass
    calls=[];tasks=[]
    class App(RefinementMixin):
        def _refinement_hold_enrollment(self): pass
        def _refinement_sync(self): self.server.state.rf_busy=False
        def _refinement_checkpoint(self): pass
        def _refinement_schedule(self, coroutine, name): tasks.append(asyncio.create_task(coroutine))
    app=App()
    app.server=SimpleNamespace(state=State(rf_busy=False,rf_experiment='exp',rf_source='source',rf_request='previous text'))
    app._refinement_workspace=SimpleNamespace(prepare=lambda *args:calls.append(args))
    async def run():
        app.generate_refinement_recipe(dict(experiment='exp',source='source',request='visible edited text'))
        await asyncio.gather(*tasks)
    asyncio.run(run())
    assert calls==[('exp','source','visible edited text')]
    assert app.server.state.rf_request=='visible edited text'
    app.generate_refinement_recipe(dict(experiment='another',source='source',request='stale selection'))
    assert 'selected project changed' in app.server.state.rf_error


def test_missing_credential_fails_before_a_native_inspection_job(tmp_path,monkeypatch):
    from gpx_recipe import providers
    w,rt=create(tmp_path);w._planner=None;w.observe([source('1')])
    monkeypatch.delenv('AZURE_ILLUMINE_API_KEY',raising=False)
    monkeypatch.delenv('RADAR_PD_REFINEMENT_KEY_FILE',raising=False)
    monkeypatch.setenv('RADAR_GPX_PLANNER_PROVIDER','illumine-luna')
    with pytest.raises(ValueError,match='session started without an Illumine credential'):
        w.prepare('exp','1','background')
    assert rt.calls==[]


def test_remote_published_metadata_can_list_a_source_without_downloading_its_full_archive(tmp_path):
    checksum='a'*64
    record=SimpleNamespace(uid='Galaxy-job',name='Run 12',local_output_dir=None,output_dataset_ids={'gpx_projects':'collection'},inputs=SimpleNamespace(instrument='PG3',ipts='IPTS-1',run_number=12,bank=2))
    payload=dict(gpx_index={'projects':[dict(stage='main_phase_anchor',status='accepted',sha256='b'*64,published_sha256=checksum,collection_path='gpx/final.gpx',collection_name='final.gpx')]})
    s=source_from_record(record,payload,None,'history',root_override=tmp_path)
    assert s['source_path'] is None and s['source_sha256']==checksum and s['collection_id']=='collection'
    assert not list(tmp_path.iterdir())


def test_galaxy_claim_is_backed_up_before_any_native_submission(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);r=approved(w,RefinementScope('exp'));w.start(r['id'],r['selection_token']);rt.kind='galaxy'
    snapshots=[]
    rt.save_workspace=lambda state:snapshots.append(state)
    original=rt.call
    def call(action,source,directory,**kwargs):
        assert snapshots and next(iter(snapshots[-1]['jobs'].values()))['status']=='running'
        kwargs['on_submitted']({'uid':'acknowledged-job'})
        assert next(iter(snapshots[-1]['jobs'].values()))['remote_action']=={'uid':'acknowledged-job'}
        return original(action,source,directory,**kwargs)
    rt.call=call
    assert w.run_next()['status']=='completed'
    assert len(snapshots)==2


def test_failed_galaxy_recovery_backup_prevents_submission(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);r=approved(w,RefinementScope('exp'));w.start(r['id'],r['selection_token']);rt.kind='galaxy'
    def fail(state):raise RuntimeError('History backup unavailable')
    rt.save_workspace=fail
    assert w.run_next()['status']=='failed'
    assert not [c for c in rt.calls if c[0]=='run']


def test_saved_recipe_can_be_preflighted_for_another_compatible_source_without_model_call(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1'),source('2')]);r=approved(w,RefinementScope('exp'))
    w._planner=lambda *a,**k:pytest.fail('Reusing a saved recipe must not call a model')
    d=w.prepare('exp','2',r['plan']['request'],reuse_recipe=r['id'])
    assert d['status']=='ready' and d['planning_calls']==0 and d['source_id']=='2'
    assert rt.calls[-1][0]=='preview'
