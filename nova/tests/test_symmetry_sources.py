import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from radar_pd_nova.refinement_sources import file_sha
from radar_pd_nova.symmetry_sources import start_choices,resolve_start,acquisition_details,recipe_source


def workspace(tmp_path):
    raw=tmp_path/'original.gpx';raw.write_bytes(b'original native GPX')
    result_dir=tmp_path/'result';result_dir.mkdir()
    final=result_dir/'polished.gpx';final.write_bytes(b'verified polished native GPX')
    stage=result_dir/'stage1.gpx';stage.write_bytes(b'verified stage native GPX')
    source=dict(id='raw',experiment='IPTS-1',scan_label='Run 123 bank 3',source_path=str(raw),
                source_sha256=file_sha(raw),sample_label='Cr sample',collection_id='ORIGINAL_COLLECTION',
                dataset_id='ORIGINAL_DATASET',project={'stage':'final_refinement'},base_rwp=25.,
                context={'original':'cached project'},inspection='original-inspection',inspection_error=None)
    result=dict(status='completed',directory=str(result_dir),files={'polished.gpx':file_sha(final),'stage1.gpx':file_sha(stage)},
                final_gpx='polished.gpx',output_sha256=file_sha(final),final_dataset_id='POLISHED_DATASET',
                stages=[dict(id='s1',title='First stage',gpx='stage1.gpx')],stage_dataset_ids={'s1':'STAGE_DATASET'})
    w=SimpleNamespace(state=dict(sources={'raw':source},jobs={'job1':dict(status='completed',source=source,result=result)}))
    w.sources=lambda experiment:[dict(s) for s in w.state['sources'].values() if s['experiment']==experiment]
    return w


def test_recipe_checkpoints_are_explicit_and_never_enroll_as_new_radar_sources(tmp_path):
    w=workspace(tmp_path);before=copy.deepcopy(w.state['sources'])
    entries=start_choices(w,'IPTS-1')
    assert len(entries)==3 and entries[0]['id']=='raw'
    final=next(e for e in entries if e.get('stage')=='final')
    source=resolve_start(w,'IPTS-1',final['id'])
    assert source['id'].startswith('recipe:') and source['origin_source_id']=='raw'
    assert source['source_sha256']==file_sha(Path(source['source_path']))
    assert source['dataset_id']=='POLISHED_DATASET' and 'collection_id' not in source and 'base_rwp' not in source
    assert not {'context','inspection','inspection_error'} & set(source)
    assert w.state['sources']==before


def test_magnetic_model_conditions_show_the_scan_range_without_downloading_a_gpx(tmp_path):
    w=workspace(tmp_path)
    w.state['sources']['raw']['metadata']={'temperature':dict(value=166.5,minimum=163.,maximum=170.,unit='K',statistic='time_weighted_step_hold')}
    final=next(e for e in start_choices(w,'IPTS-1') if e.get('stage')=='final')
    w.state['jobs']['job1']['result']['directory']=str(tmp_path/'not-recovered')
    def forbidden(*args):raise AssertionError('Condition display must not download or load a GPX')
    w.recover_job_artifacts=forbidden
    for key in ['raw',final['id']]:
        details=acquisition_details(w,'IPTS-1',key)
        assert len(details)==1 and 'time-weighted mean 166.5 K' in details[0]['value']
        assert 'recorded scan range 163–170 K' in details[0]['value']
    assert acquisition_details(w,'other-experiment',final['id'])==[]


def test_final_and_stage_dataset_references_never_fall_back_to_original(tmp_path):
    w=workspace(tmp_path);entries=start_choices(w,'IPTS-1')
    stage=next(e for e in entries if e.get('stage')=='s1')
    selected=resolve_start(w,'IPTS-1',stage['id'])
    assert selected['dataset_id']=='STAGE_DATASET' and 'collection_id' not in selected
    del w.state['jobs']['job1']['result']['stage_dataset_ids']
    selected=resolve_start(w,'IPTS-1',stage['id'])
    assert 'dataset_id' not in selected


@pytest.mark.parametrize('state',['running','failed','interrupted'])
def test_uncompleted_recipe_outputs_are_not_offered(tmp_path,state):
    w=workspace(tmp_path);w.state['jobs']['job1']['status']=state
    assert [e['id'] for e in start_choices(w,'IPTS-1')]==['raw']


def test_tampered_or_out_of_scope_checkpoint_cannot_be_inspected(tmp_path):
    w=workspace(tmp_path);entry=next(e for e in start_choices(w,'IPTS-1') if e.get('stage')=='final')
    with pytest.raises(ValueError,match='outside'):resolve_start(w,'IPTS-2',entry['id'])
    Path(w.state['jobs']['job1']['result']['directory'],'polished.gpx').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):resolve_start(w,'IPTS-1',entry['id'])


def test_changed_receipt_identity_requires_reselection(tmp_path):
    w=workspace(tmp_path);entry=next(e for e in start_choices(w,'IPTS-1') if e.get('stage')=='final')
    w.state['jobs']['job1']['result']['files']['polished.gpx']='0'*64
    with pytest.raises(ValueError,match='changed'):resolve_start(w,'IPTS-1',entry['id'])


def test_missing_local_artifacts_are_recovered_and_reverified(tmp_path):
    w=workspace(tmp_path);entry=next(e for e in start_choices(w,'IPTS-1') if e.get('stage')=='final')
    result=w.state['jobs']['job1']['result'];actual=result['directory'];result['directory']=str(tmp_path/'missing')
    w.state['jobs']['job1']['remote_action']={'id':'existing-galaxy-job'}
    called=[]
    def recover(uid):
        called.append(uid);result['directory']=actual
    w.recover_job_artifacts=recover
    assert resolve_start(w,'IPTS-1',entry['id'])['dataset_id']=='POLISHED_DATASET'
    assert called==['job1']


@pytest.mark.parametrize('change',['selection','checksum',None])
def test_native_inspection_cannot_apply_a_previous_source_to_a_new_selection(tmp_path,monkeypatch,change):
    import asyncio
    import json
    import sys
    import radar_pd_nova.symmetry_controller as controller
    w=workspace(tmp_path)
    w.runtime=SimpleNamespace(resolve=lambda source:source)
    class State(SimpleNamespace):
        def flush(self):pass
    state=State(rf_experiment='IPTS-1',sx_start_source='raw',sx_has_context=False)
    app=controller.SymmetryMixin()
    app.server=SimpleNamespace(state=state)
    app._model_preparations={};app._model_remote=lambda:None
    app._refinement_store=lambda:w
    app._symmetry_root=lambda:tmp_path/'symmetry'
    app._symmetry_transport=lambda:SimpleNamespace(command=[sys.executable],environment={})
    app._refinement_schedule=lambda coroutine,*args:asyncio.run(coroutine)
    monkeypatch.setattr(controller,'require_runtime',lambda *args:None)
    def native(command,**kwargs):
        if change=='selection':state.sx_start_source='another-source'
        context=dict(sha256='0'*64 if change=='checksum' else w.state['sources']['raw']['source_sha256'],
            runtime={'magnetic_function_sha256':'b'*64},
            phases=[dict(name='Parent',space_group='P n m a',type='nuclear',atoms=[])],
            histograms=[dict(name='PWDR sample')])
        Path(command[-1]).write_text(json.dumps(context))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(controller,'run_scientific',native)
    app.inspect_symmetry_source()
    if change:
        assert not state.sx_has_context
        assert 'changed during' in state.sx_error
    else:
        assert state.sx_has_context and app._symmetry_draft['source_id']=='raw'


@pytest.mark.parametrize('change_during',['portfolio','grouping','duplicate_poll','duplicate_before_scheduled'])
def test_experiment_change_during_refresh_reloads_instead_of_publishing_stale_results(tmp_path,monkeypatch,change_during):
    import asyncio
    import radar_pd_nova.symmetry_controller as controller
    class State(SimpleNamespace):
        def flush(self):pass
    state=State(rf_experiment='IPTS-1',sx_result_scope='experiment',sx_start_source='',rf_source='',
                sx_campaign='campaign',sx_source='')
    app=controller.SymmetryMixin();app.server=SimpleNamespace(state=state)
    app._model_remote=lambda:None
    app._symmetry_loading=False;app._symmetry_refresh_pending=False
    app._symmetry_display_identity=('campaign','IPTS-1','experiment')
    app._refinement_store=lambda:None;app._symmetry_root=lambda:tmp_path
    root=tmp_path/'campaign';root.mkdir();(root/'campaign.sqlite').touch()
    monkeypatch.setattr(controller,'start_choices',lambda *args:[])
    monkeypatch.setattr(controller,'coverage_rows',lambda rows:[])
    requested=[];published=[]
    def change():
        state.rf_experiment='IPTS-2';app.refresh_symmetry()
    def read(root,*,experiment):
        requested.append(experiment)
        if experiment=='IPTS-1' and change_during=='portfolio':change()
        if change_during=='duplicate_poll':
            assert state.sx_results_current
            app.refresh_symmetry()
        return [dict(source=experiment,parent='Parent',metadata=dict(series=experiment),status='prepared',execution_state='prepared')],[]
    monkeypatch.setattr(controller,'campaign_rows',read)
    def group(source):
        if source=='IPTS-1' and change_during=='grouping':change()
    app._symmetry_source_data=group
    app._symmetry_publish_source=lambda:published.append((state.rf_experiment,app._symmetry_all_rows[0]['source']))
    async def run():
        tasks=[]
        app._refinement_schedule=lambda coroutine,*args:tasks.append(asyncio.create_task(coroutine))
        app.refresh_symmetry()
        if change_during=='duplicate_before_scheduled':
            for _ in range(8):app.refresh_symmetry()
            assert len(tasks)==1
        while tasks:
            pending=list(tasks);tasks.clear();await asyncio.gather(*pending)
    asyncio.run(run())
    if change_during in {'duplicate_poll','duplicate_before_scheduled'}:
        assert requested==['IPTS-1'] and published==[('IPTS-1','IPTS-1')]
    else:
        assert requested==['IPTS-1','IPTS-2']
        assert published==[('IPTS-2','IPTS-2')]
    assert state.sx_results_current
    assert not app._symmetry_loading and not app._symmetry_refresh_pending


def test_model_choices_show_qualified_measured_temperature_without_changing_source_identity(tmp_path):
    w=workspace(tmp_path);source=w.state['sources']['raw']
    source['metadata']={'conditions':{'channels':[{'quantity':'temperature','value':.27992,'unit':'K','role':'measured','status':'recorded','channel':'He3'}]}}
    before=copy.deepcopy(source);entries=start_choices(w,'IPTS-1')
    assert entries[0]['title'].endswith('0.2799 K') and entries[0]['id']=='raw'
    assert w.state['sources']['raw']==before
    source['metadata']['conditions']['channels'][0]['role']='setpoint'
    assert not start_choices(w,'IPTS-1')[0]['title'].endswith('0.2799 K')


def recovered_conditions_workspace(tmp_path):
    w=workspace(tmp_path)
    w.state['jobs']['job1']['recipe_id']='r1'
    w.state['jobs']['job1']['source']=copy.deepcopy(w.state['sources']['raw'])
    w.state['sources']['raw']['metadata']={'conditions':{'channels':[
        dict(quantity='temperature',value=240.,minimum=239.,maximum=241.,unit='K',
             statistic='time_weighted_step_hold',role='measured',status='recorded')]}}
    return w


def test_derived_checkpoint_can_recover_missing_conditions_from_exact_original_bytes(tmp_path):
    from radar_pd_nova.model_campaigns import choices
    w=recovered_conditions_workspace(tmp_path);before=copy.deepcopy(w.state)
    final=next(e for e in start_choices(w,'IPTS-1') if e.get('stage')=='final')
    assert '240 K' in final['title']
    assert '239–241 K' in acquisition_details(w,'IPTS-1',final['id'])[0]['value']
    for source in [resolve_start(w,'IPTS-1',final['id']),
                   next(s for s,kind in choices(w,'IPTS-1') if s['id']==final['id'])]:
        assert source['metadata']['temperature']['value']==240.
        assert source['acquisition_metadata_recovery']['source_sha256']==w.state['sources']['raw']['source_sha256']
    assert w.state==before  # Neither job receipts nor incoming sources are rewritten.


@pytest.mark.parametrize('field',['source_sha256','experiment','scan_key','sample_key','run_number','bank',
                                  'collection_id','collection_name','internal_source_sha256','published_sha256'])
def test_recipe_condition_recovery_rejects_conflicting_original_identity(tmp_path,field):
    w=recovered_conditions_workspace(tmp_path)
    w.state['sources']['raw'][field]='DIFFERENT'
    source=recipe_source(w,w.state['jobs']['job1'])
    assert 'temperature' not in source.get('metadata',{}) and 'acquisition_metadata_recovery' not in source


@pytest.mark.parametrize('old',['scalar','rejected','recorded','observation'])
def test_recipe_recovery_preserves_original_observations_including_rejections(tmp_path,old):
    w=recovered_conditions_workspace(tmp_path)
    metadata={'scalar':{'temperature':120.},
              'recorded':{'temperature':dict(value=120.,unit='K')},
              'observation':{'temperature_observation':dict(value=120.,unit='K')},
              'rejected':{'conditions':{'channels':[dict(quantity='temperature',role='measured',status='unqualified_units')]}}}[old]
    w.state['jobs']['job1']['source']['metadata']=metadata
    source=recipe_source(w,w.state['jobs']['job1'])
    assert source['metadata']==metadata and 'acquisition_metadata_recovery' not in source


@pytest.mark.parametrize('status,role',[('needs_mapping','measured'),('recorded','setpoint'),('unqualified_units','measured')])
def test_recipe_recovery_never_promotes_an_unqualified_or_setpoint_channel(tmp_path,status,role):
    w=recovered_conditions_workspace(tmp_path)
    w.state['sources']['raw']['metadata']['conditions']['channels'][0].update(status=status,role=role)
    assert 'temperature' not in recipe_source(w,w.state['jobs']['job1']).get('metadata',{})


@pytest.mark.parametrize('restored,remote,active',[(True,True,False),(False,True,False),(True,False,False),(True,True,True)])
def test_refresh_recovers_restored_remote_evidence_without_resuming_or_submitting(tmp_path,monkeypatch,restored,remote,active):
    import asyncio
    import radar_pd_nova.symmetry_controller as controller
    from gpx_recipe.symmetry.campaign import Campaign
    class State(SimpleNamespace):
        def flush(self):pass
    state=State(rf_experiment='IPTS-1',sx_result_scope='experiment',sx_start_source='',rf_source='',
                sx_campaign='campaign',sx_source='')
    app=controller.SymmetryMixin();app.server=SimpleNamespace(state=state)
    app._models_restored=True;app._model_remote=lambda:object() if remote else None
    app._symmetry_loading=False;app._symmetry_refresh_pending=False
    app._symmetry_display_identity=None
    app._refinement_store=lambda:None;app._symmetry_root=lambda:tmp_path
    c=Campaign(tmp_path/'campaign');c.set_setting('restored',restored);c.set_setting('paused',True)
    app._model_active_campaigns={'campaign':(c,None)} if active else {}
    c.set_setting('template',{'scope':{'experiment':'IPTS-1'},'protocol':{'implementation_sha256':'old-worker'}})
    events=[]
    class ReadOnlyTransport:
        def submit_many(self,*args,**kwargs):raise AssertionError('Refreshing evidence must not submit refinements')
        def submit(self,*args,**kwargs):raise AssertionError('Refreshing evidence must not submit refinements')
    transport=ReadOnlyTransport();app._symmetry_transport=lambda campaign:transport
    original_recover=Campaign.recover
    def recover(campaign,t):
        assert t is transport and campaign.setting('paused') is True
        if active:assert 'while this campaign continues' in state.sx_message
        else:assert 'No new refinements are being submitted' in state.sx_message
        events.append('recover')
        return original_recover(campaign,t)
    monkeypatch.setattr(Campaign,'recover',recover)
    def cannot_continue(*args,**kwargs):raise AssertionError('Viewing old results must not continue their worker')
    monkeypatch.setattr(Campaign,'verify_sealed_continuation',cannot_continue)
    monkeypatch.setattr(controller,'start_choices',lambda *args:[])
    monkeypatch.setattr(controller,'coverage_rows',lambda rows:[])
    def read(root,*,experiment):
        events.append('read')
        return [],[]
    monkeypatch.setattr(controller,'campaign_rows',read)
    app._symmetry_source_data=lambda *args:None
    app._symmetry_publish_source=lambda:None
    async def run():
        tasks=[]
        app._refinement_schedule=lambda coroutine,*args:tasks.append(asyncio.create_task(coroutine))
        app.refresh_symmetry()
        await asyncio.gather(*tasks)
    asyncio.run(run())
    assert events==(['recover','read'] if restored and remote else ['read'])
    assert c.setting('paused') is True and state.sx_error==''


def test_recipe_versions_and_byte_revisions_disambiguate_the_same_scan(tmp_path):
    w=workspace(tmp_path)
    w.state['recipes']={'r1':{'version':1},'r2':{'version':2}}
    w.state['jobs']['job1']['recipe_id']='r1'
    w.state['jobs']['job2']=copy.deepcopy(w.state['jobs']['job1'])
    w.state['jobs']['job2']['recipe_id']='r2'
    choices=[e for e in start_choices(w,'IPTS-1') if e.get('stage')=='final']
    assert len({e['title'] for e in choices})==2
    assert choices[0]['title'].startswith('Recipe 1') and choices[1]['title'].startswith('Recipe 2')
    assert resolve_start(w,'IPTS-1',choices[1]['id'])['scan_label'].endswith('Recipe 2 · final')


def test_new_search_cannot_reuse_an_old_approval():
    from radar_pd_nova.symmetry_controller import SymmetryMixin
    app=SymmetryMixin();app._symmetry_draft=None;app._symmetry_approved='previous-plan'
    state=SimpleNamespace(sx_approved=True,sx_plan_ready=True,sx_plan_token='old-token',sx_setup=False)
    app.server=SimpleNamespace(state=state)
    app.new_symmetry_search()
    assert state.sx_setup and not state.sx_has_context
    assert not state.sx_approved and not state.sx_plan_ready and not state.sx_plan_token
    assert app._symmetry_approved=='' and state.sx_candidates==[]


def test_scan_navigation_is_bounded_and_never_changes_the_campaign():
    from radar_pd_nova.symmetry_controller import SymmetryMixin
    state=SimpleNamespace(sx_source_options=[{'value':'first'},{'value':'second'}],sx_source='first',sx_campaign='original')
    app=SymmetryMixin();app.server=SimpleNamespace(state=state)
    app.symmetry_adjacent_scan(-1);assert state.sx_source=='first'
    app.symmetry_adjacent_scan(1);assert state.sx_source=='second'
    app.symmetry_adjacent_scan(1);assert state.sx_source=='second'
    app.symmetry_adjacent_scan(-1);assert state.sx_source=='first' and state.sx_campaign=='original'


def test_failed_refresh_scheduling_releases_the_pending_guard():
    from radar_pd_nova.symmetry_controller import SymmetryMixin
    app=SymmetryMixin();app._symmetry_loading=False
    app.server=SimpleNamespace(state=SimpleNamespace(sx_campaign='campaign',rf_experiment='IPTS-1',sx_result_scope='experiment'))
    def fail(*args):raise RuntimeError('Event loop unavailable')
    app._refinement_schedule=fail
    with pytest.raises(RuntimeError,match='Event loop unavailable'):app.refresh_symmetry()
    assert app._symmetry_loading is False
