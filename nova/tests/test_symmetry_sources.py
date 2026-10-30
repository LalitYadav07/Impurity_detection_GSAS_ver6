import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from radar_pd_nova.refinement_sources import file_sha
from radar_pd_nova.symmetry_sources import start_choices,resolve_start


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


@pytest.mark.parametrize('change_during',['portfolio','grouping','duplicate_poll'])
def test_experiment_change_during_refresh_reloads_instead_of_publishing_stale_results(tmp_path,monkeypatch,change_during):
    import asyncio
    import radar_pd_nova.symmetry_controller as controller
    class State(SimpleNamespace):
        def flush(self):pass
    state=State(rf_experiment='IPTS-1',sx_result_scope='experiment',sx_start_source='',rf_source='',
                sx_campaign='campaign',sx_source='')
    app=controller.SymmetryMixin();app.server=SimpleNamespace(state=state)
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
        while tasks:
            pending=list(tasks);tasks.clear();await asyncio.gather(*pending)
    asyncio.run(run())
    if change_during=='duplicate_poll':
        assert requested==['IPTS-1'] and published==[('IPTS-1','IPTS-1')]
    else:
        assert requested==['IPTS-1','IPTS-2']
        assert published==[('IPTS-2','IPTS-2')]
    assert state.sx_results_current
    assert not app._symmetry_loading and not app._symmetry_refresh_pending
