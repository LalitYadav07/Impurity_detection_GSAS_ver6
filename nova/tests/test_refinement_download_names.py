"""Downloads keep verified bytes but identify the saved experiment and scan."""
import asyncio
from copy import deepcopy
import hashlib
import re
from types import SimpleNamespace

from radar_pd_nova.refinement_controller import RefinementMixin,refinement_artifact_filename,refinement_stage_positions,refinement_output_provenance


def test_recovered_recipe_keeps_original_numbers_and_prefers_approved_order():
    result={'stages':[{'id':'background'},{'id':'finish'}],
            'rejected_stages':['width'],
            'stage_outcomes':[{'id':'background'},{'id':'width'},{'id':'finish'}]}
    assert refinement_stage_positions(result)=={'background':1,'width':2,'finish':3}
    plan={'stages':[{'id':'background'},{'id':'width'},{'id':'held'},{'id':'finish'}]}
    assert refinement_stage_positions(result,plan)['finish']==4


def test_recovered_recipe_with_missing_order_is_not_renumbered():
    result={'stages':[{'id':'background'},{'id':'finish'}],'rejected_stages':['width']}
    assert refinement_stage_positions(result)=={}
    result['stage_outcomes']=[{'id':'background'},{'id':'background'}]
    assert refinement_stage_positions(result)=={}


def test_legacy_consecutive_verified_stages_keep_their_order():
    result={'stages':[{'id':'background'},{'id':'cell'}]}
    assert refinement_stage_positions(result)=={'background':1,'cell':2}


def test_output_footer_uses_approved_stage_number_after_rollback():
    result={'stages':[{'id':'background'},{'id':'finish'}],'rejected_stages':['width'],
            'stage_outcomes':[{'id':'background'},{'id':'width'},{'id':'finish'}]}
    assert 'stage 3 of the recipe' in refinement_output_provenance(result,'finish')
    result.pop('stage_outcomes')
    assert 'retained recipe stage' in refinement_output_provenance(result,'finish')
    plan={'stages':[{'id':'background'},{'id':'width'},{'id':'finish'}]}
    assert 'stage 3 of the recipe' in refinement_output_provenance(result,'finish',plan)


def test_selected_stage_defaults_to_its_own_gpx_download():
    class App(RefinementMixin):
        def _refinement_plot_changed(self):pass
    app=App()
    job={'source':{'experiment':'selected-experiment'},'result':{
        'final_gpx':'final.gpx','stages':[{'id':'finish','gpx':'stage3.gpx'}],
        'files':{'final.gpx':'final-sha','stage3.gpx':'stage-sha','execution.json':'audit-sha'}}}
    app._refinement_workspace=SimpleNamespace(state={'jobs':{'selected-job':job}})
    state=SimpleNamespace(rf_selected_job='selected-job',rf_experiment='selected-experiment',
                          rf_stage='finish',rf_download='final.gpx')
    app.server=SimpleNamespace(state=state)
    app._refinement_stage_changed()
    assert state.rf_download=='stage3.gpx'
    # A separately chosen artifact is retained while zoom/axis redraw the plot.
    state.rf_download='execution.json';app._refinement_plot_changed()
    assert state.rf_download=='execution.json'
    state.rf_stage='final';app._refinement_stage_changed()
    assert state.rf_download=='final.gpx'
    state.rf_experiment='other-experiment';state.rf_stage='finish'
    app._refinement_stage_changed()
    assert state.rf_download=='final.gpx'


def saved_job(run=64301,uid='copper-one'):
    return dict(id=uid,recipe_id='saved-recipe',recipes=['saved-recipe'],source=dict(
        experiment='experiment-id',experiment_label='PG3 · IPTS-37869',run_number=run,
        scan_label=f'{run} · Bank 2',bank='2',sample_label='copper',source_path='/private/source.gpx'))


def test_names_identify_saved_experiment_scan_bank_recipe_and_native_artifact():
    job=saved_job();recipes={'saved-recipe':{'version':3}}
    name=refinement_artifact_filename(job,'polished.gpx',recipes)
    assert name.startswith('PG3_IPTS-37869_scan-64301_bank-2_recipe-v3_polished_')
    assert name.endswith('.gpx') and re.fullmatch(r'[A-Za-z0-9._-]+',name)
    assert name==refinement_artifact_filename(deepcopy(job),'polished.gpx',deepcopy(recipes))
    assert name!=refinement_artifact_filename(saved_job(64302),'polished.gpx',recipes)
    assert name!=refinement_artifact_filename(saved_job(uid='revised-source'),'polished.gpx',recipes)
    assert name!=refinement_artifact_filename(job,'polished_atomic_coordinates.gpx',recipes)
    assert name!=refinement_artifact_filename(job,'polished.gpx',{'saved-recipe':{'version':4}})


def test_names_are_bounded_and_never_include_transport_paths_or_credentials():
    job=saved_job();job['source'].update(experiment_label='https://private.example/path?token=secret',
        scan_label='/private/sensitive/path',run_number=None,source_path='/private/never-copy-me.gpx',
        metadata={'api_key':'never-copy-me','url':'https://private.example/?token=never-copy-me'})
    name=refinement_artifact_filename(job,'../../polished.gpx',{'saved-recipe':{'version':3}})
    assert 'experiment-id_scan_bank-2_recipe-v3_polished_' in name
    assert not any(value in name for value in ['private','sensitive','secret','never-copy-me','..','/','\\','?'])
    job['source'].update(experiment_label='a'*500,run_number='b'*500,bank='c'*500)
    assert len(refinement_artifact_filename(job,'d'*500+'.gpx',{}))<220


def test_download_changes_only_filename_and_keeps_hash_gate(tmp_path):
    job=saved_job();content=b'GPX bytes are never rewritten by the download handler\x00\xff'
    path=tmp_path/'polished.gpx';path.write_bytes(content)
    job['result']={'directory':str(tmp_path),'status':'model_edited','final_gpx':'polished.gpx',
        'output_sha256':hashlib.sha256(content).hexdigest(),'files':{'polished.gpx':hashlib.sha256(content).hexdigest()}}
    recipes={'saved-recipe':{'version':3}}
    app=RefinementMixin();app._refinement_workspace=SimpleNamespace(state={'jobs':{job['id']:job},'recipes':recipes})
    app.server=SimpleNamespace(state=SimpleNamespace(rf_selected_job=job['id'],rf_download='polished.gpx',
        rf_downloads=[{'value':'polished.gpx'}],rf_error='',flush=lambda:None))
    downloads=[];app.download_file=lambda *args:downloads.append(args)
    app.download_refinement_artifact()
    assert downloads==[(refinement_artifact_filename(job,'polished.gpx',recipes),'application/octet-stream',content)]
    assert path.read_bytes()==content and job['result']['files']['polished.gpx']==hashlib.sha256(content).hexdigest()
    path.write_bytes(b'tampered')
    app.download_refinement_artifact()
    assert len(downloads)==1 and 'changed or is missing' in app.server.state.rf_error


def test_selected_saved_job_supplies_displayed_download_name(tmp_path):
    job=saved_job();job.update(message='completed',status='completed',result={
        'directory':str(tmp_path),'files':{'polished.gpx':'hash'}})
    recipes={'saved-recipe':{'version':3}}
    class App(RefinementMixin):
        def _refinement_plot_changed(self):pass
        def _refinement_trend_changed(self):pass
    app=App();app._refinement_workspace=SimpleNamespace(state={'jobs':{job['id']:job},'recipes':recipes})
    values=app._refinement_state_defaults();values.update(rf_experiment='experiment-id',rf_source='another-source',rf_request='edited draft')
    app.server=SimpleNamespace(state=SimpleNamespace(**values,flush=lambda:None))
    asyncio.run(app._select_refinement_job(job['id']))
    entry=app.server.state.rf_downloads[0]
    assert entry['value']=='polished.gpx'
    assert entry['filename']==refinement_artifact_filename(job,'polished.gpx',recipes)
    assert '64301' in entry['filename'] and 'recipe-v3' in entry['filename']
