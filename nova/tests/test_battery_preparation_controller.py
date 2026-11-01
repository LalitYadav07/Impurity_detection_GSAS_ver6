import asyncio
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from radar_pd_nova.battery_controller import BatteryMixin
from radar_pd_nova import battery_preparation as module


class State(SimpleNamespace):
    def change(self,*fields):return lambda callback:callback
    def flush(self):pass


@pytest.fixture
def app():
    app=BatteryMixin();app.server=SimpleNamespace(state=State());app._init_battery()
    return app


def test_model_change_invalidates_a_prepared_checkpoint(app):
    app._battery_prepared=(Path('/unused'),{})
    app.server.state.battery_model_ready=True
    app._battery_preparation_edited()
    assert app._battery_prepared is None
    assert not app.server.state.battery_model_ready
    assert 'Inputs changed' in app.server.state.battery_model_message


def test_added_phase_captures_width_starts_and_invalidates_the_old_model(app,tmp_path):
    path=tmp_path/'hypothesis.cif';path.write_text('Synthetic input; parsing belongs to native preparation')
    app._battery_phase_upload=SimpleNamespace(current_path=path)
    s=app.server.state;s.battery_phase_path=str(path);s.battery_phase_name='oxide';s.battery_phase_role='cathode'
    s.battery_phase_size_um='.01';s.battery_phase_microstrain_ustrain='0'
    app._battery_prepared=(Path('/old'),{});s.battery_model_ready=True
    app.add_battery_phase()
    assert not s.battery_error and not s.battery_model_ready
    assert app._battery_model_phases[0]['sample_widths']=={'size_um':.01,'microstrain_ustrain':0.}
    assert s.battery_model_phases[0]['sample_widths']=={'size_um':.01,'microstrain_ustrain':0.}
    assert 'raw' not in s.battery_model_phases[0]
    assert s.battery_phase_size_um==s.battery_phase_microstrain_ustrain==''


@pytest.mark.parametrize('value',['.0001','11','nan',True,None])
def test_invalid_width_start_does_not_add_a_phase_or_replace_a_prepared_model(app,tmp_path,value):
    path=tmp_path/'hypothesis.cif';path.write_text('Synthetic')
    app._battery_phase_upload=SimpleNamespace(current_path=path)
    s=app.server.state;s.battery_phase_path=str(path);s.battery_phase_name='oxide';s.battery_phase_role='cathode'
    s.battery_phase_size_um=value
    app._battery_prepared=(Path('/old'),{});s.battery_model_ready=True
    app.add_battery_phase()
    assert s.battery_error and not app._battery_model_phases
    assert s.battery_model_ready and app._battery_prepared is not None


def test_cells_with_one_sample_id_are_separate_and_unmixed_legacy_jobs_recover():
    from radar_pd_nova.refinement_sources import digest
    w=SimpleNamespace(history_id='history',state={'sources':{},'jobs':{}})
    a=dict(instrument='NOM',ipts='IPTS-1',sample_id='sample',cell_id='first')
    first=module.battery_experiment_id(w,a,'family')
    second=module.battery_experiment_id(w,dict(a,cell_id='second'),'family')
    assert first!=second
    legacy=digest(['battery-prepared','history','NOM','IPTS-1','sample','family'])[:24]
    w.state['sources']['old']={'experiment':legacy,'metadata':{'cell_id':'first'}}
    assert module.battery_experiment_id(w,a,'family')==legacy
    assert module.battery_experiment_id(w,dict(a,cell_id='second'),'family')==second
    w.state['jobs']['mixed']={'source':{'experiment':legacy,'metadata':{'cell_id':'second'}}}
    assert module.battery_experiment_id(w,a,'family')==first


def test_changing_the_join_invalidates_a_prepared_model_with_electrochemistry(app):
    app._battery_prepared=(Path('/unused'),{})
    app._battery_cycler_raw=b'old'
    app.server.state.battery_model_ready=True
    app._battery_binding_changed()
    assert app._battery_cycler_raw is None and app._battery_prepared is None
    assert not app.server.state.battery_model_ready


def test_switching_sample_does_not_reuse_cell_or_hypotheses(app,tmp_path):
    family=tmp_path/'shared/GSAS';family.mkdir(parents=True)
    app._battery_inventory={'instrument':'NOM','root':str(tmp_path),'product_families':[{'folder':'shared/GSAS'}]}
    app._battery_acquisitions=[{'run':2,'identity':{'sample_id':'new'}}]
    app._battery_model_sample='old';app._battery_model_phases=[{'name':'old'}]
    s=app.server.state;s.battery_family='shared/GSAS';s.battery_model_run='2';s.battery_model_cell='old cell'
    s.battery_model_review='old mapping';s.battery_phase_path='/old/cif'
    app._battery_model_run_changed()
    assert app._battery_model_phases==[] and s.battery_model_phases==[]
    assert s.battery_model_cell==s.battery_model_review==s.battery_phase_path==''


def test_powgen_inspection_resolves_the_physical_to_profile_bank_mapping(app,tmp_path):
    from test_battery_powgen import source
    path=source(tmp_path)
    app._battery_inventory={'instrument':'PG3','root':str(tmp_path),'profile_candidates':[]}
    asyncio.run(app._inspect_battery_banks_async({'path':str(path)},app._battery_generation,app._battery_model_revision))
    assert app.server.state.battery_model_bank_options[0]['value']==2
    assert app.server.state.battery_b2_profile=='powgen_catalog'
    assert app.server.state.battery_b2_profile_bank=='1'
    assert module.product_adapter('PG3') is module.powgen
    assert app._battery_profile_path(2).name=='2019A_HighRes_60HzB2_CWL1p5.instprm'
    assert any('scientific qualification pending' in note for note in app.server.state.battery_model_mapping_notes)
    app.server.state.battery_b2_profile_bank='2'
    with pytest.raises(ValueError,match='Bank label must agree'):
        app._battery_profile_path(2)
    app._battery_model_data_changed()
    assert app._battery_auto_profiles=={} and app.server.state.battery_model_mapping_notes==[]


def test_capture_rejects_source_replacement_and_oversize(tmp_path):
    source=tmp_path/'input';source.write_bytes(b'old')
    old=module.nomad.signature(source)
    source.write_bytes(b'new version')
    target=tmp_path/'copy'
    with pytest.raises(ValueError,match='changed'):
        module.capture_file(source,target,maximum=100,expected_signature=old)
    with pytest.raises(ValueError,match='limit'):
        module.capture_file(source,target,maximum=3)
    with pytest.raises(ValueError,match='content changed'):
        module.capture_file(source,target,maximum=100,expected_sha256=sha256(b'old').hexdigest())
    assert not target.exists()


def test_run_lookup_does_not_inspect_unrelated_files(tmp_path):
    (tmp_path/'NOM2tof.gsa').write_text('data')
    (tmp_path/'NOM3tof.gsa').write_text('unrelated')
    (tmp_path/'NOM_2.gsa').symlink_to(tmp_path/'NOM3tof.gsa')
    rows=module.run_sources(tmp_path,2)
    assert [Path(row['path']).name for row in rows]==['NOM2tof.gsa']
    with pytest.raises(ValueError,match='bounded'):
        module.run_sources(tmp_path,2,maximum_entries=1)


@pytest.mark.parametrize('selected',[[1,'../../escape'],[True],[1,1],[6],None])
def test_invalid_bank_selection_rejected_before_capture(app,tmp_path,monkeypatch,selected):
    monkeypatch.setenv('RADAR_PD_REFINEMENT_PYTHON','python')
    monkeypatch.setenv('RADAR_PD_BATTERY_ROOT',str(tmp_path/'jobs'))
    app._battery_bank_setup=({},SimpleNamespace(banks=[SimpleNamespace(bank=1,quality_errors=[])]))
    app.server.state.battery_model_banks=selected
    app.prepare_battery_model()
    assert 'valid physical banks' in app.server.state.battery_error
    assert not (tmp_path/'jobs').exists()


def test_stale_native_preparation_does_not_replace_revised_form(app,tmp_path,monkeypatch):
    def native(*args,**kwargs):
        app._battery_preparation_edited()
        (tmp_path/'model').mkdir()
        (tmp_path/'model/preparation.json').write_text('{"status":"prepared_needs_review","banks":[],"phases":[]}')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module,'run_scientific',native)
    asyncio.run(app._prepare_battery_model_async(tmp_path,'python',app._battery_generation,app._battery_model_revision))
    assert app._battery_prepared is None and not app.server.state.battery_model_ready
    assert not app.server.state.battery_busy


def test_changed_model_is_not_downloaded(app,tmp_path,monkeypatch):
    (tmp_path/'model').mkdir();(tmp_path/'model/prepared.gpx').write_bytes(b'changed')
    result={'schema':'radar-battery-preparation-result/v1','status':'prepared_needs_review','specification':{},'project_sha256':sha256(b'original').hexdigest()}
    (tmp_path/'model/preparation.json').write_text(json.dumps(result))
    app._battery_prepared=(tmp_path,result)
    monkeypatch.setattr(module,'validate_preparation',lambda *args:None)
    app.download_file=lambda *args:pytest.fail('Changed checkpoint was exported')
    app.download_battery_model()
    assert 'Prepared model changed' in app.server.state.battery_error


def test_saved_models_reopen_with_checks_and_cannot_select_arbitrary_paths(app,tmp_path,monkeypatch):
    monkeypatch.setenv('RADAR_PD_BATTERY_ROOT',str(tmp_path))
    job=tmp_path/'job1';(job/'model').mkdir(parents=True)
    result={'schema':'radar-battery-preparation-result/v1','status':'prepared_needs_review',
            'specification':{'acquisition':{'ipts':'IPTS-1','run':2,'sample_id':'sample','cell_id':'cell'},
                             'phases':[{'name':'oxide'}],'banks':[]},
            'project_sha256':sha256(b'project').hexdigest()}
    (job/'model/preparation.json').write_text(json.dumps(result));(job/'model/prepared.gpx').write_bytes(b'project')
    monkeypatch.setattr(module,'validate_preparation',lambda *args:None)
    app.refresh_saved_battery_models()
    assert len(app.server.state.battery_saved_models)==1
    opened=[];app.open_battery_refinement=lambda:opened.append(app._battery_prepared)
    app.server.state.battery_saved_model='../../elsewhere';app.open_saved_battery_model()
    assert not opened and 'refreshed list' in app.server.state.battery_error
    app.server.state.battery_saved_model='job1';app.open_saved_battery_model()
    assert opened==[(job,result)]
    (job/'model/prepared.gpx').write_bytes(b'corrupted');app.open_saved_battery_model()
    assert len(opened)==1 and 'changed' in app.server.state.battery_error


def test_failed_saved_profile_review_preserves_the_current_preparation(app,tmp_path,monkeypatch):
    monkeypatch.setenv('RADAR_PD_BATTERY_ROOT',str(tmp_path))
    job=tmp_path/'old';job.mkdir()
    previous=(tmp_path/'current',{'current':'prepared'})
    app._battery_prepared=previous
    app._battery_saved_models={'old':job};app.server.state.battery_saved_model='old'
    monkeypatch.setattr(module,'verify_prepared',lambda path:{'old':'historical'})
    def reject(*args):raise ValueError('Physical bank 1 needs a new preparation')
    monkeypatch.setattr(module,'check_prepared_profiles',reject)
    app.open_battery_refinement=lambda:pytest.fail('An incompatible profile reached refinement')
    app.open_saved_battery_model()
    assert app._battery_prepared==previous
    assert 'needs a new preparation' in app.server.state.battery_error


def test_saved_model_cannot_follow_a_receipt_outside_its_bundle(tmp_path):
    outside=tmp_path/'outside';outside.mkdir();(outside/'preparation.json').write_text('{}')
    job=tmp_path/'job';job.mkdir();(job/'model').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='unavailable'):
        module.verify_prepared(job)
