"""Real NeXus-shaped condition logs at the Live and saved-watch boundary."""
from dataclasses import replace
import json

import h5py
import pytest

from radar_pd_nova.snap_metadata import read_snap_scan_metadata
from radar_pd_nova.snap_conditions import binding_scope, condition_options, validate_bindings_json
from radar_pd_nova.snap_controller import SnapExperimentSettings
from radar_pd_nova.scan_conditions import read_conditions
from radar_pd_nova.powgen_watch import WatchedRun, build_submission_plan


def nexus(path, *, unit='K', specimen=123, selector='Lakeshore', values=(300.,310.), times=(0.,5.)):
    with h5py.File(path,'w') as h:
        for name,value in [('entry/run_number',b'42'),('entry/experiment_identifier',b'IPTS-1'),
                ('entry/instrument/name',b'SNAP'),('entry/sample/name',b'test specimen'),
                ('entry/sample/identifier',specimen),('entry/DASlogs/SampleId/value',specimen),
                ('entry/DASlogs/BL3:SE:IM:SelectSampleEnv/value_strings',selector.encode()),
                ('entry/title',b'temperature sweep'),('entry/start_time',b'2026-01-01T00:00:00Z'),
                ('entry/end_time',b'2026-01-01T00:01:40Z')]:
            h.create_dataset(name,data=[value])
        for name,v in [('LakeshoreSampleTemp',values),('Lakeshore2SampleTemp',[0.,0.])]:
            p='entry/DASlogs/'+name+'/'
            d=h.create_dataset(p+'value',data=v)
            if unit:d.attrs['units']=unit
            t=h.create_dataset(p+'time',data=times);t.attrs.update(units='s',start='2026-01-01T00:00:00Z')
        # No test/read is allowed to depend on loading events.
        h.create_dataset('entry/bank1_events/event_id',shape=(10000000,),dtype='i8')
    return path


def reviewed(path, *, unit='K'):
    metadata=read_snap_scan_metadata('IPTS-1',42,nexus_path=path)
    scope=binding_scope(metadata)
    return {'temperature':{'channel':'LakeshoreSampleTemp','unit':unit,'verified':True,
        'product_window_reviewed':True,
        'scope_sha256':scope['scope_sha256'],'schema_sha256':scope['schema_sha256']}}


def test_reviewed_sensor_uses_acquisition_weighting_and_survives_json_roundtrip(tmp_path):
    path=nexus(tmp_path/'run.h5')
    initial=read_snap_scan_metadata('IPTS-1',42,nexus_path=path)
    assert 'temperature' not in initial and len(condition_options(initial))==2
    binding=validate_bindings_json(json.dumps(reviewed(path)))
    result=read_snap_scan_metadata('IPTS-1',42,nexus_path=path,condition_bindings=binding)
    assert result['temperature']['value']==pytest.approx(309.5)
    assert result['temperature']['unit']=='K'
    assert result['temperature']['reviewed_binding']==binding['temperature']
    assert result['temperature']['coverage_fraction']==1
    assert result['conditions']['metadata_sha256']!=initial['conditions']['metadata_sha256']
    assert result['conditions']['window_scope']=='reviewed_full_parent_run'


@pytest.mark.parametrize('change', ['sample','environment','unit','schema'])
def test_sensor_binding_does_not_cross_specimen_environment_or_log_schema(tmp_path,change):
    path=nexus(tmp_path/'run.h5');binding=reviewed(path)
    with h5py.File(path,'a') as h:
        if change=='sample':
            h['entry/sample/identifier'][0]=456;h['entry/DASlogs/SampleId/value'][0]=456
        elif change=='environment':h['entry/DASlogs/BL3:SE:IM:SelectSampleEnv/value_strings'][0]=b'Other'
        elif change=='unit':h['entry/DASlogs/LakeshoreSampleTemp/value'].attrs['units']='C'
        else:del h['entry/DASlogs/Lakeshore2SampleTemp']
    result=read_snap_scan_metadata('IPTS-1',42,nexus_path=path,condition_bindings=binding)
    assert 'temperature' not in result
    assert any('changed' in n for n in result['metadata_notices'])


def test_moving_readback_keeps_same_binding_scope(tmp_path):
    path=nexus(tmp_path/'run.h5');binding=reviewed(path)
    with h5py.File(path,'a') as h:h['entry/DASlogs/LakeshoreSampleTemp/value'][:]=[150.,160.]
    result=read_snap_scan_metadata('IPTS-1',42,nexus_path=path,condition_bindings=binding)
    assert result['temperature']['value']==pytest.approx(159.5)


def test_reviewed_units_fill_only_missing_units_and_cannot_override_recorded_units(tmp_path):
    path=nexus(tmp_path/'no-unit.h5',unit='',values=(20.,30.))
    binding=reviewed(path,unit='C')
    result=read_snap_scan_metadata('IPTS-1',42,nexus_path=path,condition_bindings=binding)
    assert result['temperature']['value']==pytest.approx(302.65)
    assert 'source units absent' in result['temperature']['unit_provenance']
    path=nexus(tmp_path/'recorded.h5')
    result=read_snap_scan_metadata('IPTS-1',42,nexus_path=path,condition_bindings=reviewed(path,unit='C'))
    assert 'temperature' not in result
    assert any(c['status']=='unqualified_units' for c in result['conditions']['channels'])


def test_zero_kelvin_and_incomplete_log_never_become_qualified_axis(tmp_path):
    path=nexus(tmp_path/'zero.h5',values=(0.,0.))
    result=read_snap_scan_metadata('IPTS-1',42,nexus_path=path,condition_bindings=reviewed(path))
    assert 'temperature' not in result
    path=nexus(tmp_path/'partial.h5',times=(20.,50.))
    result=read_snap_scan_metadata('IPTS-1',42,nexus_path=path,condition_bindings=reviewed(path))
    assert 'temperature' not in result
    assert any(c['status']=='incomplete_coverage' for c in result['conditions']['channels'])


def test_snap_mapping_is_immutable_watch_identity_and_not_dac_pump_pressure(tmp_path):
    path=nexus(tmp_path/'run.h5');binding=reviewed(path)
    settings=SnapExperimentSettings('IPTS-1','history','config',selected_banks=(1,),
        product_family='["all",""]',condition_bindings_json=json.dumps(binding,sort_keys=True))
    assert json.loads(settings.definition().config_refs['snap_conditions'])==binding
    alternate=json.loads(json.dumps(binding));alternate['temperature']['unit']='C'
    run=WatchedRun(42,settings.definition().source_directory+'/SNAP42_all.gsa','source-revision',instrument='SNAP')
    assert build_submission_plan(settings.definition(),run).idempotency_key != build_submission_plan(
        replace(settings,condition_bindings_json=json.dumps(alternate)).definition(),run).idempotency_key
    with pytest.raises(ValueError,match='DAC pressure'):
        replace(settings,condition_bindings_json=json.dumps({'pressure':binding['temperature']}))
    with pytest.raises(ValueError,match='explicitly review'):
        binding['temperature']['verified']=False
        replace(settings,condition_bindings_json=json.dumps(binding))


def test_sensor_without_full_export_interval_review_is_not_a_qualified_axis(tmp_path):
    path=nexus(tmp_path/'run.h5');binding=reviewed(path)
    binding['temperature']['product_window_reviewed']=False
    result=read_snap_scan_metadata('IPTS-1',42,nexus_path=path,condition_bindings=binding)
    assert 'temperature' not in result
    with pytest.raises(ValueError,match='full-acquisition'):
        validate_bindings_json(json.dumps(binding))


def test_reviewed_snap_acquisition_reaches_batch_gpx_contract_without_trajectories(tmp_path,monkeypatch):
    from pathlib import Path
    from radar_pd_nova.acquisition import acquisition_context
    from radar_pd_nova import snap
    from test_snap import write_source,profile
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    from ndip_input_provenance import read_input_provenance
    from ndip_acquisition import annotate_gpx_records
    path=nexus(tmp_path/'run.h5');binding=reviewed(path)
    metadata=read_snap_scan_metadata('IPTS-1',42,nexus_path=path,condition_bindings=binding)
    acquisition=acquisition_context(metadata,instrument='SNAP',run_number=42,source=tmp_path/'SNAP42_bank_test.gsa')
    assert acquisition['product_window_verified'] and 'User-reviewed' in acquisition['product_window_basis']
    assert 'trajectory' not in acquisition['channels'][0]
    source=write_source(tmp_path/'SNAP42_bank_test.gsa')
    inst=tmp_path/'profile.instprm';inst.write_text(profile())
    staged,_,receipt=snap.stage_bank(source,snap.inspect_reduced_setup(source),2,inst,7,tmp_path/'staged',
        acknowledged=True,acquisition=acquisition,condition_bindings=binding)
    handoff=read_input_provenance(staged)
    assert handoff['condition_bindings']==binding and receipt['acquisition']==acquisition
    records=[[['PWDR test',{}],['Sample Parameters',{'Temperature':300.}],['Comments',[]]]]
    report=annotate_gpx_records(records,handoff['acquisition'])
    assert report['temperature_written'] and report['temperature_K']==pytest.approx(309.5)
    assert records[0][1][1]['Temperature']==pytest.approx(309.5)


def test_controller_freezes_sensor_receipt_restores_and_rejects_changed_source(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from test_snap import write_source,profile
    from test_nomad import Galaxy
    from radar_pd_nova import snap
    from radar_pd_nova.snap_controller import SnapWatchController
    from radar_pd_nova.powgen_watch import WatchDefinition
    path=nexus(tmp_path/'SNAP42.h5');binding=reviewed(path)
    source=write_source(tmp_path/'SNAP42_bank_test.gsa')
    monkeypatch.setattr(WatchDefinition,'source_directory',property(lambda self:str(source.parent)))
    monkeypatch.setattr('radar_pd_nova.snap_metadata.snap_nexus_path',lambda *a:path)
    settings=SnapExperimentSettings('IPTS-1','h','c',instrument_profile_dataset_id='original',
        selected_banks=(2,),subfolder='shared/SNAPRed/export/gsa/bank',
        product_family=snap.parse_filename(source).family,profile_bank=7,
        profile_acknowledged=True,allow_exploratory=True,condition_bindings_json=json.dumps(binding,sort_keys=True))
    service=Galaxy(profile())
    index=SimpleNamespace(identity=lambda run:{'sample_id':'123'})
    first=SnapWatchController(service,settings,sample_index=index)
    run=first.discover()[0];record=first.submit(run)
    receipt=record.provenance['snap_bank']
    assert receipt['conditions']['metadata_sha256']==run.scan_metadata['conditions']['metadata_sha256']
    assert receipt['condition_bindings']==binding and receipt['acquisition']['product_window_verified']
    assert receipt['acquisition']['channels'][0]['value']==pytest.approx(309.5)
    restored=SnapWatchController(service,settings,sample_index=index)
    assert restored.restore_latest_state() and not restored.discover() and len(service.jobs)==1
    assert not SnapWatchController(service,replace(settings,condition_bindings_json='{}'),sample_index=index).restore_latest_state()
    with h5py.File(path,'a') as h:h['entry/DASlogs/LakeshoreSampleTemp/value'][:]=[100.,110.]
    with pytest.raises(ValueError,match='changed after discovery'):
        first._prepare_run(run)
