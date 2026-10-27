"""Synthetic contracts reproducing independently observed SNAP archive cases."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from radar_pd_nova import snap
from radar_pd_nova.snap_controller import SnapExperimentSettings, SnapWatchController
from radar_pd_nova.snap_metadata import read_snap_scan_metadata
from radar_pd_nova.powgen_watch import WatchDefinition, WatchState, WatchedRun, discover_from_listing, LIVE_STATE_SCHEMA
from test_powgen_controller import FakeGalaxyService


def profile(bank=7, difc=6200, angle=105):
    return f'''#Bank {bank}: GSAS-II instrument parameter file
Type:PNT
Bank:{bank}
difC:{difc}
difA:2
difB:0.4
Zero:1
2-theta:{angle}
alpha:0.99
beta-0:0.02
beta-1:0.03
sig-0:1
sig-1:66
sig-2:0
sig-q:0
X:0
Y:0
'''


def write_source(path, run=42, *, state='0123456789abcdef', negative=False):
    header=f'''{{"redRecord": "/SNS/SNAP/IPTS-1/shared/SNAPRed/{state}/lite/{run}/None/ReductionRecord.json"
"attenuationMethod": null
"backgroundMethod": null}}
# Total flight path 15.5m, tth 105deg, DIFC 6200
# Data for spectrum :0
BANK 2 40 40 SLOG 6200 11000 .002 0 FXYE
'''
    rows=[]
    for i in range(40):
        x=6200+i*100
        y = '-3.25' if negative and i == 8 else str(i+1)
        rows.append(f'{x} {y} 0.5\n' if i not in {0,1,20} else f'{x} nan 0\n')
    path.write_text(header+''.join(rows))
    return path


@pytest.fixture
def source(tmp_path):
    return write_source(tmp_path/'SNAP000042_bank_test.gsa')


@pytest.fixture
def supplied(tmp_path):
    p=tmp_path/'reviewed.instprm';p.write_text(profile());return p


def test_explicit_masked_bins_are_preserved_as_missing_not_zero_measurements(source,supplied,tmp_path):
    setup=snap.inspect_reduced_setup(source)
    bank=setup.banks[0]
    assert bank.valid_bins==37 and bank.masked_bins==3
    assert bank.masked_intervals==((6200.,6300.),(8200.,8200.))
    data,inst,receipt=snap.stage_bank(source,setup,2,supplied,7,tmp_path/'staged',acknowledged=True)
    assert 'BANK 1 40 40' in data.read_text()
    assert '6200 nan 0\n' in data.read_text()
    assert 'Bank:1.0' in inst.read_text()
    assert receipt['numeric_sha256']==bank.numeric_sha256 and receipt['source_bank']==2 and receipt['profile_bank']==7
    marker=next(line for line in data.read_text().splitlines() if line.startswith('# RADAR-PD input provenance:'))
    assert json.loads(marker.split(': ',1)[1])['masked_bins']==3


def test_signed_positive_uncertainty_observations_remain_original(source,supplied,tmp_path):
    write_source(source,negative=True)
    setup=snap.inspect_reduced_setup(source)
    data,_,_=snap.stage_bank(source,setup,2,supplied,7,tmp_path/'negative',acknowledged=True)
    assert '7000 -3.25 0.5\n' in data.read_text()


@pytest.mark.parametrize('replacement,message',[
    ('6300 nan 0\n',''),
])
def test_fully_written_count_is_required(source,replacement,message):
    source.write_text(source.read_text().replace(replacement,message))
    with pytest.raises(ValueError,match='truncated'):snap.inspect_reduced_setup(source)


@pytest.mark.parametrize('old,new,message',[
    ('6400 3 0.5','6400 nan 0.5','positive-weight'),
    ('6400 3 0.5','6400 3 -1','uncertainty'),
    ('6400 3 0.5','6200 3 0.5','increasing'),
    ('BANK 2 40 40','BANK 2 41 40','count'),
    ('SLOG','CONST','SLOG/FXYE'),
    ('/lite/42/None/','/lite/43/None/','run/state'),
])
def test_corrupt_weighted_data_and_identity_fail_closed(source,old,new,message):
    source.write_text(source.read_text().replace(old,new))
    with pytest.raises(ValueError,match=message):snap.inspect_reduced_setup(source)


def test_missing_record_never_uses_latest_reduction(source,tmp_path):
    sibling=tmp_path/'2026-01-01T000000';sibling.mkdir()
    (sibling/'ReductionRecord.json').write_text('{"runNumber":"42"}')
    setup=snap.inspect_reduced_setup(source)
    assert setup.record_sha256=='' and setup.state_id=='0123456789abcdef'
    assert any('no nearest/latest' in n for n in setup.warnings)


def test_exact_record_checks_state_mode_and_run(source,tmp_path):
    root=tmp_path/'IPTS-1';record=root/'shared/SNAPRed/0123456789abcdef/lite/42/None/ReductionRecord.json'
    record.parent.mkdir(parents=True)
    body={'runNumber':'42','useLiteMode':True,'calibration':{'version':2,'runNumber':'20',
          'calculationParameters':{'instrumentState':{'id':{'hex':'0123456789abcdef'}}}},'normalization':{'runNumber':'21','version':1}}
    record.write_text(json.dumps(body));setup=snap.inspect_reduced_setup(source,ipts_root=root)
    assert setup.record_sha256 and not any('record is missing' in n for n in setup.warnings)
    body['calibration']['calculationParameters']['instrumentState']['id']['hex']='1111111111111111'
    record.write_text(json.dumps(body))
    with pytest.raises(ValueError,match='state identifier'):snap.inspect_reduced_setup(source,ipts_root=root)


def test_profile_mapping_requires_acknowledgement_and_real_labels(source,supplied):
    setup=snap.inspect_reduced_setup(source)
    with pytest.raises(ValueError,match='Confirm'):snap.validate_profile(setup,2,supplied,7)
    with pytest.raises(ValueError,match='label is absent'):snap.validate_profile(setup,2,supplied,1,acknowledged=True)
    supplied.write_text(profile(difc=4000))
    with pytest.raises(ValueError,match='incompatible'):snap.validate_profile(setup,2,supplied,7,acknowledged=True)


def test_profile_bad_lookup_tables_are_not_silently_fixed(supplied):
    supplied.write_text(profile()+'pdabc:"""1, 2, 3, 4, nan\n2, 3, 4, 5, 6"""\n')
    with pytest.raises(ValueError,match='lookup table contains nonfinite'):snap.profile_blocks(supplied)
    supplied.write_text(profile()+profile())
    with pytest.raises(ValueError,match='Duplicate'):snap.profile_blocks(supplied)


def test_source_mutation_cannot_be_staged(source,supplied,tmp_path):
    setup=snap.inspect_reduced_setup(source)
    source.write_text(source.read_text()+'\n')
    with pytest.raises(ValueError,match='changed'):snap.stage_bank(source,setup,2,supplied,7,tmp_path/'changed',acknowledged=True)


def test_folder_and_export_variants_do_not_merge(tmp_path):
    for name in ['SNAP000042_all.gsa','SNAP000042_all_reRed.gsa','SNAP000043_all.gsa']:
        (tmp_path/name).write_text('not read during discovery')
    rows=snap.directory_listing(tmp_path)
    assert len(snap.product_families(rows))==2
    definition=WatchDefinition('IPTS-1',snap.DEFAULT_FOLDER,'history','config','profile',instrument='SNAP')
    with pytest.raises(ValueError,match='Several SNAP exports'):
        discover_from_listing(definition,[{'path':Path(r['path']).name} for r in rows])
    only=snap.directory_listing(tmp_path,family=snap.parse_filename('SNAP42_all.gsa').family)
    runs=discover_from_listing(definition,[{'path':Path(r['path']).name} for r in only])
    assert [r.run_id for r in runs]==['SNAP_42','SNAP_43']
    state=WatchState('history',instrument='SNAP');assert state.schema==LIVE_STATE_SCHEMA
    assert WatchState.from_dict(state.as_dict()).instrument=='SNAP'


def test_source_path_is_confined_and_no_raw_or_total_scattering_trigger(tmp_path):
    for name in ['SNAP42.nxs.h5','SNAP42_all.csv','SNAP42_all.xye','SNAP42_all.gsa.tmp','NOM_42.gsa']:
        (tmp_path/name).write_text('not a GSAS product')
    assert not snap.directory_listing(tmp_path)
    with pytest.raises(ValueError):snap.source_directory('IPTS-1','shared/../../nexus')
    with pytest.raises(ValueError):snap.source_directory('IPTS-1','nexus')


def test_incremental_stream_handles_partial_then_complete_and_new_run(source,supplied,monkeypatch):
    monkeypatch.setattr(WatchDefinition,'source_directory',property(lambda self:str(source.parent)))
    monkeypatch.setattr(SnapWatchController,'_read_scan_metadata',lambda *a:{})
    service=FakeGalaxyService()
    settings=SnapExperimentSettings('IPTS-1',service.history_id,'config',selected_banks=(2,),subfolder='shared/SNAPRed/export/gsa/bank',
        product_family=snap.parse_filename(source).family,profile_bank=7,profile_acknowledged=True,allow_exploratory=True)
    controller=SnapWatchController(service,settings)
    complete=source.read_bytes();source.write_bytes(complete[:150])
    assert controller.discover()[0].scan_metadata['setup_error']
    assert not controller.due_submissions()
    source.write_bytes(complete);controller.discover()
    assert [r.run_id for r in controller.due_submissions()]==['SNAP_42']
    assert controller.discover()==[]
    write_source(source.parent/'SNAP000043_bank_test.gsa',43)
    assert [r.run_id for r in controller.discover()]==['SNAP_43']
    assert len(controller.due_submissions())==2


def test_unsigned_unset_id_and_unselected_temperature_never_become_axes(tmp_path):
    h5py=pytest.importorskip('h5py')
    path=tmp_path/'SNAP42.nxs.h5'
    with h5py.File(path,'w') as f:
        for name,value in [('entry/run_number',b'42'),('entry/experiment_identifier',b'IPTS-1'),
                           ('entry/instrument/name',b'SNAP'),('entry/sample/name',b'No sample'),
                           ('entry/sample/identifier',4294967295),('entry/DASlogs/SampleId/value',4294967295),
                           ('entry/title',b'Si standard, 300 K'),('entry/start_time',b'2026-01-01T00:00:00Z'),
                           ('entry/end_time',b'2026-01-01T00:01:00Z')]:
            f.create_dataset(name,data=[value])
        d=f.create_dataset('entry/DASlogs/LakeshoreSampleTemp/value',data=[300.]);d.attrs['units']='K'
        t=f.create_dataset('entry/DASlogs/LakeshoreSampleTemp/time',data=[0.]);t.attrs.update(units='s',start='2026-01-01T00:00:00Z')
    metadata=read_snap_scan_metadata('IPTS-1',42,nexus_path=path)
    assert metadata['sample_id']=='' and metadata['identity_status']=='unidentified'
    assert 'temperature' not in metadata and 'pressure' not in metadata
    assert metadata['conditions']['channels'][0]['status']=='needs_mapping'


def test_snap_raw_comparison_matches_gsas_bin_centers_and_preserves_masks(source,supplied,tmp_path):
    import numpy as np
    from radar_pd_nova.transition_input import reduced_observation
    from radar_pd_nova.transition_coordinate_mapping import mapped_calibration
    from radar_pd_nova.diffraction_coordinates import display_pattern
    from types import SimpleNamespace
    setup=snap.inspect_reduced_setup(source)
    data,profile,receipt=snap.stage_bank(source,setup,2,supplied,7,tmp_path/'mapped',acknowledged=True)
    scan={'instrument':'SNAP','ipts':'IPTS-1','run_id':'SNAP_42','bank':2,
          'source_path':str(source),'transition_source_root':str(source.parent),
          'metadata':{'sample_id':'recorded','snap_setup':setup.as_dict()}}
    measured=reduced_observation(scan)
    assert len(measured.pattern.x)==39
    np.testing.assert_array_equal(measured.pattern.x,np.arange(6250,10100,100))
    assert not measured.pattern.valid[20]
    record=SimpleNamespace(uid='fit',galaxy_job_id='fit',provenance={'snap_bank':receipt})
    controller=SimpleNamespace(records={'SNAP_42':record})
    calibration=mapped_calibration(scan,controller,{})
    assert calibration.source_bank==2 and calibration.profile_bank==1 and calibration.difb==.4
    q=display_pattern(measured.pattern,'q_inv_angstrom',calibration)
    np.testing.assert_array_equal(q.valid,measured.pattern.valid[::-1])
    np.testing.assert_array_equal(q.intensity,measured.pattern.intensity[::-1])


def test_snap_normalization_or_correction_changes_never_share_comparison_scope(source):
    from radar_pd_nova.transition_input import reduced_observation
    def scan(setup):
        return {'instrument':'SNAP','ipts':'IPTS-1','run_id':'SNAP_42','bank':2,
                'source_path':str(source),'transition_source_root':str(source.parent),
                'metadata':{'sample_id':'recorded','snap_setup':setup.as_dict()}}
    setup=snap.inspect_reduced_setup(source)
    first=reduced_observation(scan(setup))
    second=reduced_observation(scan(replace(setup,corrections='{"backgroundMethod":"subtracted"}')))
    assert first.pattern.scope!=second.pattern.scope


def test_sample_selection_and_role_filter_precede_backfill_without_reading_every_run():
    from radar_pd_nova.snap_controller import _select_runs
    calls=[]
    class Index:
        def identity(self,run):
            calls.append(run.run_number)
            return {'sample_id':'A','run_title':'empty cell' if run.run_number==4 else 'sample at fixed condition'}
    runs=[WatchedRun(n,f'/SNS/SNAP/IPTS-1/shared/SNAP{n}_all.gsa',str(n),instrument='SNAP') for n in range(1,5)]
    selected=_select_runs(runs,Index(),('A',),limit=1)
    assert [r.run_number for r in selected]==[3] and calls==[4,3]


def test_checkpoint_recovery_keeps_profile_scope_and_blocks_future_state_change(source,supplied,monkeypatch):
    from types import SimpleNamespace
    from test_nomad import Galaxy
    monkeypatch.setattr(WatchDefinition,'source_directory',property(lambda self:str(source.parent)))
    monkeypatch.setattr(SnapWatchController,'_read_scan_metadata',lambda *a:{})
    service=Galaxy(supplied.read_text())
    settings=SnapExperimentSettings('IPTS-1','h','c',instrument_profile_dataset_id='original',
        selected_banks=(2,),subfolder='shared/SNAPRed/export/gsa/bank',
        product_family=snap.parse_filename(source).family,profile_bank=7,
        profile_acknowledged=True,allow_exploratory=True)
    index=SimpleNamespace(identity=lambda run:{'sample_id':'recorded'})
    first=SnapWatchController(service,settings,sample_index=index)
    record=first.submit(first.discover()[0])
    assert record.provenance['snap_bank']['source_bank']==2
    assert service.snapshots[0].inputs.bank=='2'
    assert 'BANK 1 ' in Path(service.snapshots[0].inputs.data_path).read_text()
    restored=SnapWatchController(service,settings,sample_index=index)
    assert restored.restore_latest_state()
    assert restored._manual_setup_signature==first._manual_setup_signature
    assert not restored.discover() and len(service.jobs)==1
    assert not SnapWatchController(service,replace(settings,profile_bank=8),sample_index=index).restore_latest_state()
    assert not SnapWatchController(service,replace(settings,product_family='["bank","another"]'),sample_index=index).restore_latest_state()
    write_source(source.with_name('SNAP000043_bank_test.gsa'),43,state='1111111111111111')
    next_run=restored.discover()[0]
    with pytest.raises(ValueError,match='state, grouping, calibration'):
        restored._prepare_run(next_run)
    assert len(service.jobs)==1


def test_reprocessed_source_is_flagged_without_replacing_or_resubmitting_results(source,supplied,monkeypatch):
    from types import SimpleNamespace
    from test_nomad import Galaxy
    monkeypatch.setattr(WatchDefinition,'source_directory',property(lambda self:str(source.parent)))
    monkeypatch.setattr(SnapWatchController,'_read_scan_metadata',lambda *a:{})
    service=Galaxy(supplied.read_text())
    settings=SnapExperimentSettings('IPTS-1','h','c',instrument_profile_dataset_id='original',
        selected_banks=(2,),subfolder='shared/SNAPRed/export/gsa/bank',
        product_family=snap.parse_filename(source).family,profile_bank=7,
        profile_acknowledged=True,allow_exploratory=True,initial_backfill_limit=1)
    index=SimpleNamespace(identity=lambda run:{'sample_id':'recorded'})
    controller=SnapWatchController(service,settings,sample_index=index)
    old=controller.discover()[0]
    controller.state.discovered[old.run_id]=replace(old,scientific_summary={'scientific_status':'ok'})
    source.write_text(source.read_text().replace('6400 3 0.5','6400 4 0.5'))
    assert not controller.discover()
    flagged=controller.state.discovered[old.run_id]
    assert flagged.scientific_summary['scientific_status']=='needs_review'
    with pytest.raises(ValueError,match='changed after discovery'):
        controller._prepare_run(flagged)
    count=len(service.records)
    assert not controller.discover() and len(service.records)==count
    restored=SnapWatchController(service,settings,sample_index=index)
    assert restored.restore_latest_state()
    assert not restored.discover() and len(service.jobs)==0


@pytest.mark.parametrize('missing_profile',[False,True])
def test_completed_watch_recovers_original_profile_bank_and_hash(source,supplied,monkeypatch,missing_profile):
    from types import SimpleNamespace
    from test_nomad import Galaxy
    from radar_pd_nova.transition_coordinate_mapping import mapped_calibration
    # The original file has labels 7 and 8; analysis stages label 7 as Bank 1.
    supplied.write_text(profile(7)+profile(8,difc=8000))
    monkeypatch.setattr(WatchDefinition,'source_directory',property(lambda self:str(source.parent)))
    monkeypatch.setattr(SnapWatchController,'_read_scan_metadata',lambda *a:{})
    service=Galaxy(supplied.read_text())
    settings=SnapExperimentSettings('IPTS-1','h','c',instrument_profile_dataset_id='original',
        selected_banks=(2,),subfolder='shared/SNAPRed/export/gsa/bank',
        product_family=snap.parse_filename(source).family,profile_bank=7,
        profile_acknowledged=True,allow_exploratory=True)
    index=SimpleNamespace(identity=lambda run:{'sample_id':'recorded'})
    first=SnapWatchController(service,settings,sample_index=index)
    run=first.discover()[0];first.submit(run)
    first.state.mark_completed(run.run_id,['final-gpx'],galaxy_job_id='job');first.persist_state()
    first._staging.cleanup()  # A restarted viewer cannot use the old local copy.
    reads=[];original=service.load_text_document
    def load(identifier):
        reads.append(identifier)
        if missing_profile:raise OSError('Saved profile temporarily unavailable')
        return original(identifier)
    service.load_text_document=load
    restored=SnapWatchController(service,settings,sample_index=index)
    assert restored.restore_latest_state()
    assert len(restored.state.completed)==1 and not restored.state.submitted
    scan={'run_id':run.run_id,'bank':2,'instrument':'SNAP'}
    if missing_profile:
        with pytest.raises(ValueError,match='temporarily unavailable'):
            mapped_calibration(scan,restored,{})
    else:
        calibration=mapped_calibration(scan,restored,{})
        assert (calibration.source_bank,calibration.profile_bank,calibration.difc)==(2,7,6200)
        completed=restored.state.completed[run.run_id]
        for field,value,message in [('source_bank',3,'physical bank'),('profile_bank',8,'Bank label'),
                                    ('profile_sha256','0'*64,'analysis receipt')]:
            receipt=dict(completed.scan_metadata['snap_bank_provenance']);receipt[field]=value
            restored.state.completed[run.run_id]=replace(completed,
                scan_metadata={**completed.scan_metadata,'snap_bank_provenance':receipt})
            with pytest.raises(ValueError,match=message):mapped_calibration(scan,restored,{})
    assert reads==['original'] and len(service.snapshots)==1
