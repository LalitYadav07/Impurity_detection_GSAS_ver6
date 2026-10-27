"""Synthetic reproductions of measured HB-2A archive edge cases; no private data."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from radar_pd_nova import hb2a
from radar_pd_nova.hb2a_controller import (Hb2aExperimentSettings, Hb2aSampleIndex,
    Hb2aWatchController, discover_hb2a_samples, preflight_hb2a_experiment)
from radar_pd_nova.live_beamlines import beamline_adapter
from radar_pd_nova.diffraction_coordinates import CwCalibration, TofCalibration, coordinate_values, display_pattern
from radar_pd_nova.models import AnalysisConfig, RunRecord, RunStatus
from radar_pd_nova.powgen_watch import WatchState, WatchDefinition, discover_from_listing
from radar_pd_nova.transition_input import hb2a_observation, packet_observation
from radar_pd_nova.transition_tracking import TransitionTrackingEngine


def spice(*, exp=1000,scan=1,ipts=1,date='5/17/2026',m1=9.45,coll=0,
          name='Synthetic, sample',x_axis='2theta',footer='completed.',missing_coll=False):
    columns=['Pt.','2theta','time','monitor','m1',*([] if missing_coll else ['colltrans']),'sample_temp','active_field','hi_pressure']
    text=f'''# scan = {scan}
# date = {date}
# time = 1:00:00 PM
# proposal = {ipts}
# experiment_number = {exp}
# samplename = {name}
# scan_title = {name}, nuclear structure scan
# def_x = {x_axis}
# col_headers =
# {' '.join(columns)}
'''
    for i in range(32):
        row=[i+1,5+i*.1,10 if i%2==0 else 20,100,m1,*([] if missing_coll else [coll]),100+i,0,2]
        text+=' '.join(map(str,row))+'\n'
    if footer:
        text+=f'# 1:10:00 PM {date} scan {footer}\n'
    return text


def xye(masked=False,negative=False):
    values=[]
    for i,x in enumerate(np.linspace(5,130,1024)):
        y=20+100*np.exp(-((x-35)/.25)**2)+80*np.exp(-((x-65)/.2)**2)
        sigma=2.0
        if masked and i==20:y=sigma=0
        if negative and i==21:y=-3
        values.append(f'{x:.9f} {y:.9f} {sigma:.9f}\n')
    return ''.join(values)


@pytest.fixture
def archive(tmp_path,monkeypatch):
    root=tmp_path/'HFIR/HB2A';root.mkdir(parents=True)
    monkeypatch.setattr(hb2a,'ARCHIVE_ROOT',root)
    return root


def add(archive,*,exp=1000,scan=1,normalization='time',**kwargs):
    ipts=archive/'IPTS-1';out=ipts/'shared/autoreduce';out.mkdir(parents=True,exist_ok=True)
    raw=ipts/f'exp{exp}'/'Datafiles'/f'HB2A_exp{exp:04d}_scan{scan:04d}.dat';raw.parent.mkdir(parents=True,exist_ok=True)
    raw.write_text(spice(exp=exp,scan=scan,**kwargs))
    suffix='_autoreduced_norm_time' if normalization=='time' else '_autoreduced' if normalization=='monitor' else ''
    data=out/f'HB2A_exp{exp:04d}_scan{scan:04d}{suffix}.dat';data.write_text(xye())
    return data,raw


@pytest.fixture
def profile(tmp_path):
    p=tmp_path/'reviewed.instprm'
    p.write_text('# Synthetic profile\nType:PNC\nBank:1\nLam:2.4063\nZero:-0.01\nU:800\nV:-440\nW:240\nX:0\nY:0\nZ:0\nSH/L:0.09\n')
    return p


def settings(**kwargs):
    return Hb2aExperimentSettings('IPTS-1','history','config',instrument_profile_dataset_id='profile',
        profile_acknowledged=True,**kwargs)


class Service:
    history_id='history'
    def __init__(self,profile):
        self.profile=profile;self.checkpoints=[];self.snapshots=[]
    def upload_json_document(self,value,**_):
        self.checkpoints.append(json.loads(json.dumps(value)));return f'state-{len(self.checkpoints)}'
    def search_history_datasets(self,**_):
        return [{'id':f'state-{i+1}'} for i in reversed(range(len(self.checkpoints)))]
    def load_json_document(self,key):return self.checkpoints[int(key.split('-')[-1])-1]
    def load_text_document(self,_):return self.profile.read_text()
    def load_configuration_dataset(self,_):return AnalysisConfig(radiation='neutron',instrument_mode='cw',sample_elements=['Si'])
    def upload_document(self,*_,**__):return 'uploaded-profile'
    def create_submission_snapshot(self,config,inputs,**kwargs):
        result=(config,inputs,kwargs);self.snapshots.append(result);return result
    def submit_snapshot(self,snapshot):
        config,inputs,_=snapshot
        return RunRecord(uid='job-'+str(len(self.snapshots)),galaxy_job_id='job-'+str(len(self.snapshots)),
            history_id='history',name=config.run_name,mode=config.mode,inputs=inputs,status=RunStatus.QUEUED)


@pytest.mark.parametrize('missing_profile',[False,True])
def test_completed_cw_watch_recovers_profile_and_keeps_receipt_checks(archive,profile,missing_profile):
    from radar_pd_nova.transition_coordinate_mapping import mapped_calibration
    add(archive)
    service=Service(profile);first=Hb2aWatchController(service,settings())
    run=first.discover()[0];first.submit(run)
    first.state.mark_completed(run.run_id,['final-gpx'],galaxy_job_id='job-1');first.persist_state()
    first._staging.cleanup()
    reads=[];original=service.load_text_document
    def load(identifier):
        reads.append(identifier)
        if missing_profile:raise OSError('Saved CW profile temporarily unavailable')
        return original(identifier)
    service.load_text_document=load
    restored=Hb2aWatchController(service,settings());assert restored.restore_latest_state()
    assert len(restored.state.completed)==1 and not restored.state.submitted
    scan={'run_id':run.run_id,'bank':1,'instrument':'HB2A'}
    if missing_profile:
        with pytest.raises(ValueError,match='temporarily unavailable'):mapped_calibration(scan,restored,{})
    else:
        calibration=mapped_calibration(scan,restored,{})
        assert isinstance(calibration,CwCalibration) and calibration.wavelength==2.4063
        completed=restored.state.completed[run.run_id]
        receipt=dict(completed.scan_metadata['hb2a_pattern_provenance']);receipt['profile_sha256']='0'*64
        restored.state.completed[run.run_id]=replace(completed,
            scan_metadata={**completed.scan_metadata,'hb2a_pattern_provenance':receipt})
        with pytest.raises(ValueError,match='analysis receipt'):mapped_calibration(scan,restored,{})
    assert reads==['profile'] and len(service.snapshots)==1


def test_facility_mode_and_experiment_scan_identity_round_trip(archive):
    add(archive,exp=1000,scan=1);add(archive,exp=1001,scan=1)
    s=settings();d=s.definition();runs=discover_from_listing(d,hb2a.directory_listing(d.source_directory))
    assert [r.run_id for r in runs]==['HB2A_EXP1000_SCAN0001','HB2A_EXP1001_SCAN0001']
    state=WatchState('history',instrument='HB2A',discovered={r.run_id:r for r in runs})
    saved=WatchState.from_json(state.to_json())
    assert len(saved.discovered)==2 and saved.contains(runs[1].run_id)
    assert beamline_adapter('HB2A').facility=='HFIR' and beamline_adapter('HB2A').instrument_mode=='cw'
    with pytest.raises(ValueError):state.contains(1)
    with pytest.raises(ValueError):replace(d,facility='SNS')


@pytest.mark.parametrize('name',[None,'HB2A_exp1000_scan0001-0003_autoreduced_norm_time.dat','__HB2A_reduced.dat','HB2A_exp1000_scan0001_autoreduced_norm_time_all.dat','HB2A_exp1000_scan0001_autoreduced.gss'])
def test_only_individual_reduced_xye_products_trigger(name):
    assert hb2a.product(name or '') is None


def test_normalizations_and_legacy_exports_are_explicit_and_not_combined(archive):
    for mode in hb2a.NORMALIZATIONS:add(archive,normalization=mode)
    root=archive/'IPTS-1/shared/autoreduce'
    for mode in hb2a.NORMALIZATIONS:
        found=hb2a.directory_listing(root,normalization=mode)
        assert len(found)==1 and hb2a.product(found[0]['path']).normalization==mode


def test_legacy_xye_keeps_original_uncertainties_and_requires_explicit_family(archive):
    data,_=add(archive,normalization='legacy')
    legacy=data.with_suffix('.xye');data.rename(legacy)
    root=legacy.parent
    assert not hb2a.directory_listing(root,normalization='time')
    assert hb2a.directory_listing(root,normalization='legacy')[0]['path']==str(legacy)
    setup=hb2a.inspect_reduced_setup(legacy,ipts='IPTS-1')
    assert setup.normalization=='legacy' and setup.records==1024


def test_samples_survive_missing_setup_and_csv_commas(archive):
    add(archive,missing_coll=True)
    (archive/'IPTS-1/shared/autoreduce/IPTS-1_summary.csv').write_text('location,scan_number,samplename\nforeign,1,wrong,shifted\n')
    index=Hb2aSampleIndex('IPTS-1');r=discover_hb2a_samples('IPTS-1',index)
    assert r['scan_count']==1 and r['unidentified_count']==0
    assert 'Synthetic, sample' in r['options'][0]['title']
    assert 'sample_temp' in r['condition_channels']
    assert 'no sample ID' in index.identity(discover_from_listing(settings().definition(),hb2a.directory_listing(archive/'IPTS-1/shared/autoreduce'))[0])['identity_basis']


@pytest.mark.parametrize('date,coll,expected',[('2/22/2021',0,'IN'),('2/23/2021',0,'OUT'),('2/22/2021',80,'OUT'),('2/23/2021',80,'IN')])
def test_date_qualified_collimator_convention(archive,date,coll,expected):
    data,_=add(archive,date=date,coll=coll)
    assert hb2a.inspect_reduced_setup(data,ipts='IPTS-1').metadata['collimator']==expected


def test_ge117_requires_documented_date_and_motor_near_known_position(archive):
    data,raw=add(archive,date='4/24/2021',m1=-4.375)
    with pytest.raises(ValueError,match='Ge reflection'):hb2a.inspect_reduced_setup(data,ipts='IPTS-1')
    raw.write_text(spice(date='4/25/2021',m1=-4.375))
    assert hb2a.inspect_reduced_setup(data,ipts='IPTS-1').metadata['ge_reflection']==117
    raw.write_text(spice(m1=5.0))
    with pytest.raises(ValueError,match='Ge reflection'):hb2a.inspect_reduced_setup(data,ipts='IPTS-1')


def test_missing_collimator_needs_reviewed_choice_and_cannot_override_measurement(archive,profile):
    data,raw=add(archive,missing_coll=True)
    args=dict(profile_path=profile,profile_acknowledged=True)
    blocked=preflight_hb2a_experiment('IPTS-1',**args)
    assert not blocked['ready'] and 'colltrans is absent' in blocked['message']
    assert preflight_hb2a_experiment('IPTS-1',collimator_override='OUT',**args)['ready']
    raw.write_text(spice(coll=80))
    assert not preflight_hb2a_experiment('IPTS-1',collimator_override='OUT',**args)['ready']


def test_stopped_scan_remains_selectable_but_requires_review(archive,profile):
    add(archive,footer='stopped!!')
    assert discover_hb2a_samples('IPTS-1',Hb2aSampleIndex('IPTS-1'))['options']
    args=dict(profile_path=profile,profile_acknowledged=True)
    assert 'stopped early' in preflight_hb2a_experiment('IPTS-1',**args)['message']
    assert preflight_hb2a_experiment('IPTS-1',allow_stopped=True,**args)['ready']


def test_raw_identity_and_angle_scan_contracts_are_checked(archive):
    data,raw=add(archive)
    raw.write_text(spice(ipts=2))
    with pytest.raises(ValueError,match='disagrees'):hb2a.inspect_reduced_setup(data,ipts='IPTS-1')
    raw.write_text(spice(x_axis='sample_temp'))
    with pytest.raises(ValueError,match='fixed-angle'):hb2a.inspect_reduced_setup(data,ipts='IPTS-1')
    raw.write_text(spice(footer=''))
    with pytest.raises(ValueError,match='footer'):hb2a.inspect_reduced_setup(data,ipts='IPTS-1')


def test_conditions_require_measured_channel_units_and_review(archive):
    data,_=add(archive)
    assert 'temperature' not in hb2a.inspect_reduced_setup(data,ipts='IPTS-1').metadata
    binding={'temperature':{'channel':'sample_temp','unit':'K','reviewed':True}}
    metadata=hb2a.inspect_reduced_setup(data,ipts='IPTS-1',condition_bindings=binding).metadata
    expected=sum((10 if i%2==0 else 20)*(100+i) for i in range(32))/480
    assert metadata['temperature']['value']==pytest.approx(expected)
    assert metadata['temperature']['minimum']==100 and metadata['temperature']['maximum']==131
    binding['temperature']['reviewed']=False
    with pytest.raises(ValueError,match='review'):hb2a.inspect_reduced_setup(data,ipts='IPTS-1',condition_bindings=binding)
    binding={'magnetic_field':{'channel':'active_field','unit':'T','reviewed':True},
             'pressure':{'channel':'hi_pressure','unit':'bar','reviewed':True}}
    result=hb2a.inspect_reduced_setup(data,ipts='IPTS-1',condition_bindings=binding).metadata
    assert result['magnetic_field']['unit']=='T' and result['pressure']['unit']=='Pa'
    assert result['pressure']['value']==2e5
    assert len(result['conditions']['channels'])==2
    with pytest.raises(ValueError,match='not sample-environment'):
        hb2a.inspect_reduced_setup(data,ipts='IPTS-1',condition_bindings={'temperature':{'channel':'monitor','unit':'K','reviewed':True}})


def test_profile_review_type_wavelength_duplicates_and_physical_range(archive,profile):
    data,_=add(archive);setup=hb2a.inspect_reduced_setup(data,ipts='IPTS-1')
    with pytest.raises(ValueError,match='Confirm'):hb2a.validate_profile(setup,profile)
    text=profile.read_text()
    for old,new in [('Type:PNC','Type:PNT'),('Lam:2.4063','Lam:1.54'),('W:240','W:-1000000'),('Bank:1','Bank:1\nBank:2')]:
        profile.write_text(text.replace(old,new))
        with pytest.raises(ValueError):hb2a.validate_profile(setup,profile,acknowledged=True)
    profile.write_text(text)
    assert hb2a.validate_profile(setup,profile,acknowledged=True)['wavelength_angstrom']==2.4063


def test_staging_preserves_numeric_bytes_masks_and_revision_boundary(archive,profile,tmp_path):
    data,raw=add(archive);data.write_text(xye(masked=True))
    setup=hb2a.inspect_reduced_setup(data,ipts='IPTS-1')
    staged,inst,receipt=hb2a.stage_pattern(data,setup,profile,tmp_path/'out',acknowledged=True)
    assert staged.read_bytes().endswith(data.read_bytes()) and inst.read_bytes()==profile.read_bytes()
    assert setup.masked_points==1 and receipt['numeric_sha256']==sha256(data.read_bytes()).hexdigest()
    raw.write_text(raw.read_text().replace('Synthetic, sample','Changed specimen'))
    with pytest.raises(ValueError,match='metadata changed'):hb2a.stage_pattern(data,setup,profile,tmp_path/'out2',acknowledged=True)


def test_negative_data_never_silently_enter_a_clipping_cw_importer(archive):
    data,_=add(archive);data.write_text(xye(negative=True))
    with pytest.raises(ValueError,match='would clip'):hb2a.inspect_reduced_setup(data,ipts='IPTS-1')


def test_single_manual_profile_does_not_cross_experiments_and_full_ids_exclude(archive,profile):
    add(archive,exp=1000);add(archive,exp=1001)
    args=dict(profile_path=profile,profile_acknowledged=True)
    assert not preflight_hb2a_experiment('IPTS-1',**args)['ready']
    assert preflight_hb2a_experiment('IPTS-1',excluded_acquisitions=('HB2A_EXP1000_SCAN0001',),**args)['ready']


def test_live_arrivals_restart_scan_numbers_and_restore_new_only_cursor(archive,profile):
    add(archive,scan=50)
    service=Service(profile);controller=Hb2aWatchController(service,settings(initial_backfill_limit=0))
    assert not controller.discover()
    restored=Hb2aWatchController(service,settings(initial_backfill_limit=0))
    assert restored.restore_latest_state()
    add(archive,exp=1001,scan=1)
    new=restored.discover();assert len(new)==1 and new[0].run_id=='HB2A_EXP1001_SCAN0001'
    assert not restored.discover()
    # A late reduction from an old experiment also remains discoverable.
    add(archive,exp=1000,scan=49)
    assert restored.discover()[0].run_id=='HB2A_EXP1000_SCAN0049'


def test_partial_pattern_retries_and_completed_revisions_are_not_relabelled(archive,profile):
    data,_=add(archive);complete=data.read_bytes();data.write_bytes(complete[:100])
    service=Service(profile);controller=Hb2aWatchController(service,settings())
    broken=controller.discover()[0];assert 'setup_error' in broken.scan_metadata and not controller.due_submissions()
    data.write_bytes(complete)
    ready=controller.discover()[0];assert not ready.scan_metadata.get('setup_error')
    record=controller.submit(ready)
    assert record.inputs.instrument=='HB2A' and record.inputs.facility_root=='/HFIR'
    assert service.snapshots[0][0].instrument_mode=='cw'
    data.write_text(data.read_text().replace('20.000000000','21.000000000',1))
    controller.discover()
    assert controller.state.submitted[ready.run_id].scan_metadata.get('source_revision_warning')
    assert len(service.snapshots)==1


def test_native_cw_observation_and_q_display_keep_detectors_phase_independent(archive,profile):
    data,_=add(archive);setup=hb2a.inspect_reduced_setup(data,ipts='IPTS-1')
    scan={'run_id':'HB2A_EXP1000_SCAN0001','instrument':'HB2A','ipts':'IPTS-1','bank':1,
        'metadata':{**setup.metadata,'hb2a_setup':setup.as_dict()},'source_path':str(data),
        'transition_source_root':str(data.parent)}
    result=hb2a_observation(scan,data,data.parent)
    assert result.pattern.scope.coordinate=='two_theta_deg'
    c=CwCalibration.from_profile(profile,expected_sha256=sha256(profile.read_bytes()).hexdigest())
    converted=display_pattern(result.pattern,'q_inv_angstrom',c)
    np.testing.assert_allclose(converted.x,4*np.pi*np.sin(np.radians((result.pattern.x-c.zero)/2))/c.wavelength)
    np.testing.assert_array_equal(converted.intensity,result.pattern.intensity)
    np.testing.assert_allclose(coordinate_values(converted.x,'q_inv_angstrom','two_theta_deg',c),result.pattern.x)
    with pytest.raises(ValueError):coordinate_values(result.pattern.x,'two_theta_deg','tof_us',c)
    with pytest.raises(ValueError):coordinate_values(result.pattern.x,'two_theta_deg','d_angstrom',TofCalibration(1000))
