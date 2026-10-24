"""Synthetic format/identity fixtures, never private facility event data."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from radar_pd_nova import nomad, nomad_metadata
from radar_pd_nova.nomad_controller import (NomadExperimentSettings, NomadWatchController, preflight_nomad_experiment)
from radar_pd_nova.powgen_watch import WatchDefinition, WatchState, WatchedRun, discover_from_listing, LIVE_STATE_SCHEMA
from radar_pd_nova.models import AnalysisConfig, RunRecord, RunStatus


def profile(bank=1,difc=9910.54,angle=152.57):
    return f'''#Bank {bank}: GSAS-II instrument parameter file
Type:PNT
Bank:{bank}.0
difC:{difc}
difA:0
difB:0
Zero:0
2-theta:{angle}
fltPath:20.5
alpha:0.1
beta-0:0.01
beta-1:0.01
beta-q:0.01
sig-0:1
sig-1:1
sig-2:1
sig-q:0
X:0
Y:0
'''


@pytest.fixture
def source(tmp_path):
    path=tmp_path/'NOM_42.gsa'
    text='Sample Run: 42 Wavelength: 1.5 A\n# Synthetic standard\n# 6 Histograms\n# Instrument: Dummy\n# with Y multiplied by the bin widths.\n'
    for i,(difc,angle) in enumerate([(1428.82,15.31),(2861.51,31.2),(5606.56,66.25),(9041.71,118.28),(9910.54,152.57),(836.207,9.17)],1):
        text+=f'# Total flight path 20.5m, tth {angle}deg, DIFC {difc}\nBANK {i} 2 2 SLOG 1000 1100 .0008 0 FXYE\n1000 {-i}.5 0.1\n1100 {i}.25 0.2\n'
    path.write_text(text)
    path.with_suffix('.json').write_text(json.dumps({'Instrument':'NOMAD','Sample':{'Runs':'42','Material':'Si','DummyInfo':False},'Calibration':{'Filename':'/SNS/NOM/shared/calibration/reference.h5'},'Environment':{'Name':'InAir','Container':'PAC06'}}))
    return path


@pytest.fixture
def supplied(tmp_path):
    p=tmp_path/'test.instprm';p.write_text(profile());return p


def test_nonfirst_bank_staging_preserves_observations_and_uncertainties(source,supplied,tmp_path):
    setup=nomad.inspect_reduced_setup(source)
    data,inst,receipt=nomad.stage_bank(source,setup,5,supplied,1,tmp_path/'out',acknowledged=True)
    assert receipt['source_bank']==5 and receipt['profile_bank']==1 and receipt['staged_bank']==1
    assert b'1000 -5.5 0.1\n1100 5.25 0.2\n' in data.read_bytes()
    assert b'1000 -1.5' not in data.read_bytes()
    assert '# 1 Histograms' in data.read_text()
    assert 'DIFC 9910.54' in data.read_text() and 'DIFC 1428.82' not in data.read_text()
    assert 'Bank:1.0' in inst.read_text()
    assert receipt['numeric_sha256']==setup.banks[4].numeric_sha256
    assert setup.warnings==()  # Instrument: Dummy is an export convention, not DummyInfo.


def test_invalid_unused_bank_does_not_rewrite_or_block_selected_bank(source,supplied,tmp_path):
    source.write_text(source.read_text().replace('1100 1.25 0.2','1100 1.25 0'))
    with pytest.raises(ValueError,match='uncertainty'):
        nomad.inspect_reduced_setup(source)
    selected=nomad.inspect_reduced_setup(source,selected_bank=5)
    assert selected.banks[0].quality_errors and not selected.banks[4].quality_errors
    data,_,_=nomad.stage_bank(source,selected,5,supplied,1,tmp_path/'valid-bank',acknowledged=True)
    assert b'1000 -5.5 0.1\n1100 5.25 0.2\n' in data.read_bytes()
    with pytest.raises(ValueError,match='uncertainty'):
        nomad.inspect_reduced_setup(source,selected_bank=1)
    with pytest.raises(ValueError,match='invalid numerical'):
        nomad.validate_profile(selected,1,supplied,1,acknowledged=True)


def test_preflight_invalid_selected_bank_keeps_alternatives_visible(source,supplied,monkeypatch):
    source.write_text(source.read_text().replace('1100 1.25 0.2','1100 1.25 0'))
    monkeypatch.setattr(WatchDefinition,'source_directory',property(lambda self:str(source.parent)))
    monkeypatch.setattr('radar_pd_nova.nomad_controller.read_nomad_scan_metadata',lambda *a:{})
    args=dict(profile_path=supplied,profile_bank=1,profile_acknowledged=True)
    blocked=preflight_nomad_experiment('IPTS-1',selected_banks=(1,),**args)
    assert not blocked['ready'] and blocked['available_banks']==[1,2,3,4,5,6]
    assert blocked['bank_details'][0]['quality_errors']
    assert not blocked['bank_details'][4]['quality_errors']
    allowed=preflight_nomad_experiment('IPTS-1',selected_banks=(5,),**args)
    assert allowed['ready']


def test_unicode_sample_titles_and_profile_comments_are_preserved(source,supplied,tmp_path):
    source.write_text(source.read_text().replace('Synthetic standard','α-Al₂O₃ — synthetic'))
    supplied.write_text(profile().replace('Type:PNT','# Résolution — synthetic\nType:PNT'))
    setup=nomad.inspect_reduced_setup(source)
    data,inst,_=nomad.stage_bank(source,setup,5,supplied,1,tmp_path/'unicode',acknowledged=True)
    assert 'α-Al₂O₃ — synthetic' in data.read_text()
    assert 'Résolution — synthetic' in inst.read_text()


def test_modern_product_uses_sibling_input_json(source,tmp_path):
    data=tmp_path/'multi_banks/GSAS/NOM_42.gsa';data.parent.mkdir(parents=True)
    side=tmp_path/'multi_banks/Input/NOM_42.json';side.parent.mkdir()
    data.write_bytes(source.read_bytes());side.write_bytes(source.with_suffix('.json').read_bytes())
    setup=nomad.inspect_reduced_setup(data)
    assert setup.sidecar_path==str(side) and not setup.warnings


@pytest.mark.parametrize('bank,difc,angle',[(2,2837.58510543,31.),(3,5743.19050836,65.)])
def test_calibrated_profiles_are_not_confused_with_nominal_export_geometry(source,tmp_path,bank,difc,angle):
    supplied=tmp_path/'calibrated.instprm';supplied.write_text(profile(bank,difc,angle))
    setup=nomad.inspect_reduced_setup(source)
    data,inst,receipt=nomad.stage_bank(source,setup,bank,supplied,bank,tmp_path/'out',acknowledged=True)
    assert receipt['nominal_geometry_notice']
    assert f'difC:{difc}' in inst.read_text()
    assert 'nomad_profile_nominal_geometry_offset' in data.read_text()
    # A profile from another physical bank is still rejected.
    wrong=tmp_path/'wrong.instprm';wrong.write_text(profile(bank,difc*1.1,angle))
    with pytest.raises(ValueError,match='DIFC'):
        nomad.validate_profile(setup,bank,wrong,bank,acknowledged=True)


@pytest.mark.parametrize('change,message',[
    (lambda s:s.replace('1100 5.25 0.2','900 5.25 0.2'),'increase'),
    (lambda s:s.replace('1100 5.25 0.2','1100 5.25 0'),'uncertainty'),
    (lambda s:s.replace('1100 5.25 0.2','1100 nan 0.2'),'Nonfinite'),
    (lambda s:s.replace('1100 5.25 0.2\n',''),'truncated'),
    (lambda s:s.replace('BANK 5','BANK 4'),'duplicate'),
    (lambda s:s.replace('Sample Run: 42','Sample Run: 43'),'disagree'),
    (lambda s:s.replace('FXYE','STD'),'FXYE'),
])
def test_malformed_or_partial_banks_block(source,change,message):
    source.write_text(change(source.read_text()))
    with pytest.raises(ValueError,match=message):nomad.inspect_reduced_setup(source)


def test_profile_mapping_is_explicit_not_ordinal(source,supplied):
    setup=nomad.inspect_reduced_setup(source)
    supplied.write_text(profile(2,2861.51,31.2)+profile(5))
    nomad.validate_profile(setup,5,supplied,5,acknowledged=True)
    with pytest.raises(ValueError,match='DIFC'):nomad.validate_profile(setup,5,supplied,2,acknowledged=True)
    with pytest.raises(ValueError,match='ordinal'):nomad.validate_profile(setup,5,supplied,1,acknowledged=True)
    with pytest.raises(ValueError,match='Confirm'):nomad.validate_profile(setup,5,supplied,5)


def test_ambiguous_and_incomplete_profiles_block(supplied):
    supplied.write_text(profile()+profile())
    with pytest.raises(ValueError,match='Duplicate'):nomad.profile_blocks(supplied)
    supplied.write_text(profile().replace('sig-0:1\n',''))
    with pytest.raises(ValueError,match='complete'):nomad.profile_blocks(supplied)


@pytest.mark.parametrize('which',['data','sidecar','profile'])
def test_changed_scientific_inputs_cannot_silently_stage(source,supplied,tmp_path,which):
    setup=nomad.inspect_reduced_setup(source)
    target={'data':source,'sidecar':source.with_suffix('.json'),'profile':supplied}[which]
    if which=='profile':target.write_text(profile(difc=20000))
    else:target.write_text(target.read_text()+'\n')
    with pytest.raises(ValueError):nomad.stage_bank(source,setup,5,supplied,1,tmp_path/'out',acknowledged=True)


def test_summed_input_and_placeholder_corrections_are_distinct(source):
    side=source.with_suffix('.json');x=json.loads(side.read_text())
    x['Sample']['Runs']='41-42';side.write_text(json.dumps(x))
    with pytest.raises(ValueError,match='summed'):nomad.inspect_reduced_setup(source)
    x['Sample'].update(Runs='42',DummyInfo=True,PushPositiveLevel=100)
    side.write_text(json.dumps(x));setup=nomad.inspect_reduced_setup(source)
    assert len(setup.warnings)==2


def test_discovery_ignores_summed_pdf_unnormalized_and_nested_products(tmp_path):
    for name in ('NOM_42.gsa','NOM_Si_summed.gsa','NOM_42.xye','NOM_42.gofr'): (tmp_path/name).write_text('')
    (tmp_path/'nested').mkdir();(tmp_path/'nested/NOM_43.gsa').write_text('')
    assert [Path(r['path']).name for r in nomad.directory_listing(tmp_path)]==['NOM_42.gsa']
    for folder in ('shared/autoreduce/multi_banks_summed/GSAS','shared/autoreduce/multi_banks/GSAS_unnorm','shared/autoreduce/single_bank'):
        with pytest.raises(ValueError):WatchDefinition('IPTS-1',folder,'h','c','p',instrument='NOM')


def test_watch_identity_is_scoped_to_beamline_and_recovery_preserves_it():
    definition=WatchDefinition('IPTS-1',nomad.MODERN_FOLDER,'h','c','p',instrument='NOM')
    rows=[{'path':definition.source_directory+'/NOM_42.gsa','size':10,'modified_ns':1},
          {'path':definition.source_directory+'/PG3_42.gsa','size':10,'modified_ns':1}]
    runs=discover_from_listing(definition,rows);assert [r.run_id for r in runs]==['NOM_42']
    state=WatchState('h',instrument='NOM');state.discovered[runs[0].run_id]=runs[0]
    assert state.contains(42)
    state.mark_submitted(42,'job')
    restored=WatchState.from_json(state.to_json())
    assert restored.schema==LIVE_STATE_SCHEMA and restored.submitted['NOM_42'].instrument=='NOM'
    with pytest.raises(ValueError):discover_from_listing(definition,rows,WatchState('h'))


def test_nomad_settings_require_one_bank_and_no_monochromatic_override():
    with pytest.raises(ValueError,match='one NOMAD'):NomadExperimentSettings('IPTS-1','h','c')
    with pytest.raises(ValueError,match='broadband'):NomadExperimentSettings('IPTS-1','h','c','1.5',selected_banks=(5,))
    s=NomadExperimentSettings('IPTS-1','h','c',selected_banks=(5,),profile_bank=1)
    assert s.definition().instrument=='NOM'
    assert s.definition().config_refs!=replace(s,profile_bank=2).definition().config_refs


def test_nomad_temperature_uses_selected_summary_and_explicit_units(tmp_path,monkeypatch):
    from radar_pd_nova.powgen_metadata import _Snapshot
    import h5py
    file=tmp_path/'metadata.h5';seen=[]
    with h5py.File(file,'w') as h:
        h.create_dataset('entry/run_number', data=[42])
        h.create_dataset('entry/experiment_identifier', data=[b'IPTS-1'])
        h.create_dataset('entry/sample/identifier', data=[b'123'])
        h.create_dataset('entry/DASlogs/SampleId/value', data=[123])
    mapping={
        '/entry/run_number':_Snapshot('run',text='42'),
        '/entry/experiment_identifier':_Snapshot('ipts',text='IPTS-1'),
        '/entry/sample/identifier':_Snapshot('id',text='123'),
        '/entry/DASlogs/SampleId/value':_Snapshot('id2',values=(123,)),
        '/entry/DASlogs/BL1B:SE:SelectSampleTemp/value_strings':_Snapshot('sensor',text='Cryostream'),
        '/entry/DASlogs/BL1B:SE:SampleTemp/average_value':_Snapshot('mean',values=(26.85,),unit='C'),
        '/entry/DASlogs/BL1B:SE:SampleTemp/minimum_value':_Snapshot('min',values=(26.8,),unit='C'),
        '/entry/DASlogs/BL1B:SE:SampleTemp/maximum_value':_Snapshot('max',values=(26.9,),unit='C'),
    }
    import radar_pd_nova.powgen_metadata as pm
    def read(path,field):seen.append(field);return mapping.get(field)
    monkeypatch.setattr(pm,'_h5dump_snapshot',read);monkeypatch.setattr(pm,'_h5py_snapshot',read)
    result=nomad_metadata.read_nomad_scan_metadata('IPTS-1',42,nexus_path=file)
    assert result['temperature']['value']==300 and result['temperature']['statistic']=='mean'
    assert result['temperature']['minimum']==299.95 and result['temperature_sensor']=='Cryostream'
    assert not any('levitator' in field.lower() or 'event' in field.lower() for field in seen)
    assert 'wavelength' not in result
    mapping['/entry/DASlogs/SampleId/value']=_Snapshot('bad',values=(124,))
    with h5py.File(file,'r+') as h:
        h['entry/DASlogs/SampleId/value'][0]=124
    result=nomad_metadata.read_nomad_sample_identity('IPTS-1',42,nexus_path=file)
    assert not result['sample_id'] and result['identity_warning']


class Galaxy:
    history_id='h'
    def __init__(self,text):self.text=text;self.documents={};self.snapshots=[];self.records=[];self.jobs=[]
    def load_text_document(self,key):return self.documents.get(key,self.text)
    def upload_document(self,path,**kw):self.documents['profile-hda']=Path(path).read_text();return 'profile-hda'
    def load_configuration_dataset(self,key):return AnalysisConfig(radiation='neutron',instrument_mode='tof',sample_elements=['Si'])
    def create_submission_snapshot(self,config,inputs,**kw):
        result=SimpleNamespace(config=config,inputs=inputs,**kw);self.snapshots.append(result);return result
    def submit_snapshot(self,s):
        result=RunRecord(uid='job',name=s.config.run_name,mode=s.config.mode,history_id='h',galaxy_job_id='job',status=RunStatus.QUEUED,config=s.config,inputs=s.inputs)
        self.jobs.append(result);return result
    def upload_json_document(self,payload,**kw):self.records.append(payload);return f'state-{len(self.records)}'
    def search_history_datasets(self,**kw):return [{'id':f'state-{i}'} for i in range(len(self.records),0,-1)]
    def load_json_document(self,key):return json.loads(json.dumps(self.records[int(key.split('-')[1])-1]))
    def recent_runs(self,**kw):return self.jobs


def test_controller_submits_selected_bank_and_records_scientific_provenance(source,supplied,monkeypatch):
    monkeypatch.setattr(WatchDefinition,'source_directory',property(lambda self:str(source.parent)))
    service=Galaxy(supplied.read_text())
    settings=NomadExperimentSettings('IPTS-1','h','c',instrument_profile_dataset_id='original',
        selected_banks=(5,),profile_bank=1,profile_acknowledged=True)
    c=NomadWatchController(service,settings)
    c._read_scan_metadata=lambda run:{}
    runs=c.discover();assert [r.run_id for r in runs]==['NOM_42_BANK5']
    record=c.submit(runs[0]);snap=service.snapshots[0]
    assert snap.inputs.instrument=='NOM' and snap.inputs.bank=='5'
    assert b'1000 -5.5' in Path(snap.inputs.data_path).read_bytes()
    assert record.provenance['nomad_bank']['source_bank']==5
    assert service.records[-1]['watch']['instrument']=='NOM'
    assert service.records[-1]['watch']['nomad_profile_bank']==1
    assert c._profile_matches_watch(record,runs[0],'profile-hda')


def test_preflight_reports_corrections_and_requires_explicit_ack(source,supplied,monkeypatch):
    monkeypatch.setattr(WatchDefinition,'source_directory',property(lambda self:str(source.parent)))
    monkeypatch.setattr('radar_pd_nova.nomad_controller.read_nomad_scan_metadata',lambda *a:{})
    side=source.with_suffix('.json');x=json.loads(side.read_text());x['Sample']['DummyInfo']=True;side.write_text(json.dumps(x))
    args=dict(selected_banks=(5,),profile_path=supplied,profile_bank=1,profile_acknowledged=True)
    blocked=preflight_nomad_experiment('IPTS-1',**args)
    assert not blocked['ready'] and blocked['quality_notices']
    allowed=preflight_nomad_experiment('IPTS-1',allow_exploratory=True,**args)
    assert allowed['ready'] and allowed['analysis_count']==1


@pytest.mark.parametrize('changed',['calibration','environment','geometry'])
def test_checkpoint_recovery_preserves_mapping_and_calibration_guard(source,supplied,monkeypatch,changed):
    monkeypatch.setattr(WatchDefinition,'source_directory',property(lambda self:str(source.parent)))
    service=Galaxy(supplied.read_text())
    settings=NomadExperimentSettings('IPTS-1','h','c',instrument_profile_dataset_id='original',
        selected_banks=(5,),profile_bank=1,profile_acknowledged=True)
    first=NomadWatchController(service,settings);first._read_scan_metadata=lambda run:{}
    first.submit(first.discover()[0])
    restored=NomadWatchController(service,settings);restored._read_scan_metadata=lambda run:{}
    assert restored.restore_latest_state()
    assert 'NOM_42_BANK5' in restored.state.submitted
    assert restored._manual_setup_signature==first._manual_setup_signature
    assert not restored.discover() and not service.snapshots[1:]
    assert not NomadWatchController(service,replace(settings,profile_bank=2)).restore_latest_state()
    assert not NomadWatchController(service,replace(settings,main_cif_dataset_id='another-phase')).restore_latest_state()
    next_source=source.with_name('NOM_43.gsa')
    text=source.read_text().replace('Sample Run: 42','Sample Run: 43')
    if changed=='geometry':text=text.replace('DIFC 9910.54','DIFC 9911.54')
    next_source.write_text(text)
    side=json.loads(source.with_suffix('.json').read_text())
    side['Sample']['Runs']='43'
    if changed=='calibration':side['Calibration']['Filename']='another-calibration.h5'
    if changed=='environment':side['Environment']['Name']='Furnace'
    next_source.with_suffix('.json').write_text(json.dumps(side))
    run=restored.discover()[0]
    with pytest.raises(ValueError,match='calibration'):
        restored._prepare_run(run)


@pytest.mark.parametrize('which',['data','sidecar'])
def test_replaced_reductions_are_flagged_below_backfill_floor(source,supplied,monkeypatch,which):
    monkeypatch.setattr(WatchDefinition,'source_directory',property(lambda self:str(source.parent)))
    service=Galaxy(supplied.read_text())
    settings=NomadExperimentSettings('IPTS-1','h','c',instrument_profile_dataset_id='original',
        selected_banks=(5,),profile_bank=1,profile_acknowledged=True,initial_backfill_limit=5)
    c=NomadWatchController(service,settings);c._read_scan_metadata=lambda run:{}
    old=c.discover()[0]
    c.state.discovered[old.run_id]=replace(old,scientific_summary={'scientific_status':'ok'})
    target=source if which=='data' else source.with_suffix('.json')
    target.write_text(target.read_text()+'\n')
    assert not c.discover()  # Never creates a second job automatically.
    flagged=c.state.discovered[old.run_id]
    assert flagged.scientific_summary['scientific_status']=='needs_review'
    assert flagged.scan_metadata['source_revision_warning']
    with pytest.raises(ValueError,match='changed after discovery'):c._prepare_run(flagged)
    count=len(service.records);c.discover();assert len(service.records)==count
    restored=NomadWatchController(service,settings);restored._read_scan_metadata=lambda run:{}
    assert restored.restore_latest_state()
    restored.discover();assert len(service.records)==count


def test_staged_bank_carries_durable_review_notice(source,supplied,tmp_path):
    # The actual staged file, rather than a separate UI object, crosses into
    # Analyze so that a closed NOVA session cannot lose these notices.
    setup=nomad.inspect_reduced_setup(source)
    data,_,_=nomad.stage_bank(source,setup,5,supplied,1,tmp_path/'stage',acknowledged=True)
    marker=next(line for line in data.read_text().splitlines() if line.startswith('# RADAR-PD input provenance: '))
    receipt=json.loads(marker.split(': ',1)[1])
    assert receipt['source_bank']==5 and receipt['profile_bank']==1
    assert receipt['warnings'][0]['code']=='unverified_nomad_profile'
    assert receipt['source_sha256']==setup.source_sha256


def test_lost_ack_recovery_uses_immutable_input_identity(source,supplied,monkeypatch):
    monkeypatch.setattr(WatchDefinition,'source_directory',property(lambda self:str(source.parent)))
    service=Galaxy(supplied.read_text())
    settings=NomadExperimentSettings('IPTS-1','h','c',instrument_profile_dataset_id='original',
        selected_banks=(5,),profile_bank=1,profile_acknowledged=True)
    first=NomadWatchController(service,settings);first._read_scan_metadata=lambda run:{}
    run=first.discover()[0]
    record=first.launch_submission(run)  # Galaxy accepted it; no local acknowledgement checkpoint.
    restored=NomadWatchController(service,settings);restored._read_scan_metadata=lambda run:{}
    assert restored.restore_latest_state()
    assert restored.reconcile_recent_runs()==1
    assert restored.state.submitted[run.run_id].galaxy_job_id==record.galaxy_job_id
    side=source.with_suffix('.json');side.write_text(side.read_text()+'\n')
    revised=NomadWatchController(service,settings);revised._read_scan_metadata=lambda run:{}
    new_run=revised.discover()[0]
    assert revised._run_name(new_run)!=record.name
    assert revised.reconcile_recent_runs()==0
    assert new_run.run_id in revised.state.discovered
