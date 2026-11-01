import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from radar_pd_nova import battery_controller as module
from radar_pd_nova.battery.discovery import discover_proposal
from radar_pd_nova.battery.electrochemistry import CyclerBinding
from radar_pd_nova.battery_controller import BatteryMixin, acquisition_row, read_candidate


class State(SimpleNamespace):
    def change(self, *fields):
        return lambda callback: callback
    def flush(self): pass


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv('RADAR_PD_ONCAT_ENABLED','false')
    app=BatteryMixin();app.server=SimpleNamespace(state=State())
    app._init_battery()
    return app


def test_one_unreadable_run_does_not_hide_other_acquisitions(app,tmp_path,monkeypatch):
    monkeypatch.setattr(module,'family_runs',lambda *args:[1,2])
    def inspect(instrument,ipts,run,**kwargs):
        if run==1:raise PermissionError('Run 1 metadata is unreadable')
        return dict(run=2,status='available',fields={},identity={'sample_id':'s','identity_status':'resolved'})
    monkeypatch.setattr(module,'inspect_run',inspect)
    monkeypatch.setattr(module,'ExperimentMetadataIndex',lambda *args,**kwargs:SimpleNamespace(enrich=lambda run,identity:identity))
    inventory={'instrument':'NOM','ipts':'IPTS-42','root':str(tmp_path)}
    asyncio.run(app._inspect_battery_async(inventory,'family',2,app._battery_generation))
    assert [row['state'] for row in app.server.state.battery_rows]==['unavailable','available']
    assert 'unreadable' in app.server.state.battery_rows[0]['issues']
    assert not app.server.state.battery_busy


def record(sample="66548"):
    return {"run":138504,"status":"available","fields":{"run_title":{"value":"LNO"}},
            "identity":{"identity_status":"resolved","sample_id":sample},
            "exposure":{"start_utc":"2019-12-09T17:00:26Z","end_utc":"2019-12-09T17:00:36Z","duration_seconds":10},"issues":[]}


def test_context_change_clears_all_scientific_binding_state(app):
    app._battery_acquisitions=[record()];app._battery_series=object()
    app.server.state.battery_origin="2019-12-09T12:00:26-05:00"
    app.server.state.battery_cell="old cell"
    app.server.state.battery_cycler_path="/old/upload"
    app._battery_context_changed()
    assert app._battery_series is None and app._battery_acquisitions==[]
    assert app.server.state.battery_cell==app.server.state.battery_origin==app.server.state.battery_cycler_path==""
    assert not app.server.state.battery_export_ready


def test_slow_old_proposal_does_not_replace_new_context(app,monkeypatch):
    def old_discovery(*args,**kwargs):
        app._battery_context_changed()
        return {"product_families":[{"folder":"old","run_count":1,"formats":["gsa"]}],"cycler_candidates":[],"issues":[],"status":"available"}
    monkeypatch.setattr(module,"discover_proposal",old_discovery)
    asyncio.run(app._discover_battery_async("NOM","IPTS-21886",app._battery_generation))
    assert app._battery_inventory is None and app.server.state.battery_families==[]
    assert not app.server.state.battery_busy


def test_binding_changed_during_import_does_not_publish_stale_join(app,tmp_path,monkeypatch):
    file=tmp_path/'cell.csv';file.write_text('elapsed_seconds,voltage_v,current_a\n0,3,1\n10,4,1\n')
    app._battery_acquisitions=[record()]
    app._battery_upload=SimpleNamespace(current_path=file)
    original=module.read_cycler
    def changed(*args,**kwargs):
        result=original(*args,**kwargs);app._battery_binding_changed();return result
    monkeypatch.setattr(module,'read_cycler',changed)
    binding=CyclerBinding('66548','cell','C05','charge',clock_origin='2019-12-09T12:00:26-05:00')
    asyncio.run(app._join_battery_async(binding,('upload','',str(file)),app._battery_generation,app._battery_binding_revision))
    assert app._battery_series is None and app._battery_joins==[]
    assert app.server.state.battery_echem_summary==''


def test_unregistered_upload_path_cannot_read_server_files(app,tmp_path,monkeypatch):
    file=tmp_path/'private.csv';file.write_text('sensitive')
    app._battery_acquisitions=[record()]
    def forbidden(*args):raise AssertionError('Unregistered server file was read')
    monkeypatch.setattr(Path,'read_bytes',forbidden)
    binding=CyclerBinding('66548','cell','C05','charge')
    asyncio.run(app._join_battery_async(binding,('upload','',str(file)),app._battery_generation,app._battery_binding_revision))
    assert 'Upload a completed' in app.server.state.battery_error
    assert 'binding stopped' in app.server.state.battery_message
    assert not app.server.state.battery_busy and app.server.state.battery_operation == ''


def test_native_cycler_archive_selection_detects_revised_file(tmp_path):
    shared=tmp_path/'NOM/IPTS-21886/shared';shared.mkdir(parents=True)
    path=shared/'cell.csv';path.write_text('timestamp,voltage_v,current_a\n2019-12-09T17:00:26Z,3,1\n')
    inventory=discover_proposal('NOM','IPTS-21886',facility_root=tmp_path)
    selected=inventory['cycler_candidates'][0]
    path.write_text(path.read_text()+'2019-12-09T17:00:36Z,4,1\n')
    with pytest.raises(ValueError,match='changed'):read_candidate(inventory,selected)


def test_wrong_sample_is_rejected_before_import(app):
    app._battery_acquisitions=[record()]
    app.join_battery_cycler(dict(sample='66546',cell='cell',channel='C05',positive='charge',origin='',offset=0,clock_uncertainty='',gap=120))
    assert 'verified' in app.server.state.battery_error
    assert app._battery_task is None


@pytest.mark.parametrize('value',['12.5',12.5,True,0,201,''])
def test_invalid_preview_limit_does_not_start_background_work(app,value):
    app._battery_inventory={'root':'/not-read'}
    app.server.state.battery_inspection_limit=value
    app.inspect_battery_acquisitions()
    assert app._battery_task is None and app.server.state.battery_error


def test_second_click_does_not_invalidate_active_request(app):
    async def run():
        gate=asyncio.Event()
        app._battery_start(gate.wait())
        generation=app._battery_generation
        app.discover_battery_proposal({'instrument':'NOM','ipts':'123'})
        assert app._battery_generation==generation
        gate.set();await app._battery_task
    asyncio.run(run())


def test_recorded_identity_conflicts_and_table_notices_are_visible():
    r=record();r['identity'].update(identity_status='conflict',sample_id='',identity_warning='Different IDs',metadata_notices=['Tables disagree'])
    row=acquisition_row(r)
    assert 'Different IDs' in row['issues'] and 'Tables disagree' in row['issues']
    assert row['sample_id']==''


def test_export_uses_download_and_preserves_qualification(app):
    app._battery_acquisitions=[record()]
    files=[];app.download_file=lambda name,mime,data:files.append((name,mime,data))
    app.export_battery_evidence()
    payload=json.loads(files[0][2])
    assert files[0][:2]==('battery-evidence.json','application/json')
    assert payload['scientific_qualification']=='not_evaluated'
    assert payload['acquisitions'][0]['run']==138504


def catalog_record(scope='run', **changes):
    from radar_pd_nova.battery.catalog import SCHEMA
    result=dict(schema=SCHEMA,scope=scope,instrument='NOM',ipts='IPTS-42',status='available',message='Recorded context',
        fields={'run_title':{'value':'Changed catalog title'}},identity={'sample_id':'different','identity_status':'resolved'})
    if scope=='run':result['run']=138504
    else:result['fields']=dict(title='Proposal title',abstract='Proposal purpose')
    result.update(changes);return result


def inventory(tmp_path):
    return dict(instrument='NOM',ipts='IPTS-42',root=str(tmp_path),status='available',
        product_families=[dict(folder='shared/reduced',run_count=1,formats=['gsa'])],cycler_candidates=[],issues=[])


def mock_archive(monkeypatch, source):
    monkeypatch.setattr(module,'family_runs',lambda *args:[source['run']])
    monkeypatch.setattr(module,'inspect_run',lambda *args,**kwargs:deepcopy(source))
    monkeypatch.setattr(module,'ExperimentMetadataIndex',lambda *args,**kwargs:SimpleNamespace(enrich=lambda run,identity:identity))


def test_catalog_conflict_preserves_verified_sample_exposure_and_export(app,tmp_path,monkeypatch):
    original=record();original.update(path=str(tmp_path/'nexus/NOM_138504.nxs.h5'),source_signature={'size':1})
    mock_archive(monkeypatch,original)
    calls=[]
    def acquisition(*args,**kwargs):calls.append((args,kwargs));return catalog_record()
    app._battery_catalog_client=SimpleNamespace(begin_batch=lambda:None,acquisition=acquisition)
    asyncio.run(app._inspect_battery_async(inventory(tmp_path),'shared/reduced',1,app._battery_generation))
    result=app._battery_acquisitions[0]
    assert {key:result[key] for key in original}==original
    assert result['catalog_comparison']['conflicts']==['run_title','sample_identity']
    assert 'disagrees' in app.server.state.battery_rows[0]['issues']
    assert app.server.state.battery_sample_options[0]['value']=='66548'
    assert calls[0][1]['location']=='/SNS/NOM/IPTS-42/nexus/NOM_138504.nxs.h5'
    app._battery_catalog_proposal=catalog_record('proposal')
    files=[];app.download_file=lambda name,mime,raw:files.append(raw)
    app.export_battery_evidence();exported=json.loads(files[0])
    assert exported['catalog_proposal']['fields']['title']=='Proposal title'
    assert exported['acquisitions'][0]['catalog_comparison']==result['catalog_comparison']
    assert exported['scientific_qualification']=='not_evaluated'


def test_optional_catalog_exception_cannot_discard_file_metadata_or_expose_details(app,tmp_path,monkeypatch):
    original=record();mock_archive(monkeypatch,original)
    def fail(*args,**kwargs):raise RuntimeError('Private provider response and credential')
    app._battery_catalog_client=SimpleNamespace(begin_batch=lambda:None,acquisition=fail)
    asyncio.run(app._inspect_battery_async(inventory(tmp_path),'shared/reduced',1,app._battery_generation))
    result=app._battery_acquisitions[0]
    assert result['status']=='available' and result['identity']==original['identity']
    assert result['exposure']==original['exposure'] and result['catalog']['status']=='unavailable'
    assert 'file metadata remains usable' in app.server.state.battery_catalog_message
    assert 'Private' not in json.dumps(vars(app.server.state)) and not app.server.state.battery_error


def test_slow_old_catalog_proposal_cannot_publish_after_context_change(app,tmp_path,monkeypatch):
    monkeypatch.setattr(module,'discover_proposal',lambda *args,**kwargs:inventory(tmp_path))
    def stale(*args):app._battery_context_changed();return catalog_record('proposal')
    app._battery_catalog_client=SimpleNamespace(begin_batch=lambda:None,proposal=stale)
    asyncio.run(app._discover_battery_async('NOM','IPTS-42',app._battery_generation))
    assert app._battery_catalog_proposal is None and app._battery_inventory is None
    assert not app.server.state.battery_catalog_title and not app.server.state.battery_catalog_abstract
    assert not app.server.state.battery_export_ready


def test_slow_old_catalog_run_cannot_republish_after_family_change(app,tmp_path,monkeypatch):
    mock_archive(monkeypatch,record())
    def stale(*args,**kwargs):app._battery_family_changed();return catalog_record()
    app._battery_catalog_client=SimpleNamespace(begin_batch=lambda:None,acquisition=stale)
    asyncio.run(app._inspect_battery_async(inventory(tmp_path),'shared/reduced',1,app._battery_generation))
    assert app._battery_acquisitions==[] and app.server.state.battery_rows==[]
    assert app.server.state.battery_catalog_rows==[]


def test_proposal_failure_leaves_discovered_sources_exportable(app,tmp_path,monkeypatch):
    monkeypatch.setattr(module,'discover_proposal',lambda *args,**kwargs:inventory(tmp_path))
    def fail(*args):raise RuntimeError('Private provider failure')
    app._battery_catalog_client=SimpleNamespace(begin_batch=lambda:None,proposal=fail)
    asyncio.run(app._discover_battery_async('NOM','IPTS-42',app._battery_generation))
    assert app.server.state.battery_export_ready and len(app.server.state.battery_families)==1
    assert app._battery_catalog_proposal['status']=='unavailable'
    assert 'Private' not in app.server.state.battery_catalog_proposal_message


def test_family_change_clears_old_run_coverage_but_retains_proposal_context(app,tmp_path):
    app._battery_inventory=inventory(tmp_path)
    app._battery_catalog_proposal=catalog_record('proposal')
    app.server.state.battery_catalog_title='Proposal title'
    app.server.state.battery_catalog_proposal_message='Proposal abstract unavailable'
    app.server.state.battery_catalog_message='Context retrieved for 12 inspected runs'
    app.server.state.battery_catalog_rows=[dict(run=138504)]
    app._battery_family_changed()
    assert '12 inspected' not in app.server.state.battery_catalog_message
    assert app.server.state.battery_catalog_rows==[]
    assert app.server.state.battery_catalog_title=='Proposal title'
    assert app.server.state.battery_catalog_proposal_message=='Proposal abstract unavailable'


def test_inspection_announces_run_listing_before_filesystem_work_and_reports_failure(app,tmp_path,monkeypatch):
    app.server.state.battery_message='Found 8 reduction families.'
    def fail_listing(*args):
        assert app.server.state.battery_busy and app.server.state.battery_operation == 'inspect'
        assert 'Listing runs in shared/reduced' in app.server.state.battery_message
        assert 'Found 8' not in app.server.state.battery_message
        raise ValueError('Run-list inspection exceeded its time budget; no incomplete run list was used.')
    monkeypatch.setattr(module,'family_runs',fail_listing)
    asyncio.run(app._inspect_battery_async(inventory(tmp_path),'shared/reduced',12,app._battery_generation))
    assert 'no incomplete run list' in app.server.state.battery_error
    assert 'inspection stopped' in app.server.state.battery_message
    assert not app.server.state.battery_busy and app.server.state.battery_operation == ''


def test_inspection_reports_each_read_stage_and_finishes_with_a_review_prompt(app,tmp_path,monkeypatch):
    stages=[]
    monkeypatch.setattr(module,'family_runs',lambda *args:[138504])
    def inspect(*args,**kwargs):
        stages.append(app.server.state.battery_message)
        return record()
    def enrich(run,identity):
        stages.append(app.server.state.battery_message)
        return identity
    def acquisition(*args,**kwargs):
        stages.append(app.server.state.battery_message)
        assert 'ONCat acquisition context 1/1' in app.server.state.battery_catalog_message
        return catalog_record()
    monkeypatch.setattr(module,'inspect_run',inspect)
    monkeypatch.setattr(module,'ExperimentMetadataIndex',lambda *args,**kwargs:SimpleNamespace(enrich=enrich))
    app._battery_catalog_client=SimpleNamespace(begin_batch=lambda:None,acquisition=acquisition)
    asyncio.run(app._inspect_battery_async(inventory(tmp_path),'shared/reduced',12,app._battery_generation))
    assert stages[0].startswith('Reading NeXus metadata 1/1')
    assert stages[1].startswith('Checking sample tables and run-info XML 1/1')
    assert stages[2].startswith('Reading catalog context 1/1')
    assert app.server.state.battery_message.startswith('Inspected 1 of 1 runs')
    assert not app.server.state.battery_busy and app.server.state.battery_operation == ''


def test_empty_family_finishes_without_claiming_acquisitions_were_inspected(app,tmp_path,monkeypatch):
    monkeypatch.setattr(module,'family_runs',lambda *args:[])
    asyncio.run(app._inspect_battery_async(inventory(tmp_path),'shared/reduced',12,app._battery_generation))
    assert 'No runs remain' in app.server.state.battery_message
    assert 'No acquisition catalog context requested' in app.server.state.battery_catalog_message
    assert app.server.state.battery_rows == [] and not app.server.state.battery_error
    assert not app.server.state.battery_busy and app.server.state.battery_operation == ''


def test_join_reports_parsing_and_success_then_invalidates_its_status_on_edit(app,tmp_path,monkeypatch):
    file=tmp_path/'cell.csv';file.write_text('elapsed_seconds,voltage_v,current_a\n0,3,1\n10,4,1\n')
    app._battery_acquisitions=[record()]
    app._battery_upload=SimpleNamespace(current_path=file)
    original=module.read_cycler
    def read(*args,**kwargs):
        assert app.server.state.battery_operation == 'join'
        assert app.server.state.battery_message.startswith('Parsing cycler timestamps')
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'read_cycler',read)
    binding=CyclerBinding('66548','cell','C05','charge',clock_origin='2019-12-09T12:00:26-05:00')
    asyncio.run(app._join_battery_async(binding,('upload','',str(file)),app._battery_generation,app._battery_binding_revision))
    assert app.server.state.battery_message.startswith('Bound the cycler record to 1 inspected exposures')
    assert not app.server.state.battery_error and not app.server.state.battery_busy
    assert app.server.state.battery_operation == ''
    app._battery_binding_changed()
    assert app.server.state.battery_message.startswith('Electrochemical inputs changed')
    assert not app.server.state.battery_echem_summary
