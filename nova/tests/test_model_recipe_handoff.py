"""A structural result is an explicit recipe input, never a new live scan."""
import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpx_recipe.symmetry.contracts import digest
from radar_pd_nova import model_recipe_handoff as handoff
from radar_pd_nova.refinement_sources import RefinementScope, file_sha
from radar_pd_nova.refinement_workspace import RefinementWorkspace
from radar_pd_nova.symmetry_controller import SymmetryMixin


@pytest.fixture
def selected(tmp_path, monkeypatch):
    original=tmp_path/'original.gpx';original.write_bytes(b'original nuclear input')
    directory=tmp_path/'trial';(directory/'results').mkdir(parents=True)
    native=directory/'results/candidate.gpx';native.write_bytes(b'native structural checkpoint')
    receipt=dict(id='trial',protocol_sha256='b'*64,status='optimization_unresolved',
                 gpx='candidate.gpx',files={'results/candidate.gpx':file_sha(native)})
    row=dict(id='trial',protocol='b'*64,receipt_sha256=digest(receipt),directory=str(directory),
             gpx_relative='results/candidate.gpx',source='parent',source_sha256=file_sha(original),
             status=receipt['status'],kind='structural',name='Constructed subgroup',
             worker_revision='c'*64,native_verified=True,has_gpx=True,has_view=True,
             reason='Saved point verified, optimization not qualified')
    calls=[]
    def verified(root, expected):
        calls.append((root,expected))
        assert expected==dict(id='trial',protocol_sha256='b'*64)
        return copy.deepcopy(receipt)
    monkeypatch.setattr(handoff,'verify_receipt',verified)
    workspace=RefinementWorkspace(tmp_path/'workspace',history_id='history',
                                  runtime=SimpleNamespace(kind='local'))
    source=dict(id='parent',experiment='experiment',experiment_label='Experiment',
        scan_key='scan-123',scan_label='Run 123 bank 2',sample_key='sample',sample_label='Sample',
        run_number=123,bank='2',source_path=str(original),source_sha256=file_sha(original),
        record_uid='original-record',collection_id='original-collection',dataset_id='original-hda',
        live_watch=True,context={'old':'context'},inspection='old-inspection',
        project={'stage':'original'},scientific_warnings=['Profile calibration needs review'],
        metadata={'temperature':{'value':12.,'unit':'K'},'preparation':{'source_sha256':file_sha(original)}})
    workspace.observe([source],enroll_future=False)
    workspace.state['recipes']['active']=dict(id='active',experiment='experiment',armed=True,
        scope=RefinementScope('experiment').to_dict(),approved_utc='2000-01-01',plan={'test':'queue only'})
    return workspace,row,native,receipt,calls


def test_capture_uses_receipt_identity_and_preserves_exact_bytes(selected,tmp_path):
    workspace,row,native,_,calls=selected
    path,checksum=handoff.capture_gpx(row,tmp_path/'captured')
    assert path.read_bytes()==native.read_bytes() and checksum==file_sha(native)
    assert len(calls)==1 and path.name==row['receipt_sha256']+'.gpx'


@pytest.mark.parametrize('change',['receipt','different_gpx','no_gpx','no_view'])
def test_capture_rejects_stale_or_substituted_artifacts(selected,tmp_path,change):
    _,row,native,receipt,_=selected
    if change=='receipt':row['receipt_sha256']='0'*64
    elif change=='different_gpx':
        (native.parent/'other.gpx').write_bytes(b'other native artifact')
        receipt['files']['results/other.gpx']=file_sha(native.parent/'other.gpx')
        row['receipt_sha256']=digest(receipt);row['gpx_relative']='results/other.gpx'
    else:row['has_gpx' if change=='no_gpx' else 'has_view']=False
    with pytest.raises(ValueError):handoff.capture_gpx(row,tmp_path/'captured')
    assert not list((tmp_path/'captured').glob('*.gpx'))


def test_structural_source_keeps_acquisition_and_drops_parent_artifact_references(selected,tmp_path):
    workspace,row,_,_,_=selected
    before=copy.deepcopy(workspace.state)
    path,checksum=handoff.capture_gpx(row,tmp_path/'captured')
    source=handoff.structural_source(workspace,'experiment',row,path,checksum)
    assert source['id'].startswith('structural-model:') and source['id']!='parent'
    assert source['source_sha256']==checksum and source['source_path']==str(path)
    assert (source['run_number'],source['bank'],source['sample_key'])==(123,'2','sample')
    assert source['metadata']['temperature']=={'value':12.,'unit':'K'}
    assert 'preparation' not in source['metadata'] and 'parent_preparation' in source['metadata']
    assert source['model_checkpoint']['input_sha256']==row['source_sha256']
    assert source['model_checkpoint']['trial_id']=='trial'
    assert source['scientific_warnings'][0]=='Profile calibration needs review'
    assert any('optimization not qualified' in s for s in source['scientific_warnings'])
    for key in ['record_uid','dataset_id','collection_id','context','inspection','live_watch','model_parent_id']:
        assert key not in source
    assert workspace.state==before
    workspace.observe([source],manual=True,enroll_future=False)
    from radar_pd_nova.symmetry_sources import start_choices
    label=next(e['title'] for e in start_choices(workspace,'experiment') if e['id']==source['id'])
    assert label.startswith('Structural model · ')


@pytest.mark.parametrize('change',['magnetic','equivalent','not_native','parent_hash','source_id','experiment','captured_bytes'])
def test_unsupported_or_mismatched_source_cannot_become_recipe_input(selected,tmp_path,change):
    workspace,row,_,_,_=selected
    path,checksum=handoff.capture_gpx(row,tmp_path/'captured')
    experiment='experiment'
    if change=='magnetic':row['kind']='magnetic'
    elif change=='equivalent':row['status']='profile_equivalent'
    elif change=='not_native':row['native_verified']=False
    elif change=='parent_hash':row['source_sha256']='a'*64
    elif change=='source_id':row['source']='another-source'
    elif change=='experiment':experiment='different-experiment'
    else:path.write_bytes(b'changed after capture')
    with pytest.raises(ValueError):handoff.structural_source(workspace,experiment,row,path,checksum)


class State(SimpleNamespace):
    def flush(self):pass


def controller(workspace,row,tmp_path):
    app=SymmetryMixin()
    app.server=SimpleNamespace(state=State(sx_recipe_busy=False,sx_results_current=True,
        sx_campaign='campaign',sx_source='parent',sx_selected='trial',sx_result_scope='experiment',
        rf_experiment='experiment',sx_error=''))
    app._symmetry_rows=[row];app._refinement_internal=False
    app._symmetry_root=lambda:tmp_path/'model-root'
    app._refinement_store=lambda:workspace
    app._refinement_sync=lambda:None
    app.inspections=[];app.inspect_refinement_project=lambda:app.inspections.append(True)
    app.tasks=[];app._schedule_utility=lambda coroutine,label:app.tasks.append(coroutine)
    return app


def test_controller_opens_one_manual_input_without_enrolling_the_active_recipe(selected,tmp_path):
    workspace,row,_,_,_=selected;app=controller(workspace,row,tmp_path)
    original=copy.deepcopy(workspace.state['sources']['parent'])
    app.open_symmetry_refinement();assert app.server.state.sx_recipe_busy
    asyncio.run(app.tasks.pop())
    state=app.server.state;source=workspace.state['sources'][state.rf_source]
    assert state.rf_mode=='recipe' and state.rf_setup_mode=='expert' and not state.rf_future
    assert state.rf_scans==['source:'+source['id']] and source['manual_recipe_input']
    assert 'mode amplitudes' in state.rf_request and len(app.inspections)==1
    assert not workspace.state['jobs'] and workspace.state['recipes']['active']['armed']
    assert workspace.state['sources']['parent']==original and not state.sx_recipe_busy
    assert not state.sx_error


@pytest.mark.parametrize('change',['selection','parent_revision','upload_failure','workspace'])
def test_remote_handoff_does_not_select_stale_or_unpublished_results(selected,tmp_path,change):
    workspace,row,_,_,_=selected;app=controller(workspace,row,tmp_path)
    workspace.runtime.kind='galaxy';workspace.checkpoint=lambda:None
    uploads=[]
    def publish(path,**kwargs):
        uploads.append(Path(path).read_bytes())
        if change=='selection':app.server.state.sx_selected='different-trial'
        elif change=='parent_revision':workspace.state['sources']['parent']['source_sha256']='0'*64
        elif change=='workspace':app._refinement_store=lambda:None
        else:raise OSError('Upload interrupted')
        return 'model-hda'
    app.service=SimpleNamespace(upload_document=publish)
    app.open_symmetry_refinement();asyncio.run(app.tasks.pop())
    assert len(uploads)==1 and app.server.state.sx_error and not app.inspections
    assert len(workspace.state['sources'])==1 and not workspace.state['jobs']
    assert not app.server.state.sx_recipe_busy


def test_remote_handoff_reuses_its_published_gpx_and_checkpoints_without_submitting_a_fit(selected,tmp_path):
    workspace,row,_,_,_=selected;app=controller(workspace,row,tmp_path)
    workspace.runtime.kind='galaxy';backups=[];workspace.checkpoint=lambda:backups.append(True)
    uploads=[];app.service=SimpleNamespace(upload_document=lambda path,**kwargs:uploads.append(path) or 'model-hda')
    for _ in range(2):
        app.open_symmetry_refinement();asyncio.run(app.tasks.pop())
    source=workspace.state['sources'][app.server.state.rf_source]
    assert source['dataset_id']=='model-hda' and len(uploads)==1 and len(backups)==2
    assert len(workspace.state['sources'])==2 and not workspace.state['jobs']


def test_failed_checkpoint_can_retry_without_uploading_or_inspecting_again(selected,tmp_path):
    workspace,row,_,_,_=selected;app=controller(workspace,row,tmp_path)
    workspace.runtime.kind='galaxy';uploads=[];backups=[]
    app.service=SimpleNamespace(upload_document=lambda path,**kwargs:uploads.append(path) or 'model-hda')
    def backup():
        backups.append(True)
        if len(backups)==1:raise OSError('History checkpoint interrupted')
    workspace.checkpoint=backup
    app.open_symmetry_refinement();asyncio.run(app.tasks.pop())
    assert 'interrupted' in app.server.state.sx_error and not app.inspections
    app.open_symmetry_refinement();asyncio.run(app.tasks.pop())
    assert not app.server.state.sx_error and len(app.inspections)==1
    assert len(uploads)==1 and len(backups)==2 and not workspace.state['jobs']


def test_double_click_does_not_start_two_handoffs(selected,tmp_path):
    workspace,row,_,_,_=selected;app=controller(workspace,row,tmp_path)
    app.open_symmetry_refinement();app.open_symmetry_refinement()
    assert len(app.tasks)==1
    asyncio.run(app.tasks.pop())
    assert len(app.inspections)==1


def test_display_reset_cannot_duplicate_an_inflight_capture(selected,tmp_path):
    workspace,row,_,_,_=selected;app=controller(workspace,row,tmp_path)
    app.open_symmetry_refinement()
    app.server.state.sx_recipe_busy=False  # e.g. a navigation-triggered display reset
    app.open_symmetry_refinement()
    assert len(app.tasks)==1
    asyncio.run(app.tasks.pop())
    assert len(app.inspections)==1 and not app._symmetry_recipe_inflight
