"""Data-only structural library transport with complete scientific scope."""
import copy
import json
from pathlib import Path

import pytest

from gpx_recipe.execution import write_json
from gpx_recipe.preparation import hints,structural_hints
from gpx_recipe.preparation.operations import create_library,stage_entry,publish_entry
from gpx_recipe.structural_live.optimization import offline_native_start_bank as bank_module
from gpx_recipe.symmetry.chunks import execute_chunk
from gpx_recipe.symmetry.compiler import file_sha,package_signature
from gpx_recipe.symmetry.contracts import digest,iter_trials,trial_id,validate_protocol
from test_structural_live_offline_start_bank import source as artificial_source
from test_symmetry_contracts import make_protocol


def fixture(root,monkeypatch):
    root.mkdir()
    protocol=make_protocol(root/'inputs')
    child=root/'inputs/child.cif';child.write_text('data_artificial_child\n')
    candidate=protocol['candidates'][0]
    candidate.update(id='struct:fixture',magnetic_space_group=None,
        child=dict(path='child.cif',sha256=file_sha(child)),modes=['mode-one','mode-two'])
    protocol.update(kind='structural',sites=[],implementation_sha256=package_signature())
    protocol['policy'].update(structural_optimizer='prepared_nonlinear',refine_cell=True,
        refine_histogram_scale=True,refine_background=True)
    catalog=root/'inputs/catalog.json'
    record=json.loads(catalog.read_text());record['candidates']=[candidate]
    write_json(catalog,record);protocol['catalog']['sha256']=file_sha(catalog)
    protocol=validate_protocol(protocol)
    settings=dict(candidate=protocol['candidates'][0],mode_radius=.05,cell_radius=.005)
    parent=root/'parent.cif';parent.write_text('data_artificial_parent\n')
    profile=root/'bank.instprm';profile.write_text('# artificial profile\n')
    grid=dict(kind='PNT',lower=3000.,upper=40000.,bins=701,bank=1)
    library=create_library(root/'library',name='Artificial structural preparation',source=parent,
        banks={'test bank':dict(instrument=profile,grid=grid)},
        models=[dict(kind='structural',bank='test bank',child=child,settings=settings)])
    entry=library.document['entries'][0]['id'];attempt=library.start(entry)
    request=stage_entry(library,entry,attempt,root/'attempt')
    output=Path(request['destination']);output.mkdir()
    source,_,_,_=artificial_source(monkeypatch)
    previous=bank_module.context
    monkeypatch.setattr(bank_module,'context',lambda s:dict(previous(s),physics_sha256='e'*64))
    bank_module.build(source,output/'starts.npz',policy='joint_cell_v1')
    artifacts={name:file_sha(output/name) for name in ('starts.npz','starts.json')}
    manifest=dict(schema='radar-experiment-preparation-entry/v1',kind='structural',grid=grid,
        settings=request['settings'],input_sha256=[file_sha(request[k]) for k in ('source','instrument','child')],
        artifacts=artifacts,detail=dict(physics_sha256='e'*64),observations_included=False,scientific_result=False)
    write_json(output/'entry.json',manifest)
    publish_entry(library,entry,attempt,request,dict(schema='radar-experiment-preparation-worker/v1',
        status='prepared',kind='structural',output=str(output),artifacts=[*artifacts,'entry.json'],
        entry_sha256=file_sha(output/'entry.json'),observations_included=False,scientific_result=False))
    return library,protocol,root/'inputs'


def test_generic_capture_restore_and_candidate_subset_keep_structural_scope(tmp_path,monkeypatch):
    library,protocol,_=fixture(tmp_path/'fixture',monkeypatch)
    original=copy.deepcopy(protocol)
    target=tmp_path/'packet';document=hints.capture(library,protocol,target)
    assert document['schema']==structural_hints.SCHEMA
    assert document['eligible_entries']==len(document['entries'])==1
    assert len(hints.packet_paths(target,protocol))==3
    assert not list(target.rglob('*.gpx'))
    assert not any(p.suffix=='.py' for p in target.rglob('*'))
    received=tmp_path/'received';report=hints.restore(document,protocol,target,received)
    assert report['kind']=='structural' and report['attached']==1
    assert report['native_qualification_required']
    rows,proof=structural_hints.candidates(received,protocol,protocol['candidates'][0]['id'])
    assert len(rows)==proof['attached']==1
    assert hints.cache_for(received,protocol)[0] is None
    empty=hints.subset(document,protocol,[])
    assert empty['entries']==[] and empty['omitted_entries']==empty['eligible_entries']==1
    assert protocol==original and hints.validate(empty,protocol)==empty


def test_structural_origin_routing_survives_capture_restore_without_claiming_authority(tmp_path,monkeypatch):
    library,protocol,_=fixture(tmp_path/'fixture',monkeypatch)
    entry=next(a['sha256'] for a in library.document['entries'][0]['files'] if a['path'].endswith('/entry.json'))
    reference=dict(site='https://galaxy.invalid',dataset_id='original-certificate',sha256='a'*64,entries=[entry,'b'*64])
    library.add_evidence(reference)
    target=tmp_path/'packet';document=hints.capture(library,protocol,target)
    assert document['evidence']==[dict(reference,entries=[entry])]
    received=tmp_path/'received';proof=hints.restore(document,protocol,target,received)
    assert proof['native_qualification_required']
    assert hints.read(received,protocol)['evidence']==document['evidence']
    assert hints.subset(document,protocol,[])['evidence']==[]


@pytest.mark.parametrize('damage',['missing','checksum','receipt','well_hashed_bad_zip','receipt_list','context_list'])
def test_damaged_structural_assets_rebuild_without_removing_a_scientific_trial(tmp_path,monkeypatch,damage):
    library,protocol,inputs=fixture(tmp_path/'fixture',monkeypatch)
    document=hints.capture(library,protocol,inputs);row=document['entries'][0]
    if damage=='missing':(inputs/row['path']).unlink()
    elif damage=='checksum':(inputs/row['path']).write_bytes(b'altered')
    elif damage=='receipt':(inputs/row['receipt_path']).write_text('{}')
    else:
        receipt=json.loads((inputs/row['receipt_path']).read_text())
        if damage=='well_hashed_bad_zip':
            bad=inputs/'damaged.npz';bad.write_bytes(b'not an NPZ archive')
            row['sha256']=file_sha(bad);row['path']='_prepared_structural/'+row['sha256']+'.npz'
            bad.rename(inputs/row['path']);receipt['sha256']=row['sha256']
        elif damage=='receipt_list':receipt=[]
        else:receipt['context']=[]
        replacement=inputs/'damaged.json';write_json(replacement,receipt)
        row['receipt_sha256']=file_sha(replacement)
        row['receipt_path']='_prepared_structural/'+row['receipt_sha256']+'.json'
        replacement.rename(inputs/row['receipt_path'])
        document['snapshot_sha256']=digest({k:v for k,v in document.items() if k!='snapshot_sha256'})
    uid=trial_id(protocol,*[(c['id'],s['id']) for c,s in iter_trials(protocol)][0])
    seen=[]
    def runner(directory,**_):
        seen.append(directory.name)
        captured=directory.parent.parent/'inputs'/digest(protocol)
        rows,proof=structural_hints.candidates(captured,protocol,protocol['candidates'][0]['id'])
        assert not rows and proof['attached']==0
        return dict(state='finished',status='optimization_unresolved',wall_seconds=.1)
    result=execute_chunk(dict(schema='radar-model-chunk/v1',protocol=protocol,protocol_sha256=digest(protocol),
        trials=[uid],workers=1,dispatch_key='a'*24,seconds=240,prepared_scattering=document),
        inputs,tmp_path/'job',runner=runner,environ={})
    assert seen==[uid] and len(result['trials'])==1 and not result['pending']
    assert result['prepared_scattering']['attached']==0
    assert result['prepared_scattering']['discarded_entries']==[row['entry']]


@pytest.mark.parametrize('change',['path','candidate','receipt','duplicate','executable','snapshot'])
def test_structural_metadata_is_closed_and_bound_before_native_work(tmp_path,monkeypatch,change):
    library,protocol,_=fixture(tmp_path/'fixture',monkeypatch)
    document=hints.capture(library,protocol,tmp_path/'packet')
    if change=='path':document['entries'][0]['path']='../outside.npz'
    elif change=='candidate':document['entries'][0]['candidate']='struct:foreign'
    elif change=='receipt':document['entries'][0]['receipt_path']='_prepared_structural/arbitrary.json'
    elif change=='duplicate':document['entries']*=2;document['eligible_entries']=2
    elif change=='executable':document['entries'][0]['python']='raise RuntimeError'
    else:document['snapshot_sha256']='f'*64
    if change!='snapshot':document['snapshot_sha256']=digest({k:v for k,v in document.items() if k!='snapshot_sha256'})
    with pytest.raises(ValueError):hints.restore(document,protocol,tmp_path/'packet',tmp_path/'received')
    assert not (tmp_path/'received').exists()


def test_structural_transfer_budget_changes_preparation_coverage_only(tmp_path,monkeypatch):
    library,protocol,_=fixture(tmp_path/'fixture',monkeypatch)
    monkeypatch.setattr(structural_hints,'MAX_BYTES',1)
    document=hints.capture(library,protocol,tmp_path/'packet')
    assert document['entries']==[] and document['eligible_entries']==document['omitted_entries']==1
    assert len(list(iter_trials(protocol)))==1
    assert len(protocol['candidates'][0]['modes'])==2


def test_local_structural_dispatch_captures_preparation_before_launch_intent(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from gpx_recipe.symmetry.campaign import Campaign
    from gpx_recipe.symmetry import local_transport
    library,protocol,inputs=fixture(tmp_path/'fixture',monkeypatch)
    campaign=Campaign(tmp_path/'campaign',shared_inputs=True)
    uid=campaign.enroll(protocol,inputs,approved_hash=digest(protocol))[0]
    captured=[]
    def capture(p,root):
        assert not list(campaign.root.glob('trials/*/launch.json'))
        captured.append(root)
        return hints.capture(library,p,root)
    def launch(*args,**kwargs):
        assert len(captured)==1 and hints.read(captured[0],protocol)['entries']
        return SimpleNamespace(pid=12345)
    monkeypatch.setattr(local_transport.subprocess,'Popen',launch)
    local_transport.LocalTransport(['python'],preparation_provider=capture).submit(uid,campaign.root/'trials'/uid)
    assert (campaign.root/'trials'/uid/'launch.json').exists()


@pytest.mark.parametrize('shape',['entry_list','detail_list','receipt_list','context_list'])
def test_rehashed_portable_library_metadata_still_requires_scientific_types(tmp_path,monkeypatch,shape):
    import hashlib
    import zipfile
    from gpx_recipe.preparation.library import import_library
    library,protocol,_=fixture(tmp_path/'fixture',monkeypatch)
    library.export(tmp_path/'original.zip')
    with zipfile.ZipFile(tmp_path/'original.zip') as packet:
        data={name:packet.read(name) for name in packet.namelist()}
    document=json.loads(data['library.json']);row=document['entries'][0]
    assets={Path(a['path']).name:a['path'] for a in row['files']}
    entry=json.loads(data[assets['entry.json']])
    if shape=='entry_list':entry=[]
    elif shape=='detail_list':entry['detail']=[]
    else:
        receipt=json.loads(data[assets['starts.json']])
        if shape=='receipt_list':receipt=[]
        else:receipt['context']=[]
        data[assets['starts.json']]=json.dumps(receipt).encode()
        entry['artifacts']['starts.json']=hashlib.sha256(data[assets['starts.json']]).hexdigest()
    data[assets['entry.json']]=json.dumps(entry).encode()
    for artifact in row['files']:
        artifact['sha256']=hashlib.sha256(data[artifact['path']]).hexdigest()
    data['library.json']=json.dumps(document).encode()
    with zipfile.ZipFile(tmp_path/'uploaded.zip','w',zipfile.ZIP_DEFLATED) as packet:
        for name,body in data.items():packet.writestr(name,body)
    uploaded=import_library(tmp_path/'uploaded.zip',tmp_path/'uploaded-library')
    # Checksums establish inventory integrity, not the types or scientific
    # validity of embedded receipts. The optional-provider caller handles a
    # ValueError by rebuilding physics; AttributeError must not escape here.
    with pytest.raises(ValueError):hints.capture(uploaded,protocol,tmp_path/'model-inputs')
    assert len(list(iter_trials(protocol)))==1
