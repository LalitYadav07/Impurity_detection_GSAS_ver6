"""Portable asset selection and dispatch contracts; artificial arrays only."""
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from gpx_recipe.preparation import hints
from gpx_recipe.preparation.operations import create_library,stage_entry,publish_entry
from gpx_recipe.live.reflection_factors import ScatteringFactorCache
from gpx_recipe.execution import write_json
from gpx_recipe.symmetry.compiler import file_sha,package_signature
from gpx_recipe.symmetry.contracts import digest,iter_trials,trial_id
from gpx_recipe.symmetry.chunks import execute_chunk
from test_symmetry_contracts import make_protocol


def fixture(root,*,with_factors=True):
    root.mkdir()
    protocol=make_protocol(root/'source');protocol['implementation_sha256']=package_signature()
    source=root/'parent.cif';source.write_text('data_artificial_theory_transport\n')
    profile=root/'bank.instprm';profile.write_text('# artificial profile metadata\n')
    grid=dict(kind='PNT',lower=3000.,upper=40000.,bins=701,bank=1)
    settings=dict(candidate=protocol['candidates'][0],k=protocol['k'],sites=protocol['sites'],maximum_moment=6.,
        refine_histogram_scale=True,refine_background=True)
    library=create_library(root/'library',name='Artificial preparation',source=source,
        banks={'bank 1':dict(instrument=profile,grid=grid)},models=[dict(kind='magnetic',bank='bank 1',settings=settings)])
    entry=library.document['entries'][0]['id'];attempt=library.start(entry)
    request=stage_entry(library,entry,attempt,root/'attempt')
    output=Path(request['destination']);output.mkdir()
    np.savez(output/'response.npz',placeholder=np.arange(100.))
    artifacts={'response.npz':file_sha(output/'response.npz')};detail={}
    if with_factors:
        factors=np.arange(72.,dtype=float).reshape(4,6,3)
        # Unit transport fixture only. A new cache still treats this claim as
        # unqualified; the native gate independently tests actual amplitudes.
        owned=ScatteringFactorCache(root/'unit-factors')
        owned._admit_native('e'*64,factors,dict(all_coefficient_directions_checked=True,dimensions=3,reflections=4))
        shutil.copyfile(root/'unit-factors'/('e'*64+'.npz'),output/'scattering.npz')
        artifacts['scattering.npz']=file_sha(output/'scattering.npz')
        detail['scattering']=dict(status='prepared',identity='e'*64,artifact='scattering.npz')
    manifest=dict(schema='radar-experiment-preparation-entry/v1',kind='magnetic',settings=request['settings'],grid=grid,
        input_sha256=[file_sha(request[k]) for k in ('source','instrument')],artifacts=artifacts,detail=detail,
        observations_included=False,scientific_result=False)
    write_json(output/'entry.json',manifest)
    result=dict(schema='radar-experiment-preparation-worker/v1',status='prepared',kind='magnetic',output=str(output),
        artifacts=list(artifacts)+['entry.json'],entry_sha256=file_sha(output/'entry.json'),
        observations_included=False,scientific_result=False)
    publish_entry(library,entry,attempt,request,result)
    return library,protocol,root/'source'


def test_capture_transfers_only_needed_theory_assets_and_import_never_inherits_qualification(tmp_path,monkeypatch):
    library,protocol,_=fixture(tmp_path/'fixture')
    original=library._verify_artifacts;seen=[]
    def verify(assets):
        seen.extend(a['path'] for a in assets)
        return original(assets)
    monkeypatch.setattr(library,'_verify_artifacts',verify)
    target=tmp_path/'packet';document=hints.capture(library,protocol,target)
    assert document['eligible_entries']==len(document['entries'])==1 and document['omitted_entries']==0
    assert not any(name.endswith('response.npz') for name in seen)
    assert all('response.npz' not in name for name in hints.packet_paths(target,protocol))
    assert all(p.suffix in {'.json','.npz'} for p in target.rglob('*') if p.is_file())
    assert not any(p.suffix=='.gpx' for p in target.rglob('*'))
    received=tmp_path/'received';report=hints.restore(document,protocol,target,received)
    assert report['attached']==1 and not report['discarded_entries'] and report['native_qualification_required']
    cache,receipt=hints.cache_for(received,protocol,candidate=protocol['candidates'][0]['id'])
    assert receipt['attached']==1 and receipt['native_qualification_required']
    block,qualified=cache.lookup('e'*64)
    assert not qualified
    np.testing.assert_array_equal(block.factors,np.arange(72.).reshape(4,6,3))


def test_existing_snapshot_is_not_replaced_by_another_library_revision(tmp_path):
    library,protocol,_=fixture(tmp_path/'first');target=tmp_path/'packet'
    first=hints.capture(library,protocol,target)
    other,_,_=fixture(tmp_path/'other')
    assert hints.capture(other,protocol,target)==first
    different=copy.deepcopy(first);different['library_name']='Another revision'
    different['snapshot_sha256']=digest({k:v for k,v in different.items() if k!='snapshot_sha256'})
    with pytest.raises(ValueError,match='another preparation snapshot'):
        hints.restore(different,protocol,target,target)


@pytest.mark.parametrize('damage',['checksum','missing','oversized'])
def test_damaged_optional_assets_do_not_remove_scientific_trials(tmp_path,monkeypatch,damage):
    library,protocol,source=fixture(tmp_path/'fixture');document=hints.capture(library,protocol,source)
    path=source/document['entries'][0]['path']
    if damage=='checksum':path.write_bytes(b'changed')
    elif damage=='missing':path.unlink()
    else:monkeypatch.setattr(hints,'MAX_BYTES',5)
    uid=trial_id(protocol,*[(c['id'],s['id']) for c,s in iter_trials(protocol)][0]);seen=[]
    def runner(directory,**_):
        seen.append(directory.name)
        cache,report=hints.cache_for(directory.parent.parent/'inputs'/digest(protocol),protocol,candidate='mag:fixture')
        assert cache is None
        return dict(state='finished',status='inadequate_fit',wall_seconds=.1)
    request=dict(schema='radar-model-chunk/v1',protocol=protocol,protocol_sha256=digest(protocol),
        trials=[uid],workers=1,dispatch_key='a'*24,seconds=240,prepared_scattering=document)
    result=execute_chunk(request,source,tmp_path/'job',runner=runner,environ={})
    assert seen==[uid] and len(result['trials'])==1 and not result['pending']
    assert result['prepared_scattering']['attached']==0
    assert result['prepared_scattering']['discarded_entries']==[document['entries'][0]['entry']]


@pytest.mark.parametrize('change',['protocol','path','candidate','duplicate','execution','snapshot'])
def test_foreign_or_unsafe_hint_metadata_is_rejected_before_native_work(tmp_path,change):
    library,protocol,_=fixture(tmp_path/'fixture');packet=tmp_path/'packet'
    document=hints.capture(library,protocol,packet)
    if change=='protocol':document['protocol_sha256']='a'*64
    elif change=='path':document['entries'][0]['path']='../source.gpx'
    elif change=='candidate':document['entries'][0]['candidate']=[]
    elif change=='duplicate':document['entries']*=2;document['eligible_entries']=2
    elif change=='execution':document['entries'][0]['script']='print(1)'
    else:document['snapshot_sha256']='f'*64
    if change!='snapshot':document['snapshot_sha256']=digest({k:v for k,v in document.items() if k!='snapshot_sha256'})
    with pytest.raises(ValueError):hints.restore(document,protocol,packet,tmp_path/'target')
    assert not (tmp_path/'target').exists()


def test_legacy_partial_and_budget_limited_preparation_keep_the_full_search(tmp_path,monkeypatch):
    legacy,protocol,_=fixture(tmp_path/'legacy',with_factors=False)
    document=hints.capture(legacy,protocol,tmp_path/'legacy-packet')
    assert document['eligible_entries']==document['omitted_entries']==1 and not document['entries']
    assert hints.cache_for(tmp_path/'legacy-packet',protocol)[0] is None
    library,protocol,_=fixture(tmp_path/'factors')
    monkeypatch.setattr(hints,'MAX_BYTES',1)
    document=hints.capture(library,protocol,tmp_path/'bounded')
    assert not document['entries'] and document['omitted_entries']==1
    assert len(protocol['candidates'])==len(protocol['seeds'])==1


def test_only_the_declared_dependency_closure_can_supply_scattering(tmp_path):
    library,protocol,_=fixture(tmp_path/'fixture')
    (library.root/library.document['specification']['source']).write_text('changed parent')
    with pytest.raises(ValueError,match='changed'):
        hints.capture(library,protocol,tmp_path/'packet')


def test_manifest_and_factor_symlinks_are_not_followed(tmp_path):
    library,protocol,_=fixture(tmp_path/'fixture');root=tmp_path/'packet'
    document=hints.capture(library,protocol,root);path=root/document['entries'][0]['path']
    copy=path.with_name('copy.npz');path.rename(copy);path.symlink_to(copy)
    assert hints.cache_for(root,protocol)[0] is None
    manifest=root/hints.MANIFEST;other=manifest.with_name('copy.json');manifest.rename(other);manifest.symlink_to(other)
    with pytest.raises(ValueError):hints.read(root,protocol)


def test_corrupt_optional_manifest_cannot_reject_a_scientific_candidate(tmp_path):
    library,protocol,_=fixture(tmp_path/'fixture');root=tmp_path/'packet'
    hints.capture(library,protocol,root)
    (root/hints.MANIFEST).write_text('damaged optional metadata')
    cache,report=hints.cache_for(root,protocol,candidate=protocol['candidates'][0]['id'])
    assert cache is None and report['status']=='unavailable'
    assert report['native_qualification_required']
    assert len(protocol['candidates'])==1


def test_candidate_hint_subset_preserves_the_complete_scientific_protocol(tmp_path):
    library,protocol,_=fixture(tmp_path/'fixture');root=tmp_path/'packet'
    document=hints.capture(library,protocol,root);before=copy.deepcopy(protocol)
    selected=hints.subset(document,protocol,[protocol['candidates'][0]['id']])
    assert selected==document and selected is not document
    empty=hints.subset(document,protocol,[])
    assert empty['entries']==[] and empty['omitted_entries']==empty['eligible_entries']==1
    assert empty['protocol_sha256']==document['protocol_sha256']
    assert hints.validate(empty,protocol)==empty and protocol==before
    with pytest.raises(ValueError,match='undeclared'):hints.subset(document,protocol,['foreign'])


def test_local_launch_captures_one_immutable_selection_before_each_scientific_ticket(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import gpx_recipe.symmetry.local_transport as local
    from gpx_recipe.symmetry.campaign import Campaign
    library,protocol,source=fixture(tmp_path/'fixture')
    protocol['policy'].update(magnetic_optimizer='profiled_quadratic',moment_model='symmetry_allowed',optimizer_iterations=120)
    protocol['seeds'].append(dict(protocol['seeds'][0],id='second'))
    campaign=Campaign(tmp_path/'campaign',shared_inputs=True)
    ids=campaign.enroll(protocol,source,approved_hash=digest(protocol));seen=[]
    def capture(p,root):
        assert not any(campaign.root.glob('trials/*/launch.json'))
        seen.append(1);return hints.capture(library,p,root)
    def launched(*args,**kwargs):
        assert hints.read(campaign.root/'inputs'/digest(protocol),protocol)['entries']
        assert b'preparation_provider' not in next(campaign.root.glob('trials/*/launch.json')).read_bytes()
        return SimpleNamespace(pid=12345)
    monkeypatch.setattr(local.subprocess,'Popen',launched)
    first=local.LocalTransport(['python'],preparation_provider=capture)
    first.submit(ids[0],campaign.root/'trials'/ids[0])
    second=local.LocalTransport(['python'],preparation_provider=lambda *_:pytest.fail('Selection changed for captured inputs'))
    second.submit(ids[1],campaign.root/'trials'/ids[1])
    assert seen==[1]


def test_optional_provider_failure_still_publishes_a_local_scientific_launch(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import gpx_recipe.symmetry.local_transport as local
    from gpx_recipe.symmetry.campaign import Campaign
    _,protocol,source=fixture(tmp_path/'fixture')
    protocol['policy'].update(magnetic_optimizer='profiled_quadratic',moment_model='symmetry_allowed',optimizer_iterations=120)
    campaign=Campaign(tmp_path/'campaign',shared_inputs=True)
    uid=campaign.enroll(protocol,source,approved_hash=digest(protocol))[0]
    def damaged(*args):raise ValueError('optional data changed')
    monkeypatch.setattr(local.subprocess,'Popen',lambda *a,**k:SimpleNamespace(pid=12345))
    local.LocalTransport(['python'],preparation_provider=damaged).submit(uid,campaign.root/'trials'/uid)
    receipt=json.loads((campaign.root/'trials'/uid/'preparation-attachment.json').read_text())
    assert receipt['status']=='unavailable'
    assert (campaign.root/'trials'/uid/'launch.json').exists()
