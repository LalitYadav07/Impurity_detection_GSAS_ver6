"""Selected theory libraries follow immutable Galaxy model input packets."""
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from gpx_recipe.execution import write_json
from gpx_recipe.live.reflection_factors import ScatteringFactorCache
from gpx_recipe.preparation import hints
from gpx_recipe.preparation.operations import publish_entry,create_library
from gpx_recipe.symmetry.compiler import file_sha,package_signature
from gpx_recipe.symmetry.contracts import validate_protocol,digest
from gpx_recipe.symmetry.campaign import Campaign
from radar_pd_nova.model_runtime import GalaxyChunkTransport
from radar_pd_nova.preparation_controller import PreparationMixin
from radar_pd_nova.preparation_registry import PreparationRegistry
from test_model_runtime import remote
from test_preparation_registry import fixture,returned


def library(root):
    result,identifier,attempt,request=fixture(root)
    receipt=returned(request);output=Path(receipt['output'])
    cache=ScatteringFactorCache(root/'transport-fixture')
    cache._admit_native('e'*64,np.ones((4,6,3)),
        dict(all_coefficient_directions_checked=True,dimensions=3,reflections=4))
    shutil.copyfile(root/'transport-fixture'/('e'*64+'.npz'),output/'scattering.npz')
    manifest=json.loads((output/'entry.json').read_text())
    manifest['artifacts']['scattering.npz']=file_sha(output/'scattering.npz')
    manifest['detail']=dict(scattering=dict(status='prepared',identity='e'*64,artifact='scattering.npz'))
    write_json(output/'entry.json',manifest)
    receipt.update(artifacts=['entry.json','response.npz','scattering.npz'],entry_sha256=file_sha(output/'entry.json'))
    publish_entry(result,identifier,attempt,request,receipt)
    return result


def campaign(tmp_path,source,prepared):
    inputs=tmp_path/'scientific-inputs';inputs.mkdir()
    shutil.copyfile(source['source_path'],inputs/'source.gpx')
    model=prepared.document['entries'][0]['model'];candidate=model['settings']['candidate']
    catalog=dict(schema='radar-symmetry-catalog/v1',parent_setting_sha256='a'*64,
        k=['0','0','0'],candidates=[candidate])
    write_json(inputs/'catalog.json',catalog)
    protocol=validate_protocol(dict(kind='magnetic',source=dict(path='source.gpx',sha256=file_sha(inputs/'source.gpx')),
        catalog=dict(path='catalog.json',sha256=file_sha(inputs/'catalog.json')),parent_phase='Parent',histogram='PWDR Bank',
        parent_setting_sha256='a'*64,magnetic_function_sha256='b'*64,implementation_sha256=package_signature(),
        k=['0','0','0'],candidates=[candidate],seeds=[dict(id='first',axis='x',magnitude=3.),dict(id='second',axis='y',magnitude=3.)],
        sites=model['settings']['sites'],policy=dict(cycles=12,timeout_seconds=60,maximum_trials=2,
            maximum_cell_multiplier=8,maximum_moment=6.,fit_band_percent=.25,maximum_mode=.5,adequate_rwp_percent=12.,
            magnetic_optimizer='profiled_quadratic',optimizer_iterations=100,moment_model='symmetry_allowed')))
    owner=Campaign(tmp_path/'campaign',shared_inputs=True)
    ids=owner.enroll(protocol,inputs,approved_hash=digest(protocol))
    return owner,inputs,protocol,ids


def test_selected_library_reaches_job_without_dense_responses_keys_or_duplicate_uploads(remote,tmp_path,monkeypatch):
    runtime,service,source=remote;prepared=library(tmp_path/'preparation')
    owner,inputs,protocol,ids=campaign(tmp_path,source,prepared)
    runtime.preparation_provider=lambda p,root:hints.capture(prepared,p,root)
    transport=GalaxyChunkTransport(runtime,owner)
    transport.prepare_source(protocol,inputs,source)
    sha=digest(protocol);packet_id=transport.state['inputs'][sha]
    with zipfile.ZipFile(io.BytesIO(service.data[packet_id])) as packet:
        assert hints.MANIFEST in packet.namelist()
        assert not any(name.endswith('response.npz') or name.endswith('.gpx') for name in packet.namelist())
        document=json.loads(packet.read(hints.MANIFEST))
        assert len(document['entries'])==1 and document['observations_included'] is False
    assert not service.submissions
    monkeypatch.setattr(transport,'checkpoint',lambda **_:None)
    handle=transport.submit_many([(ids[0],owner.root/'trials'/ids[0])],workers=1)[ids[0]]
    request=transport.state['chunks'][handle['dispatch_key']]['request']
    assert request['prepared_scattering']==document
    assert transport.state['prepared_inputs'][sha]['native_qualification_required']
    assert len(service.submissions)==1
    # A later selection cannot replace the already captured source context.
    alternate=tmp_path/'another-parent.cif';alternate.write_text('data_another_parent\n')
    row=prepared.document['entries'][0]['model']
    other=create_library(tmp_path/'other-library',name='Another preparation',source=alternate,
        banks={'bank 1':dict(instrument=prepared.root/row['instrument'],grid=row['grid'])},
        models=[dict(kind='magnetic',bank='bank 1',settings=row['settings'])])
    assert other.document['id']!=prepared.document['id']
    runtime.preparation_provider=lambda p,root:hints.capture(other,p,root)
    transport.prepare_source(protocol,inputs,source)
    assert transport.state['inputs'][sha]==packet_id
    assert hints.read(owner.root/'inputs'/sha,protocol)==document
    assert all(b'PRIVATE-' not in data for data in service.data.values())


def test_optional_preparation_error_does_not_stop_the_scientific_job(remote,tmp_path,monkeypatch):
    runtime,service,source=remote;prepared=library(tmp_path/'preparation')
    owner,inputs,protocol,ids=campaign(tmp_path,source,prepared)
    def damaged(*args):raise ValueError('Corrupt selected theory artifact')
    runtime.preparation_provider=damaged
    transport=GalaxyChunkTransport(runtime,owner);transport.prepare_source(protocol,inputs,source)
    monkeypatch.setattr(transport,'checkpoint',lambda **_:None)
    handle=transport.submit_many([(ids[0],owner.root/'trials'/ids[0])],workers=1)[ids[0]]
    request=transport.state['chunks'][handle['dispatch_key']]['request']
    assert 'prepared_scattering' not in request and len(service.submissions)==1
    assert transport.state['prepared_inputs'][digest(protocol)]['status']=='unavailable'
    assert len(owner.trials())==2


def test_restored_packet_retains_theory_provenance_without_rebuilding_the_library(remote,tmp_path):
    runtime,service,source=remote;prepared=library(tmp_path/'preparation')
    owner,inputs,protocol,ids=campaign(tmp_path,source,prepared)
    runtime.preparation_provider=lambda p,root:hints.capture(prepared,p,root)
    transport=GalaxyChunkTransport(runtime,owner);transport.prepare_source(protocol,inputs,source)
    expected=hints.read(owner.root/'inputs'/digest(protocol),protocol)
    from gpx_recipe.symmetry.bundle import unpack
    archive=tmp_path/'saved-inputs.zip'
    service._download_dataset(transport.state['inputs'][digest(protocol)],archive)
    remote_packet=unpack(archive,tmp_path/'restored-packet')
    shutil.rmtree(prepared.root)
    restored=tmp_path/'fresh-scientific-context'
    actual=hints.read(remote_packet,protocol)
    result=hints.restore(actual,protocol,remote_packet,restored)
    assert actual==expected and result['attached']==1 and not service.submissions
    assert hints.cache_for(restored,protocol,candidate=protocol['candidates'][0]['id'])[0] is not None


def test_selection_caption_does_not_load_scientific_arrays(tmp_path,monkeypatch):
    prepared=library(tmp_path/'preparation');archive=tmp_path/'portable.zip';prepared.export(archive)
    registry=PreparationRegistry(tmp_path/'registry');key=registry.import_archive(archive)
    app=PreparationMixin();app._preparation_workspace=SimpleNamespace(registry=registry)
    app._preparation_store=lambda:pytest.fail('A read-only caption must not initialize a workspace')
    monkeypatch.setattr(registry,'get',lambda *args:pytest.fail('Caption must use metadata summaries'))
    assert registry.selected_key()==key
    text=app._prepared_selection_caption()
    assert prepared.document['name'] in text and '1 / 1' in text and 'compatibility' in text


@pytest.mark.parametrize('damage',[None,'manifest','array'])
def test_full_history_recovery_retains_preparation_and_all_scientific_starts(remote,tmp_path,damage):
    from gpx_recipe.symmetry.templates import make_template
    from radar_pd_nova.model_campaigns import restore_history
    from radar_pd_nova.model_runtime import GalaxyModelRuntime
    runtime,service,source=remote;prepared=library(tmp_path/'preparation')
    source.update(experiment='IPTS-1',sample_key='specimen:A',bank='1')
    old=tmp_path/'old-viewer'
    old.mkdir()
    original,inputs,protocol,_=campaign(old,source,prepared)
    # History recovery uses a durable, explicitly identified campaign name.
    owner=Campaign(old/'campaign-aaaaaaaaaaaa',shared_inputs=True)
    ids=owner.enroll(protocol,inputs,approved_hash=digest(protocol))
    template=make_template(protocol,dict(sha256=source['source_sha256']),source,[(source,'radar')],origin='radar')
    owner.set_setting('template',template);owner.set_setting('paused',True)
    runtime.preparation_provider=lambda p,root:hints.capture(prepared,p,root)
    transport=GalaxyChunkTransport(runtime,owner);transport.prepare_source(protocol,inputs,source)
    expected=hints.read(owner.root/'inputs'/digest(protocol),protocol)
    packet_id=transport.state['inputs'][digest(protocol)]
    transport.checkpoint(force=True)
    if damage is not None:
        with zipfile.ZipFile(io.BytesIO(service.data[packet_id])) as packet:
            values={name:packet.read(name) for name in packet.namelist()}
        key=hints.MANIFEST if damage=='manifest' else expected['entries'][0]['path']
        values[key]=b'changed optional preparation'
        content=io.BytesIO()
        with zipfile.ZipFile(content,'w') as packet:
            for name,value in values.items():packet.writestr(name,value)
        service.data[packet_id]=content.getvalue()
    service.data['original-data']=Path(source['source_path']).read_bytes()
    shutil.rmtree(prepared.root);shutil.rmtree(old)
    fresh=GalaxyModelRuntime(tmp_path/'fresh-runtime',service)
    restore_history(tmp_path/'new-viewer',fresh,experiment='IPTS-1')
    restored=Campaign(tmp_path/'new-viewer/campaign-aaaaaaaaaaaa',shared_inputs=True)
    recovered=GalaxyChunkTransport(fresh,restored)
    assert {row['id'] for row in restored.trials()}==set(ids)
    assert restored.setting('restored') and restored.setting('paused')
    assert not service.submissions
    state=recovered.state['prepared_inputs'][digest(protocol)]
    if damage is None:
        assert state['status']=='attached' and state['attached_entries']==1
        assert hints.read(restored.root/'inputs'/digest(protocol),protocol)==expected
    else:
        assert state['status']=='unavailable'
        if damage=='array':assert state['discarded_entries']==[expected['entries'][0]['entry']]
        else:assert state['error_type']=='JSONDecodeError'
