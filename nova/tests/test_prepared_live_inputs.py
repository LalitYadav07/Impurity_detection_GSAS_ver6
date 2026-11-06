"""Portable theory follows Live jobs without changing their scientific scope."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest

from gpx_recipe.execution import write_json
from gpx_recipe.preparation import hints
from gpx_recipe.live.packet import create_packet,unpack_packet,scientific_packet_identity,canonical
from gpx_recipe.live.contracts import file_hash
from radar_pd_nova.webmon.galaxy_runtime import GalaxyLiveRuntime
from test_prepared_model_inputs import library
from test_webmon_hardening import scientific_request
from test_webmon_galaxy_runtime import Service


def request_for(root,prepared):
    root.mkdir();request,_=scientific_request(root)
    settings=prepared.document['entries'][0]['model']['settings']
    catalog=dict(k=settings['k'],candidates=[settings['candidate']])
    write_json(Path(request['catalog']),catalog)
    write_json(Path(request['preflight']),[dict(id=settings['candidate']['id'])])
    request.update(catalog_sha256=file_hash(request['catalog']),preflight_sha256=file_hash(request['preflight']),
        sites=settings['sites'],k=settings['k'])
    return request,catalog


def test_live_packet_contains_only_bounded_theory_and_never_imports_native_qualification(tmp_path):
    prepared=library(tmp_path/'library');request,catalog=request_for(tmp_path/'source',prepared)
    root=tmp_path/'selected';context=hints.live_context(request,catalog)
    document=hints.capture(prepared,context,root)
    ordinary=tmp_path/'ordinary.zip';expected=create_packet(request,ordinary)
    request=dict(request,prepared_scattering=document)
    packet=tmp_path/'prepared.zip';full=create_packet(request,packet,prepared_root=root)
    assert full!=expected and scientific_packet_identity(packet)==expected
    with zipfile.ZipFile(packet) as archive:
        assert hints.MANIFEST in archive.namelist()
        assert not any(name.endswith('response.npz') or name.endswith('.cif') for name in archive.namelist())
        assert str(tmp_path).encode() not in archive.read('request.json')
    restored,identity=unpack_packet(packet,tmp_path/'native')
    assert identity==full
    cache,report=hints.cache_for(tmp_path/'native',hints.live_context(restored,catalog),
        candidate=catalog['candidates'][0]['id'])
    assert report['attached']==1 and report['native_qualification_required']
    assert cache.lookup('e'*64)[1] is False


def test_missing_live_theory_assets_keep_all_reviewed_candidates_available(tmp_path):
    prepared=library(tmp_path/'library');request,catalog=request_for(tmp_path/'source',prepared)
    root=tmp_path/'selected';context=hints.live_context(request,catalog)
    document=hints.capture(prepared,context,root)
    (root/document['entries'][0]['path']).unlink()
    request=dict(request,prepared_scattering=document)
    packet=tmp_path/'prepared.zip';create_packet(request,packet,prepared_root=root)
    restored,_=unpack_packet(packet,tmp_path/'native')
    cache,report=hints.cache_for(tmp_path/'native',hints.live_context(restored,catalog))
    assert cache is None and report['discarded_entries']==[document['entries'][0]['entry']]
    assert json.loads(Path(restored['catalog']).read_text())==catalog


@pytest.mark.parametrize('change',['path','candidate','script','scope'])
def test_live_theory_metadata_is_closed_and_bound_before_packing(tmp_path,change):
    prepared=library(tmp_path/'library');request,catalog=request_for(tmp_path/'source',prepared)
    root=tmp_path/'selected';document=hints.capture(prepared,hints.live_context(request,catalog),root)
    if change=='path':document['entries'][0]['path']='../elsewhere.npz'
    elif change=='candidate':document['entries'][0]['candidate']='foreign'
    elif change=='script':document['script']='print(1)'
    else:document['protocol_sha256']='a'*64
    from gpx_recipe.symmetry.contracts import digest
    document['snapshot_sha256']=digest({k:v for k,v in document.items() if k!='snapshot_sha256'})
    with pytest.raises(ValueError):create_packet(dict(request,prepared_scattering=document),tmp_path/'packet.zip',prepared_root=root)


class CapturingService(Service):
    def __init__(self,root):
        super().__init__(None);self.root=root;self.uploads=[]

    def upload_document(self,path,**kwargs):
        self.uploads.append(Path(path).read_bytes())
        with zipfile.ZipFile(io.BytesIO(self.uploads[-1])) as packet:
            manifest=json.loads(packet.read('manifest.json'));request=json.loads(packet.read('request.json'))
        identity=hashlib.sha256(canonical(manifest)).hexdigest()
        result=dict(packet_id=identity,source_revision=request['source_revision'],
            reference_revision=request['reference_revision'],rows=[])
        contents={'result.json':canonical(result)}
        files={name:hashlib.sha256(value).hexdigest() for name,value in contents.items()}
        contents['transport.json']=canonical(dict(schema='radar-live-result/v1',packet_id=identity,files=files))
        self.bundle=self.root/'result.zip'
        with zipfile.ZipFile(self.bundle,'w') as archive:
            for name,value in contents.items():archive.writestr(name,value)
        return 'request-hda'


def test_live_recovery_keeps_the_original_job_after_disposable_theory_is_lost(tmp_path):
    prepared=library(tmp_path/'library');request,_=request_for(tmp_path/'source',prepared)
    service=CapturingService(tmp_path);runtime=GalaxyLiveRuntime(tmp_path/'runtime',service)
    calls=[]
    def provider(context,root):calls.append(1);return hints.capture(prepared,context,root)
    runtime.preparation_provider=provider
    first=runtime.call(request,tmp_path/'first',dispatch_key='a'*24)
    selected=runtime.root/'prepared-inputs'/('a'*24)
    for path in selected.glob('_prepared_scattering/*.npz'):path.unlink()
    runtime.preparation_provider=lambda *_:pytest.fail('A recovered job cannot adopt another library')
    second=runtime.call(request,tmp_path/'second',dispatch_key='a'*24)
    assert first['packet_id']==second['packet_id']
    assert calls==[1] and len(service.uploads)==len(service.submissions)==1
    assert 'prepared_scattering' not in request
    changed=dict(request,upper=8.)
    with pytest.raises(ValueError,match='different live inputs'):
        runtime.call(changed,tmp_path/'different',dispatch_key='a'*24)
    assert len(service.submissions)==1


@pytest.mark.parametrize('failure',[False,True])
def test_live_without_preparation_remains_a_real_scientific_dispatch(tmp_path,failure):
    prepared=library(tmp_path/'library');request,_=request_for(tmp_path/'source',prepared)
    service=CapturingService(tmp_path);runtime=GalaxyLiveRuntime(tmp_path/'runtime',service)
    def provider(*args):
        if failure:raise ValueError('Optional library is unavailable')
    runtime.preparation_provider=provider
    runtime.call(request,tmp_path/'first',dispatch_key='b'*24)
    runtime.preparation_provider=lambda *_:pytest.fail('Selection for an acknowledged job is immutable')
    runtime.call(request,tmp_path/'second',dispatch_key='b'*24)
    assert len(service.submissions)==1
    with zipfile.ZipFile(io.BytesIO(service.uploads[0])) as archive:
        assert hints.MANIFEST not in archive.namelist()
        assert 'prepared_scattering' not in json.loads(archive.read('request.json'))


def test_sampler_timeout_is_unresolved_and_keeps_the_prepared_kernel(tmp_path,monkeypatch):
    import numpy as np
    from gpx_recipe.live import worker
    from gpx_recipe.live import validated_backend
    prepared=library(tmp_path/'library');request,catalog=request_for(tmp_path/'source',prepared)
    write_json(Path(request['preflight']),[dict(id=catalog['candidates'][0]['id'],compatible=True,dimensions=3)])
    request['preflight_sha256']=file_hash(request['preflight'])
    target=tmp_path/'request.json';write_json(target,request)
    def compiled(source,candidate,k,sites,out,**kwargs):
        out.mkdir(parents=True)
        np.savez(out/'kernel.npz',x=np.arange(4.))
        meta=dict(dimensions=3,bin_count=4,native_errors=[0.])
        write_json(out/'kernel.json',meta);return meta
    def incomplete(*args,**kwargs):raise TimeoutError('Posterior move budget exceeded; candidate remains unresolved')
    monkeypatch.setattr(worker,'compile_candidate',compiled)
    monkeypatch.setattr(worker,'infer',incomplete)
    monkeypatch.setattr(validated_backend,'active_identity',lambda:None)
    worker.execute(target)
    result=json.loads((tmp_path/'result.json').read_text());row=result['rows'][0]
    assert result['status']=='complete' and row['status']=='sampling_unresolved'
    assert row['sampling_incomplete'] and row['kernel_retained']
    assert list((tmp_path/'models').glob('*/kernel.npz'))
    assert 'posterior' not in row and 'fitted_gpx' not in row


def test_returning_from_preparation_updates_both_consumer_captions():
    from types import SimpleNamespace
    from radar_pd_nova.preparation_controller import PreparationMixin
    state=SimpleNamespace(workspace_view='preparation',setup_collapsed=True,
        sx_preparation_notice='',webmon_preparation_notice='',flush=lambda:None)
    app=PreparationMixin();app.server=SimpleNamespace(state=state)
    app._prep_return=('monitor',True);app._prepared_selection_caption=lambda:'Selected theory library'
    app.close_preparation_workspace()
    assert state.workspace_view=='monitor' and state.setup_collapsed
    assert state.sx_preparation_notice==state.webmon_preparation_notice=='Selected theory library'
