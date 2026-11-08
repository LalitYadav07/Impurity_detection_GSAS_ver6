"""Origin lookup hints survive portable files without granting authority."""
import copy
import json
from types import SimpleNamespace

import pytest
import requests

from gpx_recipe.preparation import certificates
from gpx_recipe.preparation.library import PreparationLibrary,import_library
from gpx_recipe.symmetry.compiler import file_sha,package_signature
from radar_pd_nova import preparation_evidence as evidence
from test_model_runtime import remote


def setup(tmp_path):
    path=tmp_path/'certificate.json';entry='a'*64
    path.write_text(json.dumps(dict(schema=certificates.SCHEMA,implementation_sha256=package_signature(),
        result_archive_sha256='b'*64,entries=[dict(kind='structural',entry_sha256=entry,
        physics_sha256='c'*64,artifacts={'starts.npz':'d'*64,'starts.json':'e'*64})])))
    ref=dict(site='https://galaxy.invalid',dataset_id='receipt1',sha256=file_sha(path),entries=[entry])
    calls=[]
    def download(identifier,destination,**kwargs):calls.append(identifier);destination.write_bytes(path.read_bytes())
    service=SimpleNamespace(galaxy_url=ref['site'],tool_has_input=lambda *args:True,_download_dataset=download)
    runtime=SimpleNamespace(root=tmp_path/'runtime',tool_id='model',service=service)
    request=dict(prepared_scattering=dict(evidence=[copy.deepcopy(ref)],entries=[dict(entry_sha256=entry)]))
    return path,ref,calls,runtime,request


def test_only_selected_same_site_original_receipt_is_forwarded_as_an_hda(tmp_path):
    _,ref,calls,runtime,request=setup(tmp_path)
    other=dict(ref,dataset_id='unused',entries=['f'*64]);request['prepared_scattering']['evidence'].append(other)
    assert evidence.input_ids(runtime,request)==['receipt1'] and calls==['receipt1']
    assert evidence.input_ids(runtime,request)==['receipt1'] and calls==['receipt1']
    assert not any(p.name=='authority.json' for p in runtime.root.rglob('*'))


@pytest.mark.parametrize('operation',['structural-snapshot','structural-snapshot-ranking'])
def test_structural_snapshot_routes_original_receipts_through_the_same_server_input(tmp_path,operation):
    _,_,calls,runtime,request=setup(tmp_path)
    snapshot=dict(operation=operation,payload=dict(prepared_scattering=request['prepared_scattering']))
    assert evidence.input_ids(runtime,snapshot)==['receipt1'] and calls==['receipt1']
    runtime.service.galaxy_url='https://another.invalid'
    assert evidence.input_ids(runtime,snapshot)==[] and calls==['receipt1']


def test_shared_profile_origin_is_forwarded_without_a_structural_start_bank(tmp_path):
    _,_,calls,runtime,request=setup(tmp_path)
    document=request['prepared_scattering'];row=document['entries'].pop()
    document['profile_tables']=dict(entries=[row])
    assert evidence.input_ids(runtime,request)==['receipt1'] and calls==['receipt1']


@pytest.mark.parametrize('change',['site','entry','hash','body','permission','capability'])
def test_optional_unavailable_or_edited_origins_do_not_block_science_or_contact_other_sites(tmp_path,change):
    path,ref,calls,runtime,request=setup(tmp_path)
    if change=='site':runtime.service.galaxy_url='https://another.invalid'
    elif change=='entry':request['prepared_scattering']['entries'][0]['entry_sha256']='f'*64
    elif change=='hash':request['prepared_scattering']['evidence'][0]['sha256']='f'*64
    elif change=='body':
        path.write_text('{}');request['prepared_scattering']['evidence'][0]['sha256']=file_sha(path)
    elif change=='permission':
        def forbidden(*args,**kwargs):raise requests.HTTPError('not authorized')
        runtime.service._download_dataset=forbidden
    else:runtime.service.tool_has_input=lambda *args:False
    assert evidence.input_ids(runtime,request)==[]
    if change in {'site','entry','capability'}:assert calls==[]


def test_origin_lookup_network_outage_does_not_block_submission(tmp_path):
    _,_,_,runtime,request=setup(tmp_path)
    def timeout(*args):raise requests.Timeout('offline')
    runtime.service.tool_has_input=timeout
    assert evidence.input_ids(runtime,request)==[]


def test_origin_references_roundtrip_without_becoming_library_identity_or_scientific_trust(tmp_path,monkeypatch):
    path,ref,_,_,_=setup(tmp_path)
    cif=tmp_path/'parent.cif';cif.write_text('data_public')
    library=PreparationLibrary.create(tmp_path/'library',name='Public',specification={},
        inputs={'parent.cif':cif},models=[dict(kind='structural',bank='1')])
    identity=library.document['id'];library.add_evidence(ref)
    assert library.document['id']==identity
    archive=tmp_path/'portable.zip';library.export(archive)
    imported=import_library(archive,tmp_path/'new-viewer')
    assert imported.document['evidence']==[ref]
    # The caller cannot mutate durable state through its retained reference.
    ref['entries'].append('f'*64)
    assert imported.document['evidence'][0]['entries']==['a'*64]
    before=copy.deepcopy(library.document)
    def failed():raise OSError('publication interrupted')
    monkeypatch.setattr(library,'_save',failed)
    with pytest.raises(OSError):library.add_evidence(dict(ref,entries=['c'*64]))
    assert library.document==before
    assert PreparationLibrary(library.root).document==before


def test_collected_original_receipt_becomes_a_small_routing_reference(remote,tmp_path,monkeypatch):
    runtime,service,_=remote
    certificate,_,_,_,_=setup(tmp_path)
    from radar_pd_nova.models import UtilityActionRecord,RunStatus
    job=UtilityActionRecord(uid='job1',tool_id=runtime.tool_id,name='native',galaxy_job_id='galaxy1',status=RunStatus.OK,
        outputs={'preparation_certificate':'original-receipt'})
    service.data['original-receipt']=certificate.read_bytes()
    service.refresh_utility=lambda job:job
    from radar_pd_nova import native_operations
    monkeypatch.setattr(native_operations,'recover_job',lambda *args:job)
    refs=[]
    library=SimpleNamespace(document=dict(entries=[dict(files=[dict(path='entries/id/entry.json',sha256='a'*64)])]),
        add_evidence=lambda reference:refs.append(reference))
    assert evidence.remember(library,runtime,{},tmp_path)
    assert refs==[dict(site='https://galaxy.invalid',dataset_id='original-receipt',sha256=file_sha(certificate),entries=['a'*64])]
    assert not list(tmp_path.glob('origin-*.json'))
    def forbidden(*args,**kwargs):raise requests.HTTPError('removed')
    service._download_dataset=forbidden
    assert evidence.remember(library,runtime,{},tmp_path) is False
    assert not list(tmp_path.glob('origin-*.json'))


def test_model_submission_supplies_optional_receipts_separately_from_request(remote,monkeypatch):
    runtime,service,source=remote
    monkeypatch.setattr(evidence,'input_ids',lambda *args:['receipt1','receipt2'])
    job=runtime._submit('native',source,{'operation':'inspect'},'assets1','c'*24)
    submitted=service.submissions[-1]['inputs']
    assert submitted['preparation_evidence']==[{'dataset_id':'receipt1'},{'dataset_id':'receipt2'}]
    assert 'preparation_evidence' not in next(iter(service.requests.values()))
    assert job.galaxy_job_id
