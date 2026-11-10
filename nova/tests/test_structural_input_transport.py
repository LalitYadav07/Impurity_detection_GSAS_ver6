"""Multipart training uses one durable dispatch and survives viewer loss."""
import copy
from pathlib import Path
import shutil

import pytest

from gpx_recipe.preparation import structural_jobs as jobs
from gpx_recipe.preparation.structural_input_parts import references
from radar_pd_nova.model_runtime import GalaxyModelRuntime
from radar_pd_nova.native_operations import validate_handle
from radar_pd_nova.refinement_runtime import RuntimeUnavailable
from test_model_runtime import remote
from test_structural_input_parts import captured


def setup(remote,tmp_path,monkeypatch):
    runtime,service,_=remote
    root=tmp_path/'training';request,names=captured(root,monkeypatch)
    monkeypatch.setattr(jobs,'MAX_INPUT_BYTES',12000)
    monkeypatch.setattr(service,'tool_has_input',lambda tool,key:key=='model_asset_parts',raising=False)
    source=dict(id='parent',source_sha256=request['expected_source_sha256'],source_path=str(root/'parent.cif'),dataset_id='parent-data')
    return runtime,service,source,root,request,names


def test_training_submission_reuses_parts_and_recovers_all_files_without_local_state(remote,tmp_path,monkeypatch):
    runtime,service,source,root,request,names=setup(remote,tmp_path,monkeypatch)
    payload=copy.deepcopy(request['payload'])
    for field in ['source','instrument','plan','destination']:payload[field]=str(root/payload[field])
    handle=runtime.submit_native(request['operation'],payload,root,source=source,timeout=120)
    validate_handle(runtime,handle)
    primary,more=references(handle['asset_id'])
    assert len(more)==2 and len(service.submissions)==1
    inputs=service.submissions[0]['inputs']
    assert inputs['model_assets']==dict(dataset_id=primary)
    assert inputs['model_asset_parts']==[dict(dataset_id=key) for key in more]
    # The exact parts are reused if the capture is reconstructed, without
    # uploading a second copy or submitting another scientific job.
    before=len(service.data)
    import os
    for name in names:os.utime(root/name,(1740000000,1740000000))
    assert runtime.training_packet(root,names,handle['request'])==handle['asset_id']
    assert len(service.data)==before
    fresh=GalaxyModelRuntime(tmp_path/'fresh-viewer',service)
    monkeypatch.setattr(fresh,'find_dispatch',lambda key:pytest.fail('Known job must recover directly'))
    # Only the handle and acknowledged History datasets remain after loss of
    # the originating viewer's entire local source/runtime directory.
    shutil.rmtree(root);shutil.rmtree(runtime.root)
    restored=fresh.restore_native_inputs(handle,tmp_path/'restored')
    assert set(restored)==set(handle['request']['input_hashes'])
    from gpx_recipe.symmetry.compiler import file_sha
    assert {name:file_sha(path) for name,path in restored.items()}==handle['request']['input_hashes']
    assert len(service.submissions)==1 and not list((fresh.root/'native-inputs').iterdir())


def test_old_wrapper_fails_before_upload_or_submission_without_dropping_examples(remote,tmp_path,monkeypatch):
    runtime,service,source,root,request,names=setup(remote,tmp_path,monkeypatch)
    monkeypatch.setattr(service,'tool_has_input',lambda *args:False)
    with pytest.raises(RuntimeUnavailable,match='No examples were dropped'):
        runtime.training_packet(root,names,request)
    assert not service.submissions and not service.data


def test_small_training_keeps_its_original_string_packet_identity(remote,tmp_path,monkeypatch):
    runtime,service,source,root,request,names=setup(remote,tmp_path,monkeypatch)
    monkeypatch.setattr(jobs,'MAX_INPUT_BYTES',480*1024**2)
    monkeypatch.setattr(service,'tool_has_input',lambda *args:pytest.fail('Small packets need no new wrapper'))
    key=runtime.training_packet(root,names,request)
    assert isinstance(key,str) and key==runtime.packet(root,names)


def test_multipart_descriptor_cannot_be_reused_for_a_magnetic_or_generic_operation(remote,tmp_path,monkeypatch):
    runtime,service,source,root,request,names=setup(remote,tmp_path,monkeypatch)
    reference=runtime.training_packet(root,names,request)
    with pytest.raises(ValueError,match='only support structural training'):
        runtime._submit('chunk',source,request,reference,'a'*24)
    assert not service.submissions


def test_lost_part_upload_acknowledgment_is_recovered_without_duplicate_datasets(remote,tmp_path,monkeypatch):
    runtime,service,source,root,request,names=setup(remote,tmp_path,monkeypatch)
    original=service.upload_document;calls=[]
    def upload(*args,**kwargs):
        key=original(*args,**kwargs);calls.append(key)
        if len(calls)==2:raise TimeoutError('Upload accepted, acknowledgment lost')
        return key
    monkeypatch.setattr(service,'upload_document',upload)
    with pytest.raises(TimeoutError):runtime.training_packet(root,names,request)
    assert len(service.data)==2 and not service.submissions
    reference=runtime.training_packet(root,names,request)
    assert len(service.data)==3 and len(calls)==3
    assert len(references(reference)[1])==2 and not service.submissions
