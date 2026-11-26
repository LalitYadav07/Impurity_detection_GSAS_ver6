"""A direct GPX handoff preserves job ownership and exact native bytes."""
import base64
import copy
import json
from pathlib import Path

import pytest

from radar_pd_nova.galaxy_service import GalaxyService
from radar_pd_nova.model_runtime import GalaxyChunkTransport,MODEL_TOOL_ID
from radar_pd_nova.refinement_runtime import RuntimeUnavailable
from gpx_recipe.symmetry.campaign import Campaign
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.symmetry.contracts import digest
from test_model_runtime import remote


@pytest.fixture
def published(remote,tmp_path,monkeypatch):
    runtime,service,_=remote
    campaign=Campaign(tmp_path/'campaign');transport=GalaxyChunkTransport(runtime,campaign)
    trial='a'*24;key='b'*24;body={'test':'checked synthetic protocol'};protocol=digest(body)
    request=dict(protocol=body,protocol_sha256=protocol,trials=[trial],workers=4,dispatch_key=key)
    transport.state['members'][trial]=key
    transport.state['chunks'][key]=dict(key=key,protocol=protocol,request=request,members=[trial],
        complete=True,job=dict(galaxy_job_id='job1',outputs={'result_bundle':'bundle1'}),
        result=dict(schema='radar-model-chunk-result/v1',dispatch_key=key,protocol_sha256=protocol,
                    trials=[dict(id=trial,state='finished')]))
    encoded=base64.b64encode(json.dumps(request).encode()).decode()
    job=dict(id='job1',history_id=service.history_id,tool_id=MODEL_TOOL_ID,state='ok',
        params=dict(dispatch_key=key,action='chunk',model_request_b64=encoded),
        output_collections={'reviewable_projects':dict(id='collection1')})
    monkeypatch.setattr(service,'_job_details',lambda uid:copy.deepcopy(job))
    monkeypatch.setattr(service,'_job_output_ids',lambda value:
        GalaxyService._named_output_ids(value.get('output_collections')),raising=False)
    monkeypatch.setattr(service,'_history_job_collection_ids',lambda *args,**kw:{},raising=False)
    members=[dict(name=trial,id='gpx1')]
    monkeypatch.setattr(service,'collection_elements',lambda uid:copy.deepcopy(members),raising=False)
    captured=tmp_path/'audited.gpx';captured.write_bytes(b'checked native project bytes')
    service.data['gpx1']=captured.read_bytes()
    metadata=dict(id='gpx1',state='ok',file_size=captured.stat().st_size,deleted=False,purged=False)
    monkeypatch.setattr(service,'_dataset_metadata',lambda uid:copy.deepcopy(metadata),raising=False)
    return transport,service,trial,protocol,captured,job,members,metadata


def test_published_reviewable_project_avoids_new_uploads_and_submissions(published):
    transport,service,trial,protocol,path,*_=published
    result=transport.published_project(trial,protocol,path)
    assert result==dict(dataset_id='gpx1',job_id='job1',role='reviewable_projects',sha256=file_sha(path))
    assert not service.submissions and not service.labels
    assert not list(transport.campaign.root.glob('published-gpx-*'))


@pytest.mark.parametrize('legacy',[False,True])
def test_saved_compact_checkpoint_reuses_the_authenticated_native_gpx(published,legacy):
    transport,service,trial,protocol,path,*_=published
    transport.save()
    restored=GalaxyChunkTransport(transport.runtime,transport.campaign)
    chunk=restored.state['chunks']['b'*24]
    assert 'request' not in chunk and len(chunk['request_sha256'])==64
    if legacy:chunk.pop('request_sha256')
    result=restored.published_project(trial,protocol,path)
    assert result['dataset_id']=='gpx1' and result['sha256']==file_sha(path)
    assert not service.submissions and not service.labels


@pytest.mark.parametrize('change',['members','result','fingerprint','remote_protocol','remote_dispatch','missing_members'])
def test_compact_publication_requires_all_saved_and_remote_witnesses(published,change):
    transport,service,trial,protocol,path,job,*_=published
    transport.save();chunk=transport.state['chunks']['b'*24]
    if change=='members':chunk['members']=['e'*24]
    elif change=='missing_members':chunk.pop('members')
    elif change=='result':chunk['result']['trials']=[]
    elif change=='fingerprint':chunk['request_sha256']='f'*64
    else:
        request=json.loads(base64.b64decode(job['params']['model_request_b64']))
        if change=='remote_protocol':request['protocol']={'different':'protocol'}
        else:request['dispatch_key']='f'*24
        job['params']['model_request_b64']=base64.b64encode(json.dumps(request).encode()).decode()
    with pytest.raises(ValueError):transport.published_project(trial,protocol,path)
    assert not service.submissions and not service.labels


@pytest.mark.parametrize('change',['history','tool','dispatch','request','action','bytes','size','duplicate'])
def test_changed_publication_cannot_replace_the_selected_model(published,change):
    transport,service,trial,protocol,path,job,members,metadata=published
    if change=='history':job['history_id']='another-history'
    elif change=='tool':job['tool_id']='another-tool'
    elif change=='dispatch':job['params']['dispatch_key']='d'*24
    elif change=='request':job['params']['model_request_b64']=base64.b64encode(b'{"different":"trial"}').decode()
    elif change=='action':job['params']['action']='native'
    elif change=='bytes':service.data['gpx1']=b'x'*path.stat().st_size
    elif change=='size':metadata['file_size']+=1
    else:members.append(dict(name=trial,id='gpx2'))
    with pytest.raises((ValueError,RuntimeUnavailable)):
        transport.published_project(trial,protocol,path)
    assert not service.submissions and not service.labels


@pytest.mark.parametrize('change',['legacy_request','no_collection','no_member','deleted','incomplete'])
def test_missing_older_publication_leaves_the_existing_upload_route_available(published,change):
    transport,service,trial,protocol,path,job,members,metadata=published
    if change=='legacy_request':job['params'].pop('model_request_b64')
    elif change=='no_collection':job['output_collections']={}
    elif change=='no_member':members.clear()
    elif change=='deleted':metadata['deleted']=True
    else:transport.state['chunks']['b'*24]['complete']=False
    assert transport.published_project(trial,protocol,path) is None
    assert not service.submissions and not service.labels
