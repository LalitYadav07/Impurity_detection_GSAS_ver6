import json,shutil,zipfile
from pathlib import Path
import pytest
from radar_pd_nova.webmon.galaxy_runtime import GalaxyLiveRuntime,LIVE_TOOL_ID
from radar_pd_nova.refinement_runtime import RuntimeUnavailable
from radar_pd_nova.models import UtilityActionRecord,RunStatus
from gpx_recipe.live.packet import create_packet,canonical
from gpx_recipe.live.contracts import file_hash
from test_webmon_hardening import scientific_request


class Service:
    def __init__(self,bundle,fail=False):self.bundle=bundle;self.fail=fail;self.submissions=[]
    def upload_document(self,path,**_):return 'request-hda'
    def submit_utility(self,**kw):
        self.submissions.append(kw)
        if self.fail:raise RuntimeError('connection lost')
        return UtilityActionRecord(uid='job',name='Live',tool_id=LIVE_TOOL_ID,galaxy_job_id='galaxy-job',status=RunStatus.OK,outputs={'result_bundle':'results-hda','response_cache_output':'cache-hda','verified_projects':'projects-hdca'})
    def collection_elements(self,uid):
        assert uid=='projects-hdca'
        return [{'name':'test','id':'inferred-gpx-hda'}]
    def _download_dataset(self,uid,target):assert uid=='results-hda';shutil.copy2(self.bundle,target)


def bundle_for(tmp_path,request):
    packet=tmp_path/'packet.zip';identity=create_packet(request,packet)
    root=tmp_path/'response';(root/'models/test').mkdir(parents=True)
    (root/'models/test/posterior.json').write_text('{}')
    result=dict(packet_id=identity,source_revision=request['source_revision'],reference_revision=request['reference_revision'],rows=[{'posterior':'models/test/posterior.json','published_project':'test'}])
    (root/'result.json').write_text(json.dumps(result));files={str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file()}
    (root/'transport.json').write_bytes(canonical(dict(schema='radar-live-result/v1',packet_id=identity,files=files)))
    bundle=tmp_path/'result.zip'
    with zipfile.ZipFile(bundle,'w') as z:
        for p in root.rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(root)))
    return bundle


def test_galaxy_dispatch_restores_outputs_and_reuses_acknowledged_job(tmp_path):
    src=tmp_path/'src';src.mkdir();request,_=scientific_request(src);service=Service(bundle_for(tmp_path,request))
    runtime=GalaxyLiveRuntime(tmp_path/'runtime',service)
    first=runtime.call(request,tmp_path/'call',dispatch_key='a'*24,cache_dataset_id='previous-cache')
    second=runtime.call(request,tmp_path/'call2',dispatch_key='a'*24)
    assert len(service.submissions)==1
    assert service.submissions[0]['inputs']['response_cache']=={'dataset_id':'previous-cache'}
    assert first['response_cache_dataset_id']=='cache-hda'
    assert first['rows'][0]['fitted_dataset_id']=='inferred-gpx-hda'
    assert Path(second['rows'][0]['posterior']).is_file()


def test_lost_acknowledgement_never_resubmits_ambiguous_job(tmp_path):
    src=tmp_path/'src';src.mkdir();request,_=scientific_request(src);service=Service(None,fail=True)
    runtime=GalaxyLiveRuntime(tmp_path/'runtime',service);runtime.find_dispatch=lambda _:None
    with pytest.raises(RuntimeUnavailable,match='acknowledge'):runtime.call(request,tmp_path/'first',dispatch_key='b'*24)
    with pytest.raises(RuntimeUnavailable,match='missing'):runtime.call(request,tmp_path/'second',dispatch_key='b'*24)
    assert len(service.submissions)==1


def test_changed_request_cannot_reuse_pending_dispatch_key(tmp_path):
    src=tmp_path/'src';src.mkdir();request,_=scientific_request(src);service=Service(None,fail=True)
    runtime=GalaxyLiveRuntime(tmp_path/'runtime',service);runtime.find_dispatch=lambda _:None
    with pytest.raises(RuntimeUnavailable):runtime.call(request,tmp_path/'first',dispatch_key='c'*24)
    changed=dict(request,upper=8)
    with pytest.raises(ValueError,match='different live inputs'):runtime.call(changed,tmp_path/'next',dispatch_key='c'*24)
