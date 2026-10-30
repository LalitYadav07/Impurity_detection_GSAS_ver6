import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import pytest
from radar_pd_nova.models import UtilityActionRecord,RunStatus
from radar_pd_nova.refinement_runtime import GalaxyRefinementRuntime,RuntimeUnavailable
from radar_pd_nova.refinement_sources import file_sha


class Service:
 def __init__(self,root,source):
  self.submissions=[];self.requests=[];self.error=False;self.source=source
  self.archive=root/'results.zip'
  response={'schema':'radar-refinement-worker/v1','status':'inspected','source_sha256':file_sha(source),'files':{}}
  with zipfile.ZipFile(self.archive,'w') as z:z.writestr('response.json',json.dumps(response))
 def upload_json_document(self,value,**kwargs):self.requests.append(value);return 'request-data'
 def submit_utility(self,**kwargs):
  self.submissions.append(kwargs)
  if self.error:raise TimeoutError('lost HTTP acknowledgement')
  return self.job()
 def job(self):return UtilityActionRecord(uid='utility',tool_id='neutrons_radar_pd_refine_recipe_prototype',name='native',galaxy_job_id='Galaxy-job',status=RunStatus.OK,outputs={'result_bundle':'bundle'})
 def _download_dataset(self,uid,target):
  import shutil
  shutil.copy2(self.archive if uid=='bundle' else self.source,target)


def runtime(tmp_path):
 p=tmp_path/'published.gpx';p.write_bytes(b'actual published bytes');service=Service(tmp_path,p)
 rt=GalaxyRefinementRuntime(tmp_path/'workspace',service,poll_seconds=0)
 source=dict(id='source',source_path=str(p),source_sha256=file_sha(p),dataset_id='original-data',record_uid='original-job')
 return rt,service,source


def test_fresh_dispatch_and_acknowledged_recovery_submit_exactly_once_without_crawling_history(tmp_path,monkeypatch):
 rt,s,source=runtime(tmp_path)
 monkeypatch.setattr(rt,'find_dispatch',lambda key:pytest.fail('Fresh or acknowledged work must not crawl every historical job'))
 monkeypatch.setenv('AZURE_ILLUMINE_API_KEY','SENTINEL-NOT-FOR-SCIENTIFIC-WORKER')
 actions=[]
 for index in range(2):
  response=rt.call('inspect',source,tmp_path/f'result-{index}',dispatch_key='a'*24,on_submitted=actions.append)
  assert response['status']=='inspected'
 assert len(s.submissions)==1 and len(s.requests)==1 and len(actions)==2
 assert s.submissions[0]['inputs']['source_gpx']=={'dataset_id':'original-data'}
 assert s.submissions[0]['inputs']['recipe_request']=={'dataset_id':'request-data'}
 assert 'SENTINEL' not in json.dumps(s.requests+s.submissions)


def test_parallel_recipe_and_model_inspections_capture_one_exact_gpx_without_tempfile_collision(tmp_path):
 from concurrent.futures import ThreadPoolExecutor
 import threading
 rt,s,source=runtime(tmp_path);source=dict(source,source_path=None)
 barrier=threading.Barrier(2);body=s.source.read_bytes();destinations=[]
 def download(uid,target):
  destinations.append(target)
  with target.open('wb') as f:
   f.write(body[:8]);f.flush();barrier.wait(timeout=5);f.write(body[8:])
 s._download_dataset=download
 with ThreadPoolExecutor(max_workers=2) as pool:
  results=list(pool.map(rt.resolve,[source,source]))
 assert len(set(destinations))==2
 assert {r['source_sha256'] for r in results}=={source['source_sha256']}
 assert all(Path(r['source_path']).read_bytes()==body for r in results)
 assert not list((rt.root/'downloads').glob('*.partial'))


@pytest.mark.parametrize('failure',['transfer','checksum'])
def test_failed_gpx_capture_never_publishes_or_leaves_a_partial_file(tmp_path,failure):
 rt,s,source=runtime(tmp_path);source=dict(source,source_path=None)
 def download(uid,target):
  target.write_bytes(b'incomplete or changed remote bytes')
  if failure=='transfer':raise TimeoutError('interrupted transfer')
 s._download_dataset=download
 with pytest.raises(TimeoutError if failure=='transfer' else ValueError):rt.resolve(source)
 assert not list((rt.root/'downloads').iterdir())


def test_lost_acknowledgement_is_held_then_recovered_without_second_submission(tmp_path,monkeypatch):
 rt,s,source=runtime(tmp_path);s.error=True
 with pytest.raises(RuntimeUnavailable,match='did not acknowledge'):rt.call('inspect',source,tmp_path/'first',dispatch_key='b'*24)
 monkeypatch.setattr(rt,'find_dispatch',lambda key:s.job())
 assert rt.call('inspect',source,tmp_path/'recovered',dispatch_key='b'*24)['galaxy_job_id']=='Galaxy-job'
 assert len(s.submissions)==1


def test_ambiguous_pending_submission_cannot_be_blindly_retried(tmp_path,monkeypatch):
 rt,s,source=runtime(tmp_path);s.error=True
 with pytest.raises(RuntimeUnavailable):rt.call('inspect',source,tmp_path/'first',dispatch_key='c'*24)
 monkeypatch.setattr(rt,'find_dispatch',lambda key:None)
 with pytest.raises(RuntimeUnavailable,match='No duplicate was submitted'):rt.call('inspect',source,tmp_path/'retry',dispatch_key='c'*24)
 assert len(s.submissions)==1


def test_exact_collection_element_is_required_even_when_only_one_wrong_dataset_exists(tmp_path):
 rt,s,source=runtime(tmp_path);s.collection_elements=lambda key:[dict(id='wrong',name='other.gpx')]
 with pytest.raises(ValueError,match='one exact'):rt.dataset_id(dict(collection_id='collection',collection_name='wanted.gpx'))


def test_legacy_gpx_filename_matches_one_suffixless_collection_element(tmp_path):
 rt,s,source=runtime(tmp_path)
 s.collection_elements=lambda key:[dict(id='wanted',name='03_Final_accepted_joint_cell_model'),dict(id='other',name='other')]
 assert rt.dataset_id(dict(collection_id='collection',collection_name='03_Final_accepted_joint_cell_model.gpx'))=='wanted'
 rt._collections.clear()
 s.collection_elements=lambda key:[dict(id='first',name='03_Final_accepted_joint_cell_model'),dict(id='second',name='03_Final_accepted_joint_cell_model')]
 with pytest.raises(ValueError,match='one exact'):
  rt.dataset_id(dict(collection_id='collection',collection_name='03_Final_accepted_joint_cell_model.gpx'))


def test_galaxy_stage_collection_maps_each_verified_intermediate_gpx(tmp_path):
 rt,s,source=runtime(tmp_path)
 response={'schema':'radar-refinement-worker/v1','status':'needs_review','source_sha256':source['source_sha256'],
           'files':{},'stages':[{'id':'stage-1','gpx':'verified_stage-1.gpx'}]}
 with zipfile.ZipFile(s.archive,'w') as z:z.writestr('response.json',json.dumps(response))
 s.job=lambda:UtilityActionRecord(uid='utility',tool_id='neutrons_radar_pd_refine_recipe_prototype',name='native',
     galaxy_job_id='Galaxy-job',status=RunStatus.OK,outputs={'result_bundle':'bundle','stage_projects':'stages'})
 s.collection_elements=lambda key:[dict(id='intermediate-data',name='stage-1')]
 result=rt.call('run',source,tmp_path/'stage-result',dispatch_key='d'*24)
 assert result['status']=='needs_review'
 assert result['stage_dataset_ids']=={'stage-1':'intermediate-data'}
 assert 'final_dataset_id' not in result


def test_utility_wrapper_has_no_model_or_secret_inputs_and_publishes_only_verified_projects():
 import xml.etree.ElementTree as ET
 p=Path(__file__).resolve().parents[2]/'galaxy/radar_pd_refine_recipe.xml.in';root=ET.parse(p).getroot()
 inputs={e.attrib['name'] for e in root.findall('inputs/param')}
 assert inputs=={'action','source_gpx','recipe_request','dispatch_key'}
 collection=root.find('outputs/collection');assert 'verified_' in ET.tostring(collection,encoding='unicode')
 assert '@IMMUTABLE_BATCH_IMAGE@' in p.read_text()


def test_large_workspace_restores_beyond_generic_json_preview_limit(tmp_path):
 import shutil
 rt,s,source=runtime(tmp_path);s.history_id='history'
 remote=tmp_path/'saved-workspace.gz'
 s.upload_document=lambda path,**kwargs:(shutil.copy2(path,remote) and 'saved')
 s.search_history_datasets=lambda **kwargs:[dict(id='saved',state='ok')]
 s._download_dataset=lambda uid,path:shutil.copy2(remote,path)
 s.load_json_document=lambda uid:pytest.fail('The generic 1 MB preview must not be used for workspace recovery')
 manifest=dict(schema='radar-refinement-workspace/v1',history_id='history',payload='x'*(2*1024*1024))
 rt.save_workspace(manifest)
 assert remote.stat().st_size<10000
 assert rt.recover_workspace()==manifest
