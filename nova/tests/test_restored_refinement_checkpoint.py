"""An exact recovered backup may replace one upload, never new state or access."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import threading

import pytest

from radar_pd_nova.refinement_runtime import GalaxyRefinementRuntime,RuntimeUnavailable,portable_refinement_workspace,workspace_checkpoint_fingerprint
from radar_pd_nova.refinement_workspace import RefinementWorkspace


def document(status='completed'):
    return dict(schema='radar-refinement-workspace/v1',history_id='history',updated_utc='old',
        sources={'source':dict(id='source',source_sha256='a'*64,source_path='/old/source.gpx')},
        drafts={},recipes={'recipe':{'id':'recipe'}},parallel_jobs=2,paused=False,
        jobs={'job':dict(status=status,remote_action={'galaxy_job_id':'native'},directory='/old/job',
                        result=dict(directory='/old/job',execution='execution.json',files={'execution.json':'b'*64},
                                    stage_outcomes=[dict(id='stage',metric_rows=[{'value':1}])]))})


class Service:
    def __init__(self,root,saved):
        self.history_id='history';self.galaxy_url='https://galaxy.invalid'
        self.path=root/'saved.json.gz';self.path.write_bytes(gzip.compress(json.dumps(saved).encode(),mtime=0))
        self.metadata=dict(id='saved',dataset_id='physical',history_id='history',uuid='uuid',
            state='ok',accessible=True,deleted=False,purged=False,file_size=self.path.stat().st_size,hashes=[])
        self.reads=[];self.uploads=[]
    def search_history_datasets(self,**kwargs):return [dict(id='saved',state='ok')]
    def _dataset_metadata(self,identifier):
        assert identifier=='saved';self.reads.append(identifier);return deepcopy(self.metadata)
    def _download_dataset(self,identifier,target):
        assert identifier=='saved';shutil.copyfile(self.path,target)
    def upload_checkpoint_document(self,path,**kwargs):
        self.uploads.append(json.loads(gzip.decompress(Path(path).read_bytes())))
        return 'uploaded-'+str(len(self.uploads))


def create(tmp_path,status='completed'):
    original=portable_refinement_workspace(document(status));service=Service(tmp_path,original)
    runtime=GalaxyRefinementRuntime(tmp_path/'runtime',service)
    workspace=RefinementWorkspace(tmp_path/'workspace',history_id='history',runtime=runtime)
    assert workspace.restore_remote()
    return workspace,runtime,service,original


def test_unchanged_completed_restore_reuses_exact_authorized_backup(tmp_path):
    w,r,s,original=create(tmp_path);assert not s.uploads
    assert w.checkpoint()=='saved' and w.checkpoint()=='saved'
    assert len(s.reads)==3 and not s.uploads  # Capture plus a fresh check on each reuse.
    assert workspace_checkpoint_fingerprint(w.state)==workspace_checkpoint_fingerprint(original)
    receipt=r._recovered_workspace_checkpoint
    assert receipt['downloaded_sha256']==hashlib.sha256(s.path.read_bytes()).hexdigest()
    assert receipt['metadata']['hashes']==[]  # Local SHA is not a manufactured provider hash.
    assert w.state['sources']['source']['source_path']=='/old/source.gpx'
    assert w.state['jobs']['job']['result']['directory']=='/old/job'


def test_rehydrated_verified_metric_rows_reuse_the_same_portable_checkpoint(tmp_path):
    w,r,s,original=create(tmp_path)
    stage=w.state['jobs']['job']['result']['stage_outcomes'][0]
    stage['metric_rows']=[{'value':1}]
    w.save()
    assert workspace_checkpoint_fingerprint(portable_refinement_workspace(w.state))==workspace_checkpoint_fingerprint(original)
    assert w.checkpoint()=='saved'
    assert not s.uploads


@pytest.mark.parametrize('change',['reference','hash','stage'])
def test_changed_metric_artifact_identity_still_uploads(tmp_path,change):
    w,r,s,_=create(tmp_path)
    result=w.state['jobs']['job']['result']
    stage=result['stage_outcomes'][0]
    if change=='reference':stage['metric_rows_source']['artifact']='other.execution.json'
    if change=='hash':result['files']['execution.json']='c'*64
    if change=='stage':stage['id']='different-stage'
    assert w.checkpoint()=='uploaded-1'
    assert len(s.uploads)==1 and r._recovered_workspace_checkpoint is None


def test_unreferenced_inline_metric_rows_remain_durable_changes(tmp_path):
    w,r,s,_=create(tmp_path)
    result=w.state['jobs']['job']['result'];result['files']={}
    stage=result['stage_outcomes'][0];stage.pop('metric_rows_source')
    stage['metric_rows']=[{'value':1}]
    assert w.checkpoint()=='uploaded-1'
    stage['metric_rows'][0]['value']=2
    assert w.checkpoint()=='uploaded-2'
    assert s.uploads[0]['jobs']['job']['result']['stage_outcomes'][0]['metric_rows']==[{'value':1}]
    assert s.uploads[1]['jobs']['job']['result']['stage_outcomes'][0]['metric_rows']==[{'value':2}]


def test_new_sources_remain_a_real_checkpoint_change(tmp_path):
    w,r,s,_=create(tmp_path)
    for index in range(11):
        w.state['sources'][str(index)]={'id':str(index),'source_sha256':'b'*64}
    assert w.checkpoint()=='uploaded-1'
    assert len(s.uploads[0]['sources'])==12


def test_raw_fingerprint_still_detects_hydration_for_restore_ownership():
    original=portable_refinement_workspace(document())
    hydrated=deepcopy(original)
    hydrated['jobs']['job']['result']['stage_outcomes'][0]['metric_rows']=[{'value':1}]
    assert workspace_checkpoint_fingerprint(hydrated)!=workspace_checkpoint_fingerprint(original)
    assert workspace_checkpoint_fingerprint(portable_refinement_workspace(hydrated))==workspace_checkpoint_fingerprint(original)


def test_hydration_does_not_bypass_failed_access_recheck(tmp_path):
    w,r,s,_=create(tmp_path)
    w.state['jobs']['job']['result']['stage_outcomes'][0]['metric_rows']=[{'value':1}]
    s.metadata['accessible']=False
    with pytest.raises(RuntimeUnavailable):w.checkpoint()
    assert not s.uploads and w._checkpoint_hash is None


def test_running_restore_is_changed_durable_state_and_uploads(tmp_path):
    w,r,s,original=create(tmp_path,'running');assert w.state['jobs']['job']['status']=='recovering'
    assert w.checkpoint()=='uploaded-1' and r._recovered_workspace_checkpoint is None
    assert len(s.uploads)==1 and s.uploads[0]['jobs']['job']['status']=='recovering'
    assert w.checkpoint()=='uploaded-1' and len(s.uploads)==1


@pytest.mark.parametrize('change',['paused','approval','path'])
def test_changed_state_including_paths_keeps_ordinary_upload(tmp_path,change):
    w,r,s,_=create(tmp_path)
    if change=='paused':w.state['paused']=True
    if change=='approval':w.state['recipes']['recipe']['approved']={'reviewed':'new'}
    if change=='path':w.state['sources']['source']['source_path']='/new/source.gpx'
    assert w.checkpoint()=='uploaded-1' and len(s.uploads)==1
    assert r._recovered_workspace_checkpoint is None


@pytest.mark.parametrize('field,value',[('state','error'),('accessible',False),('deleted',True),('purged',True),
    ('id','other'),('dataset_id','replaced'),('file_size',1),('hashes',[{'hash_function':'SHA256','hash_value':'f'*64}]),
    ('uuid','other'),('history_id','other')])
def test_lost_access_or_changed_physical_identity_never_reuses_or_uploads(tmp_path,field,value):
    w,r,s,_=create(tmp_path);s.metadata[field]=value
    with pytest.raises(RuntimeUnavailable):w.checkpoint()
    assert not s.uploads and w._checkpoint_hash is None


def test_denied_metadata_recheck_never_becomes_an_upload(tmp_path):
    w,r,s,_=create(tmp_path)
    def denied(_):raise PermissionError('denied')
    s._dataset_metadata=denied
    with pytest.raises(RuntimeUnavailable,match='could not be rechecked'):w.checkpoint()
    assert not s.uploads and w._checkpoint_dataset is None


@pytest.mark.parametrize('change',['history','url','service'])
def test_connection_switch_rejects_reuse_before_metadata_or_upload(tmp_path,change):
    w,r,s,_=create(tmp_path);reads=len(s.reads)
    if change=='history':s.history_id='other'
    if change=='url':s.galaxy_url='https://other.invalid'
    if change=='service':r.service=Service(tmp_path,document())
    with pytest.raises(RuntimeUnavailable,match='another Galaxy connection'):w.checkpoint()
    assert len(s.reads)==reads and not s.uploads


def test_connection_switch_during_metadata_is_not_cached(tmp_path):
    w,r,s,_=create(tmp_path);read=s._dataset_metadata
    def switched(identifier):
        result=read(identifier);s.history_id='other';return result
    s._dataset_metadata=switched
    with pytest.raises(RuntimeUnavailable,match='changed while checking'):w.checkpoint()
    assert not s.uploads and w._checkpoint_hash is None


def test_reused_cache_hits_still_check_deleted_backup(tmp_path):
    w,r,s,_=create(tmp_path);assert w.checkpoint()=='saved';s.metadata['deleted']=True
    with pytest.raises(RuntimeUnavailable):w.checkpoint()
    assert not s.uploads


def test_concurrent_unchanged_checkpoints_share_existing_hda_without_post(tmp_path):
    w,r,s,_=create(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _:w.checkpoint(),range(2)))==['saved','saved']
    assert not s.uploads and len(s.reads)==3


def test_mutation_during_recheck_is_not_marked_durable(tmp_path):
    w,r,s,_=create(tmp_path);entered=threading.Event();release=threading.Event();read=s._dataset_metadata
    def wait(identifier):
        entered.set();assert release.wait(3);return read(identifier)
    s._dataset_metadata=wait
    with ThreadPoolExecutor(max_workers=1) as pool:
        result=pool.submit(w.checkpoint);assert entered.wait(3)
        with w.lock:w.state['paused']=True;w.save()
        release.set();assert result.result(timeout=3)=='saved'
    s._dataset_metadata=read
    assert w.checkpoint()=='uploaded-1' and s.uploads[0]['paused'] is True


def test_restore_download_size_mismatch_does_not_arm_reuse(tmp_path):
    s=Service(tmp_path,document());s.metadata['file_size']+=1
    r=GalaxyRefinementRuntime(tmp_path/'runtime',s)
    with pytest.raises(RuntimeUnavailable,match='size differs'):r.recover_workspace()
    assert r._recovered_workspace_checkpoint is None and not s.uploads


def test_provider_without_metadata_interface_keeps_legacy_upload_behavior(tmp_path):
    original=portable_refinement_workspace(document());s=Service(tmp_path,original);s._dataset_metadata=None
    r=GalaxyRefinementRuntime(tmp_path/'runtime',s);w=RefinementWorkspace(tmp_path/'workspace',history_id='history',runtime=r)
    assert w.restore_remote() and w.checkpoint()=='uploaded-1'
    assert len(s.uploads)==1 and r._recovered_workspace_checkpoint is None


@pytest.mark.parametrize('name',['SHA256','MD5'])
def test_provider_hash_is_verified_against_downloaded_checkpoint(tmp_path,name):
    s=Service(tmp_path,portable_refinement_workspace(document()))
    value=hashlib.new(name.lower(),s.path.read_bytes()).hexdigest()
    s.metadata['hashes']=[dict(hash_function=name,hash_value=value)]
    r=GalaxyRefinementRuntime(tmp_path/'runtime',s);w=RefinementWorkspace(tmp_path/'workspace',history_id='history',runtime=r)
    assert w.restore_remote() and w.checkpoint()=='saved' and not s.uploads


def test_bad_provider_hash_never_arms_reuse(tmp_path):
    s=Service(tmp_path,portable_refinement_workspace(document()))
    s.metadata['hashes']=[dict(hash_function='SHA256',hash_value='f'*64)]
    r=GalaxyRefinementRuntime(tmp_path/'runtime',s)
    with pytest.raises(RuntimeUnavailable,match='advertised hash'):r.recover_workspace()
    assert r._recovered_workspace_checkpoint is None and not s.uploads


@pytest.mark.parametrize('access',['deleted','denied'])
def test_changed_content_does_not_bypass_failed_access_recheck(tmp_path,access):
    w,r,s,_=create(tmp_path);w.state['paused']=True
    if access=='deleted':s.metadata['deleted']=True
    else:
        def denied(_):raise PermissionError('denied')
        s._dataset_metadata=denied
    with pytest.raises(RuntimeUnavailable):w.checkpoint()
    assert not s.uploads and w._checkpoint_hash is None


@pytest.mark.parametrize('changed',['recipe','job','source','runtime'])
def test_concurrent_changes_are_not_overwritten_by_restore(tmp_path,changed):
    s=Service(tmp_path,portable_refinement_workspace(document()))
    r=GalaxyRefinementRuntime(tmp_path/'runtime',s);w=RefinementWorkspace(tmp_path/'workspace',history_id='history',runtime=r)
    entered=threading.Event();release=threading.Event();download=s._download_dataset
    def wait(*args):entered.set();assert release.wait(3);download(*args)
    s._download_dataset=wait
    with ThreadPoolExecutor(max_workers=1) as pool:
        result=pool.submit(w.restore_remote);assert entered.wait(3)
        with w.lock:
            if changed=='recipe':w.state['recipes']['new']={'id':'new'}
            if changed=='job':w.state['jobs']['new']={'status':'queued'}
            if changed=='source':w.state['sources']['new']={'id':'new'}
            if changed=='runtime':w.runtime=object()
            w.save()
        release.set()
        with pytest.raises(RuntimeUnavailable,match='changed during recovery'):result.result(timeout=3)
    assert 'job' not in w.state['jobs'] and 'source' not in w.state['sources']
    assert r._recovered_workspace_checkpoint is None and not s.uploads


def test_timestamp_only_save_does_not_invalidate_restore(tmp_path):
    s=Service(tmp_path,portable_refinement_workspace(document()))
    r=GalaxyRefinementRuntime(tmp_path/'runtime',s);w=RefinementWorkspace(tmp_path/'workspace',history_id='history',runtime=r)
    download=s._download_dataset
    def saved(*args):w.save();download(*args)
    s._download_dataset=saved
    assert w.restore_remote() and w.checkpoint()=='saved' and not s.uploads
