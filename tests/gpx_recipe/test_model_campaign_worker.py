"""Scientific identities survive shared staging, scheduling and viewer loss."""
import copy
import json
from pathlib import Path
import threading
import time
import zipfile

import pytest

from gpx_recipe.symmetry import chunks
from gpx_recipe.symmetry.bundle import pack,unpack,hydrate_source
from gpx_recipe.symmetry.campaign import Campaign,verify_receipt
from gpx_recipe.symmetry.compiler import package_signature,job_inputs,file_sha
from gpx_recipe.symmetry.contracts import digest,iter_trials,trial_id,validate_protocol
from gpx_recipe.symmetry.resources import allocation
from gpx_recipe.symmetry.templates import make_template,rebind
from test_symmetry_contracts import make_protocol


@pytest.fixture
def model(tmp_path):
    root=tmp_path/'source';p=make_protocol(root);p['implementation_sha256']=package_signature()
    p['seeds']=[dict(p['seeds'][0],id=str(i)) for i in range(8)];p['policy']['maximum_trials']=8
    return root,validate_protocol(p)


def test_shared_capture_is_once_thin_portable_and_tamper_checked(model,tmp_path):
    source,p=model;c=Campaign(tmp_path/'campaign',shared_inputs=True)
    ids=c.enroll(p,source,approved_hash=digest(p))
    assert len(list(c.root.rglob('source.gpx')))==1
    assert not list((c.root/'trials').rglob('runtime'))
    assert not list((c.root/'trials').rglob('catalog.json'))
    assert c.enroll(p,source,approved_hash=digest(p))==ids
    c.verify_sealed_continuation()
    directory=c.root/'trials'/ids[0];job=json.loads((directory/'job.json').read_text())
    captured=job_inputs(directory,job);(captured/'source.gpx').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='sealed input'):c.verify_sealed_continuation()


def test_shared_entrypoint_and_layout_cannot_be_replaced(model,tmp_path):
    source,p=model;c=Campaign(tmp_path/'campaign',shared_inputs=True)
    uid=c.enroll(p,source,approved_hash=digest(p))[0]
    directory=c.root/'trials'/uid;job=json.loads((directory/'job.json').read_text())
    with pytest.raises(ValueError,match='layout'):job_inputs(tmp_path/uid,job)
    (directory/'worker.py').write_text('print("changed")')
    with pytest.raises(ValueError,match='entry point'):c.verify_sealed_continuation()


def test_restore_prevalidates_full_denominator_before_mutating(model,tmp_path):
    source,p=model;c=Campaign(tmp_path/'original',shared_inputs=True);c.enroll(p,source,approved_hash=digest(p))
    snapshot=c.snapshot();snapshot['trials'].pop()
    restored=Campaign(tmp_path/'restored',shared_inputs=True)
    with pytest.raises(ValueError,match='omits'):restored.restore(snapshot,{digest(p):source})
    assert restored.trials()==[]
    snapshot=c.snapshot();snapshot['trials'][0]['state']='invented'
    with pytest.raises(ValueError,match='Unknown'):restored.restore(snapshot,{digest(p):source})
    assert restored.trials()==[]
    restored.restore(c.snapshot(),{digest(p):source})
    assert [r['id'] for r in restored.trials()]==[r['id'] for r in c.trials()]
    restored.verify_sealed_continuation()


@pytest.mark.parametrize('slots,quota,memory,expected',[
    (None,'max 100000',8192,1),('4','max 100000',8192,4),
    ('16','250000 100000',16384,2),('8','max 100000',3072,2),
    ('invalid','max 100000',4096,1),('0','max 100000',4096,1),
    (None,'400000 100000',8192,4),(None,'400000 100000',3072,2),
    (None,'50000 100000',4096,1)])
def test_allocation_uses_entitlement_and_memory(tmp_path,slots,quota,memory,expected,monkeypatch):
    monkeypatch.setattr('gpx_recipe.symmetry.resources.affinity_limit',lambda:128)
    (tmp_path/'cpu.max').write_text(quota);(tmp_path/'memory.max').write_text(str(memory*1024**2))
    env={} if slots is None else {'GALAXY_SLOTS':slots}
    value=allocation(16,environ=env,cgroup=tmp_path)
    assert value['workers']==expected and value['blas_threads']==1


def test_cgroup_v1_cpu_grant_and_memory_limit_are_respected(tmp_path,monkeypatch):
    monkeypatch.setattr('gpx_recipe.symmetry.resources.affinity_limit',lambda:128)
    cpu=tmp_path/'cpu';memory=tmp_path/'memory';cpu.mkdir();memory.mkdir()
    (cpu/'cpu.cfs_quota_us').write_text('600000')
    (cpu/'cpu.cfs_period_us').write_text('100000')
    (memory/'memory.limit_in_bytes').write_text(str(5*1024**3))
    value=allocation(8,environ={},cgroup=tmp_path)
    assert value['workers']==4  # 512 MiB reserve plus four 1024 MiB workers
    assert value['allocation_source']=='container_cpu_quota'
    assert value['allocation_reported'] and value['memory_mb']==5120


def test_visible_host_cpus_and_unlimited_v1_controls_grant_no_workers(tmp_path,monkeypatch):
    monkeypatch.setattr('os.cpu_count',lambda:128)
    (tmp_path/'cpu.cfs_quota_us').write_text('-1')
    (tmp_path/'cpu.cfs_period_us').write_text('100000')
    (tmp_path/'memory.limit_in_bytes').write_text(str(2**63-4096))
    value=allocation(16,environ={},cgroup=tmp_path)
    assert value['workers']==1 and not value['allocation_reported']
    assert value['memory_mb'] is None


def test_affinity_can_limit_but_cannot_grant_workers(tmp_path,monkeypatch):
    monkeypatch.setattr('gpx_recipe.symmetry.resources.affinity_limit',lambda:2)
    assert allocation(8,environ={'GALAXY_SLOTS':'8'},cgroup=tmp_path)['workers']==2
    assert allocation(8,environ={},cgroup=tmp_path)['workers']==1


@pytest.mark.parametrize('value',[True,0,17,1.5,'4'])
def test_invalid_worker_count_fails(value,tmp_path):
    with pytest.raises(ValueError):allocation(value,environ={},cgroup=tmp_path)


def test_insufficient_allocated_memory_is_explicit(tmp_path):
    with pytest.raises(ValueError,match='1536'):allocation(4,environ={'GALAXY_SLOTS':'4','GALAXY_MEMORY_MB':'1024'},cgroup=tmp_path)


def request(p):
    return dict(schema='radar-model-chunk/v1',protocol=p,protocol_sha256=digest(p),
        trials=[trial_id(p,c['id'],s['id']) for c,s in iter_trials(p)],workers=4,dispatch_key='d'*24,seconds=30)


def test_chunk_pool_is_bounded_replenished_and_preserves_coverage(model,tmp_path,monkeypatch):
    source,p=model;active=0;maximum=0;lock=threading.Lock();order=[]
    monkeypatch.setattr(chunks,'allocation',lambda requested,environ=None:dict(workers=3,blas_threads=1))
    def runner(directory,*,cancel):
        nonlocal active,maximum
        with lock:active+=1;maximum=max(maximum,active);order.append(directory.name)
        time.sleep(.02 if len(order)>1 else .07)
        job=json.loads((directory/'job.json').read_text())
        (directory/'result.json').write_text(json.dumps(dict(schema='radar-symmetry-trial/v1',id=job['id'],
            protocol_sha256=job['protocol_sha256'],status='technical_failure',reason='test disposition')))
        verify_receipt(directory,dict(id=job['id'],protocol_sha256=job['protocol_sha256']))
        with lock:active-=1
        return dict(state='finished',status='technical_failure')
    result=chunks.execute_chunk(request(p),source,tmp_path/'worker',runner=runner)
    assert maximum==3 and result['pending']==[] and len(result['trials'])==8
    assert {r['id'] for r in result['trials']}==set(request(p)['trials'])
    assert len(list((tmp_path/'worker').rglob('source.gpx')))==1


def test_soft_deadline_and_cancel_keep_unstarted_work_pending(model,tmp_path,monkeypatch):
    source,p=model;clock=[0.]
    monkeypatch.setattr(chunks,'allocation',lambda requested,environ=None:dict(workers=1))
    def runner(directory,*,cancel):clock[0]=40.;return dict(state='finished',status='test')
    result=chunks.execute_chunk(request(p),source,tmp_path/'deadline',runner=runner,clock=lambda:clock[0])
    assert len(result['trials'])==1 and len(result['pending'])==7
    event=threading.Event();event.set()
    result=chunks.execute_chunk(request(p),source,tmp_path/'cancelled',runner=runner,cancel=event)
    assert not result['trials'] and len(result['pending'])==8


@pytest.mark.parametrize('mutate',[
    lambda r:r.update(protocol_sha256='0'*64),lambda r:r['trials'].append(r['trials'][0]),
    lambda r:r['trials'].append('0'*24),lambda r:r.update(seconds=True),
    lambda r:r.update(script='arbitrary'),lambda r:r.update(dispatch_key='../bad')])
def test_chunk_request_cannot_change_approved_work(model,mutate):
    _,p=model;r=request(p);mutate(r)
    with pytest.raises(ValueError):chunks.validate_request(r)


@pytest.mark.parametrize('name',['../source.gpx','/absolute.gpx','a\\b.gpx','runner.py'])
def test_data_packet_rejects_escape_and_code(name,tmp_path):
    archive=tmp_path/'packet.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr(name,b'data')
    with pytest.raises(ValueError):unpack(archive,tmp_path/'out')


def test_packet_roundtrip_and_duplicate_paths(tmp_path):
    (tmp_path/'source.gpx').write_bytes(b'project');pack(tmp_path,tmp_path/'packet.zip',['source.gpx'])
    assert (unpack(tmp_path/'packet.zip',tmp_path/'out')/'source.gpx').read_bytes()==b'project'
    with zipfile.ZipFile(tmp_path/'duplicate.zip','w') as z:z.writestr('A.gpx',b'a');z.writestr('a.gpx',b'b')
    with pytest.raises(ValueError,match='duplicate'):unpack(tmp_path/'duplicate.zip',tmp_path/'duplicates')


def test_missing_gpx_copies_are_hydrated_only_from_the_exact_galaxy_source(tmp_path,monkeypatch):
    source=tmp_path/'galaxy.gpx';source.write_bytes(b'exact Galaxy source')
    checksum=file_sha(source);inputs=tmp_path/'inputs';inputs.mkdir()
    hashes={'source.gpx':checksum,'template/source.gpx':checksum}
    hydrate_source(inputs,source,hashes,checksum)
    assert all((inputs/p).read_bytes()==source.read_bytes() for p in hashes)
    # Existing legacy packets are not rewritten.
    (inputs/'source.gpx').write_bytes(b'existing captured bytes')
    hydrate_source(inputs,source,hashes,checksum)
    assert (inputs/'source.gpx').read_bytes()==b'existing captured bytes'
    for bad in ['../outside.gpx','/outside.gpx','escape\\outside.gpx']:
        with pytest.raises(ValueError,match='confined'):hydrate_source(inputs,source,{bad:checksum},checksum)
    with pytest.raises(ValueError):hydrate_source(inputs,source,{'another.gpx':checksum},'0'*64)
    monkeypatch.setattr('gpx_recipe.symmetry.bundle.MAX_BYTES',1)
    with pytest.raises(ValueError,match='expanded-size'):hydrate_source(inputs,source,{'more.gpx':checksum},checksum)


@pytest.mark.parametrize('legacy',[False,True])
def test_galaxy_entrypoint_uses_separate_source_without_reupload_and_preserves_old_packets(tmp_path,monkeypatch,legacy):
    import importlib.util
    script=Path(__file__).resolve().parents[2]/'scripts/ndip_model_campaign.py'
    spec=importlib.util.spec_from_file_location('ndip_model_campaign_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=tmp_path/'source.gpx';source.write_bytes(b'exact native source')
    assets=tmp_path/'assets.zip'
    with zipfile.ZipFile(assets,'w') as z:
        if legacy:z.write(source,'source.gpx')
    request=tmp_path/'request.json'
    request.write_text(json.dumps(dict(schema='radar-model-native/v1',expected_source_sha256=file_sha(source),
        input_hashes={'source.gpx':file_sha(source)},dispatch_key='a'*24)))
    def native(value,inputs,root,cancel):
        assert (inputs/'source.gpx').read_bytes()==source.read_bytes()
        (root/'checked.json').write_text('{"checked":true}')
        return {'checked':True},root
    monkeypatch.setattr(module,'native',native)
    result=module.work('native',source,assets,request,tmp_path/'worker',tmp_path/'output.zip')
    assert result['result']=={'checked':True}
    with zipfile.ZipFile(tmp_path/'output.zip') as z:
        assert not any(p.endswith('source.gpx') for p in z.namelist())


def template(p):
    context=dict(sha256=p['source']['sha256'],runtime=dict(native_stack_sha256='s'*64),
        phases=[dict(name='nuclear',setting_sha256='a'*64,type='nuclear',atoms=[dict(label='Mn1',type='Mn',occupancy=1.,xyz=[0.,0.,0.])],histograms=['PWDR sample'])],
        histograms=[dict(name='PWDR sample',instrument=dict(Type=['PNT','PNT',0],Bank=[2,2,0]))])
    source=dict(id='scan1',experiment='IPTS-1',sample_key='specimen:A',bank=2,source_sha256=p['source']['sha256'])
    return make_template(p,context,source,[(source,'radar')],origin='radar',future=True),source


@pytest.mark.parametrize('change,message',[
    (lambda c,s:s.update(bank=3),'scope'),(lambda c,s:s.update(sample_key='specimen:B'),'scope'),
    (lambda c,s:s.update(source_sha256='f'*64),'revision'),
    (lambda c,s:c['phases'][0].update(setting_sha256='f'*64),'setting'),
    (lambda c,s:c['phases'][0]['atoms'][0].update(occupancy=.5),'chemistry'),
    (lambda c,s:c['runtime'].update(native_stack_sha256='f'*64),'runtime')])
def test_new_acquisition_cannot_inherit_incompatible_template(model,change,message):
    _,p=model;t,s=template(p);context=copy.deepcopy(t['context']);change(context,s)
    with pytest.raises(ValueError,match=message):rebind(t,context,s,'radar')


def test_identical_bytes_in_different_acquisitions_have_distinct_trials(model):
    _,p=model;t,s=template(p);s['id']='scan2'
    rebound=rebind(t,t['context'],s,'radar')
    assert rebound['source']['sha256']==p['source']['sha256']
    assert rebound['acquisition_id']=='scan2' and digest(rebound)!=digest(p)
    assert rebound['template_sha256']==digest(t)


def test_fifty_structural_children_get_their_own_signed_starts(model):
    _,p=model;p['kind']='structural';p['sites']=[];p['candidates']=[];p['seeds']=[]
    for i in range(50):
        candidate=dict(id='struct:'+str(i),irrep='R',child=dict(path='child'+str(i)+'.cif',sha256='1'*64),modes=['R_1'],
            transformation_matrix=[[1,0,0],[0,1,0],[0,0,1]],origin_shift=[0,0,0])
        p['candidates'].append(candidate)
        for sign in [-1,1]:p['seeds'].append(dict(id=f's{i}_'+('minus' if sign<0 else 'plus'),candidate_ids=[candidate['id']],mode_values={'R_1':sign*.1}))
    p['policy']['maximum_trials']=100;p=validate_protocol(p)
    assert len(list(iter_trials(p)))==100
    p['policy']['maximum_trials']=99
    with pytest.raises(ValueError,match='budget'):validate_protocol(p)


def test_models_interleave_and_acquisitions_take_turns_without_changing_ids(model,tmp_path):
    source,p=model;p['candidates'].append(dict(p['candidates'][0],id='mag:second'))
    catalog=json.loads((source/'catalog.json').read_text());catalog['candidates']=p['candidates']
    (source/'catalog.json').write_text(json.dumps(catalog));p['catalog']['sha256']=file_sha(source/'catalog.json')
    p['policy']['maximum_trials']=16;p=validate_protocol(p)
    c=Campaign(tmp_path/'campaign',shared_inputs=True);c.enroll(p,source,approved_hash=digest(p))
    assert [r['candidate'] for r in c.trials()][:4]==['mag:fixture','mag:second']*2
    next_scan=dict(p,acquisition_id='scan2');c.enroll(next_scan,source,approved_hash=digest(next_scan))
    class Batch:
        max_active_jobs=1;chunk_size=2
        def __init__(self):self.submissions=[]
        def submit_many(self,entries,*,workers):
            self.submissions.append(json.loads((entries[0][1]/'job.json').read_text())['protocol_sha256'])
            return {key:dict(dispatch_key='chunk',key=key) for key,_ in entries}
        def poll(self,handle,directory):return dict(state='not_started')
    transport=Batch();c.tick(transport);c.tick(transport)
    assert transport.submissions==[digest(p),digest(next_scan)]
    assert len(c.trials())==32
