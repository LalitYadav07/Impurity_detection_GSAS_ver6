"""Cold native reuse requires creating-job evidence, never uploaded claims."""
import copy
import json
from pathlib import Path

import pytest

from gpx_recipe.preparation import authority, certificates
from gpx_recipe.symmetry.compiler import file_sha, package_signature


def certified(tmp_path, *, batch=False,kind='structural'):
    root=tmp_path/'worker';output=root/'native'/'prepared';output.mkdir(parents=True)
    names=('starts.npz','starts.json') if kind=='structural' else ('response.npz','scattering.npz')
    for name in names:(output/name).write_bytes(b'public artificial native output '+name.encode())
    detail=dict(physics_sha256='a'*64)
    if kind=='structural':
        detail['starting_curves']=2
        (output/'starts.json').write_text(json.dumps(dict(curve_role='native',
            interpolation=False,native_curves=2,sha256=file_sha(output/'starts.npz'))))
    if kind=='magnetic':detail['scattering']=dict(status='prepared',artifact='scattering.npz',identity='b'*64)
    manifest=dict(schema='radar-experiment-preparation-entry/v1',kind=kind,
        implementation_sha256=package_signature(),observations_included=False,scientific_result=False,
        detail=detail,artifacts={name:file_sha(output/name) for name in names})
    (output/'entry.json').write_text(json.dumps(manifest))
    result=dict(schema='radar-experiment-preparation-worker/v1',status='prepared',kind=kind,
        observations_included=False,scientific_result=False,output=str(output),entry_sha256=file_sha(output/'entry.json'))
    if batch:
        result=dict(schema='radar-preparation-batch-result/v1',entries=[
            dict(state='prepared',result=result),dict(state='pending'),dict(state='failed')])
    response=dict(schema='radar-model-native-result/v1',result=result,
        files={str(p.relative_to(root)):file_sha(p) for p in output.iterdir()})
    archive=tmp_path/'output.zip';archive.write_bytes(b'artificial packed bytes')
    certificate=tmp_path/'certificate.json'
    document=certificates.produce(response,root,archive,certificate)
    return root,output,response,archive,certificate,document


@pytest.mark.parametrize('change',['proposal','interpolation','count'])
def test_structural_proposals_cannot_receive_native_production_authority(tmp_path,change):
    root,output,response,archive,path,_=certified(tmp_path)
    receipt=json.loads((output/'starts.json').read_text())
    if change=='proposal':receipt.update(curve_role='starting_proposals',native_curves=0,starting_curves=2)
    elif change=='interpolation':receipt['interpolation']=True
    else:receipt['native_curves']=1
    (output/'starts.json').write_text(json.dumps(receipt))
    manifest=json.loads((output/'entry.json').read_text())
    manifest['artifacts']['starts.json']=file_sha(output/'starts.json')
    (output/'entry.json').write_text(json.dumps(manifest))
    response['result']['entry_sha256']=file_sha(output/'entry.json')
    response['files']={str(p.relative_to(root)):file_sha(p) for p in output.iterdir()}
    with pytest.raises(ValueError,match='native structural banks'):
        certificates.produce(response,root,archive,path)


def envelope(tmp_path, certificate):
    row=dict(certificate_path=str(certificate),tool_id='neutrons_radar_pd_model_campaign_prototype',
        tool_version='0.2.17',job_state='ok',dataset_state='ok',dataset_deleted=False,dataset_purged=False,
        job_stdout=certificates.MARKER+file_sha(certificate)+'\n')
    path=tmp_path/'server.json';path.write_text(json.dumps(dict(schema=authority.SCHEMA,records=[row])))
    return path,row


@pytest.mark.parametrize('version',['0.2.17','0.2.18','0.2.15','unknown'])
def test_only_reviewed_receipt_producer_versions_are_recognized(tmp_path,version):
    *_,certificate,_=certified(tmp_path)
    path,row=envelope(tmp_path,certificate);row['tool_version']=version
    path.write_text(json.dumps(dict(schema=authority.SCHEMA,records=[row])))
    accepted,rejected=authority.verify_envelope(path)
    if version in {'0.2.17','0.2.18'}:assert len(accepted)==1 and not rejected
    else:assert not accepted and rejected


def test_magnetic_authority_applies_only_to_exact_native_amplitude_bytes(tmp_path):
    *_,certificate,document=certified(tmp_path,kind='magnetic');path,_=envelope(tmp_path,certificate)
    entry=document['entries'][0];factor_sha=entry['artifacts']['scattering.npz']
    assert authority.magnetic_factor_origin(factor_sha) is None
    with authority.server_scope(path,tmp_path):
        origin=authority.magnetic_factor_origin(factor_sha)
        assert origin['certificate_sha256']==file_sha(certificate)
        assert origin['entry_sha256']==entry['entry_sha256']
        assert origin['reference_physics_sha256']==entry['physics_sha256']
        assert authority.magnetic_factor_origin(entry['artifacts']['response.npz']) is None
        assert authority.magnetic_factor_origin('f'*64) is None
        assert authority.magnetic_factor_origin(None) is None
    assert authority.magnetic_factor_origin(factor_sha) is None


def test_a_structural_origin_cannot_authorize_magnetic_scattering(tmp_path):
    *_,certificate,document=certified(tmp_path);path,_=envelope(tmp_path,certificate)
    with authority.server_scope(path,tmp_path):
        for sha in document['entries'][0]['artifacts'].values():
            assert authority.magnetic_factor_origin(sha) is None


@pytest.mark.parametrize('mutation',['failed_job','changed_certificate','old_implementation'])
def test_magnetic_reuse_rejects_missing_or_changed_producer_evidence(tmp_path,mutation):
    *_,certificate,document=certified(tmp_path,kind='magnetic');path,row=envelope(tmp_path,certificate)
    factor_sha=document['entries'][0]['artifacts']['scattering.npz']
    if mutation=='failed_job':
        row['job_state']='error';path.write_text(json.dumps(dict(schema=authority.SCHEMA,records=[row])))
    elif mutation=='changed_certificate':certificate.write_text(certificate.read_text()+' ')
    else:
        document['implementation_sha256']='f'*64;certificate.write_text(json.dumps(document))
        row['job_stdout']=certificates.MARKER+file_sha(certificate)
        path.write_text(json.dumps(dict(schema=authority.SCHEMA,records=[row])))
    with authority.server_scope(path,tmp_path) as report:
        assert report['verified_certificates']==0
        assert authority.magnetic_factor_origin(factor_sha) is None


@pytest.mark.parametrize('batch',[False,True])
def test_wrapper_certifies_only_completed_portable_preparation(tmp_path,batch,capsys):
    _,_,_,archive,path,document=certified(tmp_path,batch=batch)
    assert len(document['entries'])==1 and document['entries'][0]['kind']=='structural'
    assert document['result_archive_sha256']==file_sha(archive)
    certificates.announce(path)
    assert capsys.readouterr().out==certificates.MARKER+file_sha(path)+'\n'
    assert set(document)=={'schema','implementation_sha256','result_archive_sha256','entries'}


@pytest.mark.parametrize('change',['array','published_inventory','manifest_kind','implementation','observation','escaped'])
def test_certificate_rejects_changed_or_non_theoretical_outputs(tmp_path,change):
    root,output,response,archive,path,_=certified(tmp_path)
    manifest=json.loads((output/'entry.json').read_text())
    if change=='array':(output/'starts.npz').write_bytes(b'changed')
    elif change=='published_inventory':response['files']['native/prepared/starts.npz']='f'*64
    elif change=='escaped':response['result']['output']=str(tmp_path)
    else:
        if change=='manifest_kind':manifest['kind']='magnetic'
        elif change=='implementation':manifest['implementation_sha256']='f'*64
        else:manifest['observations_included']=True
        (output/'entry.json').write_text(json.dumps(manifest))
        response['result']['entry_sha256']=file_sha(output/'entry.json')
    before=path.read_bytes()
    with pytest.raises(ValueError):certificates.produce(response,root,archive,path)
    assert path.read_bytes()==before


def test_non_preparation_job_publishes_an_empty_data_only_certificate(tmp_path):
    archive=tmp_path/'result.zip';archive.write_bytes(b'not a preparation')
    document=certificates.produce({'schema':'ordinary-model-result'},tmp_path,archive,tmp_path/'receipt.json')
    assert document['entries']==[]


@pytest.mark.parametrize('field,value',[
    ('tool_id','upload1'),('tool_version','0.2.16'),('job_state','running'),
    ('dataset_state','error'),('dataset_deleted',True),('dataset_purged',True),
    ('job_stdout',''),('job_stdout',certificates.MARKER+'f'*64),
    ('job_stdout',certificates.MARKER+'f'*64+'\n'+certificates.MARKER+'a'*64),
])
def test_uploads_failed_jobs_and_self_declared_digests_do_not_authorize_reuse(tmp_path,field,value):
    *_,certificate,_=certified(tmp_path)
    path,row=envelope(tmp_path,certificate);row[field]=value
    path.write_text(json.dumps(dict(schema=authority.SCHEMA,records=[row])))
    accepted,rejected=authority.verify_envelope(path)
    assert accepted==[] and len(rejected)==1


def test_certificate_mutation_or_old_scientific_implementation_is_not_authority(tmp_path):
    *_,certificate,document=certified(tmp_path)
    path,row=envelope(tmp_path,certificate)
    document['implementation_sha256']='f'*64;certificate.write_text(json.dumps(document))
    assert authority.verify_envelope(path)[0]==[]
    row['job_stdout']=certificates.MARKER+file_sha(certificate)
    path.write_text(json.dumps(dict(schema=authority.SCHEMA,records=[row])))
    assert authority.verify_envelope(path)[0]==[]


def test_authority_is_scoped_propagated_and_cleaned_without_changing_process_environment(tmp_path,monkeypatch):
    from gpx_recipe.jobs import scientific_environment
    *_,certificate,document=certified(tmp_path)
    path,_=envelope(tmp_path,certificate)
    monkeypatch.setenv('AZURE_ILLUMINE_API_KEY','PRIVATE-DO-NOT-PROPAGATE')
    monkeypatch.setenv('BCS_API_KEY','PRIVATE-DO-NOT-PROPAGATE')
    import os
    original=dict(os.environ)
    with authority.server_scope(path,tmp_path) as report:
        assert report['verified_certificates']==1
        env=scientific_environment()
        capsule=Path(env[authority.ENV_PATH]);assert capsule.is_file()
        assert capsule.stat().st_mode&0o777==0o600
        assert env[authority.ENV_HASH]==file_sha(capsule)
        assert 'AZURE_ILLUMINE_API_KEY' not in env and 'BCS_API_KEY' not in env
        entry=document['entries'][0]
        origin=authority.structural_bank_origin(entry['artifacts']['starts.npz'],entry['artifacts']['starts.json'],entry['physics_sha256'])
        assert origin['entry_sha256']==entry['entry_sha256']
        assert authority.structural_bank_origin('f'*64,entry['artifacts']['starts.json'],entry['physics_sha256']) is None
        assert authority.structural_bank_origin(entry['artifacts']['starts.npz'],entry['artifacts']['starts.json'],'f'*64) is None
        assert authority._inherited(env) is not None
        assert dict(os.environ)==original
        with pytest.raises(RuntimeError,match='owns this worker'):
            with authority.server_scope(None,tmp_path):pass
    assert not capsule.exists() and authority.child_environment()=={}
    assert authority._inherited(env) is None


def test_empty_outer_worker_scope_suppresses_inherited_capsule(tmp_path,monkeypatch):
    *_,certificate,_=certified(tmp_path);path,_=envelope(tmp_path,certificate)
    with authority.server_scope(path,tmp_path):env=authority.child_environment();raw=Path(env[authority.ENV_PATH]).read_bytes()
    # Simulate a scientific child that legitimately inherited its launcher's
    # still-present capability. A new outer wrapper must not accept it again.
    capsule=Path(env[authority.ENV_PATH]);capsule.parent.mkdir();capsule.write_bytes(raw)
    for k,v in env.items():monkeypatch.setenv(k,v)
    assert authority.child_environment()==env
    with authority.server_scope(None,tmp_path):
        assert authority.child_environment()=={}
        assert authority.structural_bank_origin('a'*64,'b'*64,'c'*64) is None


def test_actual_scientific_child_receives_only_the_owned_origin_scope(tmp_path):
    import subprocess,sys
    from gpx_recipe.jobs import scientific_environment
    *_,certificate,document=certified(tmp_path);path,_=envelope(tmp_path,certificate)
    entry=document['entries'][0]
    arguments=[entry['artifacts']['starts.npz'],entry['artifacts']['starts.json'],entry['physics_sha256']]
    program="import json,sys;from gpx_recipe.preparation.authority import structural_bank_origin;print(json.dumps(structural_bank_origin(*sys.argv[1:])))"
    with authority.server_scope(path,tmp_path):
        result=subprocess.check_output([sys.executable,'-c',program,*arguments],env=scientific_environment(),text=True)
    assert json.loads(result)['entry_sha256']==entry['entry_sha256']


@pytest.mark.parametrize('kind',['structural','magnetic','rendered'])
@pytest.mark.parametrize('child',[False,True])
def test_each_lookup_checks_current_capsule_once_and_does_not_cache_trust(tmp_path,monkeypatch,kind,child):
    import os
    *_,certificate,document=certified(tmp_path,kind='magnetic' if kind!='structural' else 'structural')
    path,_=envelope(tmp_path,certificate);entry=document['entries'][0]
    if kind=='structural':
        lookup=lambda:authority.structural_bank_origin(entry['artifacts']['starts.npz'],
            entry['artifacts']['starts.json'],entry['physics_sha256'])
    elif kind=='magnetic':lookup=lambda:authority.magnetic_factor_origin(entry['artifacts']['scattering.npz'])
    else:lookup=lambda:authority.magnetic_response_origin(entry['artifacts']['response.npz'],entry['physics_sha256'])
    with authority.server_scope(path,tmp_path):
        env=authority.child_environment();capsule=Path(env[authority.ENV_PATH]);original=authority.package_signature
        with monkeypatch.context() as patch:
            if child:
                patch.setattr(authority,'_owned',None)
                for k,v in env.items():patch.setenv(k,v)
            calls=[]
            def signature():calls.append(1);return original()
            patch.setattr(authority,'package_signature',signature)
            assert lookup()['entry_sha256']==entry['entry_sha256']
            assert len(calls)==1
            # A second request is a fresh check, including in an owned scope.
            assert lookup() is not None and len(calls)==2
            capsule.write_bytes(capsule.read_bytes()+b' ')
            assert lookup() is None
    assert lookup() is None


@pytest.mark.parametrize('change',['schema','unexpected','duplicate_entry','oversized_entries','artifact_name'])
def test_certificate_schema_is_closed_and_bounded(tmp_path,change):
    *_,document=certified(tmp_path);document=copy.deepcopy(document)
    if change=='schema':document['schema']='unknown'
    elif change=='unexpected':document['trust_me']=True
    elif change=='duplicate_entry':document['entries']*=2
    elif change=='oversized_entries':document['entries']*=257
    else:document['entries'][0]['artifacts']['../escape.npz']='a'*64
    with pytest.raises(ValueError):certificates.validate(document)
