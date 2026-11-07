"""Artificial contracts: uploaded digests are not native production evidence."""
import copy,json
from pathlib import Path

import pytest

from gpx_recipe.preparation import authority,certificates,kernel_runtime as runtime
from gpx_recipe.preparation.tof_kernel_arrays import encode
from gpx_recipe.symmetry.compiler import file_sha,package_signature
from gpx_recipe.symmetry.contracts import digest
from test_tof_kernel_storage import data


def record():
    return dict(schema=runtime.SCHEMA,native=dict(kernel_sha256='a'*64,binary_sha256='b'*64),
        numerical=dict(schema='radar-numerical-runtime/v1',system='Linux',machine='x86_64',byteorder='little'),
        libc=['glibc','test'],libraries={'pypowder.so':'b'*64},
        cpu=[{'vendor_id':'test','cpu family':'6','model':'1','stepping':'1','flags':'sse2'}],
        floating=[0x37f,0x1f80],native_entry_offset=123)


@pytest.mark.parametrize('change',['schema','extra','native','missing_binary','offset','floating',
                                  'status_flags','cpu','system','oversized','library_path'])
def test_runtime_evidence_is_closed_bounded_and_numeric(change):
    value=record();runtime.validate_record(value)
    if change=='schema':value['schema']='other'
    elif change=='extra':value['verified']=True
    elif change=='native':value['native']['kernel_sha256']='not a digest'
    elif change=='missing_binary':value['libraries']['pypowder.so']='f'*64
    elif change=='offset':value['native_entry_offset']=True
    elif change=='floating':value['floating'][0]=True
    elif change=='status_flags':value['floating'][1]|=1
    elif change=='cpu':value['cpu'][0]['flags']=''
    elif change=='system':value['numerical']['system']='other'
    elif change=='oversized':value['cpu']*=65
    else:value['libraries']['../pypowder.so']=value['libraries'].pop('pypowder.so')
    with pytest.raises(ValueError):runtime.validate_record(value)


def kernel_certificate(root):
    directory=root/'worker';output=directory/'native'/'kernel';output.mkdir(parents=True)
    original=data();(output/'kernel.npz').write_bytes(encode(original))
    (output/'kernel.json').write_text(json.dumps(original.metadata))
    numerical=record();runtime_sha=digest(numerical)
    detail=dict(arrays_sha256=original.metadata['arrays_sha256'],kernel_runtime=numerical,
        kernel_runtime_sha256=runtime_sha,
        physics_sha256=runtime.origin_identity(original.metadata['arrays_sha256'],runtime_sha),
        native_kernel_proof=dict(method='original native kernel construction',
            verified_all_peaks=True,native_peak_calls=2))
    manifest=dict(schema='radar-experiment-preparation-entry/v1',kind='kernel',
        implementation_sha256=package_signature(),observations_included=False,scientific_result=False,
        detail=detail,artifacts={name:file_sha(output/name) for name in ['kernel.npz','kernel.json']})
    (output/'entry.json').write_text(json.dumps(manifest))
    result=dict(schema='radar-experiment-preparation-worker/v1',kind='kernel',status='prepared',
        observations_included=False,scientific_result=False,output=str(output),entry_sha256=file_sha(output/'entry.json'))
    response=dict(schema='radar-model-native-result/v1',result=result,
        files={str(p.relative_to(directory)):file_sha(p) for p in output.iterdir()})
    archive=root/'archive.zip';archive.write_bytes(b'artificial wrapper inventory')
    path=root/'certificate.json';document=certificates.produce(response,directory,archive,path)
    return directory,output,response,archive,path,document


def test_only_server_owned_exact_runtime_and_arrays_authorize_a_kernel(tmp_path):
    from test_preparation_authority import envelope
    _,output,_,_,certificate,document=kernel_certificate(tmp_path)
    detail=json.loads((output/'entry.json').read_text())['detail']
    arrays=detail['arrays_sha256'];native=detail['kernel_runtime_sha256']
    assert authority.kernel_origin(arrays,native) is None
    server,_=envelope(tmp_path,certificate)
    with authority.server_scope(server,tmp_path):
        origin=authority.kernel_origin(arrays,native)
        assert origin['entry_sha256']==document['entries'][0]['entry_sha256']
        assert origin['native_runtime_sha256']==native
        assert authority.kernel_origin('f'*64,native) is None
        assert authority.kernel_origin(arrays,'f'*64) is None
    assert authority.kernel_origin(arrays,native) is None


@pytest.mark.parametrize('child',[False,True])
def test_current_kernel_lookup_checks_authority_once_before_runtime(tmp_path,monkeypatch,child):
    from test_preparation_authority import envelope
    _,output,_,_,certificate,document=kernel_certificate(tmp_path)
    detail=json.loads((output/'entry.json').read_text())['detail']
    runtime_calls=[]
    def fingerprint():runtime_calls.append(1);return detail['kernel_runtime_sha256']
    assert authority.current_kernel_origin(detail['arrays_sha256'],fingerprint) is None and not runtime_calls
    server,_=envelope(tmp_path,certificate)
    with authority.server_scope(server,tmp_path):
        env=authority.child_environment();capsule=Path(env[authority.ENV_PATH]);original=authority.package_signature
        with monkeypatch.context() as patch:
            if child:
                patch.setattr(authority,'_owned',None)
                for k,v in env.items():patch.setenv(k,v)
            calls=[]
            def signature():calls.append(1);return original()
            patch.setattr(authority,'package_signature',signature)
            result=authority.current_kernel_origin(detail['arrays_sha256'],fingerprint)
            assert result['entry_sha256']==document['entries'][0]['entry_sha256']
            assert len(calls)==len(runtime_calls)==1
            assert authority.current_kernel_origin(detail['arrays_sha256'],lambda:'f'*64) is None
            capsule.write_bytes(capsule.read_bytes()+b' ')
            assert authority.current_kernel_origin(detail['arrays_sha256'],fingerprint) is None
            assert len(runtime_calls)==1  # Unverified origins do not fingerprint or authorize anything.


def test_non_kernel_origin_does_not_request_runtime_work(tmp_path):
    from test_preparation_authority import certified,envelope
    *_,certificate,_=certified(tmp_path);server,_=envelope(tmp_path,certificate)
    with authority.server_scope(server,tmp_path):
        assert authority.current_kernel_origin('a'*64,lambda:pytest.fail('no kernel origin')) is None
    with authority.server_scope(None,tmp_path):
        assert authority.current_kernel_origin('a'*64,lambda:pytest.fail('empty outer scope')) is None


@pytest.mark.parametrize('change',['runtime','arrays','partial_proof','record','method'])
def test_wrapper_refuses_inconsistent_kernel_certificates(tmp_path,change):
    directory,output,response,archive,path,_=kernel_certificate(tmp_path)
    before=path.read_bytes();manifest=json.loads((output/'entry.json').read_text());d=manifest['detail']
    if change=='runtime':d['kernel_runtime_sha256']='f'*64
    elif change=='arrays':d['arrays_sha256']='f'*64
    elif change=='partial_proof':d['native_kernel_proof']['native_peak_calls']=1
    elif change=='record':d['kernel_runtime']['native']['binary_sha256']='f'*64
    else:d['native_kernel_proof']['method']='uploaded verification flag'
    (output/'entry.json').write_text(json.dumps(manifest))
    response['result']['entry_sha256']=file_sha(output/'entry.json')
    response['files']['native/kernel/entry.json']=file_sha(output/'entry.json')
    with pytest.raises(ValueError):certificates.produce(response,directory,archive,path)
    assert path.read_bytes()==before


@pytest.mark.parametrize('artifacts',[{'kernel.npz':'a'*64},{'kernel.json':'a'*64},
                                    {'kernel.npz':'a'*64,'kernel.json':'b'*64,'extra.npz':'c'*64}])
def test_kernel_certificate_inventory_is_exact(artifacts):
    document=dict(schema=certificates.SCHEMA,implementation_sha256='a'*64,
        result_archive_sha256='b'*64,entries=[dict(kind='kernel',entry_sha256='c'*64,
            physics_sha256='d'*64,artifacts=artifacts)])
    with pytest.raises(ValueError):certificates.validate(document)
