"""Real native producer exits before a fresh observed-data consumer.

Galaxy creating-job metadata is simulated; the native operation, wrapper
inventory, certificate, subprocess handoff and every curve check are real.
"""
import ctypes,json,os,struct
from pathlib import Path
import shutil,subprocess,sys

import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from gpx_recipe.preparation import authority,certificates,kernel_runtime
from gpx_recipe.preparation import native_tof_kernels as kernels
from gpx_recipe.symmetry.bundle import pack,unpack
from gpx_recipe.symmetry.compiler import file_sha,package_signature,native_environment
from gpx_recipe.execution import write_json


CONSUMER='''import json,sys,numpy as np
from pathlib import Path
from test_native_profile_capture import native_source
from gpx_recipe.preparation import kernel_runtime
from gpx_recipe.preparation.tof_kernel_arrays import read,KernelData,array_identity
from gpx_recipe.preparation.kernel_store import KernelStore
from gpx_recipe.preparation.kernel_geometry import PreparedKernelBank,point_for_cell
from gpx_recipe.structural_live.optimization.native_forward import NativeTOFForward
received,work,mode=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
work.mkdir();_,s=native_source(work/'source','PNT')
manifest=json.loads((received/'native/compiled/entry.json').read_text());detail=manifest['detail']
data=read(received/'native/compiled/kernel.npz',received/'native/compiled/kernel.json')
if mode=='forged':
 arrays={k:v.copy() for k,v in data._arrays.items()};meta=data.metadata
 index=len(arrays['parameters'])//3;arrays['values'][arrays['pointers'][index]]+=1.
 meta['arrays_sha256']=array_identity(arrays);data=KernelData(arrays,meta)
if mode=='other_runtime':
 original=kernel_runtime.fingerprint
 def fingerprint(source):
  record,sha=original(source);return record,'f'*64
 kernel_runtime.fingerprint=fingerprint
if mode=='unavailable_runtime':
 def fingerprint(source):raise kernel_runtime.RuntimeUnavailable('test unavailable runtime')
 kernel_runtime.fingerprint=fingerprint
store=KernelStore.create(work/'store',detail['specification']);store.put(detail['cell_index'],data)
reader=PreparedKernelBank(store,s);engine=NativeTOFForward(s)
point=point_for_cell(s,detail['specification']['geometry']['cells'][detail['cell_index']])
s.fast.native.parms[f':{s.fast.slow.hist.id}:Temperature']=100.
expected=s.native(point).copy();actual=engine.evaluate(point,kernel=reader)
np.testing.assert_array_equal(actual,expected)
proof=reader.cached[2].diagnostics() if reader.cached is not None else None
(work/'result.json').write_text(json.dumps(dict(reader=reader.diagnostics(),forward=engine.diagnostics(),
 proof=proof,reflections=data.metadata['reflection_count'],native_array_identical=True,
 producer_exited=True)))
'''


@pytest.fixture(scope='module')
def producer(tmp_path_factory):
    root=tmp_path_factory.mktemp('kernel-native-origin');source,_=native_source(root/'inputs','PNT')
    project=G.G2Project(gpxfile=str(source));cif=root/'parent.cif';project.phase(0).export_CIF(str(cif))
    packet=root/'packet';packet.mkdir()
    shutil.copyfile(cif,packet/'parent.cif');shutil.copyfile(root/'inputs/profile.instprm',packet/'profile.instprm')
    shutil.copyfile(Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif',packet/'child.cif')
    payload=dict(source='parent.cif',instrument='profile.instprm',child='child.cif',
        grid=dict(kind='PNT',lower=3000.,upper=40000.,bins=1501,bank=1),destination='compiled',
        settings=dict(candidate=dict(id='structural:kernel-control',transformation_matrix=[[-1.,0,0],[0,-1.,0],[0,0,1.]],
            origin_shift=[0,0,0]),mode_radius=.05,cell_radius=.002,cell_index=0))
    names=['parent.cif','profile.instprm','child.cif'];pack(packet,root/'inputs.zip',names)
    message=dict(schema='radar-model-native/v1',operation='preparation-kernel',payload=payload,
        implementation_sha256=package_signature(),expected_source_sha256=file_sha(packet/'parent.cif'),
        dispatch_key='f'*24,timeout=150,input_hashes={name:file_sha(packet/name) for name in names})
    write_json(root/'request.json',message);certificate=root/'certificate.json'
    wrapper=Path(__file__).resolve().parents[2]/'scripts/ndip_model_campaign.py'
    with (root/'wrapper.log').open('w') as log:
        subprocess.run([sys.executable,str(wrapper),'native',str(packet/'parent.cif'),str(root/'inputs.zip'),
            str(root/'request.json'),str(root/'job'),str(root/'results.zip'),
            '--preparation-certificate',str(certificate)],check=True,timeout=180,
            env=native_environment(),stdout=log,stderr=subprocess.STDOUT)
        subprocess.run([sys.executable,'-m','gpx_recipe.preparation.certificates',str(certificate)],
            check=True,timeout=20,env=native_environment(),stdout=log,stderr=subprocess.STDOUT)
    document=certificates.read(certificate);assert len(document['entries'])==1
    assert document['entries'][0]['kind']=='kernel'
    received=unpack(root/'results.zip',root/'received')
    entry=json.loads((received/'native/compiled/entry.json').read_text())
    assert entry['observations_included'] is False and entry['scientific_result'] is False
    assert set(entry['artifacts'])=={'kernel.npz','kernel.json'}
    server=root/'server.json';write_json(server,dict(schema=authority.SCHEMA,records=[dict(
        certificate_path=str(certificate),tool_id='neutrons_radar_pd_model_campaign_prototype',
        tool_version='0.2.17',job_state='ok',dataset_state='ok',dataset_deleted=False,dataset_purged=False,
        job_stdout=(root/'wrapper.log').read_text())]))
    return root,received,server


@pytest.mark.parametrize('mode',['authorized','unattributed','forged','other_runtime','unavailable_runtime'])
def test_cold_worker_checks_origin_runtime_and_current_native_curve(producer,tmp_path,mode):
    _,received,server=producer
    with authority.server_scope(None if mode=='unattributed' else server,tmp_path):
        environment=native_environment()
        environment['PYTHONPATH']=os.pathsep.join([environment['PYTHONPATH'],str(Path(__file__).parent),
                                                 os.environ.get('PYTHONPATH','')])
        with (tmp_path/'consumer.log').open('w') as log:
            subprocess.run([sys.executable,'-c',CONSUMER,str(received),str(tmp_path/'consumer'),mode],
                check=True,timeout=120,env=environment,stdout=log,stderr=subprocess.STDOUT)
    result=json.loads((tmp_path/'consumer/result.json').read_text());proof=result['proof']
    assert result['producer_exited'] and result['native_array_identical']
    if mode=='authorized':
        assert proof['method']=='verified native origin and current-runtime probes'
        assert 1<=proof['native_peak_calls']<=7<result['reflections']
        assert proof['verified_all_peaks'] is False and proof['verified_all_peaks_at_origin']
    elif mode=='forged':
        assert proof is None and result['forward']['kernel_hits']==0
        assert result['reader']['bypasses'] and result['forward']['peak_calls']>0
    else:
        assert proof['method']=='complete original native peak verification'
        assert proof['verified_all_peaks'] and proof['native_peak_calls']==result['reflections']


@pytest.mark.parametrize('control',['rounding','flush_to_zero'])
def test_native_qualification_does_not_cross_a_changed_floating_control_state(tmp_path,control):
    _,s=native_source(tmp_path/'source','PNT');point=np.zeros(len(s.radii));kernel=kernels.capture(s,point)
    data=kernel.data._arrays;i=0;low,high=map(int,data['bounds'][i]);params=data['parameters'][i]
    fn=kernel._function;old=kernel_runtime.floating_state();assert old is not None
    lib=ctypes.CDLL('libm.so.6');saved=ctypes.create_string_buffer(256)
    lib.fegetenv.argtypes=[ctypes.c_void_p];lib.fegetenv.restype=ctypes.c_int
    lib.fesetenv.argtypes=[ctypes.c_void_p];lib.fesetenv.restype=ctypes.c_int
    assert lib.fegetenv(saved)==0
    try:
        if control=='rounding':
            lib.fesetround.argtypes=[ctypes.c_int];lib.fesetround.restype=ctypes.c_int
            assert lib.fesetround(0x400)==0
        else:
            raw=bytearray(saved.raw);struct.pack_into('<I',raw,28,struct.unpack_from('<I',raw,28)[0]^0x8000)
            changed=ctypes.create_string_buffer(bytes(raw));assert lib.fesetenv(changed)==0
        assert kernel_runtime.floating_state()!=old
        assert kernel.lookup(tuple(data['hkls'][i]),params,data['axis'],low,high,function=fn) is None
        assert kernels._key(kernel.data,fn) not in kernels._VERIFIED
    finally:assert lib.fesetenv(saved)==0
    assert kernel_runtime.floating_state()==old


def test_complete_requalification_recalculates_even_an_already_owned_kernel(tmp_path):
    _,s=native_source(tmp_path/'source','PNT');kernel=kernels.capture(s,np.zeros(len(s.radii)))
    cached=kernels.verify(kernel.data,s)
    assert cached.diagnostics()['native_peak_calls']==0
    fresh=kernels.verify_complete(kernel.data,s)
    assert fresh.diagnostics()['native_peak_calls']==kernel.data.metadata['reflection_count']
    assert fresh.diagnostics()['verified_all_peaks']
    assert fresh.diagnostics()['method']=='complete original native peak verification'


def test_replaced_compiled_callable_cannot_reuse_an_old_qualified_peak(tmp_path,monkeypatch):
    _,s=native_source(tmp_path/'source','PNT');point=np.zeros(len(s.radii));kernel=kernels.capture(s,point)
    data=kernel.data._arrays;low,high=map(int,data['bounds'][0])
    monkeypatch.setattr(kernel._compiled_module,'pyepsvoigt',lambda *args:pytest.fail('unexpected native call'))
    assert kernel.lookup(tuple(data['hkls'][0]),data['parameters'][0],data['axis'],low,high,
                         function=kernel._function) is None
