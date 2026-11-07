"""Portable chunk integrity, resumability and bounds do not imply science."""
import concurrent.futures
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pytest

from gpx_recipe.preparation import kernel_store as stores
from gpx_recipe.preparation import tof_kernel_arrays as arrays
from gpx_recipe.preparation.native_tof_kernels import QualifiedKernel


def sample(delta=0.):
    values=dict(hkls=np.array([[1,0,0],[0,1,0]],dtype='<i4'),
        parameters=np.array([[1000,.01,.02,4.,1.],[1100,.02,.03,5.,2.]],dtype='<f8'),
        bounds=np.array([[0,4],[1,3]],dtype='<i8'),pointers=np.array([0,4,6],dtype='<i8'),
        values=np.arange(6,dtype='<f4')+np.float32(delta),axis=np.array([900.,1000.,1100.,1200.],dtype='<f8'))
    meta=dict(schema=arrays.SCHEMA,kernel_sha256='a'*64,binary_sha256='b'*64,
        arrays_sha256=arrays.array_identity(values),reflection_count=2,
        numeric_bytes=sum(v.nbytes for v in values.values()),observations_included=False)
    return values,meta


def data(delta=0.):
    return arrays.KernelData(*sample(delta))


def create(root,count=6):
    return stores.KernelStore.create(root,dict(model_sha256='c'*64,bank='POWGEN bank 2',cell_count=count))


def test_data_roundtrip_is_immutable_and_never_a_native_permission():
    original=data();loaded=arrays.decode(arrays.encode(original),original.metadata)
    assert not hasattr(loaded,'lookup')
    for key in arrays.DTYPES:
        np.testing.assert_array_equal(loaded._arrays[key],original._arrays[key])
        with pytest.raises(ValueError):loaded._arrays[key].flags.writeable=True
    with pytest.raises(TypeError):loaded._arrays['values']=np.zeros(6)
    changed=loaded.metadata;changed['binary_sha256']='f'*64
    assert loaded.metadata['binary_sha256']=='b'*64
    with pytest.raises(TypeError,match='Only native construction'):
        QualifiedKernel(loaded,lambda:None,{'verified':True})


@pytest.mark.parametrize('change',['dtype','bounds','pointer','nonfinite','unordered','duplicate','position','width','inventory','shape'])
def test_malformed_numeric_content_cannot_be_admitted(change):
    a,m=sample()
    if change=='dtype':a['values']=a['values'].astype('f8')
    if change=='bounds':a['bounds'][0,1]=len(a['axis'])+1
    if change=='pointer':a['pointers'][1]=-1
    if change=='nonfinite':a['values'][0]=np.nan
    if change=='unordered':a['axis'][1]=a['axis'][0]
    if change=='duplicate':a['hkls'][1]=a['hkls'][0]
    if change=='position':a['parameters'][0,0]=2000001.
    if change=='width':a['parameters'][0,1]=0.
    if change=='inventory':a['extra']=np.array([1.])
    if change=='shape':a['parameters']=a['parameters'].reshape(-1)
    with pytest.raises(ValueError):arrays.KernelData(a,m)


def test_saved_qualification_and_changed_values_cannot_hide_in_metadata():
    a,m=sample();m['verified']=True
    with pytest.raises(ValueError,match='metadata'):arrays.KernelData(a,m)
    a,m=sample();a['values'][0]+=1
    with pytest.raises(ValueError,match='array content'):arrays.KernelData(a,m)


def test_object_npy_is_rejected_before_numpy_load(monkeypatch):
    stream=io.BytesIO();np.savez(stream,values=np.array([object()]))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Untrusted header reached allocation'))
    with pytest.raises(ValueError):arrays.decode(stream.getvalue(),data().metadata)


def test_regular_file_bounds_and_symlinks(tmp_path):
    p=tmp_path/'data';p.write_bytes(b'12345')
    assert arrays.read_bounded(p,5)==b'12345'
    with pytest.raises(ValueError,match='bound'):arrays.read_bounded(p,4)
    link=tmp_path/'link';link.symlink_to(p)
    with pytest.raises(ValueError,match='regular'):arrays.read_bounded(link,10)
    fifo=tmp_path/'fifo';os.mkfifo(fifo)
    with pytest.raises(ValueError):arrays.read_bounded(fifo,10)


def test_store_resume_and_exact_byte_repair_without_scientific_attestation(tmp_path):
    store=create(tmp_path/'store');original=data();store.put(0,original)
    assert stores.KernelStore(store.root).put(0,original)['state']=='unchanged'
    assert store.coverage()['stored']==1 and store.coverage()['scientific_qualification'] is False
    path=next((store.root/'chunks/0000').glob('*.npz'));path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='immutable descriptor'):store.get(0)
    assert store.put(0,original)['state']=='repaired'
    np.testing.assert_array_equal(store.get(0)._arrays['values'],original._arrays['values'])
    with pytest.raises(ValueError,match='immutable'):store.put(0,data(1.))
    assert store.coverage()['missing']==[1,2,3,4,5]


def test_concurrent_distinct_chunks_do_not_lose_updates(tmp_path):
    store=create(tmp_path/'store',8)
    def put(i):return stores.KernelStore(store.root).put(i,data(i))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(put,range(8)))
    assert store.coverage()['stored']==8 and store.coverage()['missing']==[]
    for i in range(8):np.testing.assert_array_equal(store.get(i)._arrays['values'],data(i)._arrays['values'])


def test_total_storage_budget_is_admission_not_partial_corruption(tmp_path,monkeypatch):
    store=create(tmp_path/'store');monkeypatch.setattr(stores,'MAX_STORE_BYTES',10)
    with pytest.raises(ValueError,match='total byte'):store.put(0,data())
    assert store.coverage()['stored']==0 and not (store.root/'chunks').exists()


def test_multipart_reverse_import_resume_and_deterministic_export(tmp_path,monkeypatch):
    # Reserve metadata normally while forcing several tiny fixture parts.
    monkeypatch.setattr(stores,'MAX_PART_BYTES',4*stores.MAX_INDEX+4500)
    store=create(tmp_path/'store',8)
    for i in range(6):store.put(i,data(i))
    export=store.export_parts(tmp_path/'parts')
    assert len(export['parts'])>=2
    assert store.export_parts(tmp_path/'parts')==export
    restored=None
    for part in reversed(export['parts']):
        restored=stores.KernelStore.import_part(tmp_path/'parts'/part['path'],tmp_path/'restored')
        assert restored.coverage()['scientific_qualification'] is False
    assert restored.coverage()==store.coverage()
    for i in range(6):np.testing.assert_array_equal(restored.get(i)._arrays['values'],data(i)._arrays['values'])
    first=export['parts'][0];stores.KernelStore.import_part(tmp_path/'parts'/first['path'],restored.root)
    assert restored.coverage()['stored']==6
    store.put(6,data(6))
    with pytest.raises(ValueError,match='new export directory'):store.export_parts(tmp_path/'parts')


def test_unrelated_archive_member_is_not_extracted(tmp_path):
    store=create(tmp_path/'store');store.put(0,data());part=store.export_parts(tmp_path/'parts')['parts'][0]
    path=tmp_path/'parts'/part['path']
    with zipfile.ZipFile(path,'a') as archive:archive.writestr('../escape',b'never')
    with pytest.raises(ValueError,match='unrelated'):stores.KernelStore.import_part(path,tmp_path/'restored')
    assert not (tmp_path/'escape').exists()


def test_concurrent_import_can_initialize_one_store(tmp_path,monkeypatch):
    monkeypatch.setattr(stores,'MAX_PART_BYTES',4*stores.MAX_INDEX+4500)
    store=create(tmp_path/'store')
    for i in range(6):store.put(i,data(i))
    export=store.export_parts(tmp_path/'parts')
    def receive(part):return stores.KernelStore.import_part(tmp_path/'parts'/part['path'],tmp_path/'received')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(receive,export['parts']*2))
    assert stores.KernelStore(tmp_path/'received').coverage()['stored']==6


def test_import_cannot_initialize_over_unrelated_user_files(tmp_path):
    destination=tmp_path/'unrelated';destination.mkdir();(destination/'keep.txt').write_text('keep')
    with pytest.raises(ValueError,match='unrelated'):
        stores.KernelStore.open_or_create(destination,dict(model_sha256='c'*64,bank='B2',cell_count=1))
    assert (destination/'keep.txt').read_text()=='keep'


def test_chunk_parent_symlink_cannot_redirect_storage(tmp_path):
    store=create(tmp_path/'store');outside=tmp_path/'outside';outside.mkdir()
    (store.root/'chunks').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='symbolic'):store.put(0,data())
    assert not list(outside.iterdir())


def test_cancelled_export_retains_chunks_and_can_resume(tmp_path):
    store=create(tmp_path/'store');store.put(0,data())
    calls=[]
    def cancel():
        calls.append(1)
        if len(calls)==2:raise InterruptedError('cancel')
    with pytest.raises(InterruptedError):store.export_parts(tmp_path/'parts',check_cancel=cancel)
    assert store.coverage()['stored']==1
    assert not list((tmp_path/'parts').glob('.kernel-part-*'))
    assert store.export_parts(tmp_path/'parts')['parts']


def test_cancelled_import_never_claims_unstored_chunks(tmp_path):
    store=create(tmp_path/'store');store.put(0,data());store.put(1,data(1.))
    part=store.export_parts(tmp_path/'parts')['parts'][0]
    calls=[]
    def cancel():
        calls.append(1)
        if len(calls)==3:raise InterruptedError('cancel')
    with pytest.raises(InterruptedError):
        stores.KernelStore.import_part(tmp_path/'parts'/part['path'],tmp_path/'restored',check_cancel=cancel)
    assert stores.KernelStore(tmp_path/'restored').coverage()['stored']==1
    restored=stores.KernelStore.import_part(tmp_path/'parts'/part['path'],tmp_path/'restored')
    assert restored.coverage()['stored']==2
