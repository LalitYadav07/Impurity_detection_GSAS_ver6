"""Data-only profile transport rejects altered semantics before native calls."""
import copy
import io

import numpy as np
import pytest

from gpx_recipe.preparation import tof_profile_arrays as arrays
from gpx_recipe.preparation.kernel_runtime import SCHEMA as RUNTIME
from gpx_recipe.preparation.tof_kernel_arrays import array_identity
from gpx_recipe.structural_live.optimization.tof_shape import SCHEMA,BANK_PARAMETERS,SAMPLE_PARAMETERS
from gpx_recipe.symmetry.contracts import digest


@pytest.fixture(scope='module')
def data():
    shape=dict(schema=SCHEMA,bank=dict.fromkeys(BANK_PARAMETERS,0.),sample=dict.fromkeys(SAMPLE_PARAMETERS,0.))
    shape['bank']['difC']=6200.;shape['sample']['Size;i']=1.
    runtime=dict(schema=RUNTIME,native=dict(kernel_sha256='a'*64,binary_sha256='b'*64),
        numerical=dict(schema='radar-numerical-runtime/v1',system='Linux',machine='x86_64',byteorder='little'),
        libc=['glibc','2.41'],libraries={'pypowder.so':'b'*64},
        cpu=[dict(vendor_id='public',**{'cpu family':'6','model':'1','stepping':'1','flags':'sse2'})],
        floating=[895,8064],native_entry_offset=100)
    values=dict(coefficients=np.zeros((513,8193)),halo=np.ones((513,8193),np.uint8),
        logd=np.linspace(-1,3,513),eta=np.linspace(-8,8,8193))
    meta=dict(schema=arrays.SCHEMA,algorithm=arrays.ALGORITHM,shape=shape,shape_sha256=digest(shape),
        runtime=runtime,runtime_sha256=digest(runtime),arrays_sha256=array_identity(values),
        numeric_bytes=sum(v.nbytes for v in values.values()),midpoint_checks=1,rejected_cells=0,
        native_only_stripes=0,observations_included=False,role='starting_proposals')
    return arrays.ProfileTableData(values,meta)


def test_lossless_bounded_roundtrip_and_readonly_arrays(data,tmp_path):
    raw=arrays.encode(data);copy=arrays.decode(raw,data.metadata)
    for name,value in data.arrays().items():
        assert copy.arrays()[name].tobytes()==value.tobytes()
        with pytest.raises(ValueError):copy.arrays()[name].flags.writeable=True
    mutated=copy.metadata;mutated['shape']['bank']['X']=3.
    assert copy.metadata==data.metadata


@pytest.mark.parametrize('change',['shape','runtime','role','observations','extra','algorithm','counts','bool_count'])
def test_metadata_cannot_claim_a_different_origin_or_role(data,change):
    meta=data.metadata
    if change=='shape':meta['shape']['bank']['X']+=.1
    elif change=='runtime':meta['runtime']['floating'][0]+=1
    elif change=='role':meta['role']='native_likelihood'
    elif change=='observations':meta['observations_included']=True
    elif change=='extra':meta['qualified']=True
    elif change=='algorithm':meta['algorithm']='unknown'
    elif change=='counts':meta['rejected_cells']=2
    else:meta['midpoint_checks']=True
    with pytest.raises(ValueError):arrays.validate_metadata(meta)


@pytest.mark.parametrize('change',['dtype','shape','halo','axis','nonfinite','inventory'])
def test_well_hashed_invalid_numeric_data_is_rejected(data,change):
    values=data.arrays()
    if change=='dtype':values['coefficients']=values['coefficients'].astype(np.float32)
    elif change=='shape':values['eta']=values['eta'][:-1].copy()
    elif change=='halo':values['halo']=np.full((513,8193),2,np.uint8)
    elif change=='axis':values['logd']=values['logd'].copy();values['logd'][5]+=.001
    elif change=='nonfinite':values['coefficients']=values['coefficients'].copy();values['coefficients'][0,0]=np.inf
    else:values['unexpected']=np.asarray([1.])
    meta=data.metadata;meta['arrays_sha256']=array_identity(values);meta['numeric_bytes']=sum(v.nbytes for v in values.values())
    with pytest.raises(ValueError):arrays.ProfileTableData(values,meta)


def test_files_are_read_as_exact_bounded_bytes_and_symlinks_are_rejected(data,tmp_path):
    import json
    path=tmp_path/'profile.npz';path.write_bytes(arrays.encode(data))
    receipt=tmp_path/'profile.json';receipt.write_text(json.dumps(data.metadata))
    actual,sha,meta_sha=arrays.read(path,receipt)
    assert actual.metadata==data.metadata and len(sha)==len(meta_sha)==64
    alias=tmp_path/'alias.npz';alias.symlink_to(path)
    with pytest.raises(OSError):arrays.read(alias,receipt)
    receipt.write_text(json.dumps(dict(data.metadata,role='native')))
    with pytest.raises(ValueError):arrays.read(path,receipt)


@pytest.mark.parametrize('ds',[[1.],[1.,1.],[1.,np.nan],[-1.,2.],[]])
def test_small_single_peak_or_unusable_cache_envelope_keeps_native_execution(ds,monkeypatch):
    from types import SimpleNamespace as NS
    from gpx_recipe.structural_live.optimization import tof_profile_proposals as proposals
    from gpx_recipe.live.native_peak_records import PackedPeakRecords
    refs=np.zeros((len(ds),18));refs[:,4]=ds
    source=NS(radii=np.ones(1),fast=NS(slow=NS(phase=NS(name='parent'))))
    hist={'Reflection Lists':{'parent':{'RefList':refs}}}
    packed=PackedPeakRecords(np.array([[2.,1.,1.,1.,1.]]),np.ones(1),np.array([[0,5]],np.int64),
        np.ones(1),np.array([[1,0,0]],np.int64))
    exact=NS(collector=NS(collect_packed_physical=lambda *a,**k:packed),
        _context=lambda:(None,None,None,None,None,None,None,hist,None,None))
    monkeypatch.setattr(proposals,'TOFProfileProposals',lambda *a,**k:pytest.fail('Keep the original native renderer'))
    assert proposals.for_source(source,exact=exact) is None
