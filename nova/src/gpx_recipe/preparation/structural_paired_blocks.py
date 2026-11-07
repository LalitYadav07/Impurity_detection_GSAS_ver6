"""Bounded, data-only reader for parent-conditioned structural examples.

Checksums bind bytes, not native provenance or scientific qualification.
Controls are recomputed from those bytes before training. No uploaded flag
can authorize candidate exclusion or substitute for a native fit.
"""
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from ..live.array_limits import validate_npz_bytes
from ..symmetry.contracts import validate_candidate
from .tof_kernel_arrays import read_bounded

SCHEMA='radar-structural-ranking-paired-pilot/v1'
MAX_BYTES=160*1024**2
MAX_ROWS=32768+4096+1  # Complete sampled points, per-trajectory controls and anchor.
ARRAYS={'q','theory_intensity','background','parent_intensity','parent_background',
    'parent_a','parameters','trajectory','role','geometry_groups','ambiguous_geometry',
    'displacement_rms','relative_metric_norm'}


@dataclass(frozen=True)
class PairedBlock:
    manifest:dict
    arrays:dict


def checksum(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def read(directory):
    directory=Path(directory)
    document=json.loads(read_bounded(directory/'pairs.json',8*1024**2))
    raw=read_bounded(directory/'pairs.npz',MAX_BYTES)
    if (not isinstance(document,dict) or document.get('schema')!=SCHEMA
            or document.get('observations_included') is not False
            or document.get('scientific_result') is not False
            or document.get('pairs_sha256')!=hashlib.sha256(raw).hexdigest()
            or any(not checksum(document.get(k)) for k in ('source_sha256','instrument_sha256','child_sha256'))
            or any(type(document.get(k)) is not int or not 0<=document[k]<2**32 for k in ('seed','parent_seed'))):
        raise ValueError('A bound observation-free paired structural block is required')
    validate_npz_bytes(raw,maximum_bytes=MAX_BYTES)
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        if set(archive.files)!=ARRAYS:raise ValueError('Unexpected paired structural array inventory')
        arrays={k:archive[k] for k in ARRAYS}
    q=arrays['q'];curves=arrays['theory_intensity'];n=len(curves) if curves.ndim==2 else 0
    nm=document.get('mode_dimensions');nc=document.get('cell_dimensions')
    if (type(nm) is not int or nm<0 or type(nc) is not int or not 1<=nc<=6
            or q.ndim!=1 or not 3<=len(q)<=25001 or not 3<=n<=MAX_ROWS
            or curves.shape!=(n,len(q)) or arrays['parent_intensity'].shape!=curves.shape
            or arrays['parent_background'].shape!=curves.shape or arrays['background'].shape!=q.shape
            or arrays['parameters'].shape!=(n,nm+nc) or arrays['parent_a'].shape!=(n,6)
            or arrays['geometry_groups'].shape!=(n,2) or arrays['geometry_groups'].dtype.kind not in 'iu'
            or arrays['trajectory'].shape!=(n,) or arrays['trajectory'].dtype.kind not in 'iu'
            or arrays['role'].shape!=(n,) or arrays['role'].dtype.kind not in 'iu'
            or arrays['ambiguous_geometry'].shape!=(n,) or arrays['ambiguous_geometry'].dtype!=bool
            or any(arrays[k].shape!=(n,) for k in ('displacement_rms','relative_metric_norm'))
            or any(a.dtype.kind not in 'fiub' or a.dtype.itemsize>8 or not np.isfinite(a).all() for a in arrays.values())
            or np.any(q<=0) or np.any(np.diff(q)<=0) or np.any(arrays['parameters'][0]!=0)
            or np.any(arrays['geometry_groups']<1) or np.any(arrays['geometry_groups']>230)
            or not np.array_equal(arrays['ambiguous_geometry'],arrays['geometry_groups'][:,0]!=arrays['geometry_groups'][:,1])
            or any(np.any(arrays[k]<0) for k in ('displacement_rms','relative_metric_norm'))
            or document.get('examples')!=n or document.get('bins')!=len(q)):
        raise ValueError('Invalid physical paired structural dimensions or labels')
    count=document.get('trajectories');steps=document.get('steps');interval=document.get('trajectory_slice')
    if (type(count) is not int or not 1<=count<=4096 or type(steps) is not int or not 1<=steps<=32
            or not isinstance(interval,list) or len(interval)!=2 or any(type(i) is not int for i in interval)
            or count*steps>32768 or not 0<=interval[0]<interval[1]<=count):
        raise ValueError('A complete declared paired trajectory slice is required')
    expected=np.r_[-1,np.repeat(np.arange(*interval),steps+1)]
    roles=np.r_[0,np.tile(np.r_[1,np.full(steps,2)],interval[1]-interval[0])]
    if not np.array_equal(arrays['trajectory'],expected) or not np.array_equal(arrays['role'],roles):
        raise ValueError('Paired null controls and complete child trajectories must remain together')
    source=document['source_sha256'];prefix=f'parent:{source}:{document["seed"]}:{document["parent_seed"]}:trajectory-'
    if document.get('groups')!=['parent:'+source]+[prefix+str(i) for i in expected[1:]]:
        raise ValueError('Physical paired groups must be shared across child aliases and noise siblings')
    a=arrays['parent_a'];metric=np.zeros((n,3,3));metric[:,range(3),range(3)]=a[:,:3]
    metric[:,0,1]=metric[:,1,0]=a[:,3]/2;metric[:,0,2]=metric[:,2,0]=a[:,4]/2
    metric[:,1,2]=metric[:,2,1]=a[:,5]/2
    if np.any(np.linalg.eigvalsh(metric)<=0):raise ValueError('Paired parent cells must remain positive definite')
    for trajectory in range(*interval):
        rows=np.flatnonzero(expected==trajectory)
        for name in ('parent_a','parent_intensity','parent_background'):
            if not np.array_equal(arrays[name][rows],np.broadcast_to(arrays[name][rows[0]],arrays[name][rows].shape)):
                raise ValueError('Every child trajectory needs one unchanged parent-reference context')
    # A shared raw intensity scale requires equal native physical units, not
    # merely shape agreement after an arbitrary fitted rescaling.
    for i in np.flatnonzero(roles!=2):
        child=curves[i]-arrays['background'];parent=arrays['parent_intensity'][i]-arrays['parent_background'][i]
        norm=float(parent@parent)
        if norm<=0:raise ValueError('A paired parent control has no crystalline intensity')
        scale=float(child@parent/norm)
        error=float(np.linalg.norm(child-scale*parent)/max(np.linalg.norm(child),np.finfo(float).tiny))
        if abs(scale-1)>1e-6 or error>1e-6:
            raise ValueError('Paired parent-only control does not retain the same native shape and intensity units')
    candidate=validate_candidate(document.get('candidate'))
    if (not candidate['child'] or candidate['magnetic_space_group'] is not None
            or candidate['child']['sha256']!=document['child_sha256']):
        raise ValueError('Paired examples require their exact structural child identity')
    declared=document.get('declared_child_group')
    if not isinstance(declared,dict) or type(declared.get('number')) is not int or not 1<=declared['number']<=230:
        raise ValueError('Paired examples must retain the declared native child group')
    document['candidate']=candidate
    owned={k:np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape) for k,a in arrays.items()}
    return PairedBlock(document,owned)
