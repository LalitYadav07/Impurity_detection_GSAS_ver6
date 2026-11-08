"""Group bit-identical inputs to the pinned float32 native TOF kernel.

No hkl, metric symmetry or closeness threshold determines equivalence. Equal
rounded positions are only a bucket: different translated grids are separated.
Every original reflection retains its own intensity and summation position.
"""
from dataclasses import dataclass

import numpy as np

from .native_peak_memo import _same_tof_offsets


@dataclass(frozen=True)
class Groups:
    parameters:np.ndarray
    bounds:np.ndarray
    mapping:np.ndarray
    pointers:np.ndarray
    points:int
    requested_points:int
    required_bytes:int
    offset_comparisons:int
    offset_collisions:int


def group(parameters,bounds,axis,*,maximum_bytes,base_bytes,retained_values_bytes=0):
    """Bound scratch before grouping; use the original batch when unprofitable.

    Inputs come only from the checked native batch packer. The numeric
    workspace includes the unique raw float32 profiles and an allowance for
    Python grouping records. A caller may retain one previous native plane:
    on a miss it must release that plane before allocating the new raw values.
    These buffers therefore overlap with grouping indices, never each other.
    """
    count=len(parameters)
    if type(retained_values_bytes)is not int or not 0<=retained_values_bytes<=maximum_bytes:
        raise ValueError('Bound the actual retained native plane inside the same workspace')
    if count<256:return None,'small workload'
    # Own the axis used for both grouping and native rendering. Its allocation
    # is accounted by the caller in base_bytes.
    fixed=base_bytes+28*count+8
    if fixed+retained_values_bytes>maximum_bytes:return None,'grouping index exceeds workspace'
    # A rounded-position collision must not turn one native pass into a
    # quadratic search. This limit falls back to the full original batch;
    # it never truncates reflections or weakens the equivalence test.
    requested=int(np.sum(bounds[:,1]-bounds[:,0],dtype=np.int64))
    compared_points=0
    with np.errstate(over='ignore'):
        rounded=parameters.astype(np.float32)
    if not np.isfinite(rounded).all():return None,'position outside grouping precision'
    mapping=np.empty(count,np.int64);buckets={};selected=[];points=0;comparisons=collisions=0
    bits=parameters.view(np.uint64)[:,0]
    for row,(low,high) in enumerate(bounds):
        key=(int(low),int(high),rounded[row].tobytes())
        equivalent=None
        for candidate in buckets.get(key,()):
            previous=selected[candidate]
            if bits[row]==bits[previous]:
                equivalent=candidate;break
            compared_points+=int(high-low)
            if compared_points>requested:
                return None,'offset comparison exceeds original point work'
            comparisons+=1
            if _same_tof_offsets(axis[low:high],parameters[row,0],parameters[previous,0]):
                equivalent=candidate;break
            collisions+=1
        if equivalent is None:
            equivalent=len(selected);selected.append(row);buckets.setdefault(key,[]).append(equivalent)
            points+=int(high-low)
            # 72 bytes of packed native inputs/pointers and 512 bytes of
            # grouping bookkeeping per representative, plus its raw values.
            if fixed+584*len(selected)+max(4*points,retained_values_bytes)>maximum_bytes:
                return None,'unique native shapes exceed workspace'
        mapping[row]=equivalent
    if len(selected)>.9*count:return None,'few repeated native inputs'
    indices=np.asarray(selected,np.int64)
    unique_parameters=np.ascontiguousarray(parameters[indices])
    unique_bounds=np.ascontiguousarray(bounds[indices])
    pointers=np.r_[0,np.cumsum(unique_bounds[:,1]-unique_bounds[:,0],dtype=np.int64)]
    return Groups(unique_parameters,unique_bounds,mapping,pointers,points,requested,
        fixed+584*len(selected)+max(4*points,retained_values_bytes),comparisons,collisions),None
