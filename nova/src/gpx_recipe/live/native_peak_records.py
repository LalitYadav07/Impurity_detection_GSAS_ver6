"""Owned transient scalar records; no predictions or portable authority.

Native collectors use numerical columns instead of a Python dictionary for
each reflection. Arrays are separate owned buffers, read-only during native
evaluation. The unchanged scientific collector supplies every scalar.
"""
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True,init=False,eq=False)
class PackedPeakRecords:
    def __init__(self,parameters,heights,bounds,spacings,hkls):
        arrays=(parameters,heights,bounds,spacings,hkls)
        if any(type(a) is not np.ndarray or not a.flags.c_contiguous or not a.flags.owndata for a in arrays):
            raise ValueError('Packed native records need separate owned contiguous arrays')
        n=len(heights)
        if (parameters.dtype!=np.float64 or parameters.shape!=(n,5)
                or heights.dtype!=np.float64 or heights.shape!=(n,)
                or bounds.dtype!=np.int64 or bounds.shape!=(n,2)
                or spacings.dtype!=np.float64 or spacings.shape!=(n,)
                or hkls.dtype!=np.int64 or hkls.shape!=(n,3)
                or any(not np.isfinite(a).all() for a in arrays)
                or np.any(bounds[:,0]<0) or np.any(bounds[:,1]-bounds[:,0]<2)
                or np.any(spacings<=0) or len(np.unique(hkls,axis=0))!=n):
            raise ValueError('Invalid complete packed native reflection records')
        for array in arrays:array.flags.writeable=False
        for key,array in zip(('parameters','heights','bounds','spacings','hkls'),arrays):
            object.__setattr__(self,key,array)
        object.__setattr__(self,'maximum',max(2,int(np.max(bounds[:,1]-bounds[:,0],initial=0))))
        object.__setattr__(self,'numeric_bytes',sum(a.nbytes for a in arrays))

    def __len__(self):return len(self.heights)
