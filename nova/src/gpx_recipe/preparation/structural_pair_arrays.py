"""Lossless byte-plane storage for the large native float64 curve arrays."""
import numpy as np

SCHEMA='radar-structural-ranking-pairs/v1'
ENCODING='byte-shuffle-float64-le-u32/v1'
CURVES=('theory_intensity','parent_intensity','parent_background')
MAX_ROWS=32768+4096+1
MAX_BINS=25001


def encode(arrays):
    result=dict(arrays)
    for key in CURVES:
        value=result.pop(key)
        if value.dtype!=np.dtype('<f8') or value.ndim!=2 or not value.flags.c_contiguous:
            raise ValueError('Lossless native pair packing requires ordinary float64 curve arrays')
        # Group four shuffled bytes in each uint32 storage element. This
        # preserves the generic 48-million-element guard for every admitted
        # 160 MiB block; a byte-per-element plane would exhaust it early.
        size=value.size;planes=np.zeros((8,4*((size+3)//4)),np.uint8)
        planes[:,:size]=value.view(np.uint8).reshape(-1,8).T
        result[key+'_byte_planes']=planes.view('<u4')
    return result


def decode(arrays,*,rows,bins,maximum_bytes):
    if (type(rows) is not int or type(bins) is not int or not 3<=rows<=MAX_ROWS
            or not 3<=bins<=MAX_BINS or 3*rows*bins*8>maximum_bytes):
        raise ValueError('Packed native pair dimensions exceed their numerical bound')
    result=dict(arrays)
    for key in CURVES:
        value=result.pop(key+'_byte_planes')
        if (value.dtype!=np.dtype('<u4') or value.shape!=(8,(rows*bins+3)//4)
                or not value.flags.c_contiguous):
            raise ValueError('Invalid fixed-layout native pair byte planes')
        planes=value.view(np.uint8)
        if np.any(planes[:,rows*bins:]):raise ValueError('Noncanonical native pair byte-plane padding')
        result[key]=np.ascontiguousarray(planes[:,:rows*bins].T).view('<f8').reshape(rows,bins)
    return result
