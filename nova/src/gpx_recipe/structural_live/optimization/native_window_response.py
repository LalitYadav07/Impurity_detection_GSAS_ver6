"""Compact columns captured from an unchanged original native profile pass.

This is an owned in-process matrix representation, not a portable predictor or
permission to trust uploaded arrays. No peak is calculated or approximated here.
"""
import ctypes
import os
import threading

import numpy as np

from ...live.native_peak_batch import _load, _stamp, BatchUnavailable, MAXIMUM_BYTES, DEFAULT_FLOATING
from ...preparation.kernel_runtime import floating_state

_PERMIT=object()

def _immutable(value,dtype):
    a=np.asarray(value,dtype=dtype)
    return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)


class NativeWindowResponse:
    def __init__(self,builder,raw,bounds,pointers,factors,mapping,nx,*,_permit=None):
        if _permit is not _PERMIT:raise TypeError('Only an owned native capture can construct window columns')
        self._builder=builder;self._raw=raw;self._bounds=bounds;self._pointers=pointers
        self._factors=factors;self._mapping=mapping;self.shape=(nx,len(mapping))

    @property
    def numeric_bytes(self):
        return sum(a.nbytes for a in (self._raw,self._bounds,self._pointers,self._factors,self._mapping))

    def reordered(self,order):
        order=np.asarray(order)
        if (order.dtype.kind not in 'iu' or order.shape!=(self.shape[1],)
                or not np.array_equal(np.sort(order),np.arange(self.shape[1]))):
            raise ValueError('A complete native reflection permutation is required')
        self._builder.check()
        return NativeWindowResponse(self._builder,self._raw,self._bounds,self._pointers,
            self._factors,_immutable(self._mapping[order],np.int64),self.shape[0],_permit=_PERMIT)

    def __matmul__(self,coefficients):
        builder=self._builder;builder.check()
        a=np.asarray(coefficients);vector=a.ndim==1
        if vector:a=a[:,None]
        if (a.ndim!=2 or a.shape[0]!=self.shape[1] or not 1<=a.shape[1]<=256
                or a.dtype.kind not in 'fiu'):
            raise ValueError('Complete real native response coefficients are required')
        output_size=self.shape[0]*a.shape[1]
        required=8*(a.size+output_size)+max(a.size,output_size)
        if required>builder.maximum_bytes:
            raise ValueError('Native response multiplication exceeds its workspace')
        a=np.ascontiguousarray(a,dtype=np.float64)
        if not np.isfinite(a).all():raise ValueError('Finite native response coefficients required')
        result=np.zeros((self.shape[0],a.shape[1]))
        status=builder.function(self.shape[1],len(self._factors),self.shape[0],a.shape[1],
            self._mapping,self._bounds,self._pointers,self._factors,self._raw,a,result)
        builder.check()
        if status:raise RuntimeError('Owned native response buffers changed: '+str(status))
        if not np.isfinite(result).all():raise ValueError('Native response product is nonfinite')
        builder.products+=1;builder.maximum_workspace=max(builder.maximum_workspace,required)
        return result[:,0] if vector else result


class NativeWindowBuilder:
    def __init__(self,owner_check,*,maximum_bytes=MAXIMUM_BYTES):
        if type(maximum_bytes)is not int or not 0<maximum_bytes<=MAXIMUM_BYTES:
            raise ValueError('A positive bounded native response workspace is required')
        if not callable(owner_check):raise TypeError('An owned native context check is required')
        if floating_state()!=DEFAULT_FLOATING:
            raise BatchUnavailable('Native window arithmetic controls are unqualified')
        self.owner_check=owner_check;self.pid=os.getpid();self.thread=threading.get_ident()
        self.maximum_bytes=maximum_bytes;self.library,self.receipt,self.path=_load();self.stamp=_stamp(self.path)
        try:self.function=self.library.native_window_multiply
        except AttributeError as error:raise BatchUnavailable('Installed native window helper is unavailable') from error
        f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        f32=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
        i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        self.function.argtypes=[ctypes.c_ssize_t]*4+[i64,i64,i64,f64,f32,f64,f64]
        self.function.restype=ctypes.c_int;self.products=0;self.maximum_workspace=0
        self.check()

    def check(self):
        self.owner_check()
        if (os.getpid()!=self.pid or threading.get_ident()!=self.thread
                or floating_state()!=DEFAULT_FLOATING or _stamp(self.path)!=self.stamp):
            raise RuntimeError('Native window ownership, arithmetic or installed helper changed')

    def assemble(self,columns,count,nx):
        """Only original-hook records may call this internal producer."""
        self.check()
        if type(count)is not int or type(nx)is not int or count<0 or nx<1 or not isinstance(columns,dict):
            raise ValueError('An owned native reflection inventory is required')
        total=0
        for index,(low,high,factor,raw) in columns.items():
            if (type(index)is not int or not 0<=index<count or type(low)is not int or type(high)is not int
                    or not 0<=low<high<=nx or not np.isfinite(factor)
                    or type(raw)is not np.ndarray or raw.dtype!=np.dtype('float32')
                    or raw.shape!=(high-low,) or not np.isfinite(raw).all()):
                raise ValueError('Original native windows are inconsistent')
            total+=len(raw)
        required=4*total+40*len(columns)+8*count+8
        if required>self.maximum_bytes:raise ValueError('Complete native windows exceed the response budget')
        bounds=[];factors=[];pointers=[0];mapping=np.full(count,-1,np.int64);blocks=[]
        for row,(index,(low,high,factor,raw)) in enumerate(columns.items()):
            mapping[index]=row;bounds.append((low,high));factors.append(factor)
            blocks.append(raw.tobytes());pointers.append(pointers[-1]+len(raw))
        raw=np.frombuffer(b''.join(blocks),dtype=np.float32)
        response=NativeWindowResponse(self,raw,_immutable(np.asarray(bounds).reshape(-1,2),np.int64),
            _immutable(pointers,np.int64),_immutable(factors,np.float64),_immutable(mapping,np.int64),nx,_permit=_PERMIT)
        self.check();return response


def for_source(source,*,maximum_bytes=MAXIMUM_BYTES):
    try:
        n=source.fast.native;h=n.histo_phases[0][source.fast.slow.hist.name]
        if n.controls.get(f":{h['hId']}:histType")!='PNT':return None
        return NativeWindowBuilder(n._check,maximum_bytes=maximum_bytes)
    except (BatchUnavailable,OSError,AttributeError,KeyError):return None


def reorder_response(matrix,order):
    return matrix.reordered(order) if isinstance(matrix,NativeWindowResponse) else matrix[:,order].tocsc()


def response_bytes(matrix):
    if isinstance(matrix,NativeWindowResponse):return matrix.numeric_bytes
    return matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes


def freeze_response(matrix):
    if isinstance(matrix,NativeWindowResponse):return
    for a in (matrix.data,matrix.indices,matrix.indptr):a.flags.writeable=False
