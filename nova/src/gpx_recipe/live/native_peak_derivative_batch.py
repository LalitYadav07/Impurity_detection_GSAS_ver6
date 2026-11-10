"""Bounded original-native TOF partials for optimization proposals.

This does not qualify lattice derivatives, compute an objective, or authorize
convergence. Callers retain full native acceptance and stationarity checks.
"""
import ctypes
import hashlib
import inspect
import time

import numpy as np

from .native_peak_batch import NativePeakBatch,BatchUnavailable,pack_records,MAXIMUM_BYTES

DERIVATIVE_WRAPPER_SHA256='cbc214d0890f950f36dd358f65332a5e208204ab5bf21c2c8395b5c992cdd07a'


class NativePeakDerivativeBatch(NativePeakBatch):
    def __init__(self,powder):
        super().__init__(powder)
        self.derivative_kernel=self.compiled.pydepsvoigt
        self.derivative_wrapper=powder.getdEpsVoigt
        if hashlib.sha256(inspect.getsource(self.derivative_wrapper).encode()).hexdigest()!=DERIVATIVE_WRAPPER_SHA256:
            raise BatchUnavailable('Native TOF derivative wrapper is not qualified')
        native=self.native_library.pydepsvoigt_
        capsule=ctypes.pythonapi.PyCapsule_GetPointer
        capsule.argtypes=[ctypes.py_object,ctypes.c_char_p];capsule.restype=ctypes.c_void_p
        self.derivative_address=ctypes.cast(native,ctypes.c_void_p).value
        if not self.derivative_address or self.derivative_address!=capsule(self.derivative_kernel._cpointer,None):
            raise BatchUnavailable('Native derivative entry point differs from its original wrapper')
        self.derivative_function=self.library.native_peak_derivative_columns
        f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        f32=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
        i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        self.derivative_function.argtypes=[ctypes.c_void_p,ctypes.c_ssize_t,ctypes.c_ssize_t,ctypes.c_ssize_t,
            f64,f64,f64,i64,f64,f64,ctypes.c_ssize_t,f32,f64,f64]
        self.derivative_function.restype=ctypes.c_int
        self.last_workspace=0

    def _check_derivative(self):
        self._check()
        if self.compiled.pydepsvoigt is not self.derivative_kernel or self.powder.getdEpsVoigt is not self.derivative_wrapper:
            raise BatchUnavailable('Native derivative ownership or implementation changed')

    def render_columns(self,records,axis,parameter_slopes,height_slopes,*,maximum_bytes=MAXIMUM_BYTES):
        began=time.monotonic();self._check_derivative()
        nr=len(records);dp=parameter_slopes;dh=height_slopes
        if (type(dp)is not np.ndarray or type(dh)is not np.ndarray
                or dp.dtype!=np.dtype('float64') or dh.dtype!=np.dtype('float64')
                or not dp.flags.c_contiguous or not dh.flags.c_contiguous
                or dp.ndim!=3 or dp.shape[0]!=nr or dp.shape[2]!=5
                or not 1<=dp.shape[1]<=24 or dh.shape!=dp.shape[:2]):
            raise BatchUnavailable('Complete bounded contiguous native derivative coefficients required')
        # Check sizes before allocating finite-check temporaries or outputs.
        if type(maximum_bytes)is not int or not 0<maximum_bytes<=MAXIMUM_BYTES:
            raise ValueError('Positive bounded native derivative workspace required')
        if dp.nbytes+dh.nbytes+dp.size+dh.size>maximum_bytes:
            raise BatchUnavailable('Native derivative coefficients exceed workspace')
        parameters,heights,bounds,maximum,required=pack_records(records,axis,maximum_bytes)
        nc=dp.shape[1]
        required+=20*maximum+8*nc*len(axis)+dp.nbytes+dh.nbytes+dp.size+dh.size
        if required>maximum_bytes:raise BatchUnavailable('Native derivative output exceeds workspace')
        if not np.isfinite(dp).all() or not np.isfinite(dh).all():
            raise BatchUnavailable('Native derivative coefficients must be finite')
        signal=np.zeros(len(axis));columns=np.zeros((nc,len(axis)));scratch=np.empty(7*maximum,np.float32)
        code=self.derivative_function(self.derivative_address,nr,len(axis),nc,axis,parameters,heights,bounds,
            dp,dh,maximum,scratch,signal,columns)
        self._check_derivative()
        if code:raise BatchUnavailable('Original native derivative input was refused: '+str(code))
        if not np.isfinite(signal).all() or not np.isfinite(columns).all():
            raise ValueError('Original native peak derivatives produced nonfinite values')
        self.calls+=1;self.reflections+=nr;self.points+=int(np.sum(bounds[:,1]-bounds[:,0]))
        self.last_workspace=required;self.maximum_workspace=max(self.maximum_workspace,required)
        self.seconds+=time.monotonic()-began
        return signal,columns

    def diagnostics(self):
        return dict(super().diagnostics(),method='Original installed Fortran TOF derivative batches',
            derivatives_are_proposals=True,convergence_authorized=False)


def for_powder(powder):
    try:return NativePeakDerivativeBatch(powder)
    except (BatchUnavailable,OSError,AttributeError,TypeError,ctypes.ArgumentError):return None
