"""Bounded exact batching of the installed native TOF peak routine.

The reviewed Python wrapper calls this same Fortran routine after converting
its offsets/scalars to float32. Unused wrapper quadrature and repeated Python
calls are removed. A batch may also share bit-identical compiled peak inputs;
every reflection keeps its own intensity and original summation position.
No peak formula or arithmetic precision changes. Helpers are built from
installed source at image build time; neither code, native addresses nor
permission flags are accepted from data libraries.
"""
from __future__ import annotations

import ctypes
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time

import numpy as np

from ..preparation.kernel_runtime import floating_state
from .native_peak_memo import _compiled_tof_float32
from .native_peak_groups import group as group_shapes
from .native_peak_records import PackedPeakRecords

MAXIMUM_BYTES=64*1024*1024
FLAGS=('-O3','-fPIC','-shared','-fno-fast-math','-ffp-contract=off','-frounding-math')
SCHEMA='radar-native-peak-batch-build/v1'
ABI=0x504e5402
DEFAULT_FLOATING=(895,8064)
IMAGE_HELPER_ROOTS=(Path('/opt/radar-native/gpx_recipe/live'),
    Path('/opt/radar-symmetry/gpx_recipe/live'))


class BatchUnavailable(ValueError):
    """The caller retains its original native calculation."""


def _files():
    source=Path(__file__).with_name('_native_peak_batch.c')
    # A frozen scientific bundle can contain the approved Python/C sources,
    # but never supplies executable helper code. Resolve only the image's
    # installed helper (or an ordinary installed wheel), not its job folder.
    root=next((path for path in IMAGE_HELPER_ROOTS if path.is_dir()),
        Path(sysconfig.get_path('purelib'))/'gpx_recipe/live')
    return source,root/'_native_peak_batch.so',root/'_native_peak_batch.receipt'


def _sha(path):
    path=Path(path)
    if not 0<path.stat().st_size<=1024*1024:
        raise BatchUnavailable('Native helper file exceeds its bound')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(directory=None):
    """Image-build/developer command only; never invoked by a fit or import."""
    source=Path(__file__).with_name('_native_peak_batch.c')
    # The image builder and a later frozen worker must agree on the installed
    # location. Building beside whichever source happened to be imported made
    # the reviewed-refinement image invisible to its own loader.
    default_target=_files()[1]
    target=default_target if directory is None else Path(directory)/default_target.name
    target.parent.mkdir(parents=True,exist_ok=True)
    compiler=shutil.which('cc') or shutil.which('gcc')
    if compiler is None:raise RuntimeError('A C compiler is required at image build time')
    descriptor,temporary=tempfile.mkstemp(prefix='.native-peak-batch-',suffix='.so',dir=target.parent)
    os.close(descriptor);temporary=Path(temporary)
    try:
        subprocess.run([compiler,*FLAGS,str(source),'-o',str(temporary),'-lm'],check=True,
            capture_output=True,timeout=30)
        receipt=dict(schema=SCHEMA,source_sha256=_sha(source),binary_sha256=_sha(temporary),flags=list(FLAGS))
        os.replace(temporary,target)
        stamp=target.with_suffix('.receipt')
        descriptor,pending=tempfile.mkstemp(prefix='.native-peak-batch-',suffix='.receipt',dir=target.parent)
        with os.fdopen(descriptor,'w') as stream:json.dump(receipt,stream,sort_keys=True);stream.write('\n')
        os.replace(pending,stamp)
        return receipt
    finally:
        temporary.unlink(missing_ok=True)


def _load():
    source,binary,receipt=_files()
    try:
        if receipt.stat().st_size>4096:raise BatchUnavailable('Native helper receipt exceeds its bound')
        record=json.loads(receipt.read_text())
        if (not isinstance(record,dict) or set(record)!={'schema','source_sha256','binary_sha256','flags'}
                or record['schema']!=SCHEMA or record['flags']!=list(FLAGS)
                or record['source_sha256']!=_sha(source) or record['binary_sha256']!=_sha(binary)):
            raise BatchUnavailable('Installed native helper does not match its build receipt')
        library=ctypes.CDLL(str(binary))
        abi=library.native_peak_batch_abi;abi.argtypes=[];abi.restype=ctypes.c_int
        if abi()!=ABI:raise BatchUnavailable('Unsupported native batch ABI')
        return library,record,binary
    except (OSError,ValueError,AttributeError) as error:
        if isinstance(error,BatchUnavailable):raise
        raise BatchUnavailable('Installed native peak helper is unavailable') from error


def pack_records(records,axis,maximum_bytes):
    """Bound data-only buffers before a native call; do not truncate work."""
    if type(maximum_bytes)is not int or not 0<maximum_bytes<=MAXIMUM_BYTES:
        raise ValueError('Positive bounded native batch workspace required')
    if (not isinstance(records,(dict,PackedPeakRecords)) or type(axis)is not np.ndarray
            or axis.ndim!=1 or axis.dtype!=np.dtype('float64')
            or not axis.flags.c_contiguous or not len(axis)):
        raise BatchUnavailable('Native batching requires an ordinary contiguous float64 grid')
    count=len(records)
    if count>maximum_bytes//80 or 2*axis.nbytes+80*count>maximum_bytes:
        raise BatchUnavailable('Native batch input exceeds its workspace')
    if not np.isfinite(axis).all() or np.any(np.diff(axis)<=0):
        raise BatchUnavailable('Ordered finite native grid required')
    if isinstance(records,PackedPeakRecords):
        if (records.parameters.shape!=(count,5) or records.heights.shape!=(count,)
                or records.bounds.shape!=(count,2) or np.any(records.bounds[:,0]<0)
                or np.any(records.bounds[:,1]>len(axis)) or records.maximum>np.iinfo(np.int32).max
                or np.any(records.bounds[:,1]-records.bounds[:,0]<2)
                or np.any(records.bounds[:,1]-records.bounds[:,0]>records.maximum)
                or not np.isfinite(records.parameters).all() or not np.isfinite(records.heights).all()):
            raise BatchUnavailable('Packed original native peak arguments exceed their domain')
        with np.errstate(over='ignore'):
            if not np.isfinite(records.parameters[:,1:].astype(np.float32)).all():
                raise BatchUnavailable('Native shape arguments exceed the compiled precision')
        required=2*axis.nbytes+8*records.maximum
        if required>maximum_bytes:raise BatchUnavailable('Native batch scratch exceeds its workspace')
        return records.parameters,records.heights,records.bounds,records.maximum,required
    parameters=np.empty((count,5));heights=np.empty(count);bounds=np.empty((count,2),dtype=np.int64)
    maximum=2
    for index,row in enumerate(records.values()):
        if not isinstance(row,tuple) or len(row)!=4:
            raise BatchUnavailable('Complete original native records required')
        p,height,low,high=row
        if (type(low)is bool or type(high)is bool or not isinstance(low,(int,np.integer))
                or not isinstance(high,(int,np.integer)) or not 0<=low<high<=len(axis)
                or high-low<2 or high-low>np.iinfo(np.int32).max):
            raise BatchUnavailable('Unsupported original native peak window')
        try:p=np.asarray(p,float);height=float(height)
        except (TypeError,ValueError,OverflowError) as error:
            raise BatchUnavailable('Finite original native peak arguments required') from error
        if p.shape!=(5,) or not np.isfinite(p).all() or not np.isfinite(height):
            raise BatchUnavailable('Finite original native peak arguments required')
        parameters[index]=p;heights[index]=height;bounds[index]=(low,high)
        maximum=max(maximum,int(high-low))
    with np.errstate(over='ignore'):
        if not np.isfinite(parameters[:,1:].astype(np.float32)).all():
            raise BatchUnavailable('Native shape arguments exceed the compiled precision')
    # Include packed buffers, output, grid-validation temporaries and the two
    # reusable float32 peak arrays. The caller accounts its native context and
    # already-owned input records separately.
    required=80*count+2*axis.nbytes+8*maximum
    if required>maximum_bytes:raise BatchUnavailable('Native batch scratch exceeds its workspace')
    return parameters,heights,bounds,maximum,required


def _stamp(path):
    value=Path(path).stat()
    return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns


class NativePeakBatch:
    def __init__(self,powder):
        if threading.current_thread() is not threading.main_thread() or floating_state()!=DEFAULT_FLOATING:
            raise BatchUnavailable('Native batching requires reviewed main-thread arithmetic controls')
        if not _compiled_tof_float32(getattr(powder,'pyd',None)):
            raise BatchUnavailable('Native batching requires the reviewed Fortran binary')
        self.powder=powder;self.compiled=powder.pyd;self.kernel=self.compiled.pyepsvoigt
        self.wrapper=powder.getEpsVoigt;self.native_path=Path(self.compiled.__file__).resolve()
        self.native_stamp=_stamp(self.native_path);self.pid=os.getpid()
        self.native_library=ctypes.CDLL(str(self.native_path));native=self.native_library.pyepsvoigt_
        capsule=ctypes.pythonapi.PyCapsule_GetPointer
        capsule.argtypes=[ctypes.py_object,ctypes.c_char_p];capsule.restype=ctypes.c_void_p
        address=ctypes.cast(native,ctypes.c_void_p).value
        if not address or address!=capsule(self.kernel._cpointer,None):
            raise BatchUnavailable('The native batch entry point differs from the original wrapper')
        self.address=address
        self.library,self.receipt,self.helper_path=_load();self.helper_stamp=_stamp(self.helper_path)
        self.function=self.library.native_peak_sum
        f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        f32=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
        i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        self.function.argtypes=[ctypes.c_void_p,ctypes.c_ssize_t,ctypes.c_ssize_t,
            f64,f64,f64,i64,ctypes.c_ssize_t,f32,f32,f64,ctypes.POINTER(ctypes.c_ssize_t)]
        self.function.restype=ctypes.c_int
        self.group_values=self.library.native_peak_group_values
        self.group_values.argtypes=[ctypes.c_void_p,ctypes.c_ssize_t,ctypes.c_ssize_t,
            f64,f64,i64,i64,ctypes.c_ssize_t,ctypes.c_ssize_t,f32,f32,f32,
            ctypes.POINTER(ctypes.c_ssize_t)]
        self.group_values.restype=ctypes.c_int
        self.group_sum=self.library.native_peak_group_sum
        self.group_sum.argtypes=[ctypes.c_ssize_t,ctypes.c_ssize_t,ctypes.c_ssize_t,
            f64,i64,i64,i64,ctypes.c_ssize_t,f32,f64]
        self.group_sum.restype=ctypes.c_int
        self.calls=self.reflections=self.points=self.maximum_workspace=0;self.seconds=0.
        self.native_reflections=self.requested_points=self.grouped_calls=self.duplicate_reflections=0
        self.grouped_parts=self.split_calls=0
        self.offset_comparisons=self.offset_collisions=0;self.grouping_bypasses=Counter()

    def _check(self):
        try:
            if (self.pid!=os.getpid() or threading.current_thread() is not threading.main_thread()
                    or floating_state()!=DEFAULT_FLOATING or self.powder.pyd is not self.compiled
                    or self.compiled.pyepsvoigt is not self.kernel or self.powder.getEpsVoigt is not self.wrapper
                    or _stamp(self.native_path)!=self.native_stamp or _stamp(self.helper_path)!=self.helper_stamp):
                raise BatchUnavailable('Native batch ownership, implementation or arithmetic controls changed')
        except OSError as error:raise BatchUnavailable('Native batch implementation is unavailable') from error

    def render(self,records,axis,*,maximum_bytes=MAXIMUM_BYTES):
        began=time.monotonic();self._check()
        parameters,heights,bounds,maximum,required=pack_records(records,axis,maximum_bytes)
        owned_axis=None
        if len(heights)>=256 and required+axis.nbytes<=maximum_bytes:
            owned_axis=axis.copy()
        else:self.grouping_bypasses['small or workspace-limited workload']+=1
        offsets=np.empty(maximum,np.float32);values=np.empty(maximum,np.float32)
        signal=np.zeros(len(axis));stats=Counter();workspace=required
        native_axis=axis if owned_axis is None else owned_axis

        def render_part(low,high,depth=0):
            nonlocal workspace
            groups=None;reason=None
            if owned_axis is not None:
                groups,reason=group_shapes(parameters[low:high],bounds[low:high],owned_axis,
                    maximum_bytes=maximum_bytes,base_bytes=required+owned_axis.nbytes)
                if reason is not None:self.grouping_bypasses[reason]+=1
            # Keep the global original reflection order and one output
            # accumulator. At most four sequential parts can fit a repeated
            # peak plane into the remaining workspace without adding partial
            # sums, increasing the budget or dropping any native work.
            if reason=='unique native shapes exceed workspace' and depth<2 and high-low>=512:
                stats['splits']+=1;middle=(low+high)//2
                render_part(low,middle,depth+1);render_part(middle,high,depth+1)
                return
            points=ctypes.c_ssize_t()
            self._check()
            if groups is None:
                code=self.function(self.address,high-low,len(native_axis),native_axis,
                    parameters[low:high],heights[low:high],bounds[low:high],
                    maximum,offsets,values,signal,ctypes.byref(points))
                stats['native_reflections']+=high-low;stats['requested_points']+=points.value
                workspace=max(workspace,required+(0 if owned_axis is None else owned_axis.nbytes))
            else:
                raw=np.empty(groups.points,np.float32);unique=len(groups.parameters)
                code=self.group_values(self.address,unique,len(owned_axis),owned_axis,
                    groups.parameters,groups.bounds,groups.pointers,groups.points,maximum,
                    offsets,values,raw,ctypes.byref(points))
                self._check()
                if code==4:raise ValueError('Original native peaks produced a nonfinite profile')
                if not code:
                    code=self.group_sum(high-low,unique,len(owned_axis),heights[low:high],bounds[low:high],
                        groups.mapping,groups.pointers,groups.points,raw,signal)
                stats['native_reflections']+=unique;stats['requested_points']+=groups.requested_points
                stats['grouped_parts']+=1
                stats['offset_comparisons']+=groups.offset_comparisons
                stats['offset_collisions']+=groups.offset_collisions
                workspace=max(workspace,groups.required_bytes)
            self._check()
            if code:raise BatchUnavailable('The native batch input was refused: '+str(code))
            stats['points']+=points.value

        render_part(0,len(heights))
        if owned_axis is not None and not np.array_equal(axis,owned_axis):
            raise BatchUnavailable('Native grid changed during grouped peak calculation')
        self._check()
        if not np.isfinite(signal).all():raise ValueError('Original native peaks produced a nonfinite profile')
        self.calls+=1;self.reflections+=len(heights);self.points+=stats['points']
        self.native_reflections+=stats['native_reflections'];self.requested_points+=stats['requested_points']
        self.grouped_calls+=int(stats['grouped_parts']>0);self.grouped_parts+=stats['grouped_parts']
        self.split_calls+=int(stats['splits']>0)
        self.duplicate_reflections+=len(heights)-stats['native_reflections']
        self.offset_comparisons+=stats['offset_comparisons'];self.offset_collisions+=stats['offset_collisions']
        self.maximum_workspace=max(self.maximum_workspace,workspace);self.seconds+=time.monotonic()-began
        return signal

    def diagnostics(self):
        return dict(method='Original installed Fortran TOF peaks in a bounded batch',calls=self.calls,
            reflections=self.reflections,native_points=self.points,seconds=self.seconds,
            native_reflections=self.native_reflections,requested_peak_points=self.requested_points,
            grouped_calls=self.grouped_calls,duplicate_reflections=self.duplicate_reflections,
            grouped_parts=self.grouped_parts,sequentially_split_calls=self.split_calls,
            grouping_bypasses=dict(self.grouping_bypasses),
            compiled_offset_comparisons=self.offset_comparisons,compiled_offset_collisions=self.offset_collisions,
            maximum_accounted_workspace_bytes=self.maximum_workspace,
            helper_source_sha256=self.receipt['source_sha256'],helper_binary_sha256=self.receipt['binary_sha256'],
            interpolation=False,persisted_predictions=False,reflection_summation_order_preserved=True,
            grouping_scope='identical compiled inputs within this batch; every reflection retains its own intensity')


def for_powder(powder):
    try:return NativePeakBatch(powder)
    except (BatchUnavailable,OSError,AttributeError,TypeError,ctypes.ArgumentError):return None


if __name__=='__main__':
    if sys.argv[1:]!=['--build']:raise SystemExit('Use --build only at image build time')
    print(json.dumps(build(),sort_keys=True))
