"""Native-generated kernels and exact full verification of portable chunks.

Untrusted numerical files cannot become predictions through a saved flag.
Every imported peak is recalculated by the reviewed original native kernel
before this process can reuse it. Qualification never survives serialization.
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict

import numpy as np

from .tof_kernel_arrays import KernelData, array_identity, MAX_BYTES, SCHEMA
from ..symmetry.compiler import file_sha

_PERMIT = object()
_VERIFIED = OrderedDict()
_VERIFIED_PID = os.getpid()
MAX_QUALIFICATIONS = 4096


def _key(data):
    global _VERIFIED_PID
    if _VERIFIED_PID != os.getpid():
        _VERIFIED.clear(); _VERIFIED_PID = os.getpid()
    meta = data.metadata
    return tuple(meta[k] for k in ('arrays_sha256', 'kernel_sha256', 'binary_sha256'))


def _remember(data):
    key = _key(data); _VERIFIED[key] = True; _VERIFIED.move_to_end(key)
    while len(_VERIFIED) > MAX_QUALIFICATIONS:
        _VERIFIED.popitem(last=False)


class KernelUnavailable(ValueError):
    """Keep the original native calculation for this unsupported context."""


def context(source):
    from ..structural_live.optimization.native_cell_proposals import NativeCellProposals, KERNELS
    from ..structural_live.optimization.native_tof_scalars import for_source
    from ..live.native_peak_memo import _compiled_tof_float32
    guard = NativeCellProposals(source)
    native, driver, powder, _, function, _, axis, histogram, _, _ = guard._context()
    collector = for_source(source, driver)
    if (collector is None or not _compiled_tof_float32(powder.pyd)
            or not np.array_equal(native.factor, np.ones_like(native.factor))):
        raise KernelUnavailable('Native kernel preparation requires the reviewed original TOF binary')
    identity = dict(kernel_sha256=KERNELS['PNT'][2], binary_sha256=file_sha(powder.pyd.__file__))
    return guard, collector, function, axis, identity


def _same_compiled_inputs(original, parameters, reference_axis, axis, lo, hi, low, high):
    """Compare the actual pinned F2PY inputs, including signed-zero bits."""
    parameters, axis = np.asarray(parameters, float), np.asarray(axis, float)
    return (parameters.shape == (5,) and axis.ndim == 1 and 0 <= low < high <= len(axis)
        and high-low == hi-lo and np.isfinite(parameters).all()
        and np.array_equal(original[1:].astype(np.float32).view(np.uint32),
                           parameters[1:].astype(np.float32).view(np.uint32))
        and np.array_equal(np.asarray(reference_axis[lo:hi]-original[0], np.float32).view(np.uint32),
                           np.asarray(axis[low:high]-parameters[0], np.float32).view(np.uint32)))


class QualifiedKernel:
    """Owned native qualification over immutable data, never an imported flag."""
    def __init__(self, data, function, proof, *, _permit=None):
        if _permit is not _PERMIT:
            raise TypeError('Only native construction or complete verification can qualify a kernel')
        if not isinstance(data, KernelData):
            raise TypeError('Immutable native kernel data required')
        self._data, self._function = data, function
        self._pid = os.getpid(); self._proof = dict(proof)
        self._indices = {tuple(map(int, h)): i for i, h in enumerate(data._arrays['hkls'])}
        self.hits = self.misses = 0

    @property
    def data(self):
        return self._data

    def lookup(self, hkl, parameters, axis, low, high, *, function):
        """Reuse only exactly equal compiled shape and translated-grid inputs."""
        if os.getpid() != self._pid or function is not self._function:
            self.misses += 1
            return None
        index = self._indices.get(tuple(hkl)); a = self._data._arrays
        if index is None:
            self.misses += 1
            return None
        original = a['parameters'][index]; lo, hi = a['bounds'][index]
        if not _same_compiled_inputs(original, parameters, a['axis'], axis, lo, hi, low, high):
            self.misses += 1
            return None
        start, end = a['pointers'][index:index+2]
        self.hits += 1
        return a['values'][start:end]

    def diagnostics(self):
        return dict(self._proof, hits=self.hits, misses=self.misses,
                    numeric_bytes=self._data.numeric_bytes, qualification_persisted=False)

    def native_result(self, hkl, parameters, axis, low, high, *, function):
        """The original pinned wrapper tuple, without repeating its native peak.

        The wrapper's quadrature sum depends on the actual float64 axis, not
        only its compiled float32 offsets. Recompute that sum exactly instead
        of carrying a reference-grid normalization into a new native call.
        """
        raw=self.lookup(hkl,parameters,axis,low,high,function=function)
        if raw is None:
            return None
        xdata=np.asanyarray(axis)[low:high]
        widths=np.diff(xdata);widths=np.append(widths,widths[-1])
        return raw,np.sum(raw*widths)


def _make_data(axis, records, identity):
    """Assemble the same immutable primitive format for both native producers."""
    hkls, parameters, bounds, pointers, values = [], [], [], [0], []
    for hkl, (params, low, high, raw) in records.items():
        hkls.append(hkl); parameters.append(params); bounds.append([low, high])
        values.append(raw); pointers.append(pointers[-1]+len(raw))
    arrays = dict(hkls=np.ascontiguousarray(hkls, dtype='<i4'),
        parameters=np.ascontiguousarray(parameters, dtype='<f8'),
        bounds=np.ascontiguousarray(bounds, dtype='<i8'),
        pointers=np.ascontiguousarray(pointers, dtype='<i8'),
        values=np.ascontiguousarray(np.concatenate(values), dtype='<f4'),
        axis=np.ascontiguousarray(axis, dtype='<f8'))
    metadata = dict(schema=SCHEMA, **identity, arrays_sha256=array_identity(arrays),
        reflection_count=len(hkls), numeric_bytes=sum(a.nbytes for a in arrays.values()),
        observations_included=False)
    return KernelData(arrays, metadata)


class _NativePassRecorder:
    """Record original peak outputs inside the reviewed Rietveld capture.

    This owned callback never arrives through an uploaded bundle or recipe.
    The existing capture owns the native hook, verifies its caller and restores
    it on every exit. Recorded values and arguments are immutable. Compatibility
    with a later scientific context is checked at every primitive lookup.
    """
    def __init__(self, source):
        self.guard, _, self.function, axis, self.identity = context(source)
        self.source = source
        self.axis = np.frombuffer(axis.tobytes(),dtype=axis.dtype).reshape(axis.shape)
        self.records = {}
        self.accounted = self.axis.nbytes; self.reason = None
        self.started = time.monotonic(); self.finished = False

    def observe(self, hkl, parameters, raw, low, high, remaining_bytes, native_function):
        if self.finished:
            raise ValueError('A completed native pass cannot record more kernels')
        if self.reason is not None:
            return
        if native_function is not self.function:
            raise ValueError('A recorded peak did not use its original qualified native function')
        hkl = tuple(map(int, hkl)); parameters = np.asarray(parameters, float)
        raw = np.asarray(raw)
        if (hkl in self.records or parameters.shape != (5,)
                or not 0 <= low < high <= len(self.axis)
                or raw.shape != (high-low,) or not np.isfinite(raw).all()):
            raise ValueError('Native pass returned inconsistent kernel records')
        if raw.dtype != np.dtype('float32'):
            self.reason = 'original native kernel precision is unsupported'
        additional = raw.nbytes+3*4+5*8+2*8+8+1024
        if self.accounted+additional > min(MAX_BYTES, remaining_bytes):
            self.reason = 'combined native matrix and kernel capture exceeds its byte budget'
        if self.reason is not None:
            self.records.clear(); self.accounted = 0
            return
        self.records[hkl] = (np.frombuffer(parameters.tobytes(),dtype=parameters.dtype),
            int(low),int(high),np.frombuffer(raw.tobytes(),dtype=raw.dtype))
        self.accounted += additional

    def finish(self, *, maximum_bytes):
        if self.finished:
            raise ValueError('A native pass can be qualified only once')
        self.finished = True
        if self.reason is not None or not self.records:
            return None
        self.guard._context()
        # These are outputs of the original function just called by the owned
        # Rietveld hook, not uploaded values. Repeating scattering/scalar work
        # here would not verify those output values. Actual-use lookup still
        # compares every compiled input; changed or missing peaks stay native.
        if self.accounted > maximum_bytes:
            self.reason = 'complete native matrix and kernel context exceeds its byte budget'
            self.records.clear()
            return None
        try:
            data = _make_data(self.axis, self.records, self.identity)
        except ValueError as error:
            # The native pass and its matrix are still valid. Conservative
            # portable-format limits must not reject an otherwise native fit.
            self.reason = 'recorded native kernels exceed their admission domain: '+str(error)
            self.records.clear()
            return None
        self.records.clear(); self.guard._context()
        _remember(data)
        return QualifiedKernel(data, self.function,
            dict(method='already-required original native Rietveld pass', native_peak_calls=0,
                recorded_native_peak_calls=data.metadata['reflection_count'],
                seconds=time.monotonic()-self.started, verified_all_peaks=True), _permit=_PERMIT)


def recorder_for_source(source):
    """Optional primitive recording; unsupported scopes keep their native pass."""
    from ..structural_live.optimization.native_cell_proposals import ProposalUnavailable
    try:
        return _NativePassRecorder(source)
    except (KernelUnavailable, ProposalUnavailable):
        return None


def capture(source, point, *, check_cancel=None):
    """One native geometry supplies a bounded, observation-free kernel chunk."""
    started = time.monotonic()
    if check_cancel is not None:
        check_cancel()
    guard, collector, function, axis, identity = context(source)
    records = collector.collect_physical(point, maximum_bytes=MAX_BYTES)
    if not records:
        raise KernelUnavailable('No native peak kernels occur in this reference range')
    captured = {}
    accounted = axis.nbytes
    for hkl, (params, _, low, high) in records.items():
        if check_cancel is not None:
            check_cancel()
        # Bound output BEFORE calling the native allocator.
        accounted += 4*(high-low)+3*4+5*8+2*8+8
        if accounted+8 > MAX_BYTES:
            raise KernelUnavailable('Complete native kernel chunk exceeds its byte budget')
        raw = np.asarray(function(*params, axis[low:high])[0])
        if raw.dtype != np.dtype('float32'):
            raise KernelUnavailable('Native kernel output precision changed')
        captured[hkl] = (params, low, high, raw)
    data = _make_data(axis, captured, identity)
    if check_cancel is not None:
        check_cancel()
    _remember(data)
    return QualifiedKernel(data, function, dict(method='original native kernel construction',
        native_peak_calls=len(captured), seconds=time.monotonic()-started, verified_all_peaks=True), _permit=_PERMIT)


def verify(data, source, *, check_cancel=None):
    """Recompute EVERY imported peak; metadata checks alone cannot qualify it."""
    if not isinstance(data, KernelData):
        raise TypeError('Checked native kernel data required')
    started = time.monotonic()
    if check_cancel is not None:
        check_cancel()
    guard, _, function, _, identity = context(source)
    if any(data.metadata[k] != v for k, v in identity.items()):
        raise KernelUnavailable('Portable kernel runtime differs from the reviewed native runtime')
    key = _key(data)
    if key in _VERIFIED:
        _VERIFIED.move_to_end(key)
        return QualifiedKernel(data, function, dict(method='same-process exact-content native qualification',
            native_peak_calls=0, seconds=time.monotonic()-started, verified_all_peaks=True), _permit=_PERMIT)
    arrays = data._arrays
    for index, params in enumerate(arrays['parameters']):
        if check_cancel is not None:
            check_cancel()
        guard._context() if index == 0 else None
        low, high = arrays['bounds'][index]
        start, end = arrays['pointers'][index:index+2]
        expected = np.asarray(function(*params, arrays['axis'][low:high])[0])
        if expected.dtype != np.dtype('float32') or not np.array_equal(
                expected.view(np.uint32), arrays['values'][start:end].view(np.uint32)):
            raise ValueError('Portable kernel values failed complete native verification')
    guard._context()
    if check_cancel is not None:
        check_cancel()
    _remember(data)
    return QualifiedKernel(data, function, dict(method='complete original native peak verification',
        native_peak_calls=len(arrays['hkls']), seconds=time.monotonic()-started, verified_all_peaks=True), _permit=_PERMIT)
