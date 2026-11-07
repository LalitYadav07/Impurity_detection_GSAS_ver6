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
        parameters, axis = np.asarray(parameters, float), np.asarray(axis, float)
        if (parameters.shape != (5,) or axis.ndim != 1 or not 0 <= low < high <= len(axis)
                or high-low != hi-lo or not np.isfinite(parameters).all()
                or not np.array_equal(original[1:].astype(np.float32).view(np.uint32),
                                      parameters[1:].astype(np.float32).view(np.uint32))
                or not np.array_equal(np.asarray(a['axis'][lo:hi]-original[0], np.float32).view(np.uint32),
                                      np.asarray(axis[low:high]-parameters[0], np.float32).view(np.uint32))):
            self.misses += 1
            return None
        start, end = a['pointers'][index:index+2]
        self.hits += 1
        return a['values'][start:end]

    def diagnostics(self):
        return dict(self._proof, hits=self.hits, misses=self.misses,
                    numeric_bytes=self._data.numeric_bytes, qualification_persisted=False)


def capture(source, point, *, check_cancel=None):
    """One native geometry supplies a bounded, observation-free kernel chunk."""
    started = time.monotonic()
    if check_cancel is not None:
        check_cancel()
    guard, collector, function, axis, identity = context(source)
    records = collector.collect_physical(point, maximum_bytes=MAX_BYTES)
    if not records:
        raise KernelUnavailable('No native peak kernels occur in this reference range')
    hkls, parameters, bounds, pointers, values = [], [], [], [0], []
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
    data = KernelData(arrays, metadata)
    if check_cancel is not None:
        check_cancel()
    _remember(data)
    return QualifiedKernel(data, function, dict(method='original native kernel construction',
        native_peak_calls=len(hkls), seconds=time.monotonic()-started, verified_all_peaks=True), _permit=_PERMIT)


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
