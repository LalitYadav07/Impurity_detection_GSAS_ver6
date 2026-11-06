"""Bounded exact-value reuse of reviewed native powder peak functions.

This does not approximate profiles or combine reflection intensities. The
unchanged driver retains its reflection loop, corrections and summation order.
Only identical peak-function arguments on identical native grids can reuse a
result. One owned process is required, as for the native GSAS constraint maps.
"""
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import inspect
import math
import struct
import threading
import time

import numpy as np


MAXIMUM_BYTES = 64 * 1024 * 1024
MAXIMUM_ENTRIES = 16384
PROBE_CALLS = 64
_LOCK = threading.Lock()
_DRIVER = 'caa9e0aac821043004b54a2382008b2c0d04abb97ac8b057d57be6044880944d'
_KERNELS = {
    'getEpsVoigt': ('a299231c52b148b2be785c4fd0b6d6ad8899cf5df67925adc06dc2fa297f73f3', 5),
    'getFCJVoigt3': ('263ea0a0e0a337dd89c74de570f924a23d35648c05f9f90c3e8f334c9e698c82', 4),
}


def _source_hash(function):
    try:
        return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
    except (OSError, TypeError):
        return None


class NativePeakMemo:
    """Disposable process-local cache; no uploaded or persisted predictions.

    ``maximum_bytes=0`` supplies the unchanged native reference for acceptance
    tests. This is an internal execution control, not a different user recipe.
    The byte bound includes numeric arrays and a 512-byte allowance per entry;
    the entry ceiling separately bounds Python bookkeeping. It is not a bound
    on the native adapter's complete process memory.
    """
    def __init__(self, maximum_bytes=MAXIMUM_BYTES, *, disabled_reason='disabled native reference'):
        if type(maximum_bytes) is not int or not 0 <= maximum_bytes <= MAXIMUM_BYTES:
            raise ValueError('A bounded native peak cache is required')
        if not isinstance(disabled_reason,str) or not 1<=len(disabled_reason)<=200:
            raise ValueError('A bounded native cache selection reason is required')
        self.maximum_bytes = maximum_bytes
        self.disabled_reason = disabled_reason
        self._cache = OrderedDict()
        self._bytes = 0
        self._reviewed = None
        self.calls = self.hits = self.misses = self.bypasses = self.evictions = 0
        self.driver_calls = 0
        self.adaptive_driver_bypasses = 0
        self.estimated_saved_seconds = self.measured_overhead_seconds = 0.
        self.scope_bypasses = {}

    def clear(self):
        self._cache.clear()
        self._bytes = 0

    def diagnostics(self):
        return dict(method='exact native peak arguments', calls=self.calls,
            hits=self.hits, misses=self.misses, bypasses=self.bypasses,
            evictions=self.evictions, driver_calls=self.driver_calls,
            entries=len(self._cache), accounted_cache_bytes=self._bytes,
            maximum_accounted_cache_bytes=self.maximum_bytes,
            maximum_entries=MAXIMUM_ENTRIES, scope_bypasses=dict(self.scope_bypasses),
            adaptive_driver_bypasses=self.adaptive_driver_bypasses,
            observed_peak_calls=self.calls, bypassed_driver_tails_not_counted=True,
            estimated_saved_native_seconds=self.estimated_saved_seconds,
            measured_memo_overhead_seconds=self.measured_overhead_seconds,
            adaptive_probe_calls=PROBE_CALLS,
            interpolation=False, reflection_intensities_combined=False,
            native_summation_order_preserved=True, persisted_predictions=False)

    def _bypass_scope(self, reason):
        self.scope_bypasses[reason] = self.scope_bypasses.get(reason, 0) + 1

    @contextmanager
    def evaluation(self, native_math):
        """Reuse only within a reviewed, main-thread native driver scope.

        Other native versions, active response-capture hooks, multiprocessing
        and unknown peak functions use their unchanged calculation. Restoring
        the exact previous functions on failure is part of this boundary.
        """
        driver = getattr(native_math, 'getPowderProfile', None)
        powder = getattr(native_math, 'G2pwd', None)
        mp = getattr(native_math, 'G2mp', None)
        functions = {name: getattr(powder, name, None) for name in _KERNELS}
        reason = None
        if not self.maximum_bytes:
            reason = self.disabled_reason
        elif threading.current_thread() is not threading.main_thread():
            reason = 'requires owned main thread'
        elif not callable(driver) or mp is None or mp.InitMP()[0]:
            reason = 'unsupported driver or native multiprocessing'
        else:
            compiled = getattr(powder, 'pyd', None)
            current = (driver, *functions.values(), compiled,
                       getattr(compiled, 'pyepsvoigt', None),
                       getattr(compiled, 'pypsvfcjo', None))
            if self._reviewed != current:
                # A changed implementation cannot hit entries from the
                # preceding one, even if it changes back before the next call.
                self.clear()
                if (_source_hash(driver) != _DRIVER or any(
                        _source_hash(functions[name]) != specification[0]
                        for name, specification in _KERNELS.items())):
                    reason = 'unreviewed native driver or active peak hook'
                else:
                    self._reviewed = current
        if reason is not None:
            self._bypass_scope(reason)
            yield
            return
        if not _LOCK.acquire(blocking=False):
            self._bypass_scope('another native peak scope owns this process')
            yield
            return
        grid = None
        activity = None
        owner = threading.get_ident()

        def watched(*args, **kwargs):
            nonlocal grid, activity
            old, old_activity = grid, activity
            old_peaks = {name: getattr(powder, name) for name in functions}
            grid = None
            activity = dict(calls=0, saved=0., overhead=0.)
            try:
                for name, function in wrappers.items():
                    setattr(powder, name, function)
                if threading.get_ident() == owner and len(args) > 2:
                    axis = np.asarray(args[2])
                    if (axis.dtype == np.dtype('float64') and axis.ndim == 1
                            and axis.flags.c_contiguous and len(axis) > 1):
                        # Hash every driver call, not a pointer or a GPX name.
                        # Native caller storage can be edited between calls.
                        identity = hashlib.sha256(memoryview(axis).cast('B')).digest()
                        grid = (axis, identity, axis.ctypes.data, axis.ctypes.data + axis.nbytes)
                    self.driver_calls += 1
                return driver(*args, **kwargs)
            finally:
                grid = old
                activity = old_activity
                for name, function in old_peaks.items():
                    setattr(powder, name, function)

        def wrap(name, original):
            parameters = _KERNELS[name][1]

            def peak(*args, **kwargs):
                self.calls += 1
                started = time.perf_counter()
                key = None
                if (threading.get_ident() == owner and grid is not None
                        and not kwargs and len(args) == parameters + 1):
                    axis, identity, begin, end = grid
                    values = np.asarray(args[-1])
                    if (values.dtype == axis.dtype and values.ndim == 1
                            and values.flags.c_contiguous and values.size > 1
                            and values.ctypes.data >= begin
                            and values.ctypes.data + values.nbytes <= end
                            and (values.ctypes.data - begin) % axis.itemsize == 0):
                        try:
                            scalars = tuple(map(float, args[:-1]))
                            if all(map(math.isfinite, scalars)):
                                key = (name, identity, values.ctypes.data - begin, values.size,
                                       struct.pack('!' + 'd' * parameters, *scalars))
                        except (ValueError, TypeError, OverflowError):
                            pass
                if key is None:
                    self.bypasses += 1
                    return original(*args, **kwargs)

                def completed(value, *, native_seconds=0., saved_seconds=0.):
                    # Performance-only admission: never approximate a value
                    # or omit a reflection. Stop wrapping this driver call
                    # when its first probe cannot pay for cache bookkeeping.
                    overhead = max(0., time.perf_counter() - started - native_seconds)
                    self.estimated_saved_seconds += saved_seconds
                    self.measured_overhead_seconds += overhead
                    activity['calls'] += 1
                    activity['saved'] += saved_seconds
                    activity['overhead'] += overhead
                    if (activity['calls'] % PROBE_CALLS == 0
                            and activity['saved'] < 1.25 * activity['overhead']):
                        setattr(powder, name, original)
                        self.adaptive_driver_bypasses += 1
                    return value

                if key in self._cache:
                    self.hits += 1
                    value, size = self._cache.pop(key)
                    self._cache[key] = value, size
                    # Native callers receive independent mutable arrays, as
                    # they do from the original compiled peak function.
                    return completed((value[0].copy(), value[1]), saved_seconds=value[2])
                self.misses += 1
                native_started = time.perf_counter()
                value = original(*args, **kwargs)
                native_seconds = time.perf_counter() - native_started
                if (isinstance(value, tuple) and len(value) == 2
                        and isinstance(value[0], np.ndarray)
                        and value[0].shape == values.shape
                        and value[0].dtype in (np.dtype('float32'), np.dtype('float64'))
                        and isinstance(value[1], (float, int, np.number))
                        and np.isrealobj(value[1])):
                    size = value[0].nbytes + 512
                    if size <= self.maximum_bytes:
                        while self._cache and (self._bytes + size > self.maximum_bytes
                                or len(self._cache) >= MAXIMUM_ENTRIES):
                            _, (_, removed) = self._cache.popitem(last=False)
                            self._bytes -= removed
                            self.evictions += 1
                        self._cache[key] = (value[0].copy(), value[1], native_seconds), size
                        self._bytes += size
                return completed(value, native_seconds=native_seconds)
            return peak

        wrappers = {name: wrap(name, function) for name, function in functions.items()}
        try:
            native_math.getPowderProfile = watched
            yield
        finally:
            native_math.getPowderProfile = driver
            for name, function in functions.items():
                setattr(powder, name, function)
            _LOCK.release()
