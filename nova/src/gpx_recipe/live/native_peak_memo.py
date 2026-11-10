"""Bounded exact-value reuse of reviewed native powder peak functions.

This does not approximate profiles or combine reflection intensities. The
unchanged driver retains its reflection loop, corrections and summation order.
Only identical native inputs on identical grids and arithmetic controls can
reuse a result, including the pinned TOF kernel's exact float32 conversion.
Unknown arithmetic controls retain the original native calculation. One owned process is
required, as for the native GSAS constraint maps.
"""
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import inspect
import math
import os
from pathlib import Path
import struct
import threading
import time

import numpy as np

from ..preparation.kernel_runtime import floating_state


MAXIMUM_BYTES = 64 * 1024 * 1024
MAXIMUM_ENTRIES = 16384
PROBE_CALLS = 64
_LOCK = threading.Lock()
_PROCESS_STORE = None
_DRIVER = 'caa9e0aac821043004b54a2382008b2c0d04abb97ac8b057d57be6044880944d'
# The reviewed wrapper subtracts the position in float64, then F2PY converts
# the offsets and four shape scalars to float32. Only this exact compiled
# implementation may use that equivalence; all others retain float64 keys.
_TOF_FLOAT32_BINARY = '895abf0b70909dcd4c280e01effb54e9e245d2e4ded739365841598bbf781241'
_KERNELS = {
    'getEpsVoigt': ('a299231c52b148b2be785c4fd0b6d6ad8899cf5df67925adc06dc2fa297f73f3', 5),
    'getFCJVoigt3': ('263ea0a0e0a337dd89c74de570f924a23d35648c05f9f90c3e8f334c9e698c82', 4),
}


def _source_hash(function):
    try:
        return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
    except (OSError, TypeError):
        return None


def _compiled_tof_float32(compiled):
    try:
        if type(compiled.pyepsvoigt).__name__!='fortran':return False
        return hashlib.sha256(Path(compiled.__file__).read_bytes()).hexdigest()==_TOF_FLOAT32_BINARY
    except (AttributeError,OSError,TypeError):return False


def _same_tof_offsets(axis,left,right):
    """Compare the exact compiled inputs, including signed-zero bits.

    Equal rounded positions alone are insufficient: the original wrapper
    subtracts each float64 position before its compiled float32 conversion.
    """
    return np.array_equal(np.asarray(axis-left,dtype=np.float32).view(np.uint32),
                          np.asarray(axis-right,dtype=np.float32).view(np.uint32))


class _PeakStore:
    """One bounded numeric store, independent of adapter accounting."""
    def __init__(self, maximum_bytes):
        self.maximum_bytes = maximum_bytes
        self.pid = os.getpid()
        self.cache = OrderedDict()
        self.bytes = 0
        self.reviewed = None
        self.floating = None
        self.tof_float32 = False

    def clear(self):
        self.cache.clear()
        self.bytes = 0

    def ensure_process(self):
        if self.pid != os.getpid():
            # A fork does not transfer native ownership or qualification.
            self.clear()
            self.reviewed = None
            self.floating = None
            self.pid = os.getpid()


def _process_peak_store():
    global _PROCESS_STORE
    if _PROCESS_STORE is None or _PROCESS_STORE.pid != os.getpid():
        _PROCESS_STORE = _PeakStore(MAXIMUM_BYTES)
    return _PROCESS_STORE


def shared_native_peak_memo():
    """Fresh adapter counters over the current process's exact-value store.

    This shares no native model, constraints, observations or mutable outputs.
    Spawned workers build their own store; inherited stores lose qualification.
    Explicit ``NativePeakMemo`` instances remain private regression controls.
    """
    return NativePeakMemo(_store=_process_peak_store())


class NativePeakMemo:
    """Disposable process-local cache; no uploaded or persisted predictions.

    ``maximum_bytes=0`` supplies the unchanged native reference for acceptance
    tests. This is an internal execution control, not a different user recipe.
    The byte bound includes numeric arrays and a 512-byte allowance per entry;
    the entry ceiling separately bounds Python bookkeeping. It is not a bound
    on the native adapter's complete process memory.
    """
    def __init__(self, maximum_bytes=MAXIMUM_BYTES, *, disabled_reason='disabled native reference',
                 _store=None):
        if type(maximum_bytes) is not int or not 0 <= maximum_bytes <= MAXIMUM_BYTES:
            raise ValueError('A bounded native peak cache is required')
        if not isinstance(disabled_reason,str) or not 1<=len(disabled_reason)<=200:
            raise ValueError('A bounded native cache selection reason is required')
        self.maximum_bytes = maximum_bytes
        self.disabled_reason = disabled_reason
        if _store is not None and (not isinstance(_store, _PeakStore)
                                  or _store.maximum_bytes != maximum_bytes):
            raise ValueError('The shared native cache must have the same bounded allocation')
        self._store = _store if _store is not None else _PeakStore(maximum_bytes)
        self.shared_across_contexts = _store is not None
        self.calls = self.hits = self.misses = self.bypasses = self.evictions = 0
        self.driver_calls = 0
        self.unmasked_grid_driver_calls = 0
        self.adaptive_driver_bypasses = 0
        self.estimated_saved_seconds = self.measured_overhead_seconds = 0.
        self.scope_bypasses = {}
        self.native_float32_hits = self.native_offset_checks = self.native_offset_collisions = 0

    def clear(self):
        self._store.clear()

    def diagnostics(self):
        return dict(method='exact native peak arguments', calls=self.calls,
            hits=self.hits, misses=self.misses, bypasses=self.bypasses,
            evictions=self.evictions, driver_calls=self.driver_calls,
            unmasked_grid_driver_calls=self.unmasked_grid_driver_calls,
            entries=len(self._store.cache), accounted_cache_bytes=self._store.bytes,
            maximum_accounted_cache_bytes=self.maximum_bytes,
            maximum_entries=MAXIMUM_ENTRIES, scope_bypasses=dict(self.scope_bypasses),
            shared_across_contexts=self.shared_across_contexts,
            counters_scope='this adapter', cache_scope=(
                'owned process' if self.shared_across_contexts else 'this adapter'),
            adaptive_driver_bypasses=self.adaptive_driver_bypasses,
            observed_peak_calls=self.calls, bypassed_driver_tails_not_counted=True,
            estimated_saved_native_seconds=self.estimated_saved_seconds,
            measured_memo_overhead_seconds=self.measured_overhead_seconds,
            adaptive_probe_calls=PROBE_CALLS,
            native_float32_hits=self.native_float32_hits,
            native_offset_checks=self.native_offset_checks,
            native_offset_collisions=self.native_offset_collisions,
            native_precision_equivalence='Exact compiled float32 inputs only for a pinned TOF binary; otherwise original float64 arguments',
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
        if reason is not None:
            self._bypass_scope(reason)
            yield
            return
        if not _LOCK.acquire(blocking=False):
            self._bypass_scope('another native peak scope owns this process')
            yield
            return
        try:
            with self._owned_evaluation(native_math, driver, powder, functions):
                yield
        finally:
            _LOCK.release()

    @contextmanager
    def _owned_evaluation(self, native_math, driver, powder, functions):
        # Qualification and invalidation share the same ownership lock as
        # patching. A nested or foreign scope cannot clear another owner's
        # shared cache while its native driver is using it.
        store = self._store
        if self.shared_across_contexts and store.pid != os.getpid():
            store.clear()
            store.reviewed = None
            self._store = store = _process_peak_store()
        store.ensure_process()
        state = floating_state()
        if state != store.floating or state is None:
            store.clear()
            store.floating = state
        if state is None:
            self._bypass_scope('unavailable native floating-point control state')
            yield
            return
        compiled = getattr(powder, 'pyd', None)
        current = (driver, *functions.values(), compiled,
                   getattr(compiled, 'pyepsvoigt', None),
                   getattr(compiled, 'pypsvfcjo', None))
        if store.reviewed != current:
            store.clear()
            store.reviewed = None
            if (_source_hash(driver) != _DRIVER or any(
                    _source_hash(functions[name]) != specification[0]
                    for name, specification in _KERNELS.items())):
                self._bypass_scope('unreviewed native driver or active peak hook')
                yield
                return
            store.reviewed = current
            store.tof_float32 = _compiled_tof_float32(compiled)
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
                # Foreign native code can change the thread's arithmetic
                # controls between calls in one owned evaluation scope. Peak
                # arguments alone cannot identify a result in that case.
                state = floating_state()
                if state != store.floating or state is None:
                    store.clear()
                    store.floating = state
                if state is None:
                    self._bypass_scope('unavailable native floating-point control state')
                    return driver(*args, **kwargs)
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
                if (grid is not None and type(args[2]) is np.ma.MaskedArray
                        and not np.ma.is_masked(args[2])):
                    # The reviewed driver otherwise creates masked temporary
                    # arrays for every reflection even when no mask exists.
                    # Retain the exact values/storage and original native
                    # calculation; real masks and unusual array types keep
                    # their original semantics. Histogram data are untouched.
                    args=list(args);args[2]=grid[0]
                    self.unmasked_grid_driver_calls += 1
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
                native_float32 = False
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
                                if name=='getEpsVoigt' and store.tof_float32:
                                    # Position rounding is only an index into
                                    # a possible equivalence class. Hits still
                                    # compare the compiled translated grid.
                                    packed=struct.pack('!5f',*scalars)
                                    key=(name,'native-float32',identity,values.ctypes.data-begin,values.size,packed)
                                    native_float32=True
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

                if key in store.cache and native_float32:
                    cached=store.cache[key][0]
                    if cached[3]!=scalars[0]:
                        self.native_offset_checks+=1
                        if not _same_tof_offsets(values,cached[3],scalars[0]):
                            self.native_offset_collisions+=1
                            _,removed=store.cache.pop(key);store.bytes-=removed
                if key in store.cache:
                    self.hits += 1
                    self.native_float32_hits+=int(native_float32)
                    value, size = store.cache.pop(key)
                    store.cache[key] = value, size
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
                        while store.cache and (store.bytes + size > self.maximum_bytes
                                or len(store.cache) >= MAXIMUM_ENTRIES):
                            _, (_, removed) = store.cache.popitem(last=False)
                            store.bytes -= removed
                            self.evictions += 1
                        store.cache[key] = (value[0].copy(), value[1], native_seconds, scalars[0]), size
                        store.bytes += size
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
