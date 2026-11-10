"""Avoid masked-array temporaries in the reviewed unmasked native driver.

This changes only the axis argument's array view. The original driver, peak
kernels, reflection loop, arithmetic and histogram storage remain unchanged.
Real masks, active capture hooks and unknown native drivers keep their inputs.
"""
from collections import Counter
from contextlib import contextmanager
import threading

import numpy as np

from .native_peak_memo import _DRIVER, _source_hash


_LOCK = threading.Lock()


class NativeUnmaskedAxisView:
    def __init__(self, enabled=True):
        if type(enabled) is not bool:
            raise TypeError('Native axis-view selection must be boolean')
        self.enabled = enabled
        self.driver_calls = self.converted_calls = 0
        self.bypasses = Counter()
        self._reviewed_driver = None

    def diagnostics(self):
        return dict(method='unchanged native driver with an unmasked axis view',
                    enabled=self.enabled, driver_calls=self.driver_calls,
                    converted_calls=self.converted_calls, bypasses=dict(self.bypasses),
                    histogram_arrays_modified=False, profile_interpolation=False)

    @contextmanager
    def evaluation(self, native_math):
        driver = getattr(native_math, 'getPowderProfile', None)
        mp = getattr(native_math, 'G2mp', None)
        reason = None
        if not self.enabled:
            reason = 'disabled native reference or existing peak-memo path'
        elif threading.current_thread() is not threading.main_thread():
            reason = 'requires owned main thread'
        elif not callable(driver) or mp is None or mp.InitMP()[0]:
            reason = 'unsupported driver or native multiprocessing'
        elif driver is not self._reviewed_driver:
            if _source_hash(driver) != _DRIVER:
                reason = 'unreviewed native driver or active capture hook'
            else:
                self._reviewed_driver = driver
        if reason is not None:
            self.bypasses[reason] += 1
            yield
            return
        if not _LOCK.acquire(blocking=False):
            self.bypasses['another native axis view owns this process'] += 1
            yield
            return
        owner = threading.get_ident()

        def viewed(*args, **kwargs):
            self.driver_calls += 1
            if (threading.get_ident() == owner and len(args) > 2
                    and type(args[2]) is np.ma.MaskedArray
                    and not np.ma.is_masked(args[2])):
                axis = np.asarray(np.ma.getdata(args[2]))
                if (axis.dtype == np.dtype('float64') and axis.ndim == 1
                        and axis.flags.c_contiguous):
                    args = list(args)
                    args[2] = axis
                    self.converted_calls += 1
            return driver(*args, **kwargs)

        try:
            native_math.getPowderProfile = viewed
            yield
        finally:
            native_math.getPowderProfile = driver
            _LOCK.release()
