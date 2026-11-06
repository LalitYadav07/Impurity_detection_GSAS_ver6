"""Capture exact native reflection responses during one full profile call.

The native driver still chooses positions, widths, corrections and truncation,
and its original compiled peak functions still calculate every value. This
only records their contributions in a sparse matrix. It supplies no new peak
shape and performs no interpolation. Use in an isolated scientific process.
"""
from __future__ import annotations

import sys
import threading

import numpy as np
from scipy.sparse import csc_matrix


class CaptureUnavailable(ValueError):
    """The driver cannot safely expose this response; use the native fallback."""

    def __init__(self, message, *, native_driver_calls=0):
        super().__init__(message)
        self.native_driver_calls = native_driver_calls


_capture_lock = threading.Lock()


def capture_response(driver, powder, parameters, histdict, axis, histogram,
                     phases, controls, lookup, *, histogram_name, phase_name):
    """Return a native F²-to-signal matrix without N full driver invocations.

    This deliberately supports the qualified PNC/PNT, single-phase unit-F²
    context only. Caller checks linearity and independently checks the result
    against its original Rietveld curve. Unexpected native call conventions
    fail closed; temporarily wrapped functions are restored on every exit.
    """
    code = getattr(driver, '__code__', None)
    hfx = f":{histogram['hId']}:"
    kind = controls.get(hfx + 'histType')
    if (code is None or kind not in {'PNC', 'PNT'} or hfx + 'Lam1' in parameters
            or set(phases) != {phase_name}
            or set(histogram['Reflection Lists']) != {phase_name}
            or phases[phase_name]['General'].get('Modulated')
            or not phases[phase_name]['General'].get('doPawley')):
        raise CaptureUnavailable('Native response capture requires one PNC/PNT unit-intensity phase')
    axis = np.asarray(axis, float)
    if axis.ndim != 1 or not len(axis) or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
        raise CaptureUnavailable('Native response capture requires one ordered finite grid')
    refs = histogram['Reflection Lists'][phase_name]['RefList']
    count = len(refs)
    native_mp = driver.__globals__.get('G2mp')
    if native_mp is None or (count >= 100 and native_mp.InitMP()[0]):
        raise CaptureUnavailable('Native multiprocessing needs its own response-capture adapter')
    if not _capture_lock.acquire(blocking=False):
        raise CaptureUnavailable('Another native response capture owns this process')
    originals = {}
    columns = {}
    native_peak_calls = 0

    def wrap(name):
        original = getattr(powder, name)
        originals[name] = original

        def observed(*args, **kwargs):
            nonlocal native_peak_calls
            result = original(*args, **kwargs)
            frame = sys._getframe(1)
            try:
                if frame.f_code is not code:
                    raise CaptureUnavailable('Peak call did not originate in the reviewed native driver')
                local = frame.f_locals
                if local.get('phase') != phase_name or local.get('useMP') or local.get('im') != 0:
                    raise CaptureUnavailable('Native peak phase, threading or modulation context changed')
                index, begin, end = local['iref'], local['iBeg'], local['iFin']
                if (not isinstance(index, int) or not 0 <= index < count
                        or index in columns or not 0 <= begin < end <= len(axis)):
                    raise CaptureUnavailable('Unexpected native reflection or peak-window ordering')
                if not np.array_equal(np.asarray(args[-1]), axis[begin:end]):
                    raise CaptureUnavailable('Native peak grid differs from the captured window')
                reflection = local['refl']
                if not np.array_equal(reflection[:3], refs[index, :3]):
                    raise CaptureUnavailable('Native reflection identity changed')
                if float(reflection[9]) != 1.:
                    raise CaptureUnavailable('Response capture requires unit native F squared')
                profile = np.asarray(result[0], float)
                factor = float(reflection[11])
                if (profile.shape != (end - begin,) or not np.isfinite(profile).all()
                        or not np.isfinite(factor)):
                    raise CaptureUnavailable('Nonfinite or unexpected native peak response')
                signal = factor * profile
                use = np.flatnonzero(signal)
                columns[index] = (use.astype(np.int32) + begin, signal[use].copy())
                native_peak_calls += 1
                return result
            finally:
                del frame

        setattr(powder, name, observed)

    try:
        wrap('getFCJVoigt3' if kind == 'PNC' else 'getEpsVoigt')
        try:
            signal, background = driver(parameters, histdict, axis, [], histogram,
                phases, controls, lookup, histogram=histogram_name)
        except CaptureUnavailable as exc:
            exc.native_driver_calls = 1
            raise
    finally:
        for name, original in originals.items():
            setattr(powder, name, original)
        _capture_lock.release()
    indices, values, pointers = [], [], [0]
    for index in range(count):
        rows, data = columns.get(index, (np.empty(0, np.int32), np.empty(0)))
        indices.append(rows)
        values.append(data)
        pointers.append(pointers[-1] + len(data))
    matrix = csc_matrix((np.concatenate(values) if values else np.empty(0),
                         np.concatenate(indices) if indices else np.empty(0, np.int32),
                         np.asarray(pointers)), shape=(len(axis), count))
    signal = np.asarray(signal, float)
    background = np.asarray(background, float)
    if (signal.shape != axis.shape or background.shape != axis.shape
            or not np.isfinite(signal).all() or not np.isfinite(background).all()):
        raise CaptureUnavailable('Native driver returned an invalid profile or background',
                                 native_driver_calls=1)
    rebuilt = matrix @ np.ones(count)
    error = float(np.linalg.norm(rebuilt - signal) / max(np.linalg.norm(signal), 1e-30))
    if not np.isfinite(error) or error > 1e-11:
        raise CaptureUnavailable(f'Captured contributions disagree with the native unit profile: {error}',
                                 native_driver_calls=1)
    return matrix, background, dict(
        method='single native profile pass', native_driver_calls=1,
        native_peak_calls=native_peak_calls, reflections=count,
        unit_profile_relative_error=error, interpolation=False,
        native_peak_functions_restored=True)
