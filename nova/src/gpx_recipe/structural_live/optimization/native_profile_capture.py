"""Capture exact native reflection responses during one full profile call.

The native driver still chooses positions, widths, corrections and truncation,
and its original compiled peak functions still calculate every value. This
only records their contributions in a sparse matrix. It supplies no new peak
shape and performs no interpolation. Use in an isolated scientific process.
"""
from __future__ import annotations

import sys
import threading
import hashlib
import inspect

import numpy as np
from scipy.sparse import csc_matrix


class CaptureUnavailable(ValueError):
    """The driver cannot safely expose this response; use the native fallback."""

    def __init__(self, message, *, native_driver_calls=0):
        super().__init__(message)
        self.native_driver_calls = native_driver_calls


_capture_lock = threading.Lock()

# Reviewed native PNC/PNT loops call the original peak functions even for
# zero F². This is necessary to retain dormant reflections without division
# by F². Unknown drivers keep using the independent full-native path.
RIETVELD_DRIVERS={'caa9e0aac821043004b54a2382008b2c0d04abb97ac8b057d57be6044880944d'}


def capture_response(driver, powder, parameters, histdict, axis, histogram,
                     phases, controls, lookup, *, histogram_name, phase_name,
                     maximum_bytes=None):
    return _capture_response(driver,powder,parameters,histdict,axis,histogram,
        phases,controls,lookup,histogram_name=histogram_name,phase_name=phase_name,
        maximum_bytes=maximum_bytes)[:3]


def capture_rietveld_response(driver,powder,parameters,histdict,axis,histogram,
                              phases,controls,lookup,*,histogram_name,phase_name,
                              evaluate,maximum_bytes,observe_peak=None,packed_builder=None):
    """Record unit-F² columns during an already-needed native evaluation.

    ``evaluate`` is an owned in-process scientific callback, never serialized
    or accepted from a user recipe. It must invoke exactly this native driver
    once. Its full native prediction is returned unchanged as the fourth item.
    """
    if not callable(evaluate):raise TypeError('An owned native evaluator is required')
    return _capture_response(driver,powder,parameters,histdict,axis,histogram,
        phases,controls,lookup,histogram_name=histogram_name,phase_name=phase_name,
        maximum_bytes=maximum_bytes,evaluate=evaluate,observe_peak=observe_peak,packed_builder=packed_builder)


def _capture_response(driver,powder,parameters,histdict,axis,histogram,
                      phases,controls,lookup,*,histogram_name,phase_name,
                      maximum_bytes=None,evaluate=None,observe_peak=None,packed_builder=None):
    """Return a native F²-to-signal matrix without N full driver invocations.

    This deliberately supports the qualified PNC/PNT, single-phase unit-F²
    context only. Caller checks linearity and independently checks the result
    against its original Rietveld curve. Unexpected native call conventions
    fail closed; temporarily wrapped functions are restored on every exit.
    """
    code = getattr(driver, '__code__', None)
    unit=evaluate is None
    if observe_peak is not None and (unit or not callable(observe_peak)):
        raise TypeError('Only an owned Rietveld pass can record original peak outputs')
    hfx = f":{histogram['hId']}:"
    kind = controls.get(hfx + 'histType')
    if packed_builder is not None:
        from .native_window_response import NativeWindowBuilder
        if not isinstance(packed_builder,NativeWindowBuilder) or unit or kind!='PNT':
            raise TypeError('Packed responses require an owned native TOF Rietveld capture')
        packed_builder.check()
    if (code is None or kind not in {'PNC', 'PNT'} or hfx + 'Lam1' in parameters
            or set(phases) != {phase_name}
            or set(histogram['Reflection Lists']) != {phase_name}
            or phases[phase_name]['General'].get('Modulated')
            or bool(phases[phase_name]['General'].get('doPawley'))!=unit):
        raise CaptureUnavailable('Native response capture requires one qualified PNC/PNT phase')
    if not unit:
        try:source_hash=hashlib.sha256(inspect.getsource(driver).encode()).hexdigest()
        except (OSError,TypeError):source_hash=None
        pfx=f"{phases[phase_name]['pId']}::";phfx=f"{phases[phase_name]['pId']}:{histogram['hId']}:"
        if (source_hash not in RIETVELD_DRIVERS or driver.__globals__.get('getPowderProfile') is not driver
                or phases[phase_name]['General'].get('Type')!='nuclear' or parameters.get(pfx+'isMag')
                or parameters.get(phfx+'Extinction')!=0. or controls.get('PhasePartials')
                or controls.get(hfx+'Exclude')
                or any(np.ma.getmaskarray(v).any() for v in histogram['Data'][:3])):
            raise CaptureUnavailable('Rietveld response capture requires the reviewed unmasked linear native driver')
    axis = np.asarray(axis, float)
    if axis.ndim != 1 or not len(axis) or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
        raise CaptureUnavailable('Native response capture requires one ordered finite grid')
    refs = histogram['Reflection Lists'][phase_name]['RefList']
    count = len(refs)
    if maximum_bytes is not None and (type(maximum_bytes) is not int or maximum_bytes < 1):
        raise ValueError('A positive native response byte budget is required')
    native_mp = driver.__globals__.get('G2mp')
    if native_mp is None or ((not unit or count >= 100) and native_mp.InitMP()[0]):
        raise CaptureUnavailable('Native multiprocessing needs its own response-capture adapter')
    if not _capture_lock.acquire(blocking=False):
        raise CaptureUnavailable('Another native response capture owns this process')
    originals = {}
    columns = {}
    native_peak_calls = 0
    stored_entries = 0
    native_result=None;driver_calls=0;driver_result=None

    def wrap(name):
        original = getattr(powder, name)
        originals[name] = original

        def observed(*args, **kwargs):
            nonlocal native_peak_calls, stored_entries
            result = original(*args, **kwargs)
            frame = sys._getframe(1)
            try:
                if frame.f_code is not code:
                    raise CaptureUnavailable('Peak call did not originate in the reviewed native driver')
                local = frame.f_locals
                if local.get('phase') != phase_name or local.get('useMP') or local.get('im') != 0:
                    raise CaptureUnavailable('Native peak phase, threading or modulation context changed')
                current=histogram['Reflection Lists'][phase_name]['RefList']
                index, begin, end = local['iref'], local['iBeg'], local['iFin']
                if (not isinstance(index, int) or not 0 <= index < len(current)
                        or index in columns or not 0 <= begin < end <= len(axis)):
                    raise CaptureUnavailable('Unexpected native reflection or peak-window ordering')
                if not np.array_equal(np.asarray(args[-1]), axis[begin:end]):
                    raise CaptureUnavailable('Native peak grid differs from the captured window')
                reflection = local['refl']
                if not np.array_equal(reflection[:3], current[index, :3]):
                    raise CaptureUnavailable('Native reflection identity changed')
                if unit and float(reflection[9]) != 1.:
                    raise CaptureUnavailable('Response capture requires unit native F squared')
                profile = np.asarray(result[0]) if packed_builder is not None else np.asarray(result[0],float)
                factor = float(reflection[11])
                if (profile.shape != (end - begin,) or not np.isfinite(profile).all()
                        or not np.isfinite(factor)):
                    raise CaptureUnavailable('Nonfinite or unexpected native peak response')
                if packed_builder is not None:
                    if profile.dtype!=np.dtype('float32'):
                        raise CaptureUnavailable('Native TOF window precision is unsupported')
                    stored_entries+=len(profile)
                    required=stored_entries*4+40*(len(columns)+1)+8*len(current)+8
                    if required>maximum_bytes:
                        raise CaptureUnavailable('Native profile response exceeds its allocated byte budget')
                    columns[index]=(int(begin),int(end),factor,np.frombuffer(profile.tobytes(),dtype=np.float32))
                else:
                    signal = factor * profile
                    use = np.flatnonzero(signal)
                    stored_entries += len(use)
                    required=stored_entries*12+(len(current)+1)*8
                    if maximum_bytes is not None and required > maximum_bytes:
                        raise CaptureUnavailable('Native profile response exceeds its allocated byte budget')
                    columns[index] = (use.astype(np.int32) + begin, signal[use].copy())
                if observe_peak is not None:
                    remaining=maximum_bytes-required
                    observe_peak(tuple(reflection[:3]),np.asarray(args[:-1],float),
                        np.asarray(result[0]),begin,end,remaining,original)
                native_peak_calls += 1
                return result
            finally:
                del frame

        setattr(powder, name, observed)

    def watched(*args,**kwargs):
        nonlocal driver_calls,driver_result
        driver_calls+=1
        if (driver_calls!=1 or len(args)<7 or args[4] is not histogram or args[5] is not phases
                or not np.array_equal(np.asarray(args[2]),axis)):
            raise CaptureUnavailable('Owned evaluation changed the native phase, histogram or grid')
        driver_result=driver(*args,**kwargs)
        return driver_result

    try:
        wrap('getFCJVoigt3' if kind == 'PNC' else 'getEpsVoigt')
        try:
            if unit:
                signal, background = driver(parameters, histdict, axis, [], histogram,
                    phases, controls, lookup, histogram=histogram_name)
            else:
                driver.__globals__['getPowderProfile']=watched
                native_result=evaluate()
                if driver_calls!=1 or driver_result is None:
                    raise CaptureUnavailable('Owned evaluation did not call its reviewed native driver once')
                signal,background=driver_result
        except CaptureUnavailable as exc:
            exc.native_driver_calls = 1
            raise
    finally:
        if not unit:driver.__globals__['getPowderProfile']=driver
        for name, original in originals.items():
            setattr(powder, name, original)
        _capture_lock.release()
    refs=histogram['Reflection Lists'][phase_name]['RefList'];count=len(refs)
    if packed_builder is not None:
        matrix=packed_builder.assemble(columns,count,len(axis))
    else:
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
    rebuilt = matrix @ (np.ones(count) if unit else refs[:,9])
    error = float(np.linalg.norm(rebuilt - signal) / max(np.linalg.norm(signal), 1e-30))
    if not np.isfinite(error) or error > 1e-11:
        raise CaptureUnavailable(f'Captured contributions disagree with the native profile: {error}',
                                 native_driver_calls=1)
    return matrix, background, dict(
        method='single native profile pass' if unit else 'already-required native Rietveld pass', native_driver_calls=1,
        native_peak_calls=native_peak_calls, reflections=count,
        unit_profile_relative_error=error, interpolation=False,
        native_peak_functions_restored=True,no_F_squared_division=True,
        response_storage='original native windows' if packed_builder is not None else 'CSC columns'),native_result
