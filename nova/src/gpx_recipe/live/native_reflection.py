"""Exact magnetic response from unchanged native amplitudes and peak profiles.

GSAS-II computes every scattering amplitude and every peak shape. Recording
those amplitudes before squaring avoids repeating the full powder driver for
each moment cross term. The caller retains independent full-profile checks.
Only the reviewed, fixed-geometry PNC/PNT single-magnetic-phase case is used;
other contexts retain the existing checked native compiler.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path
import sys
import threading
import time

import numpy as np

from .validated_backend import PINNED_PRODUCTION_STRMATH
from ..structural_live.optimization.native_profile_capture import CaptureUnavailable, capture_response

MAX_PROFILE_BYTES = 64*1024*1024
MAX_AMPLITUDE_BYTES = 64*1024*1024
MAX_QUADRATIC_BYTES = 128*1024*1024
_trace_lock = threading.Lock()


class ReflectionCompilerUnavailable(ValueError):
    """This accelerator cannot represent the context; use the native fallback."""


def _amplitude_marker(function):
    """Locate a reviewed observation point without replacing native code."""
    try:
        lines, first = inspect.getsourcelines(function)
    except (OSError, TypeError) as exc:
        raise ReflectionCompilerUnavailable('Native amplitude source is unavailable') from exc
    marker = 'refl.T[9] = np.sum(fams**2,axis=0)+np.sum(fbms**2,axis=0)'
    found = [first+i for i, line in enumerate(lines) if line.partition('#')[0].strip() == marker]
    if len(found) != 1:
        raise ReflectionCompilerUnavailable('Native magnetic amplitude layout needs review')
    return found[0]


def _amplitudes(function, reflection, metric, hfx, pfx, symmetry, controls, parameters,
                marker, *, check):
    """Observe completed complex-vector blocks in one unchanged native call."""
    if threading.current_thread() is not threading.main_thread() or sys.gettrace() is not None:
        raise ReflectionCompilerUnavailable('Native amplitude capture requires an untraced owned process')
    count = len(reflection['RefList'])
    result = np.empty((count, 6))
    occupied = np.zeros(count, dtype=bool)
    if not _trace_lock.acquire(blocking=False):
        raise ReflectionCompilerUnavailable('Another native amplitude capture owns this process')

    def trace(frame, event, _arg):
        if frame.f_code is not function.__code__:
            return None
        if event == 'line' and frame.f_lineno == marker:
            local = frame.f_locals
            begin, end = local.get('iBeg'), local.get('iFin')
            if (type(begin) is not int or type(end) is not int or not 0 <= begin < end <= count
                    or occupied[begin:end].any()):
                raise ReflectionCompilerUnavailable('Native magnetic block identity/order changed')
            real, imaginary = np.asarray(local.get('fams')), np.asarray(local.get('fbms'))
            if (real.shape != (3, end-begin) or imaginary.shape != real.shape
                    or not np.isfinite(real).all() or not np.isfinite(imaginary).all()):
                raise ReflectionCompilerUnavailable('Native magnetic amplitude block is invalid')
            result[begin:end, :3] = real.T
            result[begin:end, 3:] = imaginary.T
            occupied[begin:end] = True
        return trace

    try:
        check()
        sys.settrace(trace)
        function(reflection, metric, hfx, pfx, symmetry, controls, parameters)
    finally:
        sys.settrace(None)
        _trace_lock.release()
    if not occupied.all():
        raise ReflectionCompilerUnavailable('Native magnetic amplitudes did not cover every reflection')
    actual = np.asarray(reflection['RefList'][:, 9], float)
    discrepancy = np.max(abs(np.sum(result**2, axis=1)-actual))/max(float(np.max(abs(actual))), 1e-30)
    if not np.isfinite(discrepancy) or discrepancy > 1e-12:
        raise ReflectionCompilerUnavailable('Captured amplitudes differ from native magnetic intensities')
    return result


def compile_reflection_quadratic(*, math, lattice, histogram, phases, histdict,
                                 parameters, controls, basis, check, histogram_name,
                                 verify_coefficients=False):
    """Return unweighted bin-by-moment-by-moment response at current nuisance values.

    ``basis`` is the shared audited native homogeneous moment map. The caller
    supplies the current histogram scale and owns restoration of native state.
    Observations and weights do not enter this calculation.
    """
    started = time.monotonic()
    check()
    code = getattr(math.MagStructureFactor2, '__code__', None)
    if (code is None or hashlib.sha256(Path(math.__file__).read_bytes()).hexdigest() != PINNED_PRODUCTION_STRMATH
            or Path(code.co_filename).resolve() != Path(math.__file__).resolve()):
        raise ReflectionCompilerUnavailable('Native amplitude capture requires the reviewed production function')
    if sys.gettrace() is not None or threading.current_thread() is not threading.main_thread():
        raise ReflectionCompilerUnavailable('Native amplitude capture requires an untraced owned process')
    selected = [name for name, phase in phases.items() if phase['General']['Type'] == 'magnetic']
    hfx = f":{histogram['hId']}:"
    if len(selected) != 1 or controls.get(hfx+'histType') not in {'PNC', 'PNT'}:
        raise ReflectionCompilerUnavailable('One commensurate neutron magnetic phase is required')
    name = selected[0];phase = phases[name]
    if (phase['General'].get('Modulated') or phase['General'].get('doPawley')
            or any(n != name for n, _, _ in basis.atoms)):
        raise ReflectionCompilerUnavailable('The moment map requires a different reflection adapter')
    pfx = f"{phase['pId']}::";phfx = f"{phase['pId']}:{histogram['hId']}:"
    if parameters.get(phfx+'Extinction') != 0. or parameters.get(phfx+'LeBail'):
        raise ReflectionCompilerUnavailable('Intensity-dependent corrections need the full native compiler')
    if controls.get('PhasePartials'):
        raise ReflectionCompilerUnavailable('Native partial-output side effects need the full native compiler')
    x = np.asarray(np.ma.getdata(histogram['Data'][0]), float)
    lower, upper = controls[hfx+'Limits']
    fit_slice = slice(int(np.searchsorted(x, lower)), int(np.searchsorted(x, upper))+1)
    axis = x[fit_slice]
    reference = copy.deepcopy(histogram['Reflection Lists'][name])
    count, dimension = len(reference['RefList']), basis.components.shape[-1]
    if (count < 1 or not 1 <= dimension <= 24
            or count*6*dimension*8 > MAX_AMPLITUDE_BYTES
            or len(axis)*dimension*dimension*8 > MAX_QUADRATIC_BYTES):
        raise ReflectionCompilerUnavailable('Native reflection response exceeds its bounded numeric allocation')
    local_phase = copy.deepcopy(phase);local_phase['General']['doPawley'] = True
    local_histogram = copy.copy(histogram)
    local_histogram['Reflection Lists'] = {name: copy.deepcopy(reference)}
    local_histogram['Reflection Lists'][name]['RefList'][:, 9] = 1.
    profile_parameters = parameters.copy();lookup = {}
    for index, row in enumerate(reference['RefList']):
        h, k, l = map(int, row[:3])
        lookup[pfx+f'{h},{k},{l}'] = index
        profile_parameters[pfx+f'PWLref:{index}'] = 1.
    before_profile = time.monotonic()
    try:
        response, _, profile_proof = capture_response(math.getPowderProfile, math.G2pwd,
            profile_parameters, histdict, axis, local_histogram, {name: local_phase}, controls,
            lookup, histogram_name=histogram_name, phase_name=name, maximum_bytes=MAX_PROFILE_BYTES)
    except CaptureUnavailable as exc:
        raise ReflectionCompilerUnavailable(str(exc)) from exc
    profile_seconds = time.monotonic()-before_profile
    function = math.MagStructureFactor2
    marker = _amplitude_marker(function)
    a = np.array([parameters[pfx+f'A{i}'] for i in range(6)])+math.GetDij(phfx, phase['General']['SGData'], parameters)
    metric, _ = lattice.A2Gmat(a)
    private_parameters = parameters.copy()
    keys = [keys for _, _, keys in basis.atoms]
    calls = 0
    verification_calls=0

    def amplitudes(point):
        nonlocal calls
        check()
        for row, values in zip(keys, basis.components@point):
            private_parameters.update(zip(row, values))
        reflection = copy.deepcopy(reference)
        result = _amplitudes(function, reflection, metric, hfx, pfx, phase['General']['SGData'],
            controls, private_parameters, marker, check=check)
        calls += 1
        return result

    before_amplitudes = time.monotonic()
    amplitude_basis = np.empty((count, 6, dimension))
    for index in range(dimension):
        point = np.eye(dimension)[index]
        amplitude_basis[:, :, index] = (amplitudes(point)-amplitudes(-point))/2.
    coefficient_validation=None
    if verify_coefficients:
        from .quadratic_qualification import check_factor_coefficients
        def intensities(point):
            nonlocal verification_calls
            check()
            for row,values in zip(keys,basis.components@point):
                private_parameters.update(zip(row,values))
            reflection=copy.deepcopy(reference)
            function(reflection,metric,hfx,pfx,phase['General']['SGData'],controls,private_parameters)
            verification_calls+=1
            return reflection['RefList'][:,9]
        try:coefficient_validation=check_factor_coefficients(amplitude_basis,intensities)
        except ValueError as error:raise ReflectionCompilerUnavailable(str(error)) from error
    quadratic = np.empty((len(axis), dimension, dimension))
    # Stream symmetric cross terms. A large reflection inventory must not
    # allocate another dense reflection-by-D-by-D tensor as temporary storage.
    for i in range(dimension):
        for j in range(i, dimension):
            values = response@np.sum(amplitude_basis[:, :, i]*amplitude_basis[:, :, j], axis=1)
            quadratic[:, i, j] = quadratic[:, j, i] = values
    if not np.isfinite(quadratic).all():
        raise ReflectionCompilerUnavailable('Nonfinite native magnetic response')
    check()
    return quadratic, dict(method='native_reflection_amplitudes', dimensions=dimension,
        reflections=count, bins=len(axis), native_amplitude_calls=calls,
        coefficient_structure_factor_calls=verification_calls,
        native_total_structure_factor_calls=calls+verification_calls,
        native_profile_calls=profile_proof['native_driver_calls'], profile_capture=profile_proof,
        profile_seconds=profile_seconds, amplitude_assembly_seconds=time.monotonic()-before_amplitudes,
        amplitude_bytes=amplitude_basis.nbytes, response_bytes=quadratic.nbytes,
        profile_bytes=response.data.nbytes+response.indices.nbytes+response.indptr.nbytes,
        coefficient_validation=coefficient_validation,
        seconds=time.monotonic()-started, observation_values_used=False, native_functions_replaced=False,
        tracing_restored=sys.gettrace() is None, fit_slice=[fit_slice.start, fit_slice.stop])


def compile_profile_reflections(native, reference, indices, *,verify_coefficients=False):
    """Use the shared physical response in a forward-only native adapter."""
    from .native_profile import NativeReducedProfile
    from .moment_basis import from_native
    if not isinstance(native, NativeReducedProfile) or len(native.parts) != 1:
        raise ReflectionCompilerUnavailable('Reflection compiler requires one direct native histogram')
    name, _ = native.parts[0]
    histogram = native.histo_phases[0][name]
    hfx = f":{histogram['hId']}:"
    names = [native.vary[i] for i in indices]
    nuisance = [key for i, key in enumerate(native.vary) if i not in indices]
    if (any('::AM' not in key and not key.startswith('::constr') for key in names)
            or any(key != hfx+'Scale' and not key.startswith(hfx+'Back;') for key in nuisance)):
        raise ReflectionCompilerUnavailable('Complete moment basis and fixed geometry are required')
    basis = from_native(native, reference, indices, nuisance=nuisance)
    zero = np.asarray(reference, float).copy();zero[indices] = 0.
    base = native.profile(zero).copy()
    quadratic, proof = compile_reflection_quadratic(math=native.math, lattice=native.lattice,
        histogram=histogram, phases=native.histo_phases[1], histdict=native.histdict,
        parameters=native.parms, controls=native.controls, basis=basis,
        check=native._check, histogram_name=name,verify_coefficients=verify_coefficients)
    if len(base) != len(quadratic):
        raise ReflectionCompilerUnavailable('Native profile and reflection grids differ')
    # errRefine's saved calculated array is zero at its native excluded bins.
    # Keep that adapter convention here, after compiling the full physical
    # response. Weighted-residual callers apply their own native mask instead.
    excluded = np.ma.getmaskarray(histogram['Data'][0])[native.parts[0][1]]
    quadratic[excluded] = 0.
    return dict(base=base, quadratic=quadratic.transpose(1, 2, 0), diagnostics=proof)
