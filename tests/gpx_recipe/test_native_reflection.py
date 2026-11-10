"""Qualify shared native amplitude/profile reuse, including failure recovery."""
import copy
import hashlib
import json
import sys

import numpy as np
import pytest

pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profiled import artificial_project
from gpx_recipe.live.native_profile import NativeReducedProfile
from gpx_recipe.live.native_kernel_policy import compile_native_magnetic_kernel
from gpx_recipe.live import native_reflection as module
from gpx_recipe.live.moment_basis import from_native
from gpx_recipe.symmetry.native_profiled import NativeMagneticStartingPoint


@pytest.mark.parametrize('constrained,masked', [(False, False), (True, False), (True, True)])
def test_full_native_profiles_and_all_cross_terms_match(tmp_path, constrained, masked):
    _, path = artificial_project(tmp_path, sites=2, constrained=constrained, masked=masked)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    native = NativeReducedProfile(path)
    indices = [i for i, name in enumerate(native.vary) if '::AM' in name or name.startswith('::constr')]
    compiled = compile_native_magnetic_kernel(native, native.x0, indices, method='reflection')
    assert compiled['diagnostics']['selected_method'] == 'reflection'
    proof = compiled['diagnostics']['native_reflection_diagnostics']
    json.dumps(compiled['diagnostics'], allow_nan=False)
    assert proof['native_amplitude_calls'] == 2*len(indices)
    assert proof['native_profile_calls'] == 1 and proof['tracing_restored']
    assert not proof['observation_values_used'] and not proof['native_functions_replaced']
    q = compiled['quadratic']
    points = [np.zeros(len(indices)), np.ones(len(indices))*.0001,
        *np.random.default_rng(9003).normal(size=(5, len(indices)))*2.]
    for point in points:
        values = native.x0.copy();values[indices] = point
        direct = native.profile(values)
        predicted = compiled['base']+np.einsum('i,ijb,j->b', point, q, point)
        np.testing.assert_allclose(predicted, direct, rtol=1e-8, atol=1e-8)
    point = np.random.default_rng(5101).uniform(.4, 1.7, len(indices))
    values = native.x0.copy();values[indices] = point
    _, jac = native.profile(values, jacobian=True)
    expected = 2*np.einsum('ijb,j->ib', q, point)
    np.testing.assert_allclose(expected, jac[indices], rtol=1e-7, atol=1e-7)
    # Also check the adapter against actual finite differences; agreement
    # between two arrays with the same mask convention is insufficient.
    for index in indices:
        plus = values.copy();minus = values.copy()
        plus[index] += 1e-4;minus[index] -= 1e-4
        finite = (native.profile(plus)-native.profile(minus))/2e-4
        np.testing.assert_allclose(jac[index], finite, rtol=1e-5, atol=1e-7)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize('masked,constrained', [(False, False), (True, False), (True, True)])
def test_model_exploration_applies_identical_weights_masks_and_nuisances(tmp_path, masked, constrained):
    project, path = artificial_project(tmp_path, sites=2, constrained=constrained, masked=masked)
    project.histograms()[0].data['data'][0]['wtFactor'] = 2.5
    project.save()
    accelerated = NativeMagneticStartingPoint(path, refine_scale=True, refine_background=True, maximum_moment=6.)
    fast, proof = accelerated.build()
    assert proof['compiler'] == 'native_reflection_amplitudes', proof
    assert proof['native_derivative_calls'] == 1
    original = NativeMagneticStartingPoint(path, refine_scale=True, refine_background=True, maximum_moment=6.)
    slow, control = original._build(use_derivative=False)
    assert control['compiler'] == 'native_profiles'
    for name in ('offset', 'nuclear', 'quadratic', 'background'):
        np.testing.assert_allclose(getattr(fast, name), getattr(slow, name), rtol=1e-8, atol=1e-6)
    assert fast.normalization == slow.normalization
    assert max(proof['heldout_relative_errors']) < 1e-8


def test_existing_tracer_is_preserved_and_auto_retains_native_fallback(tmp_path):
    _, path = artificial_project(tmp_path)
    native = NativeReducedProfile(path)
    indices = [i for i, name in enumerate(native.vary) if '::AM' in name]
    tracer = lambda *args: None
    try:
        sys.settrace(tracer)
        result = compile_native_magnetic_kernel(native, native.x0, indices)
        assert sys.gettrace() is tracer
    finally:
        sys.settrace(None)
    assert result['diagnostics']['selected_method'] == 'central'
    assert 'untraced' in result['diagnostics']['fallback_reason']


def test_native_failure_restores_tracing_and_capture_ownership(tmp_path, monkeypatch):
    _, path = artificial_project(tmp_path)
    native = NativeReducedProfile(path)
    indices = [i for i, name in enumerate(native.vary) if '::AM' in name]
    basis = from_native(native, native.x0, indices, nuisance=[k for i, k in enumerate(native.vary) if i not in indices])
    histogram = native.histo_phases[0][native.parts[0][0]]
    phase = next(p for p in native.histo_phases[1].values() if p['General']['Type'] == 'magnetic')
    function = native.math.MagStructureFactor2
    marker = module._amplitude_marker(function)
    reflection = copy.deepcopy(histogram['Reflection Lists'][phase['General']['Name']])
    pfx = f"{phase['pId']}::";hfx = f":{histogram['hId']}:"
    metric, _ = native.lattice.A2Gmat([native.parms[pfx+f'A{i}'] for i in range(6)])
    def broken(*args):
        raise RuntimeError('induced amplitude failure')
    with monkeypatch.context() as patch:
        patch.setattr(native.math, 'GetAtomFXU', broken)
        with pytest.raises(RuntimeError, match='induced amplitude failure'):
            module._amplitudes(function, reflection, metric, hfx, pfx, phase['General']['SGData'],
                native.controls, native.parms.copy(), marker, check=native._check)
    assert sys.gettrace() is None
    result = compile_native_magnetic_kernel(native, native.x0, indices, method='reflection')
    assert result['diagnostics']['native_reflection_diagnostics']['tracing_restored']
    assert basis.components.shape[-1] == len(indices)


def test_response_memory_limit_restores_peak_functions_and_uses_fallback(tmp_path, monkeypatch):
    _, path = artificial_project(tmp_path)
    native = NativeReducedProfile(path)
    indices = [i for i, name in enumerate(native.vary) if '::AM' in name]
    original = native.math.G2pwd.getFCJVoigt3
    monkeypatch.setattr(module, 'MAX_PROFILE_BYTES', 1)
    result = compile_native_magnetic_kernel(native, native.x0, indices)
    assert result['diagnostics']['selected_method'] == 'central'
    assert 'byte budget' in result['diagnostics']['fallback_reason']
    assert native.math.G2pwd.getFCJVoigt3 is original
    assert sys.gettrace() is None
