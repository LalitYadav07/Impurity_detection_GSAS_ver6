"""Native controls for portable unweighted physics and fresh scan binding."""
import hashlib
import numpy as np
import pytest

G = pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profiled import artificial_project
from gpx_recipe.live.native_profile import NativeReducedProfile
from gpx_recipe.symmetry.native_profiled import NativeMagneticStartingPoint
from gpx_recipe.preparation.magnetic import ARRAYS, load
from gpx_recipe.preparation.native_magnetic import compile_response, physics_identity, qualify_response


@pytest.mark.parametrize('constrained', [False, True])
def test_portable_response_requires_fresh_native_checks_after_load(tmp_path, constrained):
    _, path = artificial_project(tmp_path, sites=2, constrained=constrained)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    native = NativeReducedProfile(path)
    response = compile_response(native)
    assert response.quadratic.shape[1] == (3 if constrained else 6)
    assert response.proof['observations_included'] is False
    destination = tmp_path/'theory.npz'
    manifest = response.save(destination)
    assert manifest['scientific_result'] is False
    with np.load(destination, allow_pickle=False) as data:
        assert set(data.files) == set(ARRAYS)|{'manifest'}
    restored = load(destination, physics_sha256=response.physics_sha256)
    kwargs = dict(x=response.x, y=np.ones(len(response.x)), weights=np.ones(len(response.x)),
        fit_mask=np.ones(len(response.x), bool), normalization_mask=np.ones(len(response.x), bool),
        physics_sha256=response.physics_sha256, maximum_moment=6., fixed_scale=120.)
    with pytest.raises(ValueError, match='Qualify'):
        restored.bind(**kwargs)
    validation = qualify_response(restored, native)
    assert max(validation['independent_magnetic_errors']) < 1e-8
    restored.bind(**kwargs)
    for name in ARRAYS:
        np.testing.assert_array_equal(getattr(restored, name), getattr(response, name))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_theory_only_zero_observations_are_not_replaced_or_used_as_evidence(tmp_path):
    project, path = artificial_project(tmp_path, sites=2)
    hist = project.histograms()[0]
    hist.data['data'][1][1][:] = 0.
    hist.data['data'][1][2][:] = 1.
    hist.data['data'][0]['Dummy'] = True
    project.save()
    before = path.read_bytes()
    native = NativeReducedProfile(path)
    response = compile_response(native)
    assert np.any(response.nuclear) and np.any(response.quadratic)
    assert not np.any(native.histo_phases[0][native.parts[0][0]]['Data'][1])
    assert not np.any(native.counts)
    assert path.read_bytes() == before


def test_observations_and_variances_change_likelihood_not_prepared_physics(tmp_path):
    project, path = artificial_project(tmp_path, sites=2)
    response = compile_response(NativeReducedProfile(path))
    hist = project.histograms()[0]
    x, y, w = hist.data['data'][1][:3]
    masks = np.ones(len(x), bool)
    kwargs = dict(x=x, fit_mask=masks, normalization_mask=masks,
        physics_sha256=response.physics_sha256, maximum_moment=6., fixed_scale=120.)
    first = response.bind(y=y, weights=w, **kwargs)
    hist.data['data'][1][1] = y*1.1+2.
    hist.data['data'][1][2] = w*.3
    hist.data['data'][0]['wtFactor'] = 2.5
    project.save()
    changed = NativeReducedProfile(path)
    assert physics_identity(changed) == response.physics_sha256
    qualify_response(response, changed)
    second = response.bind(y=y*1.1+2., weights=w*.3, weight_factor=2.5, **kwargs)
    assert not np.array_equal(first.offset, second.offset)
    assert first.normalization != second.normalization


def test_bound_objective_matches_native_weights_masks_and_normalization(tmp_path):
    project, path = artificial_project(tmp_path, sites=2)
    response = compile_response(NativeReducedProfile(path))
    hist = project.histograms()[0]
    x = hist.data['data'][1][0]
    hist.data['Limits'].append([float(x[170]), float(x[190])])
    hist.data['data'][0]['wtFactor'] = 2.5
    project.save()
    assert physics_identity(NativeReducedProfile(path)) == response.physics_sha256
    direct = NativeMagneticStartingPoint(path, refine_scale=True, refine_background=True, maximum_moment=6.)
    moment = direct.initial[direct.moment_indices]*.8
    bg = direct.initial[direct.background_indices]*1.3
    actual = direct.residual(moment, 120., bg)
    sl = slice(None)
    y, w = (a[sl] for a in hist.data['data'][1][1:3])
    model = response.bind(x=x[sl], y=y, weights=w, fit_mask=~direct.residual_mask,
        normalization_mask=np.ones(len(y), bool), physics_sha256=response.physics_sha256,
        maximum_moment=6., weight_factor=2.5)
    predicted = model.offset+120.*(model.nuclear+np.einsum('i,nij,j->n', moment,
        model.quadratic, moment))+model.background@bg
    np.testing.assert_allclose(predicted, actual, rtol=1e-8, atol=1e-6)
    assert model.normalization == pytest.approx(direct.histogram['Residuals']['sumwYo'], rel=1e-12)


def test_native_fit_range_change_requires_response_rendering(tmp_path):
    project, path = artificial_project(tmp_path, sites=2)
    response = compile_response(NativeReducedProfile(path))
    hist = project.histograms()[0]
    x = hist.data['data'][1][0]
    hist.data['Limits'][1] = [float(x[30]), float(x[-31])]
    project.save()
    assert physics_identity(NativeReducedProfile(path)) != response.physics_sha256
    with pytest.raises(ValueError, match='grid or fit range'):
        response.bind(x=response.x[30:-30], y=np.ones(len(x)-60), weights=np.ones(len(x)-60),
            fit_mask=np.ones(len(x)-60, bool), normalization_mask=np.ones(len(x)-60, bool),
            physics_sha256=response.physics_sha256, maximum_moment=6.)


def test_native_constraints_can_project_one_shared_response_without_refitting(tmp_path):
    full = tmp_path/'full'; full.mkdir()
    restricted = tmp_path/'restricted'; restricted.mkdir()
    _, common_path = artificial_project(full, sites=2, constrained=False)
    _, child_path = artificial_project(restricted, sites=2, constrained=True)
    common = compile_response(NativeReducedProfile(common_path))
    child = compile_response(NativeReducedProfile(child_path))
    projection, _, rank, _ = np.linalg.lstsq(common.moment_maps.reshape(-1, 6),
        child.moment_maps.reshape(-1, 3), rcond=None)
    assert rank == 6
    np.testing.assert_allclose(common.moment_maps@projection, child.moment_maps, atol=1e-12)
    projected = np.einsum('ia,nij,jb->nab', projection, common.quadratic, projection)
    np.testing.assert_allclose(projected, child.quadratic, rtol=1e-8, atol=1e-6)
    for moment in np.random.default_rng(931).normal(size=(11, 3)):
        np.testing.assert_allclose(common.predict(projection@moment, scale=120., background=[3., .2, .1]),
            child.predict(moment, scale=120., background=[3., .2, .1]), rtol=1e-8, atol=1e-6)


def test_another_native_adapter_invalidates_measured_global_maps(tmp_path):
    _, path = artificial_project(tmp_path)
    measured = NativeMagneticStartingPoint(path, refine_scale=True, refine_background=True, maximum_moment=6.)
    NativeReducedProfile(path)
    with pytest.raises(RuntimeError, match='process-global'):
        measured.residual(measured.initial[measured.moment_indices], measured.source_scale,
            measured.initial[measured.background_indices])
