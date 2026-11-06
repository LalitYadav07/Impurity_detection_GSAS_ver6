"""Native controls for portable unweighted physics and fresh scan binding."""
import hashlib
import json
import time
import numpy as np
import pytest

G = pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profiled import artificial_project
from gpx_recipe.live.native_profile import NativeReducedProfile
from gpx_recipe.symmetry.native_profiled import NativeMagneticStartingPoint
from gpx_recipe.preparation.magnetic import ARRAYS, MagneticResponse, load
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
    coefficients=validation['coefficient_validation']
    assert coefficients['all_coefficient_directions_checked']
    assert coefficients['symmetric_coefficients']==response.quadratic.shape[1]*(response.quadratic.shape[1]+1)//2
    assert coefficients['maximum_tolerance_ratio']<=1.
    assert coefficients['prepared_tensor_comparison']['maximum_tolerance_ratio']<=1.
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


def test_all_coefficients_are_checked_at_the_complete_24_variable_limit(tmp_path):
    project,path=artificial_project(tmp_path,sites=4)
    nuclear,magnetic=project.phases()
    for i,xyz in enumerate([(.19,.47,.63),(.51,.28,.79),(.62,.83,.17),(.87,.52,.39)],5):
        nuclear.add_atom(*xyz,'Fe','Fe'+str(i))
        magnetic.add_atom(*xyz,'Fe+3','Fe'+str(i))
        magnetic.data['Atoms'][-1][2]='M'
        magnetic.data['Atoms'][-1][7:10]=[1.,-1.5,.7]
    project.index_ids();project.save()
    native=NativeReducedProfile(path)
    response=compile_response(native)
    assert response.quadratic.shape[1]==24
    proof=response.proof['coefficient_validation']
    assert proof['symmetric_coefficients']==300 and proof['native_jacobian_calls']==0
    assert proof['native_structure_factor_calls']==1153
    assert proof['all_coefficient_directions_checked']
    assert proof['maximum_tolerance_ratio']<=1.
    # Retain the slower full native powder Jacobian as an independent oracle.
    from gpx_recipe.live.quadratic_qualification import check_coefficients
    indices=[i for i,k in enumerate(native.vary) if '::AM' in k or k.startswith('::constr')]
    backgrounds=[i for i,k in enumerate(native.vary) if ':Back;' in k]
    histogram=native.histo_phases[0][native.parts[0][0]];scale=f":{histogram['hId']}:Scale"
    scale_index=native.vary.index(scale) if scale in native.vary else None
    original=native.x0.copy();source_scale=native.parms[scale]
    def derivatives(moment):
        values=original.copy();values[indices]=moment;values[backgrounds]=0.
        native.parms[scale]=1.
        if scale_index is not None:values[scale_index]=1.
        return native.profile(values,jacobian=True)[1][indices]
    began=time.monotonic()
    try:independent=check_coefficients(response.quadratic,derivatives)
    finally:
        native.parms[scale]=source_scale;native.profile(original)
    oracle_seconds=time.monotonic()-began
    assert max(independent['maximum_tolerance_ratios'])<=1.
    # Construct a response corruption invisible to the original seven random
    # probes. The complete native-coefficient check must still reject it.
    rng=np.random.default_rng(30917);probes=[]
    for _ in range(7):
        probes.append(rng.normal(size=24)*2.);rng.normal(size=len(backgrounds));rng.uniform(.3,3.)
    _,_,vh=np.linalg.svd(np.asarray(probes),full_matrices=True);unseen=vh[-1]
    bad=response.quadratic.copy();bad[len(bad)//2]+=1e-3*np.outer(unseen,unseen)
    forged=MagneticResponse(**{name:(bad if name=='quadratic' else getattr(response,name)) for name in ARRAYS},
        physics_sha256=response.physics_sha256,proof=response.proof)
    with pytest.raises(ValueError,match='independently rebuilt native coefficients'):
        qualify_response(forged,native)
    assert forged._native_seal is None
    (tmp_path/'qualification.json').write_text(json.dumps(dict(dimensions=24,
        compile_and_qualification_seconds=response.proof['seconds'],
        independent_full_powder_jacobian_seconds=oracle_seconds,
        independent_full_powder_jacobian=independent,
        coefficient_validation=proof,compiler=response.proof['compiler']),indent=2)+'\n')


def test_intensity_dependent_corrections_do_not_get_a_quadratic_preparation_seal(tmp_path):
    _,path=artificial_project(tmp_path)
    native=NativeReducedProfile(path)
    magnetic=next(p for p in native.histo_phases[1].values() if p['General']['Type']=='magnetic')
    histogram=native.histo_phases[0][native.parts[0][0]]
    native.parms[f"{magnetic['pId']}:{histogram['hId']}:Extinction"]=.1
    with pytest.raises(ValueError,match='Intensity-dependent'):
        compile_response(native)


def test_unavailable_reflection_check_automatically_retains_full_native_jacobian_checks(tmp_path,monkeypatch):
    from gpx_recipe.live import native_reflection
    _,path=artificial_project(tmp_path)
    original=native_reflection.compile_profile_reflections
    def unavailable(*args,**kwargs):
        if kwargs.get('verify_coefficients'):
            raise native_reflection.ReflectionCompilerUnavailable('Injected reflection capture unavailable')
        return original(*args,**kwargs)
    monkeypatch.setattr(native_reflection,'compile_profile_reflections',unavailable)
    response=compile_response(NativeReducedProfile(path))
    proof=response.proof['coefficient_validation']
    assert proof['all_coefficient_directions_checked'] and proof['native_jacobian_calls']==6
    assert proof['fallback_reason']=='Injected reflection capture unavailable'
