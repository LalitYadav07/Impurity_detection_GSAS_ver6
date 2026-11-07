"""Real native geometry/profile updates of observation-free prepared physics."""
import json
from pathlib import Path

import numpy as np
import pytest

G = pytest.importorskip('GSASII.GSASIIscriptable')
from test_preparation_theory_native import inputs
from gpx_recipe.live.native_profile import NativeReducedProfile
from gpx_recipe.live.reflection_factors import ScatteringFactorCache, digest_array
from gpx_recipe.preparation.native_magnetic import compile_response, adapt_response
from gpx_recipe.symmetry.native_magnetic import construct_theory
from gpx_recipe.symmetry.worker import Artifacts


def model(root, kind):
    root.mkdir()
    reference, protocol, candidate, seed = inputs(root, kind)
    directory = root/'candidate'
    directory.mkdir()
    result = construct_theory(reference['path'], Artifacts(directory), protocol, candidate, seed)
    return directory/result['gpx']


def compare_predictions(response, native):
    indices = [i for i, k in enumerate(native.vary) if '::AM' in k or k.startswith('::constr')]
    backgrounds = [i for i, k in enumerate(native.vary) if ':Back;' in k]
    histogram = native.histo_phases[0][native.parts[0][0]]
    scale = f":{histogram['hId']}:Scale"
    scale_index = native.vary.index(scale) if scale in native.vary else None
    rng = np.random.default_rng(901253)
    for _ in range(5):
        values = native.x0.copy()
        values[indices] = rng.normal(size=len(indices))*1.8
        values[backgrounds] = rng.normal(size=len(backgrounds))
        common_scale = float(rng.uniform(.5, 2.))
        native.parms[scale] = common_scale
        if scale_index is not None:
            values[scale_index] = common_scale
        direct = native.profile(values)
        predicted = response.predict(values[indices], scale=common_scale, background=values[backgrounds])
        np.testing.assert_allclose(predicted, direct, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_changed_recorded_conditions_reuse_the_same_full_magnetic_response(tmp_path,kind):
    from gpx_recipe.preparation.identity import ACQUISITION_PARAMETERS
    from gpx_recipe.preparation.native_magnetic import physics_identity
    path=model(tmp_path/'source',kind);cache=ScatteringFactorCache()
    first=compile_response(NativeReducedProfile(path),factor_cache=cache)
    project=G.G2Project(gpxfile=str(path))
    sample=project.histograms()[0].data['Sample Parameters']
    for index,name in enumerate(sorted(ACQUISITION_PARAMETERS)):sample[name]=118.+index
    changed=tmp_path/'new-conditions.gpx';project.save(str(changed))
    native=NativeReducedProfile(changed)
    assert physics_identity(native)==first.physics_sha256
    current,receipt=adapt_response(first,native,factor_cache=cache)
    assert receipt['action']=='same_context_requalified'
    assert current.physics_sha256==first.physics_sha256
    np.testing.assert_array_equal(current.quadratic,first.quadratic)
    compare_predictions(current,native)
    histogram=native.histo_phases[0][native.parts[0][0]]
    assert histogram['Sample Parameters']['Temperature']==sample['Temperature']


@pytest.mark.parametrize('kind', ['PNC', 'PNT'])
def test_changed_widths_reuse_scattering_but_render_and_check_new_native_peaks(tmp_path, kind):
    path = model(tmp_path/'source', kind)
    cache = ScatteringFactorCache(tmp_path/'factors')
    first = compile_response(NativeReducedProfile(path), factor_cache=cache)
    before = first.quadratic.copy()
    project = G.G2Project(gpxfile=str(path))
    profile = project.histograms()[0].data['Instrument Parameters'][0]
    for name, value in [('X', .1), ('Y', .08), ('Z', .03)]:
        profile[name][0] = profile[name][1] = value
    changed = tmp_path/'changed-profile.gpx'
    project.save(str(changed))
    native = NativeReducedProfile(changed)
    current, receipt = adapt_response(first, native, factor_cache=cache)
    assert receipt['action'] == 'current_context_rendered'
    assert receipt['scattering_factors_reused'] and receipt['scattering_qualification_reused']
    assert current.physics_sha256 != first.physics_sha256
    assert not np.array_equal(current.quadratic, before)
    proof = current.proof['compiler']['native_reflection_diagnostics']
    assert proof['native_amplitude_calls'] == proof['coefficient_structure_factor_calls'] == 0
    assert proof['native_profile_calls'] == 1
    compare_predictions(current, native)
    np.testing.assert_array_equal(first.quadratic, before)
    (tmp_path/'adaptation.json').write_text(json.dumps(receipt, indent=2)+'\n')


@pytest.mark.parametrize('change', ['cell', 'position', 'adp', 'ion'])
def test_changed_scattering_dependencies_recompute_instead_of_reusing_old_amplitudes(tmp_path, change):
    path = model(tmp_path/'source', 'PNT')
    cache = ScatteringFactorCache()
    first = compile_response(NativeReducedProfile(path), factor_cache=cache)
    project = G.G2Project(gpxfile=str(path))
    for phase in project.phases():
        magnetic = phase.data['General']['Type'] == 'magnetic'
        if change == 'cell':
            phase.data['General']['Cell'][1] *= 1.003
            phase.data['General']['Cell'][2] *= .998
        elif magnetic:
            atom = phase.data['Atoms'][0]
            cx, _, _, cia = phase.data['General']['AtomPtrs']
            if change == 'position':
                atom[cx] += .007
            elif change == 'adp':
                atom[cia+1] += .01
            else:
                atom[1] = 'Fe+2'
                G.SetupGeneral(phase.data, None)
    changed = tmp_path/('changed-'+change+'.gpx')
    project.save(str(changed))
    native = NativeReducedProfile(changed)
    current, receipt = adapt_response(first, native, factor_cache=cache)
    # The compiler first creates current factors; the separate qualification
    # step can then reuse those current factors. Inspect the creation receipt.
    proof = current.proof['compiler']['native_reflection_diagnostics']
    assert not proof['scattering_factors_reused']
    assert proof['native_amplitude_calls'] == 6
    assert proof['coefficient_structure_factor_calls'] == 19
    assert not receipt['scattering_factors_reused']
    assert proof['scattering_identity'] != first.proof['compiler']['native_reflection_diagnostics']['scattering_identity']
    assert current.physics_sha256 != first.physics_sha256
    compare_predictions(current, native)


def test_imported_factor_file_rechecks_native_and_rebuilds_a_forged_hint(tmp_path):
    path = model(tmp_path/'source', 'PNT')
    directory = tmp_path/'factors'
    first = compile_response(NativeReducedProfile(path), factor_cache=ScatteringFactorCache(directory))
    identity = first.proof['compiler']['native_reflection_diagnostics']['scattering_identity']
    target = directory/(identity+'.npz')
    with np.load(target, allow_pickle=False) as data:
        factors = data['factors'].copy()
        manifest = json.loads(str(data['manifest']))
    factors *= 2.
    manifest['sha256'] = digest_array(factors)  # Valid integrity does not establish native physics.
    np.savez_compressed(target, factors=factors, manifest=np.array(json.dumps(manifest)))
    native = NativeReducedProfile(path)
    recovered, receipt = adapt_response(first, native, factor_cache=ScatteringFactorCache(directory))
    assert receipt['action'] == 'same_context_requalified'
    assert recovered._native_seal is not None
    # The corrupt but checksummed factor hint was replaced by a checked native block.
    with np.load(target, allow_pickle=False) as data:
        np.testing.assert_allclose(data['factors'], factors/2., rtol=0, atol=0)
    compare_predictions(recovered, native)


def test_actual_exclusions_are_restored_and_bound_with_fresh_native_weights(tmp_path):
    from test_native_profiled import artificial_project
    from gpx_recipe.symmetry.native_profiled import NativeMagneticStartingPoint
    project, path = artificial_project(tmp_path)
    cache = ScatteringFactorCache()
    first = compile_response(NativeReducedProfile(path), factor_cache=cache)
    histogram = project.histograms()[0]
    arrays = histogram.data['data'][1]
    mask = np.arange(len(arrays[0])) % 73 == 0
    for i in range(3):
        arrays[i] = np.ma.array(arrays[i], mask=mask)
    histogram.data['Limits'].append([45., 48.])
    histogram.data['data'][0]['wtFactor'] = 2.5
    project.save()
    native = NativeReducedProfile(path)
    actual = native.histo_phases[0][native.parts[0][0]]
    original_data = actual['Data']
    originals = actual['Data'][:3]
    values = [np.ma.getdata(a).copy() for a in originals]
    masks = [np.ma.getmaskarray(a).copy() for a in originals]
    exclusion_key = f":{actual['hId']}:Exclude"
    exclusions = native.controls[exclusion_key]
    response, receipt = adapt_response(first, native, factor_cache=cache)
    assert receipt['action'] == 'same_context_requalified'
    assert native.controls[exclusion_key] is exclusions
    assert actual['Data'] is original_data
    for original, current, value, source_mask in zip(originals, actual['Data'][:3], values, masks):
        np.testing.assert_array_equal(np.ma.getdata(current), value)
        np.testing.assert_array_equal(np.ma.getmaskarray(current), source_mask)
    measured = NativeMagneticStartingPoint(path, refine_scale=True, refine_background=True, maximum_moment=6.)
    moment = measured.initial[measured.moment_indices]*.8
    background = measured.initial[measured.background_indices]*1.2
    direct = measured.residual(moment, 120., background)
    objective = response.bind(x=np.ma.getdata(arrays[0]), y=np.ma.getdata(arrays[1]),
        weights=np.ma.getdata(arrays[2]), fit_mask=~measured.residual_mask,
        # The native GPX reader clears stored row masks, then applies the
        # explicit exclusion intervals to fitting. Its norm is taken before
        # those exclusions. All finite observations in this control remain
        # in that norm, including points excluded from fitting.
        normalization_mask=np.ones(len(arrays[0]), bool), physics_sha256=response.physics_sha256,
        maximum_moment=6., weight_factor=2.5)
    predicted = objective.offset + 120.*(objective.nuclear + np.einsum(
        'i,nij,j->n', moment, objective.quadratic, moment)) + objective.background@background
    np.testing.assert_allclose(predicted, direct, rtol=1e-8, atol=1e-6)
    assert objective.normalization == pytest.approx(measured.histogram['Residuals']['sumwYo'], rel=1e-12)


def test_failed_native_check_also_restores_measurement_masks(tmp_path, monkeypatch):
    from test_native_profiled import artificial_project
    _, path = artificial_project(tmp_path, masked=True)
    native = NativeReducedProfile(path)
    actual = native.histo_phases[0][native.parts[0][0]]
    original_data = actual['Data']
    originals = actual['Data'][:3].copy()
    exclusions = dict(native.controls)
    # Obtain compatible physical preparation without changing the source file.
    from gpx_recipe.preparation.native_magnetic import _unmasked_physical_context
    with _unmasked_physical_context(native):
        prepared = compile_response(native)
    def broken(*args, **kwargs):
        raise RuntimeError('Injected native calculation failure')
    monkeypatch.setattr(native, 'profile', broken)
    with pytest.raises(RuntimeError, match='Injected native'):
        adapt_response(prepared, native)
    assert actual['Data'] is original_data
    for original, current in zip(originals, actual['Data'][:3]):
        np.testing.assert_array_equal(np.ma.getdata(current), np.ma.getdata(original))
        np.testing.assert_array_equal(np.ma.getmaskarray(current), np.ma.getmaskarray(original))
    key = f":{actual['hId']}:Exclude"
    assert native.controls[key] is exclusions[key]


def test_model_exploration_uses_prepared_factors_with_fresh_observations(tmp_path):
    from test_native_profiled import artificial_project
    from gpx_recipe.symmetry.native_profiled import NativeMagneticStartingPoint
    project, path = artificial_project(tmp_path)
    cache = ScatteringFactorCache()
    compile_response(NativeReducedProfile(path), factor_cache=cache)
    histogram = project.histograms()[0]
    histogram.data['data'][1][1] *= 1.03
    histogram.data['data'][1][2] *= .7
    project.save()
    native = NativeMagneticStartingPoint(path, refine_scale=True, refine_background=True, maximum_moment=6.)
    objective, receipt = native.build(factor_cache=cache)
    assert receipt['compiler'] == 'native_reflection_amplitudes'
    proof = receipt['compiler_diagnostics']
    assert proof['scattering_factors_reused'] and proof['scattering_qualification_reused']
    assert proof['native_amplitude_calls'] == proof['coefficient_structure_factor_calls'] == 0
    point = native.initial[native.moment_indices]*.9
    background = native.initial[native.background_indices]*1.1
    actual = native.residual(point, 120., background)
    predicted = objective.offset + 120.*(objective.nuclear + np.einsum(
        'i,nij,j->n', point, objective.quadratic, point)) + objective.background@background
    np.testing.assert_allclose(predicted, actual, rtol=1e-8, atol=1e-6)
