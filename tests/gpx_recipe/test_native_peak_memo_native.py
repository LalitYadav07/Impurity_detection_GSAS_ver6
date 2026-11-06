"""Original native peak reuse retains complete profiles and native Jacobians."""
import numpy as np
import pytest

pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from test_native_profiled import artificial_project
from gpx_recipe.live.native_peak_memo import NativePeakMemo
from gpx_recipe.live.native_profile import NativeReducedProfile
from gpx_recipe.symmetry.native_profiled import NativeMagneticStartingPoint
from gpx_recipe.structural_live.optimization.captured_cell_derivative import CapturedCellDerivative


@pytest.mark.parametrize('kind',['PNC','PNT'])
@pytest.mark.parametrize('context',['plain','excluded','extinction'])
def test_changed_cells_profiles_and_native_derivatives_are_identical(tmp_path,kind,context):
    _,source=native_source(tmp_path/'inputs',kind)
    n=source.fast.native;raw=NativePeakMemo(0)
    assert n.cell_indices and n.peak_memo.maximum_bytes==0
    memo=NativePeakMemo()  # Exercise the helper explicitly, not its automatic policy.
    hfx=f":{source.fast.slow.hist.id}:"
    name=source.fast.slow.phase.name
    if context=='excluded':
        axis=n.histo_phases[0][source.fast.slow.hist.name]['Data'][0]
        n.controls[hfx+'Exclude']=[[float(axis[170]),float(axis[190])]]
    elif context=='extinction':
        phase=n.histo_phases[1][name]
        n.parms[f"{phase['pId']}:{source.fast.slow.hist.id}:Extinction"] = .001
    driver=n.math.getPowderProfile
    kernel=n.math.G2pwd.getEpsVoigt if kind=='PNT' else n.math.G2pwd.getFCJVoigt3
    for i in range(3):
        point=np.linspace(-.03,.05,len(source.radii))*source.radii*(i+1)
        # The complete native path is the reference, not another memo owner.
        n.peak_memo=raw;expected=source.native(point)
        n.peak_memo=memo;actual=source.native(point)
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(source.native(point),expected)
    for profile in [0.,.1]:
        n.parms[hfx+'X']=profile
        n.peak_memo=raw;expected,jac=n.profile(n.x0,jacobian=True)
        n.peak_memo=memo;actual,actual_jac=n.profile(n.x0,jacobian=True)
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(actual_jac,jac)
    assert memo.hits>0 and memo.misses>0
    assert memo.diagnostics()['accounted_cache_bytes']<=memo.maximum_bytes
    assert n.math.getPowderProfile is driver
    assert (n.math.G2pwd.getEpsVoigt if kind=='PNT' else n.math.G2pwd.getFCJVoigt3) is kernel


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_required_pass_response_capture_remains_owned_and_exact(tmp_path,kind):
    _,source=native_source(tmp_path/'inputs',kind)
    n=source.fast.native;cache=CapturedCellDerivative(source)
    point=np.linspace(-.04,.06,len(source.radii))*source.radii
    actual=cache.native_evaluate(point)
    assert cache.has_response(point)
    curve,jac=cache.mode_jacobian(point)
    n.peak_memo=NativePeakMemo(0)
    expected,derivatives=source.native_mode_jacobian(point)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_allclose(curve,expected,rtol=1e-10,atol=1e-8)
    np.testing.assert_allclose(jac,derivatives,rtol=2e-6,atol=2e-7)
    assert cache.diagnostics()['maximum_capture_relative_error']<1e-9


@pytest.mark.parametrize('masked',[False,True])
def test_model_magnetic_residuals_include_secondary_nuclear_phase_and_masks(tmp_path,masked):
    _,path=artificial_project(tmp_path,sites=2,constrained=True,masked=masked)
    native=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    assert native.peak_memo.maximum_bytes>0
    memo=native.peak_memo;raw=NativePeakMemo(0)
    # Other native cases may already have filled the process-wide shape store.
    # This control deliberately exercises both cold admission and later hits.
    memo.clear()
    for factor in [.3,1.,1.7]:
        moments=native.initial[native.moment_indices]*factor
        background=native.initial[native.background_indices]*.9
        native.peak_memo=raw;expected=native.residual(moments,120.,background)
        native.peak_memo=memo;actual=native.residual(moments,120.,background)
        np.testing.assert_array_equal(actual,expected)
    assert memo.hits>0 and memo.misses>0
    assert bool(memo.unmasked_grid_driver_calls)==(not masked)


def test_refinable_profile_uses_original_native_rendering_by_default(tmp_path):
    project,path=artificial_project(tmp_path,sites=1)
    project.histograms()[0].set_refinements({'Instrument Parameters':['X']})
    project.save()
    native=NativeReducedProfile(path)
    assert not native.cell_indices and native.peak_memo.maximum_bytes==0
    native.profile(native.x0)
    assert native.peak_memo.diagnostics()['scope_bypasses']=={
        'variable-cell/profile workload uses original native rendering':1}


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_fresh_native_contexts_share_exact_shapes_and_invalidate_changed_geometry(tmp_path,kind,monkeypatch):
    from GSASII import GSASIIscriptable as G, GSASIIlattice as lattice
    from gpx_recipe.live import native_peak_memo as module
    from test_preparation_adaptation_native import model
    monkeypatch.setattr(module,'_PROCESS_STORE',None)
    path=model(tmp_path/'source',kind)
    first=NativeReducedProfile(path)
    first.profile(first.x0)
    first_proof=first.peak_memo.diagnostics()
    assert first_proof['shared_across_contexts'] and first_proof['misses']>0
    second=NativeReducedProfile(path)
    assert second.peak_memo is not first.peak_memo
    assert second.peak_memo._store is first.peak_memo._store
    assert second.peak_memo.diagnostics()['calls']==0
    for factor in [.7,1.3]:
        point=second.x0.copy()
        for i,name in enumerate(second.vary):
            if '::AM' in name or name.startswith('::constr'):point[i]*=factor
        memo=second.peak_memo
        second.peak_memo=NativePeakMemo(0);expected=second.profile(point)
        second.peak_memo=memo;actual=second.profile(point)
        np.testing.assert_array_equal(actual,expected)
    assert second.peak_memo.hits>0
    with pytest.raises(RuntimeError,match='Native global constraint state was replaced'):
        first.profile(first.x0)
    project=G.G2Project(gpxfile=str(path))
    for phase in project.phases():
        cell=phase.data['General']['Cell']
        cell[1]*=1.001
        cell[7]=lattice.calc_V(lattice.cell2A(cell[1:7]))
    profile=project.histograms()[0].data['Instrument Parameters'][0]
    key='sig-1' if kind=='PNT' else 'U'
    profile[key][0]=profile[key][1]=float(profile[key][1])*1.1
    changed=tmp_path/'changed.gpx';project.save(str(changed))
    third=NativeReducedProfile(changed);memo=third.peak_memo
    third.peak_memo=NativePeakMemo(0);expected=third.profile(third.x0)
    third.peak_memo=memo;actual=third.profile(third.x0)
    np.testing.assert_array_equal(actual,expected)
    assert memo.misses>0  # Previous cell/profile values are never approximate hits.
    assert memo.diagnostics()['accounted_cache_bytes']<=module.MAXIMUM_BYTES


@pytest.mark.parametrize('masked',[False,True])
def test_shared_shapes_do_not_share_measurements_or_native_constraint_ownership(tmp_path,masked,monkeypatch):
    from gpx_recipe.live import native_peak_memo as module
    monkeypatch.setattr(module,'_PROCESS_STORE',None)
    project,path=artificial_project(tmp_path,sites=2,constrained=True,masked=masked)
    options=dict(refine_scale=True,refine_background=True,maximum_moment=6.)
    first=NativeMagneticStartingPoint(path,**options)
    seed=first.initial[first.moment_indices]
    bg=first.initial[first.background_indices]
    before=first.residual(seed,first.source_scale,bg)
    project.histograms()[0].data['data'][1][1]+=2.5
    project.save()
    second=NativeMagneticStartingPoint(path,**options)
    memo=second.peak_memo
    second.peak_memo=NativePeakMemo(0)
    expected=second.residual(seed,second.source_scale,bg)
    second.peak_memo=memo
    actual=second.residual(seed,second.source_scale,bg)
    np.testing.assert_array_equal(actual,expected)
    assert not np.array_equal(actual,before)
    assert memo.hits>0
    assert second.peak_memo._store is first.peak_memo._store
    with pytest.raises(RuntimeError,match='process-global'):
        first.residual(seed,first.source_scale,bg)
