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
    for factor in [.3,1.,1.7]:
        moments=native.initial[native.moment_indices]*factor
        background=native.initial[native.background_indices]*.9
        native.peak_memo=raw;expected=native.residual(moments,120.,background)
        native.peak_memo=memo;actual=native.residual(moments,120.,background)
        np.testing.assert_array_equal(actual,expected)
    assert memo.hits>0 and memo.misses>0


def test_refinable_profile_uses_original_native_rendering_by_default(tmp_path):
    project,path=artificial_project(tmp_path,sites=1)
    project.histograms()[0].set_refinements({'Instrument Parameters':['X']})
    project.save()
    native=NativeReducedProfile(path)
    assert not native.cell_indices and native.peak_memo.maximum_bytes==0
    native.profile(native.x0)
    assert native.peak_memo.diagnostics()['scope_bypasses']=={
        'variable-cell/profile workload uses original native rendering':1}
