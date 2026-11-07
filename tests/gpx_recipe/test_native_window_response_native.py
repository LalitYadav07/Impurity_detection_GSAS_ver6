"""Original GSAS-II passes and SciPy independently check packed columns."""
import numpy as np
import pytest
G=pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from gpx_recipe.structural_live.optimization.native_profile_capture import capture_rietveld_response,CaptureUnavailable
from gpx_recipe.structural_live.optimization import native_window_response as windows
from gpx_recipe.structural_live.optimization.captured_cell_derivative import CapturedCellDerivative


def capture(source,point,builder=None):
    n=source.fast.native;name=source.fast.slow.hist.name;phase=source.fast.slow.phase.name
    return capture_rietveld_response(n.math.getPowderProfile,n.math.G2pwd,n.parms,n.histdict,
        source.fast.x[source.fast.slice],n.histo_phases[0][name],n.histo_phases[1],n.controls,n.pawley,
        histogram_name=name,phase_name=phase,evaluate=lambda:source.native(point),
        maximum_bytes=64*1024*1024,packed_builder=builder)


@pytest.mark.parametrize('shifted',[False,True])
def test_original_native_curve_and_complete_unit_columns_survive_compaction(tmp_path,shifted):
    _,s=native_source(tmp_path/'inputs','PNT');builder=windows.for_source(s)
    assert builder is not None,'The publication image must contain the matching installed helper'
    p=np.zeros(len(s.radii))
    if shifted:p=np.array([.07,-.11,.13,-.17,.09])*s.radii
    reference,bg,proof,native=capture(s,p)
    packed,actual_bg,actual_proof,actual=capture(s,p,builder)
    np.testing.assert_array_equal(actual,native);np.testing.assert_array_equal(actual_bg,bg)
    assert isinstance(packed,windows.NativeWindowResponse)
    assert actual_proof['native_driver_calls']==1 and actual_proof['no_F_squared_division']
    rng=np.random.default_rng(104)
    for outputs in [1,2,16]:
        coefficients=rng.normal(size=(reference.shape[1],outputs))
        np.testing.assert_allclose(packed@coefficients,reference@coefficients,rtol=1e-13,atol=1e-10)
        order=rng.permutation(reference.shape[1])
        np.testing.assert_allclose(packed.reordered(order)@coefficients,
            reference[:,order]@coefficients,rtol=1e-13,atol=1e-10)
    assert packed.numeric_bytes<reference.data.nbytes+reference.indices.nbytes+reference.indptr.nbytes
    assert builder.products>=7


def test_shared_capture_uses_packed_windows_and_preserves_native_mode_derivatives(tmp_path):
    _,s=native_source(tmp_path/'inputs','PNT');cache=CapturedCellDerivative(s)
    p=np.array([.07,-.11,.13,-.17,.09])*s.radii
    curve=cache.native_evaluate(p);actual,jac=cache.mode_jacobian(p)
    native,expected=s.native_mode_jacobian(p)
    np.testing.assert_allclose(curve,native,rtol=1e-11,atol=1e-8)
    np.testing.assert_allclose(actual,native,rtol=1e-11,atol=1e-8)
    np.testing.assert_allclose(jac,expected,rtol=2e-6,atol=2e-7)
    assert cache.diagnostics()['response_storage']=={'original native windows':1}
    assert cache.captured_native_calls==1 and not cache.capture_fallbacks


def test_unavailable_installed_helper_retains_original_capture(tmp_path,monkeypatch):
    _,s=native_source(tmp_path/'inputs','PNT')
    def missing():raise windows.BatchUnavailable('Test missing image helper')
    monkeypatch.setattr(windows,'_load',missing)
    cache=CapturedCellDerivative(s);point=np.zeros(len(s.radii))
    np.testing.assert_array_equal(cache.native_evaluate(point),s.native(point))
    assert cache.diagnostics()['response_storage']=={'CSC columns':1}
    assert not cache.capture_fallbacks


def test_window_capture_budget_failure_restores_original_native_functions(tmp_path):
    _,s=native_source(tmp_path/'inputs','PNT');n=s.fast.native;driver=n.math.getPowderProfile;peak=n.math.G2pwd.getEpsVoigt
    builder=windows.for_source(s);builder.maximum_bytes=1
    with pytest.raises((ValueError,CaptureUnavailable),match='budget'):
        capture(s,np.zeros(len(s.radii)),builder)
    assert n.math.getPowderProfile is driver and n.math.G2pwd.getEpsVoigt is peak
    assert np.isfinite(s.native(np.zeros(len(s.radii)))).all()


def test_parent_cancellation_retains_peaks_that_appear_after_a_displacement(tmp_path):
    from gpx_recipe.execution import freeze
    from gpx_recipe.live.native_profile import NativeReducedProfile
    native_source(tmp_path/'profile-inputs','PNT')
    project=G.G2Project(newgpx=str(tmp_path/'cancellation.gpx'))
    phase=project.add_phase(phasename='Public cancellation control',spacegroup='P 1',
        cell=[4.6,5.2,6.3,90.,90.,90.])
    phase.add_atom(0.,0.,0.,'Fe','Fe1');phase.add_atom(.5,.5,.5,'Fe','Fe2')
    hist=project.add_simulated_powder_histogram('public dormant control',
        str(tmp_path/'profile-inputs/profile.instprm'),3.,40.,Npoints=701,scale=10.,phases=[phase])
    freeze(project);ct=phase.data['General']['AtomPtrs'][1];phase.data['Atoms'][1][ct+1]='X'
    hist.data['data'][1][1][:]=1.;project.save()
    n=NativeReducedProfile(project.filename,refresh_cell_reflections=True)
    histogram=n.histo_phases[0][hist.name];sl=n.parts[0][1]
    matrix,bg,proof,initial=capture_rietveld_response(n.math.getPowderProfile,n.math.G2pwd,n.parms,n.histdict,
        np.asarray(histogram['Data'][0][sl],float),histogram,n.histo_phases[1],n.controls,n.pawley,
        histogram_name=hist.name,phase_name=phase.name,evaluate=lambda:n.profile(n.x0),
        maximum_bytes=64*1024*1024,packed_builder=windows.NativeWindowBuilder(n._check))
    before=histogram['Reflection Lists'][phase.name]['RefList'][:,9].copy();dormant=before<1e-18
    assert dormant.any() and np.max(matrix@dormant.astype(float))>0.
    values=n.x0.copy();index=next(i for i,key in enumerate(n.vary) if '::dAx:1' in key);values[index]+=.037
    changed=n.profile(values);after=histogram['Reflection Lists'][phase.name]['RefList'][:,9]
    assert np.max(after[dormant])>1e-3 and proof['no_F_squared_division']
    np.testing.assert_allclose(matrix@before+bg,initial,rtol=1e-11,atol=1e-9)
    np.testing.assert_allclose(matrix@after+bg,changed,rtol=1e-11,atol=1e-9)
