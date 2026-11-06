"""Independent native controls for capture during the original Rietveld call."""
import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from gpx_recipe.structural_live.optimization.captured_cell_derivative import CapturedCellDerivative
from gpx_recipe.structural_live.optimization.profile_cache import NativeProfileBank
from gpx_recipe.structural_live.optimization import native_profile_capture as capture


@pytest.mark.parametrize('kind',['PNC','PNT'])
@pytest.mark.parametrize('shifted',[False,True])
def test_one_required_native_pass_supplies_all_mode_columns_and_unseen_states(tmp_path,monkeypatch,kind,shifted):
    _,s=native_source(tmp_path/'inputs',kind);n=s.fast.native
    driver=n.math.getPowderProfile;peak_name='getFCJVoigt3' if kind=='PNC' else 'getEpsVoigt'
    peak=getattr(n.math.G2pwd,peak_name)
    point=np.r_[np.array([.07,-.09])*s.radii[:2],
        np.array([.11,-.13,.08])*s.radii[2:] if shifted else np.zeros(3)]
    original=s.native;calls=[]
    def native(p):calls.append(np.asarray(p).copy());return original(p)
    monkeypatch.setattr(s,'native',native)
    monkeypatch.setattr(s,'snapshot',lambda *_:pytest.fail('No second snapshot/profile construction'))
    monkeypatch.setattr(NativeProfileBank,'matrix',lambda *_:pytest.fail('No second Pawley response construction'))
    cache=CapturedCellDerivative(s)
    full=cache.native_evaluate(point)
    assert cache.builds==1,cache.diagnostics()
    assert len(calls)==cache.captured_native_calls==1
    assert n.math.getPowderProfile is driver and getattr(n.math.G2pwd,peak_name) is peak
    curve,jac=cache.mode_jacobian(point)
    assert len(calls)==1 and cache.derivative_calls==1
    np.testing.assert_allclose(curve,full,rtol=1e-11,atol=1e-8)
    direct,reference=s.native_mode_jacobian(point)
    np.testing.assert_allclose(direct,curve,rtol=1e-11,atol=1e-8)
    np.testing.assert_allclose(jac,reference,rtol=2e-6,atol=2e-7)
    for i in range(s.mode_dimensions):
        step=s.radii[i]*1e-5;lo=point.copy();hi=point.copy();lo[i]-=step;hi[i]+=step
        numerical=(s.native(hi)-s.native(lo))/(2*step)
        assert np.linalg.norm(numerical-jac[i])/max(np.linalg.norm(numerical),1e-20)<2e-4
    unseen=point.copy();unseen[:2]=np.array([-.12,.04])*s.radii[:2]
    np.testing.assert_allclose(cache.evaluate(unseen),s.native(unseen),rtol=1e-10,atol=1e-8)
    proof=cache.diagnostics()
    assert proof['entries']==proof['captured_native_calls']==1
    assert proof['maximum_capture_relative_error']<1e-9
    assert proof['numeric_cache_bytes']<=proof['maximum_numeric_cache_bytes']
    assert not proof['capture_fallbacks']


@pytest.mark.parametrize('reason',['unknown_driver','memory','mask'])
def test_capture_limitations_restore_original_functions_and_use_native_derivatives(tmp_path,monkeypatch,reason):
    _,s=native_source(tmp_path/'inputs','PNC');n=s.fast.native
    driver=n.math.getPowderProfile;peak=n.math.G2pwd.getFCJVoigt3
    if reason=='unknown_driver':monkeypatch.setattr(capture,'RIETVELD_DRIVERS',set())
    if reason=='mask':n.controls[f":{s.fast.slow.hist.id}:Exclude"]=[[30.,35.]]
    cache=CapturedCellDerivative(s,maximum_bytes=1 if reason=='memory' else 64*1024*1024)
    point=np.zeros(len(s.radii));actual=cache.native_evaluate(point)
    np.testing.assert_array_equal(actual,s.native(point))
    curve,jac=cache.mode_jacobian(point);reference,derivative=s.native_mode_jacobian(point)
    np.testing.assert_array_equal(curve,reference);np.testing.assert_array_equal(jac,derivative)
    assert not cache.cache and cache.capture_fallbacks
    assert cache.capture_attempts==1
    assert n.math.getPowderProfile is driver and n.math.G2pwd.getFCJVoigt3 is peak


def test_native_exception_releases_both_driver_and_peak_hooks(tmp_path):
    _,s=native_source(tmp_path/'inputs','PNC');n=s.fast.native
    driver=n.math.getPowderProfile;peak=n.math.G2pwd.getFCJVoigt3
    name=s.fast.slow.hist.name;phase=s.fast.slow.phase.name
    def broken():raise RuntimeError('induced owned native failure')
    with pytest.raises(RuntimeError,match='owned native failure'):
        capture.capture_rietveld_response(driver,n.math.G2pwd,n.parms,n.histdict,
            s.fast.x[s.fast.slice],n.histo_phases[0][name],n.histo_phases[1],n.controls,n.pawley,
            histogram_name=name,phase_name=phase,evaluate=broken,maximum_bytes=64*1024*1024)
    assert n.math.getPowderProfile is driver and n.math.G2pwd.getFCJVoigt3 is peak
    cache=CapturedCellDerivative(s);cache.native_evaluate(np.zeros(len(s.radii)))
    assert cache.builds==1


def test_exact_cell_cache_evicts_without_rounding_geometry_or_losing_mode_changes(tmp_path):
    _,s=native_source(tmp_path/'inputs','PNC');cache=CapturedCellDerivative(s,max_cells=2)
    for offset in [0.,.12,-.17,.21]:
        point=np.zeros(len(s.radii));point[-1]=offset*s.radii[-1]
        full=cache.native_evaluate(point)
        changed=point.copy();changed[0]=.06*s.radii[0]
        np.testing.assert_allclose(cache.evaluate(changed),s.native(changed),rtol=1e-10,atol=1e-8)
        assert len(cache.cache)<=2 and cache.numeric_bytes<=cache.maximum_bytes
    assert cache.captured_native_calls==4


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_zero_parent_intensity_keeps_a_column_that_can_generate_a_new_peak(tmp_path,kind):
    from gpx_recipe.execution import freeze
    from gpx_recipe.live.native_profile import NativeReducedProfile
    # P1 explicitly retains all reciprocal points. Identical atoms related
    # by a half translation cancel at odd h+k+l; moving one atom makes those
    # reflections observable without changing the metric or instrument.
    native_source(tmp_path/'profile-inputs',kind)
    p=G.G2Project(newgpx=str(tmp_path/'dormant.gpx'))
    phase=p.add_phase(phasename='Public cancellation control',spacegroup='P 1',
        cell=[4.6,5.2,6.3,90.,90.,90.])
    phase.add_atom(0.,0.,0.,'Fe','Fe1');phase.add_atom(.5,.5,.5,'Fe','Fe2')
    lower,upper=(12.,125.) if kind=='PNC' else (3.,40.)
    h=p.add_simulated_powder_histogram('public dormant control',str(tmp_path/'profile-inputs/profile.instprm'),
        lower,upper,Npoints=701,scale=10.,phases=[phase])
    freeze(p);ct=phase.data['General']['AtomPtrs'][1]
    phase.data['Atoms'][1][ct+1]='X'
    h.data['data'][1][1][:]=1.;p.save()
    n=NativeReducedProfile(p.filename,refresh_cell_reflections=True)
    hist=n.histo_phases[0][h.name];driver=n.math.getPowderProfile
    sl=n.parts[0][1];axis=np.asarray(hist['Data'][0][sl],float)
    matrix,bg,proof,native=capture.capture_rietveld_response(driver,n.math.G2pwd,n.parms,n.histdict,
        axis,hist,n.histo_phases[1],n.controls,n.pawley,histogram_name=h.name,phase_name=phase.name,
        evaluate=lambda:n.profile(n.x0),maximum_bytes=64*1024*1024)
    refs=hist['Reflection Lists'][phase.name]['RefList'];before=refs[:,9].copy()
    dormant=(before<1e-18)&(np.asarray(matrix.getnnz(axis=0)).ravel()>0)
    assert np.any(dormant) and proof['no_F_squared_division']
    values=n.x0.copy();index=next(i for i,name in enumerate(n.vary) if '::dAx:1' in name)
    values[index]+=.037
    changed=n.profile(values);after=hist['Reflection Lists'][phase.name]['RefList'][:,9]
    assert np.max(after[dormant])>1e-3
    np.testing.assert_allclose(matrix@before+bg,native,rtol=1e-11,atol=1e-9)
    np.testing.assert_allclose(matrix@after+bg,changed,rtol=1e-11,atol=1e-9)
