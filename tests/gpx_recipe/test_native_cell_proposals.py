"""Native controls for streamed proposal directions, never an exact Jacobian."""
import numpy as np
import pytest

pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from gpx_recipe.structural_live.optimization import native_cell_proposals as module
from gpx_recipe.structural_live.optimization.native_profile_capture import _capture_lock


def allowed(source,point):
    if np.any(abs(point)>1):return False
    try:source.check(point*source.radii)
    except ValueError:return False
    return True


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_streamed_native_anchor_and_cell_directions_restore_the_scientific_state(tmp_path,kind):
    _,s=native_source(tmp_path/'inputs',kind);n=s.fast.native
    point=np.linspace(-.03,.04,len(s.radii));steps=np.full(len(point),.004)
    engine=module.NativeCellProposals(s);driver=n.math.getPowderProfile
    kernel=module.KERNELS[kind][0];original=getattr(n.math.G2pwd,kernel)
    anchor,columns,usable=engine.columns(point,steps,allowed=lambda p:allowed(s,p))
    np.testing.assert_array_equal(anchor,s.native(point*s.radii))
    assert columns.shape==(s.cell_dimensions,len(anchor))
    assert np.any(usable),engine.diagnostics()
    for index in np.flatnonzero(usable):
        upper=point.copy();lower=point.copy();column=s.mode_dimensions+index
        upper[column]+=.004;lower[column]-=.004
        reference=(s.native(upper*s.radii)-s.native(lower*s.radii))/.008
        # A useful proposal direction is distinct from exact derivative or
        # covariance qualification. Accepted scores always remain native.
        cosine=float(reference@columns[index]/max(np.linalg.norm(reference)*np.linalg.norm(columns[index]),1e-30))
        assert cosine>.99
    assert n.math.getPowderProfile is driver and getattr(n.math.G2pwd,kernel) is original
    assert _capture_lock.acquire(blocking=False);_capture_lock.release()
    proof=engine.diagnostics()
    assert proof['maximum_accounted_workspace_bytes']<=proof['workspace_budget_bytes']
    assert proof['full_native_anchor_calls']==1
    assert proof['scalar_driver_calls']+proof['batched_scalar_probe_calls']==2*s.cell_dimensions
    assert proof['batched_scalar_anchor_checks']==int(kind=='PNT')
    assert not proof['uncertainty_derivatives_qualified']


@pytest.mark.parametrize('reason',['memory','mask','driver','derivative'])
def test_unsupported_contexts_keep_original_native_functions_and_fallback(tmp_path,monkeypatch,reason):
    _,s=native_source(tmp_path/'inputs','PNT');n=s.fast.native;driver=n.math.getPowderProfile
    peak=n.math.G2pwd.getEpsVoigt;derivative=n.math.G2pwd.getdEpsVoigt
    if reason=='mask':n.controls[f":{s.fast.slow.hist.id}:Exclude"]=[[4000.,5000.]]
    if reason=='driver':monkeypatch.setattr(module,'RIETVELD_DRIVERS',set())
    if reason=='derivative':monkeypatch.setattr(module,'KERNELS',dict(module.KERNELS,PNT=(*module.KERNELS['PNT'][:3],'not-qualified')))
    engine=module.NativeCellProposals(s,maximum_bytes=1 if reason=='memory' else 64*1024*1024)
    with pytest.raises(module.ProposalUnavailable):
        engine.columns(np.zeros(len(s.radii)),np.full(len(s.radii),.004),allowed=lambda p:allowed(s,p))
    assert n.math.getPowderProfile is driver and n.math.G2pwd.getEpsVoigt is peak
    assert n.math.G2pwd.getdEpsVoigt is derivative
    assert _capture_lock.acquire(blocking=False);_capture_lock.release()
    assert np.isfinite(s.native(np.zeros(len(s.radii)))).all()
    assert engine.diagnostics()['fallbacks']==1


def test_failure_during_scalar_probes_restores_original_full_profile(tmp_path,monkeypatch):
    # Keep coverage of the original scalar-driver fallback as well as the
    # separate batched-collector failure controls below.
    monkeypatch.setattr(module,'scalar_collector_for_source',lambda *args:None)
    _,s=native_source(tmp_path/'inputs','PNT');n=s.fast.native
    point=np.zeros(len(s.radii));expected=s.native(point);original=s.native;calls=[]
    def fail_second_probe(at):
        calls.append(np.asarray(at).copy())
        if len(calls)==2:raise RuntimeError('Induced scalar probe failure')
        return original(at)
    monkeypatch.setattr(s,'native',fail_second_probe)
    engine=module.NativeCellProposals(s);driver=n.math.getPowderProfile;peak=n.math.G2pwd.getEpsVoigt
    with pytest.raises(RuntimeError,match='scalar probe failure'):
        engine.columns(point,np.full(len(point),.004),allowed=lambda p:allowed(s,p))
    assert len(calls)==3 and np.array_equal(calls[-1],point)
    assert n.math.getPowderProfile is driver and n.math.G2pwd.getEpsVoigt is peak
    np.testing.assert_array_equal(original(point),expected)
    assert _capture_lock.acquire(blocking=False);_capture_lock.release()


def test_deadline_between_probes_restores_native_state_before_propagating(tmp_path):
    _,s=native_source(tmp_path/'inputs','PNT');point=np.zeros(len(s.radii))
    expected=s.native(point);checks=[]
    def deadline():
        checks.append(True)
        if len(checks)==4:raise TimeoutError('Induced cell allocation deadline')
    engine=module.NativeCellProposals(s)
    with pytest.raises(TimeoutError,match='allocation deadline'):
        engine.columns(point,np.full(len(point),.004),allowed=lambda p:allowed(s,p),check_time=deadline)
    np.testing.assert_array_equal(s.native(point),expected)
    assert engine.anchor_calls==1
    assert _capture_lock.acquire(blocking=False);_capture_lock.release()
