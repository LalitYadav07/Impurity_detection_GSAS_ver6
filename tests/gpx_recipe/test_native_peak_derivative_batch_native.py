import numpy as np
import pytest

powder=pytest.importorskip('GSASII.GSASIIpwd')
from gpx_recipe.live import native_peak_batch as helper
from gpx_recipe.live import native_peak_derivative_batch as batch


@pytest.fixture
def installed(tmp_path,monkeypatch):
    from pathlib import Path
    source=Path(helper.__file__).with_name('_native_peak_batch.c')
    helper.build(tmp_path)
    monkeypatch.setattr(helper,'_files',lambda:(source,tmp_path/'_native_peak_batch.so',tmp_path/'_native_peak_batch.receipt'))
    return batch.NativePeakDerivativeBatch(powder)


@pytest.mark.parametrize('columns',[1,4,6])
@pytest.mark.parametrize('case',['overlap','translated','mixed_widths','zero_height','empty'])
def test_original_peak_values_and_all_proposal_partials(installed,case,columns):
    rng=np.random.default_rng(720191);axis=np.linspace(-15.,30.,601)
    if case=='translated':axis+=100000.
    records={}
    for i in range(17 if case!='empty' else 0):
        position=float(rng.uniform(-1,2))+(100000. if case=='translated' else 0.)
        shape=np.exp(rng.uniform(-3,2,size=4)) if case=='mixed_widths' else np.array([.5,.25,1.,.125])
        low=int(rng.integers(0,40));high=int(rng.integers(300,602))
        records[(i,0,0)]=(np.r_[position,shape],0. if case=='zero_height' else float(rng.uniform(-3,4)),low,high)
    dp=rng.normal(size=(len(records),columns,5));dh=rng.normal(size=(len(records),columns))
    expected=np.zeros(len(axis));derivatives=np.zeros((columns,len(axis)))
    for row,(p,height,low,high) in enumerate(records.values()):
        values=powder.getdEpsVoigt(*p,axis[low:high]);native=powder.getEpsVoigt(*p,axis[low:high])[0]
        np.testing.assert_array_equal(values[0],native)
        expected[low:high]+=np.float64(height)*native
        partials=np.asarray(values[1:],float)
        for column in range(columns):
            derivatives[column,low:high]+=dh[row,column]*native.astype(float)+height*(dp[row,column]@partials)
    actual,result=installed.render_columns(records,axis,dp,dh)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_allclose(result,derivatives,rtol=1e-12,atol=1e-12)
    assert installed.calls==1 and installed.reflections==len(records)
    assert installed.points==sum(row[3]-row[2] for row in records.values())
    assert installed.maximum_workspace<=helper.MAXIMUM_BYTES
    assert installed.diagnostics()['derivatives_are_proposals']
    assert not installed.diagnostics()['convergence_authorized']


@pytest.mark.parametrize('changed',['compiled','wrapper'])
def test_replaced_derivative_function_cannot_reuse_its_pointer(installed,monkeypatch,changed):
    if changed=='compiled':monkeypatch.setattr(powder.pyd,'pydepsvoigt',lambda:None)
    else:monkeypatch.setattr(powder,'getdEpsVoigt',lambda:None)
    with pytest.raises(batch.BatchUnavailable,match='changed'):
        installed.render_columns({},np.arange(8.),np.empty((0,1,5)),np.empty((0,1)))


def test_missing_installed_helper_uses_the_original_path(monkeypatch):
    def missing():raise helper.BatchUnavailable('No installed helper')
    monkeypatch.setattr(helper,'_load',missing)
    assert batch.for_powder(powder) is None


def test_complete_native_lattice_columns_reuse_an_owned_anchor(tmp_path,installed):
    from test_native_profile_capture import native_source
    from gpx_recipe.structural_live.optimization.native_cell_proposals import NativeCellProposals
    _,source=native_source(tmp_path/'source','PNT')
    point=np.zeros(len(source.radii));steps=np.full(len(point),.004)
    allowed=lambda p:bool(np.all(abs(p)<=1) and np.isfinite(source.chemistry.log_kernel((p*source.radii)[None,:])[0]))
    reference=NativeCellProposals(source).columns(point,steps,allowed=allowed)
    engine=NativeCellProposals(source)
    actual=engine.columns(point,steps,allowed=allowed,native_anchor=reference[0])
    np.testing.assert_array_equal(actual[0],reference[0])
    np.testing.assert_array_equal(actual[2],reference[2])
    np.testing.assert_allclose(actual[1],reference[1],rtol=1e-10,atol=1e-10)
    assert engine.derivative_batches==engine.reused_native_anchors==1
    assert engine.anchor_calls==0
    assert engine.maximum_accounted_bytes<=engine.maximum_bytes


def test_unqualified_anchor_falls_back_to_the_original_native_proposal(tmp_path,installed):
    from test_native_profile_capture import native_source
    from gpx_recipe.structural_live.optimization.native_cell_proposals import NativeCellProposals
    _,source=native_source(tmp_path/'source','PNT')
    point=np.zeros(len(source.radii));steps=np.full(len(point),.004)
    allowed=lambda p:bool(np.all(abs(p)<=1) and np.isfinite(source.chemistry.log_kernel((p*source.radii)[None,:])[0]))
    reference=NativeCellProposals(source).columns(point,steps,allowed=allowed)
    engine=NativeCellProposals(source)
    actual=engine.columns(point,steps,allowed=allowed,native_anchor=reference[0]+1.)
    np.testing.assert_array_equal(actual[0],reference[0])
    np.testing.assert_array_equal(actual[1],reference[1])
    assert engine.derivative_batches==0 and engine.batch_fallbacks==1 and engine.anchor_calls==1


def test_batched_probe_timeout_restores_original_native_state_and_lock(tmp_path,installed):
    from test_native_profile_capture import native_source
    from gpx_recipe.structural_live.optimization.native_cell_proposals import NativeCellProposals
    from gpx_recipe.structural_live.optimization.native_profile_capture import _capture_lock
    _,source=native_source(tmp_path/'source','PNT')
    point=np.zeros(len(source.radii));anchor=source.native(point).copy();checks=[]
    allowed=lambda p:bool(np.all(abs(p)<=1) and np.isfinite(source.chemistry.log_kernel((p*source.radii)[None,:])[0]))
    def deadline():
        checks.append(True)
        if len(checks)==4:raise TimeoutError('Batched native proposal deadline')
    engine=NativeCellProposals(source)
    with pytest.raises(TimeoutError,match='deadline'):
        engine.columns(point,np.full(len(point),.004),allowed=allowed,check_time=deadline,native_anchor=anchor)
    assert engine.anchor_calls==1 and engine.derivative_batches==0
    np.testing.assert_array_equal(source.native(point),anchor)
    assert _capture_lock.acquire(blocking=False);_capture_lock.release()
