import numpy as np
import pytest

from gpx_recipe.structural_live.optimization.cell_derivative_model import CellDerivativeModel


def test_reuses_all_lattice_columns_without_turning_them_into_an_objective():
    model=CellDerivativeModel(1,3)
    modes=np.array([[2.,1.,0.]])
    cells=np.array([[0.,3.,1.],[1.,0.,2.]])
    jac=np.r_[modes,cells]
    calls=[]
    def build():calls.append(True);return cells
    zero=np.zeros(3);base=np.ones(3)
    model.observe_modes(zero,base,modes)
    np.testing.assert_array_equal(model.columns(zero,base,np.ones(3),build),cells)
    point=np.array([.02,.03,-.01]);rate=base+jac.T@point
    np.testing.assert_allclose(model.columns(point,rate,np.ones(3),build),cells,atol=1e-14)
    assert len(calls)==1 and model.last_reused
    assert model.diagnostics()['uncertainty_derivatives_qualified'] is False
    assert not hasattr(model,'predict')


def test_new_mode_derivative_does_not_erase_the_observed_prediction_error():
    model=CellDerivativeModel(1,2)
    zero=np.zeros(2);base=np.zeros(3);cells=np.ones((1,3));modes=np.ones((1,3))
    model.observe_modes(zero,base,modes)
    model.columns(zero,base,np.ones(3),lambda:cells)
    point=np.array([.02,.02]);rate=np.array([3.,1.,2.])
    model.observe_modes(point,rate,modes)
    calls=[]
    result=model.columns(point,rate,np.ones(3),lambda:(calls.append(True) or cells*2))
    assert calls and model.rejected_predictions==1 and not model.last_reused
    np.testing.assert_array_equal(result,cells*2)


def test_moving_anchor_cannot_extend_the_native_derivative_domain_indefinitely():
    model=CellDerivativeModel(0,1,max_reuses=10,maximum_step=.2)
    builds=[]
    for value in [0.,.1,.19,.21]:
        model.columns(np.array([value]),np.array([value]),np.ones(1),lambda:(builds.append(True) or np.ones((1,1))))
    assert len(builds)==2 and model.reuses==2


def test_masked_bins_do_not_force_a_rebuild_but_fitted_mismatches_do():
    model=CellDerivativeModel(0,1)
    model.columns([0.],[0.,0.],[1.,0.],lambda:np.ones((1,2)))
    model.columns([.1],[.1,100.],[1.,0.],lambda:pytest.fail('Excluded bins must not control derivative proposal reuse'))
    assert model.last_reused
    calls=[]
    model.columns([.15],[10.,100.],[1.,0.],lambda:(calls.append(True) or np.ones((1,2))))
    assert calls and not model.last_reused


def test_reuse_age_and_explicit_invalidation_force_native_rebuilds():
    model=CellDerivativeModel(0,1,max_reuses=1);calls=[]
    def build():calls.append(True);return np.ones((1,1))
    for value in [0.,.01,.02]:model.columns([value],[value],[1.],build)
    assert len(calls)==2
    model.invalidate();model.columns([.02],[.02],[1.],build)
    assert len(calls)==3 and model.latest_modes is None


def test_invalid_native_columns_cannot_become_a_reusable_proposal():
    model=CellDerivativeModel(0,2)
    with pytest.raises(ValueError,match='All finite native lattice'):
        model.columns([0.,0.],[0.],[1.],lambda:np.array([[np.nan],[0.]]))
    assert model.model is None


def test_cache_memory_limit_uses_fresh_native_columns_without_removing_modes():
    model=CellDerivativeModel(2,3,maximum_bytes=8)
    model.observe_modes(np.zeros(3),np.zeros(5),np.ones((2,5)))
    assert model.latest_modes is None
    calls=[]
    for _ in range(2):
        columns=model.columns(np.zeros(3),np.zeros(5),np.ones(5),
            lambda:(calls.append(True) or np.ones((1,5))))
        np.testing.assert_array_equal(columns,np.ones((1,5)))
    assert len(calls)==2 and model.model is None and not model.last_reused
    assert model.diagnostics()['memory_bypasses']==2


def test_joint_native_recovery_uses_the_same_proposal_model_and_original_cost():
    from test_structural_live_alternating_native_fit import setup
    from gpx_recipe.structural_live.optimization.feasible_gauss_newton import fit_feasible_gn
    from gpx_recipe.structural_live.optimization.fit_fast_native import residual_and_factor
    s,t,o,z,c,bank,_=setup();bank.diagnostics=lambda:dict(test_fixture=True)
    result,rate=fit_feasible_gn(s,t,o,seconds=2.,starts=1,maxiter=12,
        recycle_cells=True,box_steps=True,multiscale=True,offline_bank=bank,
        precomputed_proposal_steps=True)
    proposal=result['cell_derivative_recycling']['proposal_model']
    assert proposal['builds']>0 and proposal['reuses']>0
    assert result['best']['deviance']==pytest.approx(residual_and_factor(o,rate)[0]@residual_and_factor(o,rate)[0])
    assert not result['global_optimum_certified']


def test_fresh_peak_proposal_cannot_claim_native_stationarity_even_at_same_point():
    model=CellDerivativeModel(0,1);calls=[]
    def native():calls.append(True);return np.ones((1,3))
    model.columns([0.],[0.,0.,0.],np.ones(3),native,propose=lambda:np.zeros((1,3)))
    assert model.last_requires_native_retry and not model.last_reused and not calls
    model.columns([0.],[0.,0.,0.],np.ones(3),native)
    assert model.last_requires_native_retry
    result=model.columns([0.],[0.,0.,0.],np.ones(3),native,force_native=True,
                         propose=lambda:pytest.fail('Fresh native retry must bypass the proposal'))
    np.testing.assert_array_equal(result,np.ones((1,3)))
    assert calls and not model.last_requires_native_retry
    assert model.diagnostics()['forced_native_builds']==1


def test_unavailable_peak_proposal_preserves_all_native_columns():
    from gpx_recipe.structural_live.optimization.native_cell_proposals import ProposalUnavailable
    model=CellDerivativeModel(0,2)
    def unavailable():raise ProposalUnavailable('Changed native inventory')
    expected=np.arange(6.).reshape(2,3)
    np.testing.assert_array_equal(model.columns([0.,0.],[0.,0.,0.],np.ones(3),
        lambda:expected,propose=unavailable),expected)
    assert not model.last_requires_native_retry and model.diagnostics()['proposal_fallbacks']==1


def test_a_programming_failure_in_a_proposal_is_not_disguised_as_native_fallback():
    model=CellDerivativeModel(0,1)
    def broken():raise RuntimeError('Induced scientific callback failure')
    with pytest.raises(RuntimeError,match='scientific callback'):
        model.columns([0.],[0.],[1.],lambda:pytest.fail('Do not conceal the failure'),propose=broken)
