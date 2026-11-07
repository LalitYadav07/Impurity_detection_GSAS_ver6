from types import SimpleNamespace
import numpy as np
import pytest
from gpx_recipe.structural_live.optimization.feasible_gauss_newton import select_ensemble_starts


def candidates():
    return [(float(i),np.array([0.,i*.3,0.,0.,0.])) for i in range(4)]+[(100.,np.array([.8,0.,0.,0.,0.]))]


def test_displacement_reservation_survives_cost_sorting_without_adding_starts():
    prior=candidates()
    old=select_ensemble_starts([],prior,4,3)
    assert all(p[0]==0 for p in old)
    selected=select_ensemble_starts([],prior,4,3,mode_dimensions=1)
    assert len(selected)==4 and any(p[0]==.8 for p in selected)
    np.testing.assert_array_equal(selected[0],prior[0][1])
    assert all(np.linalg.norm(a[:3]-b[:3])>.2 for i,a in enumerate(selected) for b in selected[i+1:])


def test_warm_parent_guesses_cannot_remove_every_displaced_prior():
    warm=[(-3.,np.array([0.,-.8,0.,0.,0.])),(-2.,np.array([0.,-.4,0.,0.,0.]))]
    selected=select_ensemble_starts(warm,candidates(),4,3,mode_dimensions=1)
    assert len(selected)==4 and any(p[0]==.8 for p in selected)
    np.testing.assert_array_equal(selected[0],warm[0][1])


@pytest.mark.parametrize('modes',[None,0])
def test_legacy_and_zero_mode_selection_remain_unchanged(modes):
    prior=candidates()
    expected=select_ensemble_starts([],prior,4,3)
    actual=select_ensemble_starts([],prior,4,3,mode_dimensions=modes)
    np.testing.assert_array_equal(actual,expected)


def test_single_start_keeps_its_declared_single_best_guess():
    prior=candidates()
    actual=select_ensemble_starts([],prior,1,3,mode_dimensions=1)
    assert len(actual)==1
    np.testing.assert_array_equal(actual[0],prior[0][1])


@pytest.mark.parametrize('modes',[True,-1,4,1.5,'1'])
def test_ambiguous_or_invalid_displacement_blocks_are_refused(modes):
    with pytest.raises(ValueError,match='Mode dimensions'):
        select_ensemble_starts([],candidates(),4,3,mode_dimensions=modes)


def test_second_order_feature_is_recovered_when_parent_derivative_is_zero(monkeypatch):
    from test_structural_live_alternating_native_fit import setup
    from gpx_recipe.structural_live.optimization import profiled_native_fit as module
    from gpx_recipe.structural_live.observations import CumulativeCounts
    source,table,observation,truth,cache,_,_=setup()
    # A separate fixed nuclear reflection identifies the common scale. Without
    # it, this toy's scale/background freedom makes amplitude recovery
    # non-identifiable even when its diffraction curve is fitted exactly.
    nuclear=.02*np.exp(-((observation.x-.92)/.04)**2)
    old_native=source.native;old_jacobian=source.native_mode_jacobian
    source.native=lambda p:old_native(p)+nuclear
    source.native_mode_jacobian=lambda p:(source.native(p),old_jacobian(p)[1])
    cache=SimpleNamespace(evaluate=source.native,mode_jacobian=source.native_mode_jacobian,diagnostics=lambda:{})
    rate=np.exp(truth[-2])*(source.native(truth[:3]*source.radii)-table.renderer.background)+.002*np.exp(truth[-1])
    counts=np.random.default_rng(99021).poisson(rate*1e8)
    observation=CumulativeCounts.from_counts(observation.acquisition,'identified-second-order',0,1e8,
        observation.x,counts,1.,np.ones(len(counts),bool))
    # Several better-ranked cell-only guesses hide the deliberately worse
    # displaced guess. At the parent the emergent peak's derivative is zero.
    points=np.array([[0.,.12+j*.21,-.1] for j in [-2,-1,0,1,2]]+[[0.,.12,.11],[.9,.12,-.1]])
    bank=SimpleNamespace(points=points,curves=np.array([source.native(p*source.radii) for p in points]),
        proposal_cell_steps=np.full(3,.002))
    selector=module.select_ensemble_starts
    def legacy(warm,prior,starts,dimensions,**kwargs):return selector(warm,prior,starts,dimensions)
    monkeypatch.setattr(module,'select_ensemble_starts',legacy)
    old,_=module.fit(source,table,observation,bank,seconds=5,starts=4,evaluator=cache,qualification_iterations=8)
    assert all(a['initial_parameters'][0]==0 for a in old['attempts'])
    monkeypatch.setattr(module,'select_ensemble_starts',selector)
    new,_=module.fit(source,table,observation,bank,seconds=5,starts=4,evaluator=cache,qualification_iterations=8)
    assert any(a['initial_parameters'][0]==.9 for a in new['attempts'])
    assert len(new['attempts'])==len(old['attempts'])==4
    assert new['best']['deviance']<.25*old['best']['deviance']
    assert abs(new['best']['parameters'][0]-truth[0])<.02
    assert new['all_allowed_modes_retained'] and not new['model_exclusion_authorized']
