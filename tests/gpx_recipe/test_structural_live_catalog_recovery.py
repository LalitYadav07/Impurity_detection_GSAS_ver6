import numpy as np
import pytest
from gpx_recipe.structural_live.optimization.catalog_recovery_policy import CatalogRecoveryPolicy,round_candidates,prior_start_exclusions,optimization_status,select_recovery_candidates


def test_catalog_coverage_does_not_drop_bad_or_failed_families():
    rows=[dict(candidate_id='a',round_index=0,terminal=True,deviance=1),
          dict(candidate_id='b',round_index=0,terminal=True,state='failed')]
    assert round_candidates(['a','b','c'],rows,0)==['c']
    assert round_candidates(['a','b','c'],rows,1)==['a','b','c']
    with pytest.raises(ValueError):round_candidates(['a','a'],[],0)


def test_requested_restarts_do_not_imply_actual_coverage():
    row=dict(round_index=0,terminal=True,fit=dict(attempts=[dict(initial_parameters=[0,0,0],iterations=0)]))
    result=optimization_status([row])
    assert not result['independent_start_allocation_completed']
    assert not result['model_exclusion_authorized']
    assert not result['coverage_rounds_complete']


def test_second_round_avoids_repeating_initial_native_starts(tmp_path,monkeypatch):
    from test_structural_live_offline_start_bank import source
    from gpx_recipe.structural_live.optimization.offline_native_start_bank import build,OfflineNativeStartBank
    from gpx_recipe.structural_live.optimization.feasible_gauss_newton import fit_feasible_gn
    s,t,o,_=source(monkeypatch);p=tmp_path/'bank.npz';build(s,p);bank=OfflineNativeStartBank(s,p)
    first,_=fit_feasible_gn(s,t,o,seconds=5,starts=2,maxiter=2,multiscale=True,offline_bank=bank,balanced_restarts=True)
    exclusions=prior_start_exclusions(first)
    second,_=fit_feasible_gn(s,t,o,seconds=5,starts=2,maxiter=2,multiscale=True,offline_bank=bank,
        balanced_restarts=True,screening_exclusions=exclusions)
    assert exclusions and second['previous_starts_excluded']>0
    for new in prior_start_exclusions(second):
        assert all(np.linalg.norm(np.array(new[:-2])-np.array(old[:-2]))>.2 for old in exclusions)
    with pytest.raises(ValueError,match='complete finite'):
        fit_feasible_gn(s,t,o,screening_exclusions=[[0]])


@pytest.mark.parametrize('kwargs',[dict(first_starts=0),dict(second_starts=1),dict(second_seconds=0),dict(second_seconds=float('nan'))])
def test_invalid_coverage_policy(kwargs):
    with pytest.raises(ValueError):CatalogRecoveryPolicy(**kwargs)


def test_short_rounds_reuse_prepared_proposals_and_leave_time_for_optimization():
    policy=CatalogRecoveryPolicy()
    assert policy.fit_options(0)==dict(seconds=8.,starts=1,precomputed_proposal_steps=True,balanced_restarts=True)
    assert policy.fit_options(1)==dict(seconds=20.,starts=2,precomputed_proposal_steps=True,balanced_restarts=True)
    with pytest.raises(ValueError):policy.fit_options(2)


def test_pareto_allocation_keeps_simple_families_outside_top_three():
    rows=[dict(candidate_id=cid,state='fitted',deviance=cost,mode_dimensions=m,cell_dimensions=1)
          for cid,cost,m in [('a',10,10),('b',11,9),('c',12,8),('simple',15,2),('dominated',16,3),('sentinel',30,20)]]
    chosen=select_recovery_candidates(rows)
    assert chosen==['a','b','c','simple','sentinel']
    assert select_recovery_candidates(list(reversed(rows)))==chosen
    with pytest.raises(ValueError):select_recovery_candidates(rows+[rows[0]])
    with pytest.raises(ValueError):select_recovery_candidates([dict(rows[0],deviance=float('nan'))])
    assert select_recovery_candidates([dict(candidate_id='failed',state='failed')])==[]


def test_extra_round_does_not_erase_completed_coverage_and_failure_is_visible():
    rows=[dict(round_index=i,terminal=True,state='fitted' if i!=1 else 'failed') for i in range(3)]
    status=optimization_status(rows)
    assert status['coverage_rounds_complete']
    assert not status['coverage_rounds_successful']
    assert status['failed_rounds']==[1]
    assert not status['model_exclusion_authorized']
