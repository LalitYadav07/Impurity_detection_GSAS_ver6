from types import SimpleNamespace
import numpy as np
import pytest
from test_structural_live_multiaxis_fit import fixture
from gpx_recipe.structural_live.optimization import offline_native_start_bank as module
from gpx_recipe.structural_live.optimization.feasible_gauss_newton import fit_feasible_gn
from gpx_recipe.structural_live.optimization.fit_fast_native import residual_and_factor


def source(monkeypatch):
    s,t,o,truth=fixture();s.fast=SimpleNamespace(x=o.x)
    s.chemistry=SimpleNamespace(log_kernel=lambda points:np.zeros(len(points)))
    s.identity='candidate-one'
    monkeypatch.setattr(module,'context',lambda s:dict(candidate=s.identity,radii=s.radii.tolist(),axis=s.fast.x.tolist()))
    return s,t,o,truth


def test_precomputed_native_curves_do_not_quantize_the_continuous_fit(tmp_path,monkeypatch):
    s,t,o,truth=source(monkeypatch);path=tmp_path/'bank.npz';receipt=module.build(s,path)
    bank=module.OfflineNativeStartBank(s,path)
    for p,y in zip(bank.points,bank.curves):np.testing.assert_array_equal(y,s.native(p*s.radii))
    assert receipt['native_curves']==73 and not receipt['interpolation']
    result,rate=fit_feasible_gn(s,t,o,multiscale=True,offline_bank=bank,seconds=5,starts=4,maxiter=40)
    estimate=np.array(result['best']['parameters'])
    # This fixture has a scale/background/structural-amplitude degeneracy;
    # parameter identity is not an identifiable target. Test the attainable
    # native likelihood and that the solution leaves the discrete start bank.
    true_rate=np.exp(truth[-2])*(s.native(truth[:3]*s.radii)-t.renderer.background)+.002*np.exp(truth[-1])
    residual,_=residual_and_factor(o,true_rate)
    assert result['exact_native_deviance']<=residual@residual+1.
    assert np.min(np.linalg.norm(bank.points-estimate[:3],axis=1))>.01
    assert result['direct_screening_calls']==0 and result['offline_screening_calls']==73
    assert result['global_optimum_certified'] is False and np.isfinite(rate).all()


def test_changed_candidate_domain_and_tampered_cache_are_rejected(tmp_path,monkeypatch):
    s,_,_,_=source(monkeypatch);path=tmp_path/'bank.npz';module.build(s,path)
    s.identity='candidate-two'
    with pytest.raises(ValueError,match='context'):module.OfflineNativeStartBank(s,path)
    s.identity='candidate-one';s.radii=s.radii*1.01
    with pytest.raises(ValueError,match='context'):module.OfflineNativeStartBank(s,path)
    s.radii=s.radii/1.01
    path.write_bytes(path.read_bytes()+b'tampered')
    with pytest.raises(ValueError,match='checksum'):module.OfflineNativeStartBank(s,path)


def test_chemically_infeasible_starts_are_accounted_for(tmp_path,monkeypatch):
    s,_,_,_=source(monkeypatch)
    s.chemistry=SimpleNamespace(log_kernel=lambda points:np.where(points[:,0]>=0,0.,-np.inf))
    path=tmp_path/'bank.npz';r=module.build(s,path);b=module.OfflineNativeStartBank(s,path)
    assert r['native_curves']+r['chemically_infeasible_points']==r['proposed_points']
    assert r['chemically_infeasible_points']>0 and np.all(b.points[:,0]>=0)
    with pytest.raises(ValueError,match='immutable'):module.build(s,path)


def test_prepared_secants_remove_online_selection_without_changing_native_target(tmp_path,monkeypatch):
    from gpx_recipe.structural_live.optimization.constrained_native_fit import FeasibleNativeObjective
    s,t,o,_=source(monkeypatch);path=tmp_path/'bank.npz';receipt=module.build(s,path)
    bank=module.OfflineNativeStartBank(s,path)
    assert receipt['proposal_step_selection']['observations_used'] is False
    assert receipt['proposal_step_selection']['derivative_qualified'] is False
    def forbidden(*args):raise AssertionError('online selection should have been prepared')
    monkeypatch.setattr(FeasibleNativeObjective,'select_proposal_steps',forbidden)
    result,rate=fit_feasible_gn(s,t,o,multiscale=True,offline_bank=bank,adaptive_cells=True,precomputed_proposal_steps=True,seconds=5.,starts=4)
    residual,_=residual_and_factor(o,rate)
    assert result['precomputed_proposal_steps'] and result['exact_native_deviance']==pytest.approx(residual@residual)


def test_joint_cell_bank_is_explicit_and_cannot_be_reinterpreted(tmp_path,monkeypatch):
    s,_,_,_=source(monkeypatch);path=tmp_path/'bank.npz';receipt=module.build(s,path,policy='joint_cell_v1')
    bank=module.OfflineNativeStartBank(s,path)
    assert len(bank.points)==185 and receipt['policy']=='joint_cell_v1'
    import json
    metadata=path.with_suffix('.json');record=json.loads(metadata.read_text());record['policy']='multiscale_v1';metadata.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='shapes'):module.OfflineNativeStartBank(s,path)
