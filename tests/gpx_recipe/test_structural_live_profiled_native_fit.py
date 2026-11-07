import numpy as np
import pytest
from types import SimpleNamespace
from test_structural_live_alternating_native_fit import setup
from gpx_recipe.structural_live.optimization.fit_fast_native import residual_and_factor
from gpx_recipe.structural_live.optimization import profiled_native_fit as module

def test_conditional_native_fit_preserves_objective_and_all_modes():
    s,t,o,z,c,b,_=setup()
    result,rate=module.fit(s,t,o,b,seconds=2,starts=1,evaluator=c)
    residual,_=residual_and_factor(o,rate)
    assert result['best']['deviance']==pytest.approx(residual@residual)
    assert result['all_allowed_modes_retained'] and len(result['best']['parameters'])==5
    assert result['independent_cache_checks']>0
    assert not result['model_exclusion_authorized']
    assert np.all(np.diff([p['deviance'] for p in result['trace']])<0)
    assert result['best']['deviance']<150


def test_joint_native_updates_preserve_the_same_observation_objective_and_all_coordinates():
    s,t,o,z,c,b,_=setup()
    result,rate=module.fit(s,t,o,b,seconds=2,starts=1,evaluator=c,joint_steps=True)
    residual,_=residual_and_factor(o,rate)
    assert result['best']['deviance']==pytest.approx(residual@residual)
    assert result['method']=='profiled_native_joint' and result['all_allowed_modes_retained']
    assert len(result['best']['parameters'])==5
    assert result['blocks'] and {row['kind'] for row in result['blocks']}=={'joint'}
    assert result['independent_cache_checks']>0
    assert result['best']['deviance']<150
    assert np.all(np.diff([p['deviance'] for p in result['trace']])<0)


def test_joint_updates_cannot_skip_native_verification_of_the_mode_response():
    s,t,o,z,c,b,_=setup()
    c.evaluate=lambda p:s.native(p)+.001*np.sin(o.x*23)
    with pytest.raises(ValueError,match='disagrees'):
        module.fit(s,t,o,b,seconds=2,starts=1,evaluator=c,joint_steps=True)


def test_joint_native_derivative_reference_keeps_every_coordinate():
    s,t,o,z,c,b,_=setup()
    def forbidden(point):raise AssertionError('The native derivative reference cannot use a mode-response matrix')
    c.evaluate=forbidden
    result,rate=module.fit(s,t,o,b,seconds=2,starts=1,evaluator=c,
        joint_steps=True,force_native_mode_derivatives=True)
    assert result['independent_cache_checks']==0
    assert result['best']['deviance']==pytest.approx(residual_and_factor(o,rate)[0]@residual_and_factor(o,rate)[0])
    assert result['best']['deviance']<150 and len(result['best']['parameters'])==5


def test_joint_update_control_must_be_an_explicit_boolean():
    with pytest.raises(ValueError,match='boolean'):
        module.fit(None,None,None,None,joint_steps='true')

def test_profiled_cache_error_is_not_hidden_by_scale_or_background():
    s,t,o,z,c,b,_=setup()
    c.evaluate=lambda p:s.native(p)+.001*np.sin(o.x*23)
    with pytest.raises(ValueError,match='disagrees'):
        module.fit(s,t,o,b,seconds=2,starts=1,evaluator=c)

def test_screening_timeout_keeps_exact_conditional_incumbent(monkeypatch):
    s,t,o,z,c,b,_=setup();clock=[0.];original=s.native
    monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    def slow(point):
        value=original(point);clock[0]=3.;return value
    s.native=slow
    result,rate=module.fit(s,t,o,b,seconds=1,starts=2,evaluator=c)
    assert result['starts']==0 and not result['attempts']
    residual,_=residual_and_factor(o,rate)
    assert result['best']['deviance']==pytest.approx(residual@residual)


def test_short_start_updates_lattice_before_expensive_displacement_derivatives(monkeypatch):
    s,t,o,z,c,b,_=setup();clock=[0.]
    monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    original=c.mode_jacobian
    def expensive_modes(point):
        value=original(point);clock[0]+=2.;return value
    c.mode_jacobian=expensive_modes
    result,rate=module.fit(s,t,o,b,seconds=1.,starts=1,evaluator=c)
    assert result['blocks'][0]['kind']=='cell'
    assert result['blocks'][0]['accepted_costs']
    assert result['attempts'][0]['reason']=='per_start_time_budget'
    assert result['best']['deviance']==pytest.approx(residual_and_factor(o,rate)[0]@residual_and_factor(o,rate)[0])


def test_forced_native_derivatives_do_not_build_a_single_step_response_cache():
    s,t,o,z,c,b,_=setup()
    def forbidden(point):raise AssertionError('Exact-native fitting must not build a fixed-cell response matrix')
    c.evaluate=forbidden
    result,rate=module.fit(s,t,o,b,seconds=2.,starts=1,evaluator=c,force_native_mode_derivatives=True)
    residual,_=residual_and_factor(o,rate)
    assert result['independent_cache_checks']==0
    assert result['best']['deviance']==pytest.approx(residual@residual)
    assert result['best']['deviance']<150
    assert {'cell','modes'}<={block['kind'] for block in result['blocks']}


def test_recycled_zero_score_is_rebuilt_before_declaring_lattice_stationarity(monkeypatch):
    from gpx_recipe.structural_live.optimization.cell_derivative_model import CellDerivativeModel
    instances=[]
    class ZeroOnce(CellDerivativeModel):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.injected=False;instances.append(self)
        def columns(self,point,rate,weights,build):
            if self.builds and not self.injected:
                self.injected=True;self.last_reused=True
                return np.zeros((self.dimensions-self.modes,len(rate)))
            return super().columns(point,rate,weights,build)
    monkeypatch.setattr(module,'CellDerivativeModel',ZeroOnce)
    s,t,o,z,c,b,_=setup()
    result,rate=module.fit(s,t,o,b,seconds=2,starts=1,evaluator=c,force_native_mode_derivatives=True)
    assert instances[0].injected
    assert result['cell_derivative_proposals']['invalidations']>=2
    assert result['cell_derivative_proposals']['builds']>=2
    assert result['best']['deviance']==pytest.approx(residual_and_factor(o,rate)[0]@residual_and_factor(o,rate)[0])
    assert np.all(np.diff([p['deviance'] for p in result['trace']])<0)


def test_exact_profile_memo_reuses_repeated_geometry_with_fresh_nuisance_fits():
    s,t,o,z,c,b,_=setup()
    result,rate=module.fit(s,t,o,b,seconds=2,starts=1,evaluator=c,force_native_mode_derivatives=True)
    assert result['exact_profile_memo']['hits']>0
    assert result['exact_profile_memo']['entries']<=2
    assert result['cell_derivative_proposals']['reuses']>0
    assert result['best']['deviance']==pytest.approx(residual_and_factor(o,rate)[0]@residual_and_factor(o,rate)[0])


def test_exact_profile_memo_cannot_bypass_native_context_ownership():
    s,t,o,z,c,b,_=setup()
    def replaced():raise RuntimeError('Native constraint owner was replaced')
    s.fast.native=SimpleNamespace(_check=replaced)
    with pytest.raises(RuntimeError,match='constraint owner'):
        module.fit(s,t,o,b,seconds=2,starts=1,evaluator=c,force_native_mode_derivatives=True)


def test_owned_capture_receives_required_native_calls_without_replacing_acceptance():
    s,t,o,z,c,b,_=setup();direct=[];original=s.native
    def native(point):
        direct.append(np.asarray(point).copy())
        return original(point)
    c.native_evaluate=native
    result,rate=module.fit(s,t,o,b,seconds=2,starts=1,evaluator=c)
    residual,_=residual_and_factor(o,rate)
    assert 0<len(direct)<result['full_native_calls']
    assert result['native_cell_probe_calls']==result['full_native_calls']-len(direct)
    assert result['independent_cache_checks']>0
    assert result['best']['deviance']==pytest.approx(residual@residual)


def test_zero_peak_proposal_is_retried_natively_before_stationarity(monkeypatch):
    s,t,o,z,c,b,_=setup()
    class ZeroProposal:
        anchor_calls=0
        def columns(self,point,steps,*,allowed,check_time=None):
            self.anchor_calls+=1
            curve=s.native(point*s.radii)
            return curve,np.zeros((s.cell_dimensions,len(curve))),np.ones(s.cell_dimensions,dtype=bool)
        def diagnostics(self):return dict(anchors=self.anchor_calls)
    proposal=ZeroProposal()
    monkeypatch.setattr(module,'cell_proposals_for_source',lambda source:proposal)
    result,rate=module.fit(s,t,o,b,seconds=2.,starts=1,evaluator=c)
    assert proposal.anchor_calls>0
    assert result['cell_derivative_proposals']['proposal_builds']>0
    assert result['cell_derivative_proposals']['forced_native_builds']>0
    assert result['best']['deviance']<150
    residual,_=residual_and_factor(o,rate)
    assert result['best']['deviance']==pytest.approx(residual@residual)


def test_peak_proposal_anchor_cannot_change_the_native_objective(monkeypatch):
    s,t,o,z,c,b,_=setup()
    class BadProposal:
        anchor_calls=0
        def columns(self,point,steps,*,allowed,check_time=None):
            self.anchor_calls+=1
            curve=s.native(point*s.radii)+.001*(1.+.1*np.sin(o.x*23))
            return curve,np.zeros((s.cell_dimensions,len(curve))),np.ones(s.cell_dimensions,dtype=bool)
        def diagnostics(self):return {}
    monkeypatch.setattr(module,'cell_proposals_for_source',lambda source:BadProposal())
    with pytest.raises(ValueError,match='anchor changed'):
        module.fit(s,t,o,b,seconds=2.,starts=1,evaluator=c)


def test_outer_small_progress_exit_also_requires_original_native_cell_directions(monkeypatch):
    s,t,o,z,c,b,_=setup()
    class FreshProposal:
        anchor_calls=0
        def columns(self,point,steps,*,allowed,check_time=None):
            self.anchor_calls+=1;curve=s.native(point*s.radii);rows=[]
            for column in range(s.mode_dimensions,len(point)):
                upper=point.copy();lower=point.copy();upper[column]+=1e-5;lower[column]-=1e-5
                rows.append((s.native(upper*s.radii)-s.native(lower*s.radii))/2e-5)
            return curve,np.asarray(rows),np.ones(s.cell_dimensions,dtype=bool)
        def diagnostics(self):return {}
    def tiny(j,residual,*args,**kwargs):
        gradient=j.T@residual
        return -1e-11*gradient/max(np.linalg.norm(gradient),1e-30),dict(information_score=1.)
    monkeypatch.setattr(module,'cell_proposals_for_source',lambda source:FreshProposal())
    monkeypatch.setattr(module,'bounded_damped_step',tiny)
    result,rate=module.fit(s,t,o,b,seconds=2.,starts=1,evaluator=c)
    assert result['cell_derivative_proposals']['proposal_builds']>0
    assert result['cell_derivative_proposals']['forced_native_builds']>0
    assert result['attempts'][0]['reason']=='local_block_progress_small'
