import numpy as np
import pytest
from scipy.optimize import minimize
from test_structural_live_multiaxis_fit import fixture
from gpx_recipe.structural_live.optimization.profiled_nuisance import ConditionalNuisance

@pytest.mark.parametrize('point',[[.5,.12,-.1],[.35,.12,-.1],[.9,.12,-.1]])
def test_poisson_profile_agrees_with_independent_bounded_minimization(point):
    s,t,o,z=fixture();signal=s.native(np.array(point)*s.radii)-t.renderer.background
    conditional=ConditionalNuisance(o,signal,np.zeros(len(signal)),np.full(len(signal),.002))
    cost,rate,_,_=conditional.solve()
    def objective(x):
        gradient,_,_,_=conditional.gradient_hessian(x)
        return conditional.value(x)[0],2*gradient
    ref=minimize(objective,[1.,1.],jac=True,method='L-BFGS-B',bounds=list(zip(conditional.lower,conditional.upper)),
        options=dict(ftol=1e-14,gtol=1e-8,maxiter=1000,maxls=50))
    assert cost<=ref.fun+1e-6
    assert conditional.projected_score<1e-4 and np.isfinite(rate).all()

def test_implicit_profile_derivative_includes_the_design_derivative():
    s,t,o,z=fixture();point=np.array([.49,.11,-.08])*s.radii
    curve,jac=s.native_mode_jacobian(point)
    conditional=ConditionalNuisance(o,curve-t.renderer.background,np.zeros(len(curve)),np.full(len(curve),.002))
    _,rate,r,factor=conditional.solve();analytic=conditional.rate_jacobian(jac)[0]
    step=1e-6;lo=point.copy();hi=point.copy();lo[0]-=step;hi[0]+=step
    def value(p):
        c=ConditionalNuisance(o,s.native(p)-t.renderer.background,np.zeros(len(curve)),np.full(len(curve),.002))
        return c.solve()[1]
    numerical=(value(hi)-value(lo))/(2*step)
    np.testing.assert_allclose(analytic,numerical,rtol=2e-4,atol=2e-6)

def test_masked_bins_do_not_change_conditional_coefficients():
    s,t,o,z=fixture();curve=s.native(z[:-2]*s.radii)
    mask=o.mask.copy();mask[:5]=False
    from dataclasses import replace
    o=replace(o,mask=mask)
    def solve(signal):
        c=ConditionalNuisance(o,signal,np.zeros(len(curve)),np.full(len(curve),.002));c.solve();return c.coefficients
    signal=curve-t.renderer.background;expected=solve(signal)
    changed=signal.copy();changed[:5]=1e9
    np.testing.assert_allclose(solve(changed),expected,rtol=1e-12,atol=1e-12)

@pytest.mark.parametrize('kind',['gaussian','nonunit_counts','zero_counts','singular'])
def test_conditional_edge_cases_against_independent_optimizer(kind):
    from gpx_recipe.structural_live.observations import CumulativeObservation,CumulativeCounts
    s,t,o,z=fixture();signal=s.native(z[:-2]*s.radii)-t.renderer.background
    bg=np.full(len(signal),.002);fixed=np.full(len(signal),.0002)
    if kind=='gaussian':
        o=CumulativeObservation(o.acquisition,o.protocol_sha256,o.revision,o.exposure,o.x,o.y,o.variance,o.mask)
    else:
        counts=np.zeros(len(signal),dtype=int) if kind=='zero_counts' else o.counts
        scale=np.linspace(.7,1.3,len(signal)) if kind=='nonunit_counts' else 1.
        o=CumulativeCounts.from_counts(o.acquisition,o.protocol_sha256,0,o.exposure,o.x,counts,scale,o.mask)
    if kind=='singular':signal=bg*2
    c=ConditionalNuisance(o,signal,fixed,bg);cost,*_=c.solve()
    def objective(x):return c.value(x)[0],2*c.gradient_hessian(x)[0]
    ref=minimize(objective,[1.,1.],jac=True,method='L-BFGS-B',bounds=list(zip(c.lower,c.upper)),options=dict(ftol=1e-14,gtol=1e-9,maxiter=1000,maxls=50))
    assert cost<=ref.fun+1e-5
    assert c.projected_score<1e-4


@pytest.mark.parametrize('direction',[-1.,1.])
def test_gaussian_profile_is_independent_of_screened_nuisance_start(direction):
    """A tiny prepared-curve error must not seed another native objective."""
    from gpx_recipe.structural_live.observations import CumulativeObservation
    s,t,counts,z=fixture()
    observation=CumulativeObservation(counts.acquisition,counts.protocol_sha256,
        counts.revision,counts.exposure,counts.x,counts.y,counts.variance,counts.mask)
    signal=s.native(z[:-2]*s.radii)-t.renderer.background
    fixed=np.full(len(signal),.0002)
    background=np.full(len(signal),.002)
    reference=ConditionalNuisance(observation,signal,fixed,background)
    expected=reference.solve()
    # This perturbation is within the old early-exit tolerance. It models the
    # screening coefficients arriving from two numerically close start banks.
    initial=reference.coefficients+direction*np.array([1e-11,-1e-11])
    actual=ConditionalNuisance(observation,signal,fixed,background)
    got=actual.solve(initial)
    np.testing.assert_array_equal(actual.coefficients,reference.coefficients)
    np.testing.assert_array_equal(got[1],expected[1])
    assert got[0]==expected[0]
