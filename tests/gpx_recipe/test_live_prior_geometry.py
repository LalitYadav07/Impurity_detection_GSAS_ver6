"""Physical prior invariance and replica-health adversarial controls."""
import time

import numpy as np
import pytest

from gpx_recipe.live.statistics import QuadraticObservation,compare_replicas,sample_posterior


def model(forms,quadratic=None,lower=0.,upper=1.):
    n=forms.shape[1]
    if quadratic is None:quadratic=np.zeros((n,n,3))
    return QuadraticObservation(np.zeros(quadratic.shape[-1]),quadratic,
        np.zeros(quadratic.shape[-1]),np.ones(quadratic.shape[-1]),forms,lower,upper)


class CountedRandom:
    def __init__(self,seed,maximum=500_000):
        self.rng=np.random.default_rng(seed);self.proposals=0;self.maximum=maximum
    def _count(self,size):
        if isinstance(size,tuple) and len(size)==2:
            self.proposals+=size[0]
            if self.proposals>self.maximum:raise AssertionError('Prior proposal explosion')
    def normal(self,*args,**kwargs):
        self._count(kwargs.get('size'));return self.rng.normal(*args,**kwargs)
    def uniform(self,*args,**kwargs):
        self._count(kwargs.get('size',args[2] if len(args)>2 else None))
        return self.rng.uniform(*args,**kwargs)
    def random(self,*args,**kwargs):return self.rng.random(*args,**kwargs)


def orbit_forms(orbits):
    n=3*orbits;forms=np.zeros((orbits,n,n))
    for i in range(orbits):forms[i,3*i:3*i+3,3*i:3*i+3]=np.eye(3)
    return forms


def test_dense_equivalent_constraint_basis_does_not_revert_to_a_huge_uniform_box():
    forms=orbit_forms(8);rng=np.random.default_rng(20261106)
    rotation,_=np.linalg.qr(rng.normal(size=(24,24)))
    dense=np.einsum('ia,sij,jb->sab',rotation,forms,rotation)
    current=model(dense);counted=CountedRandom(19)
    draws=current.prior(counted,64,deadline=time.monotonic()+20)
    assert current.valid(draws).all() and counted.proposals<=500_000
    assert current.prior_diagnostics[0]['complete_original_constraints_checked']
    physical=draws@rotation.T
    assert np.max(np.linalg.norm(physical.reshape(-1,8,3),axis=2))<=1.+1e-12


def test_equivalent_bases_preserve_the_uniform_physical_shell_prior():
    forms=orbit_forms(2);rng=np.random.default_rng(661)
    rotation,_=np.linalg.qr(rng.normal(size=(6,6)))
    transform=rotation@np.diag([.3,.5,1.,1.5,2.,3.])
    changed=np.einsum('ia,sij,jb->sab',transform,forms,transform)
    original=model(forms,lower=.3);equivalent=model(changed,lower=.3)
    a=original.prior(np.random.default_rng(18),12000)
    b=equivalent.prior(np.random.default_rng(19),12000)@transform.T
    expected=(1-.3**5)/(1-.3**3)/5
    for physical in (a,b):
        np.testing.assert_allclose(np.mean(physical,axis=0),0.,atol=.025)
        np.testing.assert_allclose(np.mean(physical**2,axis=0),expected,atol=.02)
        np.testing.assert_allclose(np.mean(physical[:,0]*physical[:,3]),0.,atol=.02)
        radii=np.linalg.norm(physical.reshape(-1,2,3),axis=2)
        assert np.min(radii)>=.3-1e-12 and np.max(radii)<=1.+1e-12
        # Uniform 3D volume makes (r³-L³)/(U³-L³) uniform, rather than r.
        for orbit in range(2):
            quantiles=np.sort((radii[:,orbit]**3-.3**3)/(1-.3**3))
            upper=np.arange(1,len(quantiles)+1)/len(quantiles)
            lower=np.arange(len(quantiles))/len(quantiles)
            assert max(np.max(upper-quantiles),np.max(quantiles-lower))<.025


def test_prior_deadline_and_invalid_sample_count_remain_explicit():
    current=model(np.ones((1,1,1)))
    with pytest.raises(TimeoutError):current.prior(np.random.default_rng(1),64,deadline=0.)
    for n in (0,-1,True,1.5):
        with pytest.raises(ValueError):current.prior(np.random.default_rng(1),n)


def test_moderately_ill_conditioned_coordinates_preserve_the_physical_ball():
    rng=np.random.default_rng(107);rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
    transform=np.diag([.01,1.,100.])@rotation
    current=model((transform.T@transform)[None])
    draws=current.prior(np.random.default_rng(108),8000)
    physical=draws@transform.T
    assert np.max(np.linalg.norm(physical,axis=1))<=1.+1e-8
    np.testing.assert_allclose(np.mean(physical**2,axis=0),.2,atol=.02)


def test_unqualified_numerical_prior_geometry_fails_explicitly():
    current=model(np.diag([1e-4,1e10])[None])
    with pytest.raises(ValueError,match='cannot safely bound'):
        current.prior(np.random.default_rng(1),64)


def test_magnitude_agreement_cannot_hide_different_directions_and_predictions():
    q=np.zeros((2,2,3));q[0,0]=[1.,0.,.2];q[1,1]=[0.,1.,.2]
    current=model(np.eye(2)[None],q,upper=2.)
    a=dict(samples=np.tile([1.,0.],(64,1)));b=dict(samples=np.tile([0.,1.],(64,1)))
    result=compare_replicas(current,a,b)
    assert result['median_discrepancy_in_interval_widths']==0.
    assert not result['predictive_agreement']['passed']
    assert not result['physical_second_moment_agreement']['passed']
    assert not result['screen_passed'] and not result['global_convergence_proven']


def test_global_sign_symmetry_is_preserved_by_every_replica_screen():
    rng=np.random.default_rng(89);samples=rng.normal(0.,.2,(128,3))
    current=model(np.eye(3)[None],upper=3.)
    result=compare_replicas(current,dict(samples=samples),dict(samples=-samples))
    assert result['screen_passed'] and not result['global_convergence_proven']


def test_replica_screens_are_invariant_under_equivalent_native_coordinates():
    rng=np.random.default_rng(90);a=rng.normal(0.,.2,(128,3));b=a+[.1,0.,0.]
    amplitudes=rng.normal(size=(5,3));q=np.einsum('ni,nj->ijn',amplitudes,amplitudes)
    forms=np.eye(3)[None];rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
    transform=rotation@np.diag([.4,1.,2.])
    first=model(forms,q,upper=3.)
    second=model(np.einsum('ia,sij,jb->sab',transform,forms,transform),
        np.einsum('ia,ijn,jb->abn',transform,q,transform),upper=3.)
    original=compare_replicas(first,dict(samples=a),dict(samples=b))
    changed=compare_replicas(second,dict(samples=np.linalg.solve(transform,a.T).T),
        dict(samples=np.linalg.solve(transform,b.T).T))
    assert original['screen_passed']==changed['screen_passed']
    for name in ('predictive_agreement','physical_second_moment_agreement'):
        for key in ('mean_discrepancy_in_spread','relative_spread_discrepancy'):
            assert original[name][key]==pytest.approx(changed[name][key],rel=1e-9,abs=1e-12)


def test_posterior_retains_prior_proposal_diagnostics_without_a_qualification_claim():
    current=model(np.ones((1,1,1)),upper=2.)
    result=sample_posterior(current,particles=64,moves=2,max_seconds=5.)
    assert result['prior_diagnostics'] and not result['qualified']
    assert result['history'][-1]['beta']==1.
