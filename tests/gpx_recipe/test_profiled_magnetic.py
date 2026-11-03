"""Public artificial profiles; no proposal data or native GPX fixtures."""
import numpy as np
import pytest
from gpx_recipe.symmetry.profiled_magnetic import ProfiledMagneticObjective


def fixture_model(*,dimension=3,sites=1,fixed_scale=None,free_background=True,ceiling=6.,truth=None):
    rng=np.random.default_rng(67241+dimension+sites)
    n=127
    factors=rng.normal(size=(n,4,dimension))
    q=np.einsum('nki,nkj->nij',factors,factors)/4
    nuclear=rng.uniform(1,4,size=n)
    b=np.column_stack([np.ones(n),np.linspace(-1,1,n)])
    maps=np.zeros((sites,3,dimension))
    for i in range(dimension):maps[i//3%sites,i%3,i]=1
    # Oblique coordinates and coupled independent sites remain fully free.
    maps[:,0,:]+=.2*maps[:,2,:]
    if np.linalg.matrix_rank(maps.reshape(-1,dimension))!=dimension:
        raise ValueError('Insufficient independent synthetic moment maps')
    truth=np.linspace(.8,1.6,dimension) if truth is None else np.asarray(truth,float)
    scale=1.7 if fixed_scale is None else fixed_scale
    bg=np.array([.3,-.2])
    observed=scale*(nuclear+np.einsum('i,nij,j->n',truth,q,truth))+b@bg
    offset=-observed if free_background else -observed+b@bg
    model=ProfiledMagneticObjective(offset=offset,nuclear=nuclear,quadratic=q,
        background=b if free_background else np.empty((n,0)),moment_maps=maps,
        maximum_moment=ceiling,normalization=float(observed@observed),fixed_scale=fixed_scale)
    return model,truth,scale,bg


@pytest.mark.parametrize('fixed_scale',[None,1.7])
@pytest.mark.parametrize('free_background',[False,True])
@pytest.mark.parametrize('dimension,sites',[(1,1),(2,1),(3,1),(5,2),(6,2)])
def test_recovers_full_independent_moments_and_only_declared_nuisances(dimension,sites,fixed_scale,free_background):
    model,truth,scale,bg=fixture_model(dimension=dimension,sites=sites,
        fixed_scale=fixed_scale,free_background=free_background)
    fit=model.fit(truth*.65)
    assert fit.success,fit.message
    assert fit.weighted_cost<1e-8
    # Time reversal is indistinguishable in unpolarized powder intensity.
    assert min(np.linalg.norm(fit.coefficients-truth),np.linalg.norm(fit.coefficients+truth))<2e-5
    assert fit.scale==pytest.approx(scale,rel=1e-5)
    if free_background:assert fit.background==pytest.approx(bg,abs=2e-5)
    else:assert len(fit.background)==0


def test_profiles_exact_gradient_in_oblique_multisite_basis():
    model,*_=fixture_model(dimension=5,sites=2)
    z=np.array([.3,-.8,1.1,.7,1.2]);value,gradient=model.value_gradient(z)
    numeric=[]
    for j in range(len(z)):
        step=np.eye(len(z))[j]*1e-5
        numeric.append((model.value_gradient(z+step)[0]-model.value_gradient(z-step)[0])/2e-5)
    assert gradient==pytest.approx(numeric,rel=1e-7,abs=1e-10)
    assert np.isfinite(value)


def test_moment_ceiling_applies_to_each_site_and_is_never_relaxed():
    model,*_=fixture_model(dimension=5,sites=2,ceiling=.5)
    fit=model.fit(np.ones(5)*4)
    assert fit.success,fit.message
    assert fit.maximum_moment<=.50000001
    assert min(model._constraint(model.to_physical@fit.coefficients))>=-1e-8
    assert fit.weighted_cost>1.


def test_zero_components_are_free_and_a_sign_seed_can_recover_them():
    model,truth,*_=fixture_model(truth=[0.,2.,0.])
    fits=[model.fit(np.array(seed)) for seed in [[1.,1.,1.],[-1.,2.,-.5],[.3,-1.,.7]]]
    assert all(f.success for f in fits)
    best=min(fits,key=lambda f:f.weighted_cost)
    assert best.weighted_cost<1e-8
    assert min(np.linalg.norm(best.coefficients-truth),np.linalg.norm(best.coefficients+truth))<2e-5


def test_rejects_nonidentifiable_moment_map_and_background():
    model,*_=fixture_model()
    kwargs=dict(offset=model.offset,nuclear=model.nuclear,quadratic=model.quadratic,
        background=model.background,moment_maps=model.moment_maps,
        maximum_moment=6.,normalization=model.normalization)
    with pytest.raises(ValueError,match='moment map is rank deficient'):
        ProfiledMagneticObjective(**{**kwargs,'moment_maps':np.zeros_like(model.moment_maps)})
    with pytest.raises(ValueError,match='background basis is rank deficient'):
        ProfiledMagneticObjective(**{**kwargs,'background':np.ones((len(model.offset),2))})


def test_native_heldout_must_agree_and_budget_exhaustion_is_not_success():
    model,truth,scale,bg=fixture_model()
    assert model.heldout_error(truth,scale,bg,np.zeros(len(model.offset)))<1e-12
    assert model.heldout_error(truth,scale,bg,np.ones(len(model.offset)))>.1
    fit=model.fit(np.ones(3)*.01,max_iterations=1)
    assert not fit.success


def test_fixed_background_has_no_hidden_columns_and_scale_stays_common():
    model,truth,scale,_=fixture_model(fixed_scale=1.7,free_background=False)
    fit=model.fit(truth*.7)
    assert fit.success
    assert fit.scale==1.7
    assert fit.background.size==0
    assert fit.weighted_cost<1e-8


def test_zero_gradient_at_zero_moment_is_not_mistaken_for_a_minimum():
    model,*_=fixture_model()
    fit=model.fit(np.zeros(3))
    assert not fit.success
    assert fit.local_descent_found
