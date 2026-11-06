"""Public algebraic controls, including a random-probe blind direction."""
import numpy as np
import pytest

from gpx_recipe.live.quadratic_qualification import check_coefficients,check_factor_coefficients,compare_coefficients


@pytest.mark.parametrize('dimension',[1,3,7,24])
def test_independent_coefficient_check_covers_the_full_basis(dimension):
    rng=np.random.default_rng(103+dimension)
    factors=rng.normal(size=(19,3,dimension));q=np.einsum('nki,nkj->nij',factors,factors)
    calls=[]
    def derivatives(point):
        calls.append(point.copy())
        return 2*np.einsum('nij,j->in',q,point)
    result=check_coefficients(q,derivatives)
    assert result['symmetric_coefficients']==dimension*(dimension+1)//2
    assert len(calls)==result['native_jacobian_calls']==2*dimension
    assert result['all_coefficient_directions_checked']


def test_corruption_in_a_direction_unseen_by_seven_probes_is_rejected():
    dimension=24;rng=np.random.default_rng(104)
    probes=rng.normal(size=(7,dimension));_,_,vh=np.linalg.svd(probes,full_matrices=True)
    unseen=vh[-1];truth=np.eye(dimension)[None,:,:]
    corrupted=truth+10*np.outer(unseen,unseen)[None,:,:]
    delta=np.einsum('bi,nij,bj->bn',probes,corrupted-truth,probes)
    assert np.max(abs(delta))<1e-12
    with pytest.raises(ValueError,match='independent native derivatives'):
        check_coefficients(corrupted,lambda p:2*np.einsum('nij,j->in',truth,p))


@pytest.mark.parametrize('bad',['shape','nonfinite'])
def test_missing_or_nonfinite_native_derivatives_cannot_qualify_a_response(bad):
    q=np.eye(3)[None,:,:]
    with pytest.raises(ValueError,match='different basis or nonfinite'):
        check_coefficients(q,lambda _:np.ones((2,1)) if bad=='shape' else np.full((3,1),np.nan))


def test_native_zero_guard_offsets_cancel_in_central_coefficient_checks():
    q=np.array([[[2.,.3],[.3,1.]]])
    result=check_coefficients(q,lambda p:2*np.einsum('nij,j->in',q,p)+1e-8)
    assert max(result['maximum_tolerance_ratios'])<1.


@pytest.mark.parametrize('dimension',[1,3,7,24])
def test_complete_reflection_intensity_checks_include_every_mixed_coefficient(dimension):
    factors=np.random.default_rng(104+dimension).normal(size=(13,6,dimension))
    native=lambda p:np.sum((factors@p)**2,axis=1)
    proof=check_factor_coefficients(factors,native)
    assert proof['all_coefficient_directions_checked']
    assert proof['native_structure_factor_calls']==1+2*dimension**2
    assert proof['symmetric_coefficients']==dimension*(dimension+1)//2


def test_complete_reflection_checks_reject_a_wrong_amplitude_basis():
    factors=np.random.default_rng(105).normal(size=(13,6,7));wrong=factors.copy();wrong[0,0,0]+=.01
    with pytest.raises(ValueError,match='complete native reflection coefficients'):
        check_factor_coefficients(wrong,lambda p:np.sum((factors@p)**2,axis=1))


def test_native_rebuilt_tensor_comparison_checks_all_entries():
    q=np.eye(24)[None,:,:];bad=q.copy();bad[0,23,0]=bad[0,0,23]=1e-4
    assert compare_coefficients(q,q)['maximum_tolerance_ratio']==0.
    with pytest.raises(ValueError,match='independently rebuilt'):
        compare_coefficients(bad,q)
