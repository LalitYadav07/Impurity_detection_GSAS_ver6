"""Named coverage must not inherit a Gram eigensystem's mixed directions."""
import numpy as np
import pytest

from gpx_recipe.preparation.structural_mode_directions import from_basis
from gpx_recipe.structural_live.optimization.physical_prior_geometry import orthonormal_lift


def test_correlated_physical_normalization_keeps_each_named_mode_isolated():
    gram=np.array([[2.,.6],[.6,1.]])
    old=orthonormal_lift(gram)
    assert np.all(np.count_nonzero(abs(old)>1e-8,axis=0)==2)
    result=from_basis(np.eye(2),np.empty((2,0)),gram,['mode-a','mode-b'])
    np.testing.assert_array_equal(result.physical!=0,np.eye(2,dtype=bool))
    np.testing.assert_allclose(np.einsum('ni,ij,nj->n',result.physical,gram,result.physical),1.)
    np.testing.assert_allclose(result.physical,np.diag([1/np.sqrt(2),1.]))
    assert not result.gauge_only.any()


def test_every_native_axis_reconstructs_modulo_only_the_supplied_gauges():
    rng=np.random.default_rng(4);basis=np.linalg.qr(rng.normal(size=(6,6)))[0]
    mapping=basis[:,:4]@np.diag([1e-9,3.,1e5,1e-4]);gauges=basis[:,4:]
    gram=np.array([[1.,.1,0,0],[.1,2.,0,0],[0,0,.5,.1],[0,0,.1,1.]])
    result=from_basis(mapping,gauges,gram,[f'native-{i}' for i in range(6)])
    reconstructed=result.physical@mapping.T+result.gauge@gauges.T
    np.testing.assert_allclose(reconstructed,np.diag(result.native_scale),rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(np.einsum('ni,ij,nj->n',result.physical,gram,result.physical),1.)
    assert np.linalg.matrix_rank(result.physical)==4
    assert result.maximum_reconstruction_error<1e-10


def test_pure_origin_gauges_remain_explicit_null_controls():
    result=from_basis(np.array([[1.],[0.]]),np.array([[0.],[1.]]),np.array([[4.]]),['physical','origin'])
    np.testing.assert_array_equal(result.gauge_only,[False,True])
    np.testing.assert_array_equal(result.physical,[[.5],[0.]])
    np.testing.assert_array_equal(result.gauge,[[0.],[1.]])


def test_weak_physical_coordinate_is_not_converted_to_a_gauge():
    result=from_basis(np.diag([1e-12,1e12]),np.empty((2,0)),np.eye(2),['small-unit','large-unit'])
    assert not result.gauge_only.any()
    np.testing.assert_allclose(result.physical,np.eye(2))
    np.testing.assert_allclose(result.native_scale,[1e-12,1e12])


def test_fully_gauge_and_empty_models_keep_their_complete_inventories():
    result=from_basis(np.empty((3,0)),np.eye(3),np.empty((0,0)),['origin-x','origin-y','origin-z'])
    assert result.physical.shape==(3,0) and result.gauge_only.all()
    np.testing.assert_array_equal(result.gauge,np.eye(3))
    empty=from_basis(np.empty((0,0)),np.empty((0,0)),np.empty((0,0)),[])
    assert empty.physical.shape==(0,0) and empty.labels==()


def test_named_display_labels_do_not_replace_indexed_native_identity():
    result=from_basis(np.eye(2),np.empty((2,0)),np.eye(2),['repeated display','repeated display'])
    np.testing.assert_array_equal(result.physical,np.eye(2))
    for value in [result.physical,result.gauge,result.native_scale,result.gauge_only]:
        assert not value.flags.writeable
        with pytest.raises(ValueError):value.flags.writeable=True


@pytest.mark.parametrize('change',['missing-gauge','lost-coordinate','rank','nonfinite','metric','labels','string-labels'])
def test_invalid_bases_fail_without_partial_mode_coverage(change):
    mapping=np.eye(2);gauges=np.empty((2,0));gram=np.eye(2);labels=['one','two']
    if change=='missing-gauge':mapping=np.ones((3,2));gauges=np.empty((3,0));labels=['one','two','three']
    elif change=='lost-coordinate':mapping=mapping[:,:1]
    elif change=='rank':mapping[:,1]=mapping[:,0]
    elif change=='nonfinite':mapping[0,0]=np.nan
    elif change=='metric':gram[0,0]=0.
    elif change=='labels':labels=['one']
    else:labels='ab'
    with pytest.raises(ValueError):from_basis(mapping,gauges,gram,labels)
