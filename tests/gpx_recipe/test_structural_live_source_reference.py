import numpy as np
import pytest

from gpx_recipe.structural_live.source_reference import numeric_precision, project_rounded_site


def threefold():
    r=np.array([[0,-1,0],[1,-1,0],[0,0,1]])
    return [(np.eye(3),np.zeros(3)),(r,np.zeros(3)),(r@r,np.zeros(3))]


def test_printed_precision_is_distinct_from_reported_uncertainty():
    assert numeric_precision('0.3333')==(.3333,.00005,0.)
    np.testing.assert_allclose(numeric_precision('0.0005(4)'),[.0005,.00005,.0004])
    np.testing.assert_allclose(numeric_precision('1.20(3)e-2'),[.012,.00005,.0003])
    assert numeric_precision('0')==(0.,0.,0.)
    with pytest.raises(ValueError):numeric_precision('?')


@pytest.mark.parametrize('digits',[4,5,6])
def test_source_thirds_project_within_written_rounding(digits):
    x=np.round([1/3,2/3,.25],digits);precision=np.full(3,.5*10.**-digits)
    result,stabilizers=project_rounded_site(x,precision,threefold())
    np.testing.assert_allclose(result,[1/3,2/3,.25],rtol=0,atol=1e-12)
    assert len(stabilizers)==3
    assert np.all(abs(result-x)<=precision)


def test_resolved_displacement_is_not_snapped_to_special_site():
    x=np.array([.33331,.66666,.25])
    result,stabilizers=project_rounded_site(x,[.000005]*3,threefold())
    np.testing.assert_array_equal(result,x)
    assert len(stabilizers)==1


def test_exact_integer_and_free_axis_are_preserved():
    ops=[(np.eye(3),np.zeros(3)),(np.diag([-1,-1,1]),np.zeros(3))]
    x=[0.,0.,.13987]
    result,_=project_rounded_site(x,[0.,0.,.000005],ops)
    np.testing.assert_array_equal(result,x)


def test_inconsistent_nearby_operations_fail_instead_of_averaging():
    ops=[(np.eye(3),np.zeros(3)),(-np.eye(3),np.zeros(3)),(-np.eye(3),[.00005,0,0])]
    with pytest.raises(ValueError,match='cannot satisfy'):
        project_rounded_site([0.,0.,0.],[.00005]*3,ops)
