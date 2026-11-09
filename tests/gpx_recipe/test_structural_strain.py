"""Independent lattice equations, including nonorthogonal parent cells."""
import numpy as np
import pytest

from gpx_recipe.preparation.structural_strain import StrainProjection,cell_metric


def tetragonal():
    # Public artificial version of a tetragonal -> orthorhombic ISO basis.
    matrix=np.zeros((6,3));matrix[5,0]=1;matrix[:2,1]=1;matrix[2,2]=1
    basis=np.array([[1,1,0],[-1,1,0],[0,0,1.]])
    return StrainProjection(cell_metric([4.,4.,9.,90.,90.,90.]),basis,matrix,[np.sqrt(2),1/np.sqrt(2),1])


def test_independent_orthorhombic_formula_recovers_strain_and_thermal_modes():
    projection=tetragonal();m=np.array([.012,-.004,.006])
    common=1+m[1]/np.sqrt(2);shear=m[0]/np.sqrt(2)
    # Independently derived child lengths, rather than using forward().
    lengths=[np.sqrt(2)*4*(common+shear),np.sqrt(2)*4*(common-shear),9*(1+m[2])]
    metric=cell_metric(lengths+[90.,90.,90.]);result=projection.project(metric)
    np.testing.assert_allclose(result['amplitudes'],m,rtol=1e-12,atol=1e-14)
    assert result['reconstruction_relative_error']<1e-12
    assert result['units']=='dimensionless'
    assert not result['experimental_mode_absence_qualified']


def test_parent_expansion_is_not_mislabeled_as_orthorhombic_order():
    projection=tetragonal();cell=cell_metric([4.01,4.01,9.02,90.,90.,90.])
    child=projection.basis@cell@projection.basis.T
    result=projection.project(child)
    assert abs(result['amplitudes'][0])<1e-12
    assert result['amplitudes'][1]>0 and result['amplitudes'][2]>0
    current=projection.project(child,parent_metric=cell)
    np.testing.assert_allclose(current['amplitudes'],0.,atol=1e-12)


@pytest.mark.parametrize('basis',[
    np.eye(3),[[1,1,0],[-1,1,0],[0,0,1]],[[1,2,0],[-1,0,0],[-1/3,-2/3,1/3]],
])
def test_full_strain_tensor_recovers_mixed_modes_in_a_triclinic_parent(basis):
    parent=cell_metric([4.,5.,7.,78.,94.,103.])
    projection=StrainProjection(parent,basis,np.eye(6),np.ones(6))
    rng=np.random.default_rng(872199)
    for _ in range(20):
        e=rng.uniform(-.015,.015,6)
        # Independent TOPAS convention B_strained = B_parent S.
        s=np.eye(3)+np.diag(e[:3]);s[1,2]=s[2,1]=e[3]/2
        s[0,2]=s[2,0]=e[4]/2;s[0,1]=s[1,0]=e[5]/2
        t=np.asarray(basis);child=t@s@parent@s@t.T
        np.testing.assert_allclose(projection.project(child)['amplitudes'],e,rtol=1e-10,atol=1e-13)


def test_missing_shear_is_an_error_not_an_absent_mode_prediction():
    projection=tetragonal();s=np.eye(3);s[0,2]=s[2,0]=.002
    child=projection.basis@s@projection.parent@s@projection.basis.T
    with pytest.raises(ValueError,match='outside the declared'):
        projection.project(child)


def test_normalization_is_preserved_even_for_very_different_mode_units():
    matrix=np.eye(6);norm=np.geomspace(1e-8,1e8,6)
    projection=StrainProjection(np.eye(3),np.eye(3),matrix,norm)
    amplitudes=np.array([.001,-.002,.003,-.004,.005,-.006])/norm
    result=projection.project(projection.forward(amplitudes))
    np.testing.assert_allclose(result['amplitudes'],amplitudes,rtol=1e-10,atol=1e-16)


@pytest.mark.parametrize('case',['dependent','zero','nonfinite','bad_basis','bad_normalization'])
def test_invalid_or_incomplete_coordinates_are_not_silently_dropped(case):
    matrix=np.eye(6);basis=np.eye(3);norm=np.ones(6)
    if case=='dependent':matrix[:,1]=matrix[:,0]
    elif case=='zero':matrix[:,1]=0
    elif case=='nonfinite':matrix[0,0]=np.nan
    elif case=='bad_basis':basis[1]=basis[0]
    else:norm[0]=0
    with pytest.raises(ValueError):StrainProjection(np.eye(3),basis,matrix,norm)


@pytest.mark.parametrize('cell',[[0,4,5,90,90,90],[4,4,5,90,90,180],
                              [4,4,5,10,10,170],[4,4,5,np.nan,90,90]])
def test_invalid_crystal_metrics_are_rejected(cell):
    with pytest.raises(ValueError):cell_metric(cell)


def test_projection_owns_immutable_inputs_and_rejects_negative_stretch():
    parent=np.eye(3);basis=np.eye(3);matrix=np.eye(6);norm=np.ones(6)
    projection=StrainProjection(parent,basis,matrix,norm)
    parent[:]=0;basis[:]=0;matrix[:]=0;norm[:]=0
    np.testing.assert_allclose(projection.project(np.eye(3))['amplitudes'],0,atol=1e-14)
    with pytest.raises(ValueError):projection.forward([-2,0,0,0,0,0])


def test_asymmetric_input_cannot_be_repaired_by_averaging():
    metric=np.eye(3);metric[0,1]=1e-8
    with pytest.raises(ValueError,match='symmetric'):
        tetragonal().project(metric)
