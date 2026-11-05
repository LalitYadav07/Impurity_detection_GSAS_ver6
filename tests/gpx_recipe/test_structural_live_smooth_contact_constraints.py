import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.optimize import Bounds,minimize

from gpx_recipe.structural_live.optimization.bounded_chemical_prior import BoundedChemicalGeometry
from gpx_recipe.structural_live.optimization.chemical_geometry import ChemicalPotential,ContactRule,periodic_contacts
from gpx_recipe.structural_live.optimization.physical_prior_geometry import native_reciprocal_matrix
from gpx_recipe.structural_live.optimization.smooth_contact_constraints import SmoothContactConstraints,audit_local_kkt


class Catalog:
    fingerprint='independent-geometric-control'
    def potential(self,species):
        return ChemicalPotential([ContactRule(('X','X'),'analytic test',hard_min=2.)]),tuple(species)


def setup(skew=False):
    base=np.array([[0.,0.,0.],[.5,0.,0.]])
    mapping=np.zeros((2,3,2));mapping[1,0,0]=1.;mapping[1,1,1]=.3
    a=np.array([.04,.04,.04,0.,.006 if skew else 0.,0.])
    cell=np.zeros((6,1));cell[0,0]=.001;cell[4,0]=.0002
    geometry=BoundedChemicalGeometry(base,mapping,a,cell,['X','X'],[1.,1.],np.eye(3),[1,1,1],[.18,.12,.3],Catalog())
    return SmoothContactConstraints(geometry),base,mapping,a,cell


def test_contact_derivatives_include_coordinate_cell_cross_terms():
    c,*_=setup(skew=True);x=np.array([.06,.04,.13]);h=1e-5
    numerical=np.column_stack([(c.values(x+np.eye(3)[j]*h)-c.values(x-np.eye(3)[j]*h))/(2*h)for j in range(3)])
    assert np.allclose(c.jacobian(x),numerical,rtol=2e-7,atol=1e-9)
    weights=np.arange(1,c.count+1,dtype=float)
    hessian=np.column_stack([((c.jacobian(x+np.eye(3)[j]*h)-c.jacobian(x-np.eye(3)[j]*h)).T@weights)/(2*h)for j in range(3)])
    assert np.allclose(c.hessian(x,weights),hessian,rtol=2e-7,atol=1e-8)


def test_bounded_contact_inequalities_agree_with_independent_full_enumeration():
    c,base,mapping,a,cell=setup(skew=True);rng=np.random.default_rng(43)
    allowed=[]
    for x in rng.uniform([-.18,-.12,-.3],[.18,.12,.3],size=(60,3)):
        xyz=base+mapping@x[:2];direct=np.linalg.inv(native_reciprocal_matrix(a+cell@x[2:]))
        full=periodic_contacts(xyz,direct,2.)
        expected=not any(distance<2. for _,_,_,distance in full)
        actual=bool(np.all(c.values(x)>=0))
        assert actual==expected;allowed.append(actual)
    assert any(allowed) and not all(allowed)


def test_trust_solver_stays_feasible_and_recovers_known_boundary_optimum():
    base=np.array([[0.,0.,0.],[.5,0.,0.]])
    mapping=np.zeros((2,3,1));mapping[1,0,0]=1.
    g=BoundedChemicalGeometry(base,mapping,np.array([.04,.04,.04,0.,0.,0.]),np.empty((6,0)),
                              ['X','X'],[1.,1.],np.eye(1),[1],[.18],Catalog())
    c=SmoothContactConstraints(g);seen=[]
    def objective(x):
        assert np.min(c.values(x))>=-1e-12
        seen.append(float(x[0]));return (x[0]-.16)**2
    gradient=lambda x:np.array([2*(x[0]-.16)])
    result=minimize(objective,[0.],method='trust-constr',jac=gradient,hess=lambda x:np.array([[2.]]),
                    bounds=Bounds([-.18],[.18],keep_feasible=True),constraints=[c.scipy_constraint()],
                    options={'gtol':1e-12,'barrier_tol':1e-10,'maxiter':250,
                             'initial_barrier_parameter':1e-9,'initial_barrier_tolerance':1e-9})
    assert result.success and abs(result.x[0]-.1)<2e-5
    audit=audit_local_kkt(result,gradient(result.x),c,np.array([-.18]),np.array([.18]),tolerance=1e-6)
    assert audit['first_order_conditions_passed']
    assert len(seen)>2 and max(seen)<=.1


def test_no_contact_case_and_wrong_domain_are_explicit():
    c,*_=setup()
    with pytest.raises(ValueError,match='declared domain'):c.values([.3,0.,0.])
    with pytest.raises(ValueError):c.hessian([0.,0.,0.],np.zeros(c.count+1))


def test_bad_stationarity_cannot_be_hidden_by_solver_success_flag():
    result=SimpleNamespace(x=np.array([0.]),v=[np.zeros(1)],success=True)
    audit=audit_local_kkt(result,np.array([3.]),None,np.array([-1.]),np.array([1.]))
    assert not audit['first_order_conditions_passed']
    assert audit['stationarity_infinity_norm']==3.


def test_small_lagrangian_gradient_does_not_certify_complementarity():
    # A finite barrier solution can have an exact Lagrangian gradient while
    # remaining short of a true bound optimum. Audit both conditions.
    result=SimpleNamespace(x=np.array([.09]),v=[np.array([.14])],success=True)
    audit=audit_local_kkt(result,np.array([-.14]),None,np.array([-.18]),np.array([.1]),tolerance=1e-6)
    assert audit['stationarity_infinity_norm']==0.
    assert audit['complementarity_residual']>.001
    assert not audit['first_order_conditions_passed']
