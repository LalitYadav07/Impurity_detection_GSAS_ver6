from types import SimpleNamespace
import numpy as np
from gpx_recipe.structural_live.optimization.chemistry_sqp import contacts,step

def test_contact_derivatives_include_cell_and_coordinate_changes():
    g=SimpleNamespace(baseline=np.array([[1.,.1,0],[.1,1.2,0],[0,0,.8]]),
        cell_directions=np.array([np.eye(3)*.1,np.diag([.3,.1,-.05])]),
        delta=np.array([[1.2,.2,.3],[.1,1.4,.3]]),
        directions=np.random.default_rng(12).normal(size=(2,3,2))*.1,threshold=np.array([1.,1.]),
        offsets=[0,1,2],radii=[.2,.03])
    z=np.array([.1,-.2]);radii=np.array([.2,.03]);value,j=contacts(g,z,radii)
    for i in range(2):
        dz=np.zeros(2);dz[i]=1e-5
        numerical=(contacts(g,z+dz,radii)[0]-contacts(g,z-dz,radii)[0])/2e-5
        np.testing.assert_allclose(j[:,i],numerical,rtol=1e-7,atol=1e-9)

def test_coupled_step_moves_along_a_contact_boundary():
    p,info=step(np.eye(2),np.array([1.,0.]),np.zeros(2),np.array([0.]),np.array([[1.,-1.]]),radius=1.)
    assert p[0]<-.49 and p[1]<-.49
    assert p[0]-p[1]>=-1e-9 and info['linear_constraints_satisfied']
    assert np.sum((np.array([1.,0.])+p)**2)<.51
