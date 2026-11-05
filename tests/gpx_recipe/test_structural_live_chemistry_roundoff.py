import numpy as np
import pytest
from gpx_recipe.structural_live.optimization.chemical_geometry import ChemicalPotential,ContactRule,hard_contact_comparison,ChemistryUnqualified

def potential(distance):
    p=ChemicalPotential([ContactRule(('A','B'),'test',hard_min=1)])
    return p.evaluate([[0,0,0],[distance/10,0,0]],np.eye(3)*100,['A','B'],[1,1])

def test_roundoff_boundary_is_recorded_separately():
    r=potential(np.nextafter(1.,0.))
    assert r['feasible'] and len(r['roundoff_boundary_contacts'])==1
    assert r['roundoff_boundary_contacts'][0]['comparison']['arithmetic_allowance_angstrom']<1e-12

def test_resolved_short_contact_still_rejected():
    r=potential(1-1e-10)
    assert not r['feasible'] and len(r['hard_violations'])==1

def test_exact_boundary_and_long_contact_are_feasible():
    assert potential(1.)['feasible'] and potential(1.001)['feasible']

def test_large_cancellation_is_unresolved_not_silently_accepted():
    with pytest.raises(ChemistryUnqualified,match='precision'):
        hard_contact_comparison(1.,1.,[1e14,0,0],[1e14+1,0,0],[0,0,0],np.eye(3))

def test_observed_export_roundoff_has_only_arithmetic_allowance():
    minimum=3.1558323967977424;distance=3.1558323967977415
    r=hard_contact_comparison(distance,minimum,[0,0,0],[.5,.5,.5],[0,0,0],np.eye(3)*(distance**2/.75))
    assert r['roundoff_boundary'] and not r['violates']
    assert r['arithmetic_allowance_angstrom']<1e-12
