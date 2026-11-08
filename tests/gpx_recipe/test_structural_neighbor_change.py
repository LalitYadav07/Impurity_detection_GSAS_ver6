import numpy as np
import pytest

from gpx_recipe.preparation.structural_neighbor_change import from_parent


def example():
    xyz=np.array([[0.,0,0],[.2,0,0]])
    g=np.eye(3)*100
    return xyz,g,from_parent(xyz,g,[1.,1.])


@pytest.mark.parametrize('fraction,strong',[(0.,False),(.005,False),(.019,False),(.021,True),(-.021,True)])
def test_two_angstrom_contact_reports_exact_physical_fraction(fraction,strong):
    xyz,g,graph=example();changed=xyz.copy();changed[1,0]*=1+fraction
    d=graph.compare(changed,g)
    assert d['maximum_relative_change']==pytest.approx(abs(fraction),abs=1e-14)
    assert d['maximum_absolute_change_angstrom']==pytest.approx(2*abs(fraction),abs=1e-14)
    assert d['strong_neighbor_distance_change'] is strong
    assert not d['screening_qualified']
    np.testing.assert_array_equal(xyz,graph.coordinates)
    with pytest.raises(ValueError):graph.pairs.flags.writeable=True


def test_two_percent_arithmetic_boundary_does_not_become_an_exact_physical_decision():
    xyz,g,graph=example();changed=xyz.copy();changed[1,0]*=1.02
    d=graph.compare(changed,g)
    assert d['threshold_boundary_contacts']==1
    assert d['strong_neighbor_distance_change'] is None


def test_parent_cell_change_is_removed_by_the_matched_parent_reference():
    xyz,g,graph=example();warm=g*1.03**2
    assert graph.compare(xyz,warm)['strong_neighbor_distance_change'] is True
    d=graph.compare(xyz,warm,reference_coordinates=xyz,reference_metric=warm)
    assert d['maximum_relative_change']==0. and d['strong_neighbor_distance_change'] is False


def test_common_translation_is_not_a_distortion_and_original_images_remain_matched():
    xyz,g,graph=example()
    assert graph.compare(xyz+np.array([.3,-.5,1.]),g)['maximum_relative_change']<1e-14
    crossing=np.array([[.95,0,0],[1.15,0,0]])
    first=from_parent(crossing,g,[1.,1.]);d=first.compare(crossing+np.array([.04,0,0]),g)
    assert d['maximum_relative_change']<1e-14


def test_graph_includes_contacts_nearest_to_either_endpoint():
    xyz=np.array([[0.,0,0],[.1,0,0],[.4,0,0]])
    graph=from_parent(xyz,np.eye(3)*100,[1.,1.,1.])
    assert {tuple(p) for p in graph.pairs}=={(0,1),(1,2)}


def test_skew_cell_uses_periodic_image_enumeration_instead_of_fractional_rounding():
    g=np.array([[100.,90.,0.],[90.,100.,0.],[0.,0.,100.]])
    xyz=np.array([[0.,0,0],[.49,.49,0]])
    graph=from_parent(xyz,g,[1.,1.])
    d=graph.compare(xyz,g)
    assert d['largest_change']['reference_angstrom']<3.
    assert any(any(v!=0 for v in row) for row in graph.images)


def test_coincident_partial_sites_remain_an_explicit_unresolved_contact_case():
    xyz=np.array([[0.,0,0],[0.,0,0],[.2,0,0]])
    graph=from_parent(xyz,np.eye(3)*100,[.5,.5,1.])
    changed=xyz.copy();changed[2,0]*=1.05
    d=graph.compare(changed,np.eye(3)*100)
    assert d['excluded_coincident_contacts']==1 and d['partially_occupied_sites']==2
    assert d['strong_neighbor_distance_change'] is None


@pytest.mark.parametrize('case',['no_occupancy','invalid_occupancy','wrong_shape','nonfinite','bad_metric','incomplete_reference'])
def test_invalid_geometry_cannot_be_reported_as_a_weak_distortion(case):
    xyz,g,graph=example()
    with pytest.raises(ValueError):
        if case=='no_occupancy':from_parent(xyz,g,[0.,0.])
        elif case=='invalid_occupancy':from_parent(xyz,g,[1.,-1.])
        elif case=='wrong_shape':graph.compare(xyz[:1],g)
        elif case=='nonfinite':graph.compare(xyz*np.nan,g)
        elif case=='bad_metric':graph.compare(xyz,-g)
        else:graph.compare(xyz,g,reference_coordinates=xyz)
