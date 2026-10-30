import numpy as np
import pytest
from gpx_recipe.symmetry.k_periodicity import compatible_arms,projection_fraction,star_vectors,map_parent_orbits


def ops(translations=((0,0,0),)):
    return [dict(rotation=np.eye(3,dtype=int),translation=t) for t in translations]


def test_smaller_cell_cannot_masquerade_as_thirds_plus_half_k():
    k=['1/3','1/3','1/2'];identity=[(np.eye(3),[0,0,0])]
    assert not compatible_arms(k,ops(),np.diag([1,1,2]),identity,[1])
    assert compatible_arms(k,ops(),[[1,-1,0],[1,2,0],[0,0,2]],identity,[1])


def test_antitranslation_accepts_half_k_and_excludes_gamma():
    child=[(np.eye(3),[0,0,0]),(np.eye(3),[0,0,.5])]
    assert compatible_arms(['0','0','1/2'],ops(),np.diag([1,1,2]),child,[1,-1])
    assert not compatible_arms(['0','0','0'],ops(),np.diag([1,1,2]),child,[1,-1])


def test_precomputed_parent_star_keeps_exact_translation_decisions():
    child=[(np.eye(3),[0,0,0]),(np.eye(3),[0,0,.5])];k=['0','0','1/2']
    direct=compatible_arms(k,ops(),np.diag([1,1,2]),child,[1,-1])
    cached=compatible_arms(k,ops(),np.diag([1,1,2]),child,[1,-1],star=star_vectors(k,ops()))
    assert direct==cached


def test_rotated_arm_is_checked_in_the_original_parent_basis():
    r=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    parent=[dict(rotation=np.linalg.matrix_power(r,n),translation=[0,0,0]) for n in range(4)]
    arms=compatible_arms(['1/2','0','0'],parent,np.diag([1,2,1]),[(np.eye(3),[0,0,0])],[1])
    assert arms and all(abs(q[0])<1e-8 and abs(q[1])==.5 for q in arms)


def test_actual_antiferromagnetic_seed_matches_half_k_but_ferromagnetic_does_not():
    positions=[[0,0,0],[0,0,1]];arms=star_vectors(['0','0','1/2'],ops())
    assert projection_fraction(positions,[[0,0,1],[0,0,-1]],arms,[[0,0,0]])['fraction']==pytest.approx(1.)
    assert projection_fraction(positions,[[0,0,1],[0,0,1]],arms,[[0,0,0]])['fraction']==pytest.approx(0.)


def test_thirds_sinusoidal_order_and_unwanted_zero_k_component():
    positions=[[0,0,n] for n in range(3)];arms=star_vectors(['0','0','1/3'],ops())
    values=np.cos(2*np.pi*np.arange(3)/3)
    moments=np.column_stack([np.zeros(3),np.zeros(3),values])
    assert projection_fraction(positions,moments,arms,[[0,0,0]])['fraction']==pytest.approx(1.)
    moments[:,2]+=1
    assert projection_fraction(positions,moments,arms,[[0,0,0]])['fraction']==pytest.approx(1/3)


def test_centred_parent_integral_k_is_not_gamma():
    centring=[[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0]]
    moments=[[0,0,(-1 if t[0]==.5 else 1)] for t in centring]
    arms=star_vectors(['1','0','0'],ops(centring))
    assert projection_fraction(centring,moments,arms,centring)['fraction']==pytest.approx(1.)
    gamma=star_vectors(['0','0','0'],ops(centring))
    assert projection_fraction(centring,moments,gamma,centring)['fraction']==pytest.approx(0.)


def test_multiple_chemical_basis_sites_have_independent_fourier_phases():
    positions=[[0,0,0],[0,0,1],[.2,0,.3],[.2,0,1.3]]
    moments=[[0,0,1],[0,0,-1],[0,0,-2],[0,0,2]]
    arms=star_vectors(['0','0','1/2'],ops())
    assert projection_fraction(positions,moments,arms,[[0,0,0]])['fraction']==pytest.approx(1.)


def test_rounded_thirds_do_not_invent_separate_translation_basis_sites():
    reference=np.array([1/3,2/3,.1]);token=tuple(reference)
    offsets=np.array([[0,0,0],[1,0,0],[0,1,0]])
    positions=reference+offsets+np.array([[1e-6,-1e-6,0],[-1e-6,1e-6,0],[0,0,1e-6]])
    before=positions.copy()
    points,error=map_parent_orbits(positions,['Co']*3,{('Co',token):reference},[[0,0,0]],3)
    np.testing.assert_allclose(points,reference+offsets)
    assert 0<error<2e-6
    np.testing.assert_array_equal(positions,before)


@pytest.mark.parametrize('positions,expected,message',[
    ([[0,0,0],[0,0,0]],2,'duplicates'),
    ([[0,0,0]],2,'every parent translation'),
    ([[.1,0,0]],1,'cannot be mapped'),
])
def test_incomplete_or_duplicate_translation_orbits_are_not_scored(positions,expected,message):
    with pytest.raises(ValueError,match=message):
        map_parent_orbits(np.array(positions),['Mn']*len(positions),{('Mn',(0.,0.,0.)):np.zeros(3)},[[0,0,0]],expected)


def test_truly_ambiguous_close_basis_sites_require_review():
    basis={('Mn',(0.,0.,0.)):np.zeros(3),('Mn',(1e-6,0.,0.)):np.array([1e-6,0,0])}
    with pytest.raises(ValueError,match='ambiguous'):
        map_parent_orbits(np.zeros((1,3)),['Mn'],basis,[[0,0,0]],1)
