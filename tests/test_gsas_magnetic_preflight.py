import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from gsas_magnetic_preflight import (moment_basis, direction_in_basis, setting_fingerprint, transform_k_vector,
    direct_unit_metric, magnetic_magnitude, transform_native_moment, integer_supercell_translations)


@pytest.mark.parametrize('t', [[[1,1,0],[-1,1,0],[0,0,1]], [[2,0,0],[0,3,0],[0,0,1]],
                              [[-2,0,0],[0,1,0],[0,0,1]], [[1,4,0],[0,1,0],[0,0,1]]])
def test_integer_cosets_cover_the_actual_cell_index(t):
    result=integer_supercell_translations(t)
    assert result['index']==round(abs(np.linalg.det(t)))
    offsets=[np.linalg.inv(t).T@np.array(n) for n in result['parent_translations']]
    keys={tuple(np.round(v%1.,8)) for v in offsets}
    assert len(keys)==result['index']


def test_rotated_doubled_cell_has_two_cosets_despite_matrix_entries_at_most_one():
    result=integer_supercell_translations([[1,1,0],[-1,1,0],[0,0,1]])
    assert result['index']==2
    assert ['1/2','1/2','0'] in result['child_offsets']


def test_hexagonal_moment_magnitude_uses_the_cell_metric():
    cell=[5.,5.,12.,90.,90.,120.]
    assert magnetic_magnitude([1,1,0],cell)==pytest.approx(1.)
    assert magnetic_magnitude([1,-1,0],cell)==pytest.approx(np.sqrt(3.))


def test_parent_b_becomes_child_a_for_the_actual_bilbao_pbnm_transform():
    t=[[0,1,0],[0,0,-1],[-1,0,0]]
    np.testing.assert_allclose(transform_native_moment([0,3,0],[5.53,5.73,7.67,90,90,90],t),[3,0,0],atol=1e-12)


@pytest.mark.parametrize('t', [[[0,1,0],[0,0,1],[1,0,0]], [[2,0,0],[0,1,0],[0,0,1]],
                              [[1,1,0],[0,1,0],[0,0,1]], [[-1,0,0],[0,1,0],[0,0,1]]])
def test_nonorthogonal_supercell_and_handedness_preserve_moment_norm(t):
    cell=np.array([4.,5.,6.,76.,103.,117.]);moment=np.array([1.2,-.3,2.1])
    child=transform_native_moment(moment,cell,t)
    g=np.diag(cell[:3])@direct_unit_metric(cell)@np.diag(cell[:3])
    child_g=np.array(t)@g@np.array(t).T;lengths=np.sqrt(np.diag(child_g))
    assert child@(child_g/np.outer(lengths,lengths))@child==pytest.approx(magnetic_magnitude(moment,cell)**2)


def test_invalid_magnetic_metric_is_not_silently_normalized():
    with pytest.raises(ValueError):magnetic_magnitude([1,0,0],[5,5,5,1,1,179])


def test_reciprocal_basis_conversion_preserves_fourier_phase():
    t=[[0,0,1],[1,0,0],[0,1,0]]
    transformed=transform_k_vector(['0','0','1/2'],t)
    assert transformed==['1/2','0','0']
    parent=np.array([.137,.231,.347]);child=np.linalg.inv(t).T@parent
    np.testing.assert_allclose(np.dot([0,0,.5],parent),np.dot([.5,0,0],child),rtol=0,atol=1e-14)


def test_incommensurate_looking_values_are_not_rounded_to_a_small_cell():
    with pytest.raises(ValueError,match='rational'):
        transform_k_vector([0.,0.,.4999],np.eye(3,dtype=int).tolist())
    with pytest.raises(ValueError,match='budget'):
        transform_k_vector(['0','0','0.333333'],np.eye(3,dtype=int).tolist())


def test_coupled_components_form_one_amplitude_not_two_axis_flags():
    basis=moment_basis([1,1,0],[1.,-2.,0.])
    assert basis.shape==(3,1)
    assert direction_in_basis(basis,[1,-2,0])
    assert not direction_in_basis(basis,[1,0,0])
    assert not direction_in_basis(basis,[1,2,0])


def test_forbidden_site_has_no_seedable_amplitude():
    basis=moment_basis([0,0,0],[0.,0.,0.])
    assert basis.shape==(3,0)
    assert not direction_in_basis(basis,[1,0,0])


@pytest.mark.parametrize('direction',[[0,0,0],[np.nan,0,1],[1,0],[np.inf,1,0]])
def test_zero_or_ambiguous_direction_fails(direction):
    with pytest.raises(ValueError):direction_in_basis(np.eye(3),direction)


def test_basis_membership_does_not_depend_on_vector_magnitude():
    assert direction_in_basis(moment_basis([1,2,0],[1,1,0]),[4,-3,0])
    assert not direction_in_basis(moment_basis([1,2,0],[1,1,0]),[4,-3,.01])


def test_setting_fingerprint_includes_origin_and_centering():
    rotation=np.diag([-1,1,-1])
    group=dict(SGInv=False,SGCen=np.array([[0.,0.,0.]]),SGOps=[(np.eye(3),np.zeros(3)),(rotation,np.array([0.,.5,0.]))])
    shifted={**group,'SGOps':[(np.eye(3),np.zeros(3)),(rotation,np.array([.25,.5,0.]))]}
    centered={**group,'SGCen':np.array([[0.,0.,0.],[.5,.5,.5]])}
    assert len({setting_fingerprint(group),setting_fingerprint(shifted),setting_fingerprint(centered)})==3
