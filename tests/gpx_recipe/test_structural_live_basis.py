import numpy as np
import pytest

from gpx_recipe.structural_live.mode_basis import origin_gauges


def test_two_free_p1_atoms_have_three_translations_and_three_relative_modes():
    b = np.eye(6).reshape(2, 3, 6)
    result = origin_gauges(b, [np.eye(3)])
    assert result['gauge_rank'] == result['physical_rank'] == 3
    moved = b @ result['gauge_basis']
    np.testing.assert_allclose(moved[0], moved[1], atol=1e-12)
    np.testing.assert_allclose(result['gauge_basis'].T @ result['physical_basis'], 0, atol=1e-12)


def test_single_atom_centrosymmetric_coordinate_cannot_be_removed_as_translation():
    result = origin_gauges(np.eye(3).reshape(1, 3, 3), [np.eye(3), -np.eye(3)])
    assert result['gauge_rank'] == 0
    assert result['physical_rank'] == 3


def test_polar_two_site_z_modes_have_one_gauge():
    b = np.zeros((2, 3, 2)); b[0, 2, 0] = .04; b[1, 2, 1] = .07
    fourfold = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    result = origin_gauges(b, [np.eye(3), fourfold])
    assert result['gauge_rank'] == result['physical_rank'] == 1


def test_strain_only_does_not_invent_a_displacement_parameter():
    result = origin_gauges(np.zeros((3, 3, 0)), [np.eye(3)])
    assert result['physical_basis'].shape == (0, 0)


def test_null_native_mapping_is_not_discarded_as_a_gauge():
    with pytest.raises(ValueError, match='unexplained rank'):
        origin_gauges(np.zeros((2, 3, 1)), [np.eye(3)])
