"""Real dependency checks run in the isolated SeeK-path audit environment."""
import warnings
import numpy as np
import pytest
pytest.importorskip('seekpath')
import spglib
from gpx_recipe.commensurate_k import discover_points
from gpx_recipe.commensurate_k import recover_points


@pytest.mark.parametrize('cell,positions,group', [
    (np.diag([4., 4., 4.]), [[0, 0, 0]], 221),
    (np.diag([4., 4., 4.]), [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]], 225),
    (np.diag([4., 4., 7.]), [[0, 0, 0], [.5, .5, .5]], 139),
    (np.diag([3., 4., 5.]), [[0, 0, 0], [.5, .5, 0]], 65),
    ([[3., 0, 0], [0, 4., 0], [-1.7101007166, 0, 4.698463104]], [[0, 0, 0], [.5, .5, 0]], 12),
])
def test_real_points_match_full_original_setting(cell, positions, group):
    structure = (cell, positions, [1] * len(positions))
    native = spglib.get_symmetry_dataset(structure)
    ops = [dict(rotation=r.tolist(), translation=t.tolist()) for r, t in zip(native.rotations, native.translations)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = discover_points(structure, ops, expected_space_group=group)
    assert result['candidates'][0]['k'] == ['0', '0', '0']
    if group == 225:
        assert ['0', '1', '0'] in [r['k'] for r in result['candidates']]
    if group == 139:
        assert ['0', '0', '1'] in [r['k'] for r in result['candidates']]
        assert result['excluded']  # metric-dependent vertices are not rounded guesses


def test_wrong_parent_group_and_translation_inventory_refused():
    structure = (np.diag([4., 4., 4.]), [[0, 0, 0]], [1])
    ops = [dict(rotation=np.eye(3).tolist(), translation=[0, 0, 0])]
    with pytest.raises(ValueError, match='space group'):
        discover_points(structure, ops, expected_space_group=225)
    with pytest.raises(ValueError, match='inventory'):
        discover_points(structure, ops, expected_space_group=221)


def test_ultraclose_sites_are_refused_if_point_generator_cannot_keep_them():
    positions=np.array([[2e-7,0,0],[1-2e-7,0,0]])
    structure=(np.diag([5.,6.,7.]),positions,[1,1]);before=positions.copy()
    native=spglib.get_symmetry_dataset(structure,symprec=1e-8)
    ops=[dict(rotation=r.tolist(),translation=t.tolist()) for r,t in zip(native.rotations,native.translations)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # spglib resolves the two sites, but SeeK-path's primitive-cell
        # construction cannot. Never merge or move atoms just to get a path.
        with pytest.raises(ValueError,match='No tested coordinate precision'):
            recover_points(structure,ops,expected_space_group=native.number)
    np.testing.assert_array_equal(positions,before)


def test_recovery_uses_the_exact_unchanged_parent_inventory():
    positions=np.array([[0,0,0],[.5,.5,.5]])
    structure=(np.diag([4.,4.,7.]),positions,[1,1]);before=positions.copy()
    native=spglib.get_symmetry_dataset(structure)
    ops=[dict(rotation=r.tolist(),translation=t.tolist()) for r,t in zip(native.rotations,native.translations)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');result=recover_points(structure,ops,expected_space_group=native.number)
    assert result['parent_space_group']==139 and result['symprec']==1e-5
    np.testing.assert_array_equal(positions,before)
