import copy
from types import SimpleNamespace

import CifFile
import pytest

from gpx_recipe.symmetry.parent_export import export_exact_parent


def parent():
    atoms = [['O1', 'O', '', .3471682937405186, .6528317062594814, .2198746391720548, .625]]
    data = {'General': {'AtomPtrs': [3, 1, 7, 9],
                       'Cell': [False, 7.123456789012345, 7.123456789012345, 19.9876543210987, 90., 90., 120., 0.]},
            'Atoms': atoms}
    # Deliberately reproduce table-style rounding and asymmetric s.u. output.
    text = '''data_parent
_cell_length_a 7.123(4)
_cell_length_b 7.123(4)
_cell_length_c 19.988(4)
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
_space_group_name_H-M_alt 'R -3 m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
O1 O .3472(8) .65283 .21987(22) .625
'''
    return SimpleNamespace(data=data, export_CIF=lambda path: open(path, 'w').write(text))


def test_numerical_parent_export_preserves_exact_refined_site_and_cell(tmp_path):
    phase = parent(); before = copy.deepcopy(phase.data)
    target = tmp_path/'parent.cif'
    receipt = export_exact_parent(phase, target)
    block = CifFile.ReadCif(str(target))['parent']
    xyz = [float(block['_atom_site_fract_'+axis][0]) for axis in 'xyz']
    assert xyz == phase.data['Atoms'][0][3:6]
    assert abs(sum(xyz[:2])-1.) < 1e-15
    assert float(block['_cell_length_a']) == pytest.approx(phase.data['General']['Cell'][1], rel=1e-12)
    assert block['_cell_length_a'] == block['_cell_length_b']
    assert float(block['_atom_site_occupancy'][0]) == .625
    assert block['_space_group_name_H-M_alt'] == 'R -3 m'
    assert phase.data == before and receipt['source_coordinates_unchanged']


def test_cell_serialization_removes_only_symmetry_roundoff(tmp_path):
    phase = parent()
    phase.data['General']['Cell'][1:3] = [7.123456789012345, 7.123456789012346]
    phase.data['General']['Cell'][6] = 119.99999999999999
    target = tmp_path/'parent.cif'
    export_exact_parent(phase, target)
    block = CifFile.ReadCif(str(target))['parent']
    assert block['_cell_length_a'] == block['_cell_length_b']
    assert float(block['_cell_angle_gamma']) == 120.


def test_export_rejects_mismatched_site_labels(tmp_path):
    phase = parent(); phase.data['Atoms'][0][0] = 'another'
    with pytest.raises(ValueError, match='site identities'):
        export_exact_parent(phase, tmp_path/'parent.cif')
