"""Real long-flight-path TOF ranges must survive preparation admission."""
import pytest

from gpx_recipe.preparation.reference import validate_grid, MAXIMUM_TOF_MICROSECONDS
from gpx_recipe.preparation.worker import validate_request


@pytest.mark.parametrize('bounds', [(11301.913503367501, 239409.), (220000., 300000.),
                                  (3000., MAXIMUM_TOF_MICROSECONDS)])
def test_long_tof_range_survives_the_same_worker_admission(bounds):
    grid = validate_grid(kind='PNT', lower=bounds[0], upper=bounds[1], bins=3818, bank=1)
    request = dict(source='parent.cif', instrument='bank.instprm', grid=grid,
                   settings={}, destination='prepared')
    accepted = validate_request('reference', request)
    assert accepted['grid'] == grid
    assert accepted['grid']['upper'] == bounds[1]


@pytest.mark.parametrize('kind,lower,upper', [('PNT', 3000., float('inf')),
    ('PNT', 3000., MAXIMUM_TOF_MICROSECONDS+1), ('PNT', 3., 3.), ('PNC', 10., 180.)])
def test_extended_tof_support_retains_finite_and_physical_admission(kind, lower, upper):
    with pytest.raises(ValueError):
        validate_grid(kind=kind, lower=lower, upper=upper)


def test_planner_silicon_handoff_retains_diamond_symmetry_and_absolute_pattern(tmp_path):
    """An expanded P1 cell plots correctly but cannot seed subgroup discovery."""
    from pathlib import Path
    import numpy as np
    G = pytest.importorskip('GSASII.GSASIIscriptable')
    from GSASII import GSASIIstrMain as MAIN
    from gpx_recipe.planning import gsas as planning
    from gpx_recipe.preparation.reference import create_reference

    # The actual Planner example is passed through the preparation importer.
    example = Path(planning.__file__).parent / 'data/examples/silicon.cif'
    profile = (Path(planning.__file__).parents[2] / 'radar_pd_nova/data/powgen_calibrations/'
               '2025_Feb_HighRes_60HzB2_CWL1p5.instprm')
    explicit = tmp_path / 'independent-expanded-cell.cif'
    coordinates = [(.125, .125, .125), (.125, .625, .625), (.625, .125, .625),
                   (.625, .625, .125), (.875, .875, .875), (.875, .375, .375),
                   (.375, .875, .375), (.375, .375, .875)]
    explicit.write_text("""data_expanded_diamond
_cell_length_a 5.4312
_cell_length_b 5.4312
_cell_length_c 5.4312
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
_symmetry_space_group_name_H-M 'P 1'
loop_
_space_group_symop_operation_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_adp_type
_atom_site_U_iso_or_equiv
""" + ''.join(f'Si{i} Si {x} {y} {z} 1 Uiso .0055\n'
              for i, (x, y, z) in enumerate(coordinates, 1)))

    curves = []
    for name, cif in [('planner', example), ('expanded', explicit)]:
        record = create_reference(cif, profile, tmp_path / name, kind='PNT',
                                  lower=12425.58, upper=163403.15, bins=2500, bank=1)
        project = G.G2Project(gpxfile=record['path'])
        phase, hist = project.phase(0), project.histogram(0)
        if name == 'planner':
            assert phase.data['General']['SGData']['SpGrp'].replace(' ', '') == 'Fd-3m'
            assert len(phase.data['Atoms']) == 1
            assert phase.data['General']['NoAtoms'] == {'Si': 8.}
        hist.data['data'][0]['Dummy'] = False
        hist.data['data'][1][1][:] = 1.
        hist.data['Sample Parameters']['Scale'] = [1., False]
        project.save()
        curves.append(np.asarray(MAIN.DoNoFit(record['path'], hist.name), dtype=float))
    assert np.isfinite(curves).all() and np.max(curves[0]) > 0
    np.testing.assert_allclose(curves[0], curves[1], rtol=2e-8,
                               atol=2e-8 * np.max(curves[0]))
