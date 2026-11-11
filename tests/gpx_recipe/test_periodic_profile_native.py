"""Centered magnetic profiles versus explicit atoms from independent BNS ops."""
import copy
import json

import numpy as np
import pytest

G = pytest.importorskip('GSASII.GSASIIscriptable')
from GSASII import GSASIIspc as S, GSASIIlattice as L
from test_preparation_theory_native import inputs
from gpx_recipe.preparation.reference import create_reference
from gpx_recipe.native_forward import forward_profile
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.symmetry.magnetic_identity import authoritative_operations
from gpx_recipe.symmetry.native_magnetic import construct_theory
from gpx_recipe.symmetry.worker import Artifacts
from gsas_magnetic_preflight import setting_fingerprint


@pytest.mark.parametrize('kind', ['PNC', 'PNT'])
@pytest.mark.parametrize('anisotropic', [False, True], ids=['isotropic', 'anisotropic'])
@pytest.mark.parametrize('symbol,bns,origin', [
    ("Cmm2", '35.165', [.5, 0., 0.]),
    ("Cm'm'2", '35.168', [.5, 0., 0.]),
    ("Cm'", '8.34', [.25, .25, 0.]),
])
def test_centered_profile_matches_explicit_atoms_from_bns_database(tmp_path, kind, symbol, bns, origin, anisotropic):
    root = tmp_path/'public'; root.mkdir()
    unused, _, _, _ = inputs(root, kind)
    error, parent_sg = S.SpcGroup('P -4 21 m'); assert not error
    operations = '\n'.join("'"+op.lower()+"'" for op in S.AllOps(parent_sg)[0])
    cif = root/'parent.cif'
    cif.write_text(f'''data_public_centered_control
_cell_length_a 6.8
_cell_length_b 6.8
_cell_length_c 4.6
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P -4 21 m'
_space_group_IT_number 113
loop_
_space_group_symop_operation_xyz
{operations}
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_U_iso_or_equiv
Fe1 Fe .13731 .63731 .21879 1 .01
''')
    bounds = (12., 130.) if kind == 'PNC' else (3000., 40000.)
    reference = create_reference(cif, root/'bank.instprm', root/'centered-reference',
        kind=kind, lower=bounds[0], upper=bounds[1], bins=701)
    source = reference['path']
    project = G.G2Project(gpxfile=source)
    parent = project.phase(reference['parent_phase'])
    if anisotropic:
        cia = parent.data['General']['AtomPtrs'][3]
        atom = parent.data['Atoms'][0]
        atom[cia] = 'A'
        atom[cia+2:cia+8] = [.01, .01, .015, .002, 0., 0.]
        G.SetupGeneral(parent.data, None)
        project.save()
    original = file_sha(source)
    protocol = dict(source={'sha256': original}, parent_phase=parent.name,
        histogram=reference['histogram'], parent_setting_sha256=setting_fingerprint(parent.data['General']['SGData']),
        k=['0', '0', '0'], sites=[dict(label='Fe1', form_factor='Fe+3', lande_g=2.)],
        policy=dict(maximum_cell_multiplier=8, magnetic_d_min=0., cycles=1,
            maximum_moment=6., site_policy='all', refine_histogram_scale=False,
            refine_background=False, moment_model='symmetry_allowed'))
    candidate = dict(id='public-centered', magnetic_space_group=symbol, bns_number=bns,
        bns_lattice='', origin_shift=origin,
        transformation_matrix=[[1., -1., 0.], [1., 1., 0.], [0., 0., 1.]])
    seed = dict(axis='x', magnitude=1.7, parent_direction=None, independent_mode_seed=17, site_signs={})
    output = tmp_path/'compact'; output.mkdir()
    result = construct_theory(source, Artifacts(output), protocol, candidate, seed)
    assert result['audit']['passed'] and result['periodicity']['passed']
    compact = output/result['gpx']
    project = G.G2Project(gpxfile=str(compact))
    phase = next(p for p in project.phases() if p.data['General']['Type'] == 'magnetic')
    general = phase.data['General']; cx, ct, cs, _ = general['AtomPtrs']
    cell = general['Cell'][1:7]
    direct = L.cell2AB(cell)[0]; unit_axes = direct @ np.diag(1/np.asarray(cell[:3]))
    inverse_axes = np.linalg.inv(unit_axes); explicit = []
    # This expansion uses the independent BNS database, not native GenAtom,
    # ExpandCell, GetUnique, or the adapter being tested. The axial-vector
    # transform includes parity, time reversal, and the crystal-axis metric.
    for atom in phase.data['Atoms']:
        for operation in sorted(authoritative_operations(candidate)):
            rotation = np.asarray(operation[:9]).reshape(3, 3)
            translation = np.asarray(operation[9:12]); spin = operation[12]
            position = (rotation @ atom[cx:cx+3] + translation) % 1.
            moment = inverse_axes @ (spin*np.linalg.det(rotation)*direct @ rotation
                @ np.linalg.inv(direct) @ unit_axes @ atom[cx+4:cx+7])
            matches = [row for row in explicit if np.max(np.abs((row[cx:cx+3]-position+.5)%1.-.5)) < 1e-9]
            if matches:
                np.testing.assert_allclose(matches[0][cx+4:cx+7], moment, rtol=0, atol=1e-9)
                continue
            row = copy.deepcopy(atom)
            row[ct-1] = 'Fe_explicit_'+str(len(explicit))
            row[cx:cx+3] = position.tolist(); row[cx+4:cx+7] = moment.tolist()
            row[cs] = '1'; row[cs+1] = 1; row[-1] += len(explicit)+1000
            explicit.append(row)
    assert len(explicit) == 8
    assert any(np.linalg.norm(a[cx+4:cx+7]) > .1 for a in explicit)
    phase.data['Atoms'] = explicit
    general['SGData'] = L.makeBilbaoPhase(['P 1', ''], np.zeros(3), np.eye(3), True)['SGData']
    for hap in phase.data['Histograms'].values():
        assert not any(hap['HStrain'][0]) and not any(hap['HStrain'][1])
        count = len(S.HStrainNames(general['SGData']))
        hap['HStrain'] = [[0.]*count, [False]*count]
    G.SetupGeneral(phase.data, None)
    independent = tmp_path/'explicit-atoms.gpx'; project.save(str(independent))
    calculated, _ = forward_profile(compact, protocol['histogram'])
    expected, _ = forward_profile(independent, protocol['histogram'])
    np.testing.assert_allclose(calculated, expected, rtol=1e-8, atol=1e-8)
    control = G.G2Project(gpxfile=str(compact))
    magnet = next(p for p in control.phases() if p.data['General']['Type'] == 'magnetic')
    for atom in magnet.data['Atoms']:
        atom[cx+4:cx+7] = [0., 0., 0.]
    zero_path = tmp_path/'zero-moments.gpx'; control.save(str(zero_path))
    nuclear, _ = forward_profile(zero_path, protocol['histogram'])
    signal = calculated-nuclear
    assert np.max(np.abs(signal)) > 1e-7*max(1., np.max(np.abs(calculated)))
    np.testing.assert_allclose(signal, expected-nuclear, rtol=1e-8, atol=1e-12)
    assert file_sha(source) == original
    (tmp_path/'profile-comparison.json').write_text(json.dumps(dict(
        kind=kind, bns=bns, anisotropic=anisotropic, explicit_atoms=len(explicit),
        maximum_absolute_difference=float(np.max(np.abs(calculated-expected))),
        maximum_magnetic_signal=float(np.max(np.abs(signal))),
        scope='Public artificial model; independently expanded BNS operations; no experimental inference'), indent=2))
