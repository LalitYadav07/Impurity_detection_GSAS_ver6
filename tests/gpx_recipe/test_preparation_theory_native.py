"""Begin before measurement with only a public CIF and bank profile."""
from pathlib import Path
import numpy as np
import pytest

G = pytest.importorskip('GSASII.GSASIIscriptable')
from gpx_recipe.preparation.reference import create_reference
from gpx_recipe.preparation.native_magnetic import compile_response
from gpx_recipe.live.native_profile import NativeReducedProfile
from gpx_recipe.symmetry.native_magnetic import construct_theory
from gpx_recipe.symmetry.worker import Artifacts
from gpx_recipe.symmetry.compiler import file_sha
from gsas_magnetic_preflight import setting_fingerprint


CIF = '''data_public_artificial
_cell_length_a 4.7
_cell_length_b 5.6
_cell_length_c 6.8
_cell_angle_alpha 79
_cell_angle_beta 93
_cell_angle_gamma 104
_space_group_name_H-M_alt 'P 1'
_space_group_IT_number 1
loop_
_space_group_symop_operation_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_U_iso_or_equiv
_atom_site_occupancy
Fe1 Fe .13 .21 .34 .005 1
O1 O .37 .29 .19 .005 1
'''


def inputs(root, kind):
    cif = root/'parent.cif'; cif.write_text(CIF)
    instrument = root/'bank.instprm'
    if kind == 'PNC':
        fields = ('Lam:1.54\nZero:0\nU:.12\nV:0\nW:.08\n'
            'X:0\nY:0\nZ:0\nSH/L:.002\nPolariz.:0\nAzimuth:0\n')
        bounds = (12., 130.)
    else:
        fields = ('difC:6200\ndifA:2\ndifB:.4\nZero:1\n2-theta:105\nfltPath:15.5\nalpha:.99\n'
            'beta-0:.02\nbeta-1:.03\nbeta-q:0\nsig-0:1\nsig-1:66\nsig-2:0\nsig-q:0\n'
            'X:0\nY:0\nZ:0\nAzimuth:0\n')
        bounds = (3000., 40000.)
    instrument.write_text('#GSAS-II instrument parameter file\nType:'+kind+'\nBank:1\n'+fields)
    reference = create_reference(cif, instrument, root/'reference', kind=kind,
        lower=bounds[0], upper=bounds[1], bins=701)
    project = G.G2Project(gpxfile=reference['path'])
    phase = project.phase(reference['parent_phase'])
    protocol = dict(source={'sha256': file_sha(reference['path'])}, parent_phase=phase.name,
        histogram=reference['histogram'], parent_setting_sha256=setting_fingerprint(phase.data['General']['SGData']),
        k=['0', '0', '0'], sites=[dict(label='Fe1', form_factor='Fe+3', lande_g=2.)],
        policy=dict(maximum_cell_multiplier=8, magnetic_d_min=0., cycles=1, maximum_moment=6.,
                    site_policy='all', refine_histogram_scale=True, refine_background=True,
                    moment_model='symmetry_allowed'))
    candidate = dict(id='public:p1', magnetic_space_group='P1', bns_number='1.1', bns_lattice='',
        origin_shift=[0., 0., 0.], transformation_matrix=np.eye(3).tolist())
    seed = dict(axis='x', magnitude=1.7, parent_direction=None, independent_mode_seed=17, site_signs={})
    return reference, protocol, candidate, seed


@pytest.mark.parametrize('kind', ['PNC', 'PNT'])
def test_cif_to_audited_magnetic_physics_has_no_measured_scan(tmp_path, kind):
    reference, protocol, candidate, seed = inputs(tmp_path, kind)
    before = Path(reference['path']).read_bytes()
    output = tmp_path/'candidate'; output.mkdir()
    result = construct_theory(reference['path'], Artifacts(output), protocol, candidate, seed)
    assert result['status'] == 'theory_constructed'
    assert result['observations_included'] is False and result['no_refinement'] is True
    assert result['audit']['passed'] and result['periodicity']['passed']
    assert not (output/'baseline-refresh.json').exists()
    model = output/result['gpx']
    response = compile_response(NativeReducedProfile(model))
    assert response.quadratic.shape[1] == 3 and response.background.shape[1] > 0
    assert np.any(response.quadratic)
    saved = G.G2Project(gpxfile=str(model))
    assert not np.any(saved.histograms()[0].data['data'][1][1])
    assert saved.histograms()[0].data['data'][0]['Dummy'] is True
    assert Path(reference['path']).read_bytes() == before


def test_theory_constructor_cannot_relabel_measured_counts(tmp_path):
    reference, protocol, candidate, seed = inputs(tmp_path, 'PNC')
    project = G.G2Project(gpxfile=reference['path'])
    project.histograms()[0].data['data'][1][1][20] = 1.
    project.save()
    output = tmp_path/'candidate'; output.mkdir()
    with pytest.raises(ValueError, match='observation-free reference'):
        construct_theory(reference['path'], Artifacts(output), protocol, candidate, seed)
