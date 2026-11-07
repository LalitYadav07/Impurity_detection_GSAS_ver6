"""One theory-only TOF table serves different structures and measured grids."""
import numpy as np
import pytest

pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from gpx_recipe.preparation.reference import create_reference
from gpx_recipe.live.native_profile import NativeReducedProfile
from gpx_recipe.structural_live.optimization.tof_profile_table import CompiledTOFTable
from gpx_recipe.structural_live.optimization import tof_profile_table
from gpx_recipe.structural_live.optimization.tof_profile_proposals import TOFProfileProposals
from gpx_recipe.structural_live.optimization.native_cell_proposals import ProposalUnavailable


@pytest.fixture(scope='module')
def theory_table(tmp_path_factory):
    root=tmp_path_factory.mktemp('cell-free-tof-table')
    cif=root/'public-parent.cif'
    cif.write_text('''data_public_control
_symmetry_space_group_name_H-M 'P m -3 m'
_cell_length_a 3.4
_cell_length_b 3.4
_cell_length_c 3.4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_U_iso_or_equiv
Si1 Si 0 0 0 1 0.01
''')
    instrument=root/'bank.instprm'
    instrument.write_text('''#GSAS-II instrument parameter file
Type:PNT
Bank:1
difC:6200
difA:2
difB:.4
Zero:1
2-theta:105
fltPath:15.5
alpha:.99
beta-0:.02
beta-1:.03
beta-q:0
sig-0:1
sig-1:66
sig-2:0
sig-q:0
X:0
Y:0
Z:0
Azimuth:0
''')
    reference=create_reference(cif,instrument,root/'theory',kind='PNT',lower=3000.,upper=40000.,bins=701)
    native=NativeReducedProfile(reference['path'])
    # Neither a child CIF nor a structural-mode source nor a measured scan
    # exists during this compilation. Even the dummy reference arrays are
    # unavailable to the compiler; its interval is an explicit planned input.
    class NoArrays:
        def __getitem__(self,key):raise AssertionError('No scan arrays belong in profile compilation')
        def __iter__(self):raise AssertionError('No scan arrays belong in profile compilation')
        def __array__(self,*a,**k):raise AssertionError('No scan arrays belong in profile compilation')
    native.histo_phases[0][reference['histogram']]['Data']=NoArrays()
    return CompiledTOFTable.from_reference(native,reference['histogram'],reference['parent_phase'],
        tof_range=(3000.,40000.))


def test_theory_table_rebinds_to_another_structure_grid_and_changing_cell(theory_table,tmp_path,monkeypatch):
    _,source=native_source(tmp_path/'observed','PNT')
    before={k:v.tobytes() for k,v in theory_table.arrays().items()}
    def forbidden(*a,**k):pytest.fail('The prepared table must not be rebuilt for another structure or cell')
    monkeypatch.setattr(CompiledTOFTable,'__init__',forbidden)
    # The ordinary renderer selects the matching prepared component itself;
    # callers do not inject private initialization or a fast/accurate option.
    proposal=TOFProfileProposals(source)
    assert proposal.table is theory_table and proposal.diagnostics()['owned_table_reused']
    assert proposal.coefficients is theory_table.coefficients
    rng=np.random.default_rng(291057)
    for point in [np.zeros(len(source.radii)),*[rng.uniform(-.6,.6,len(source.radii))*source.radii for _ in range(4)]]:
        actual=proposal.evaluate(point);expected=source.native(point)
        hist=source.fast.slow.hist.data['data'][1]
        background=np.zeros(len(expected));background[source.fast.slice]=hist[4][source.fast.slice]
        delta=actual-expected
        assert np.linalg.norm(delta)/max(np.linalg.norm(expected-background),1e-30)<=1e-8
        assert np.max(abs(delta)*np.sqrt(hist[2]))<=.01
    assert before=={k:v.tobytes() for k,v in theory_table.arrays().items()}
    import json
    assert json.loads(json.dumps(proposal.diagnostics()))==proposal.diagnostics()
    assert set(theory_table.arrays())=={'coefficients','halo','logd','eta'}
    for array in theory_table.arrays().values():
        with pytest.raises(ValueError):array.flags.writeable=True


def test_other_profile_or_unattributed_arrays_cannot_claim_owned_reuse(theory_table,tmp_path):
    _,source=native_source(tmp_path/'input','PNT')
    with pytest.raises(TypeError,match='owned native'):TOFProfileProposals(source,table=theory_table.arrays())
    source.fast.native.parms[f':{source.fast.slow.hist.id}:X']+=.1
    with pytest.raises(ProposalUnavailable,match='differs from the current'):
        TOFProfileProposals(source,table=theory_table)


@pytest.mark.parametrize('change',['process','thread','floating','compiled'])
def test_table_ownership_and_native_runtime_remain_required(theory_table,monkeypatch,change):
    if change=='process':monkeypatch.setattr(theory_table,'_pid',-1)
    elif change=='thread':monkeypatch.setattr(theory_table,'_thread',object())
    elif change=='floating':monkeypatch.setattr(tof_profile_table,'floating_state',lambda:(1919,16256))
    else:monkeypatch.setattr(theory_table._powder.pyd,'pyepsvoigt',lambda *a:None)
    with pytest.raises(ProposalUnavailable):theory_table.check_runtime()
