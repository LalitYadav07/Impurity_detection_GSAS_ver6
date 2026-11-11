"""Independent position counts at cell boundaries; no diffraction fitting."""
import numpy as np
import pytest
import copy

G=pytest.importorskip('GSASII.GSASIIscriptable')
S=pytest.importorskip('GSASII.GSASIIspc')
from gpx_recipe.symmetry.space_group_patch import corrected_source


@pytest.mark.parametrize('source',['','def GenAtom():\n    pass\n','raise RuntimeError("unreviewed")\n'])
def test_unknown_source_cannot_be_patched(source):
    with pytest.raises(ValueError,match='Unreviewed'):
        corrected_source(source)


@pytest.mark.parametrize('symbol',['P m m 2','C m m 2'])
@pytest.mark.parametrize('epsilon',[-5e-6,5e-6])
def test_near_boundary_mirror_has_the_same_native_site_as_the_exact_mirror(symbol,epsilon):
    error,sg=S.SpcGroup(symbol);assert not error
    exact=S.SytSym([0.,.173,.317],sg)
    nearby=S.SytSym([epsilon,.173,.317],sg)
    assert nearby[:2]==exact[:2]
    assert nearby[3]==exact[3]
    assert len(list(S.GenAtom([epsilon,.173,.317],sg,Move=True)))==exact[1]


@pytest.mark.parametrize('symbol',['P m m 2','C m m 2'])
def test_distinct_off_mirror_site_is_not_merged(symbol):
    error,sg=S.SpcGroup(symbol);assert not error
    near=S.SytSym([0.,.173,.317],sg)
    far=S.SytSym([.001,.173,.317],sg)
    assert far[1]==2*near[1]
    assert len(list(S.GenAtom([.001,.173,.317],sg,Move=True)))==far[1]


def test_general_orbit_is_invariant_under_integer_cell_translation():
    error,sg=S.SpcGroup('P 21/c');assert not error
    a=list(S.GenAtom([.137,.231,.347],sg,All=True))
    b=list(S.GenAtom([1.137,-1.769,3.347],sg,All=True))
    assert len(a)==len(b)
    for first,second in zip(a,b):
        np.testing.assert_allclose((np.asarray(first[0])-second[0]+.5)%1.-.5,0.,atol=1e-12)
        assert first[1]==second[1] and first[-1]==second[-1]


@pytest.mark.parametrize('symbol,origin',[("Cmm2",[.5,0.,0.]),("Cm'm'2",[.5,0.,0.]),
    ("Cm",[.25,.25,0.]),("Cm'",[.25,.25,0.])])
@pytest.mark.parametrize('roundoff',[0.,1e-5])
def test_rotated_centered_child_preserves_parent_sites_and_native_functions(tmp_path,symbol,origin,roundoff):
    from GSASII import GSASIIscriptable as G, GSASIIlattice as L
    from gsas_magnetic_preflight import preflight_magnetic_sites,setting_fingerprint
    error,parent_sg=S.SpcGroup('P -4 21 m');assert not error
    operations='\n'.join("'"+op.lower()+"'" for op in S.AllOps(parent_sg)[0])
    path=tmp_path/'public-parent.cif'
    path.write_text(f'''data_public_control
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
Fe1 Fe 0.13731 {0.63731+roundoff:.5f} 0.21879 1 0.01
''')
    project=G.G2Project(newgpx=str(tmp_path/'public-parent.gpx'))
    parent=project.add_phase(str(path),phasename='public').data
    original=copy.deepcopy(parent)
    functions=(L.TransformPhase,L.ExpandCell,L.GetUnique)
    basis=np.array([[1,-1,0],[1,1,0],[0,0,1.]])
    child,report=preflight_magnetic_sites(parent,dict(candidate_id='public-centered',
        magnetic_space_group=symbol,bns_lattice='',transformation_matrix=basis.tolist(),origin_shift=origin),
        magnetic_sites=['Fe1'],site_policy='all',expected_parent_setting=setting_fingerprint(parent['General']['SGData']))
    assert report['compatible']
    cx,_,cs,_=child['General']['AtomPtrs']
    assert sum(a[cs+1] for a in child['Atoms'])==8
    px=parent['General']['AtomPtrs'][0]
    parent_points=[row[0] for row in S.GenAtom(parent['Atoms'][0][px:px+3],parent['General']['SGData'],All=True)]
    expected=[np.linalg.inv(basis).T@(xyz+np.asarray(shift)+origin) for xyz in parent_points
              for shift in [(0,0,0),(1,0,0)]]
    for atom in child['Atoms']:
        assert min(np.max(np.abs((atom[cx:cx+3]-point+.5)%1.-.5)) for point in expected)<1e-12
    assert parent['Atoms']==original['Atoms']
    assert (L.TransformPhase,L.ExpandCell,L.GetUnique)==functions
