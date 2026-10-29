from types import SimpleNamespace
import sys
import numpy as np
import pytest
from gpx_recipe.symmetry.magnetic_identity import authoritative_operations,verify_native_operators,operation_key,GreyGroup,UnverifiedMagneticSetting
from gpx_recipe.symmetry.magnetic_identity import require_collinear_line


def database(monkeypatch,operations):
    seen=[]
    def group(identifier):seen.append(identifier);return SimpleNamespace(symmetry_ops=[SimpleNamespace(rotation_matrix=r,translation_vector=t,time_reversal=s) for r,t,s in operations])
    monkeypatch.setitem(sys.modules,'pymatgen.symmetry.maggroups',SimpleNamespace(MagneticSpaceGroup=group))
    return seen


def test_bns_grey_group_forbids_moments_even_when_its_compact_name_looks_like_a_screw(monkeypatch):
    seen=database(monkeypatch,[(np.eye(3),[0,0,0],1),(np.eye(3),[0,0,0],-1)])
    with pytest.raises(GreyGroup,match='pure time reversal'):authoritative_operations(dict(bns_number='38.188',magnetic_space_group="Amm21'"))
    assert seen==[[38,188]]


def test_antitranslation_is_distinct_from_pure_time_reversal(monkeypatch):
    database(monkeypatch,[(np.eye(3),[0,0,0],1),(np.eye(3),[0,0,.5],-1)])
    result=authoritative_operations(dict(bns_number='1.3',magnetic_space_group='P_S1'))
    assert len(result)==2


def test_rotations_translations_and_spin_all_belong_to_the_verified_setting(monkeypatch):
    identity=np.eye(3);mirror=np.diag([-1,1,1]);authority={operation_key(identity,[0,0,0],1),operation_key(mirror,[0,0,.5],-1)}
    native=SimpleNamespace(AllOps=lambda _: (None,None,[(identity,np.zeros(3)),(mirror,np.array([0,0,.5]))]))
    monkeypatch.setitem(sys.modules,'GSASII',SimpleNamespace(GSASIIspc=native))
    assert verify_native_operators({},dict(SpnFlp=[1,-1]),authority=authority)['operation_count']==2
    with pytest.raises(UnverifiedMagneticSetting):verify_native_operators({},dict(SpnFlp=[1,1]),authority=authority)


def test_a_shared_asymmetric_axis_cannot_silently_generate_noncollinear_mates():
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    require_collinear_line([rotation],np.eye(3),[1,1,1],[0,0,1])
    with pytest.raises(UnverifiedMagneticSetting,match='noncollinear'):require_collinear_line([rotation],np.eye(3),[1,1,1],[1,0,0])


def test_symmetry_coupled_diagonal_is_not_replaced_with_a_cartesian_axis():
    swap=np.array([[0,1,0],[1,0,0],[0,0,1]])
    require_collinear_line([swap],np.eye(3),[1,1,1],[1,1,0])
    with pytest.raises(UnverifiedMagneticSetting):require_collinear_line([swap],np.eye(3),[1,1,1],[1,0,0])
