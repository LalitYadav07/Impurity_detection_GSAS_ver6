"""Joint cell propagation: preserve correlations, reject unqualified blocks."""
import copy

import numpy as np
import pytest

from scripts.ndip_lattice import add_cross_phase_covariance


def fixture():
    ja=np.zeros((7,2));ja[0]=[1,0];ja[2]=[0,1];ja[6]=[2,3]
    jb=np.zeros((7,1));jb[0]=[4];jb[1]=[4];jb[6]=[9]
    rows=[dict(phase_key="a",cell_variable_names=["0::A0","0::A2"],cell_jacobian=ja.tolist(),
               covariance_status="symmetry_jacobian",covariance_fresh=True),
          dict(phase_key="b",cell_variable_names=["1::A0"],cell_jacobian=jb.tolist(),
               covariance_status="symmetry_jacobian",covariance_fresh=True)]
    covariance=dict(freshCOV=True,varyList=["0::A0","0::A2","1::A0"],
        covMatrix=[[4e-6,1e-6,2e-6],[1e-6,9e-6,-1e-6],[2e-6,-1e-6,16e-6]])
    return rows,covariance,ja,jb


def test_joint_cell_block_matches_independent_linear_propagation():
    rows,covariance,ja,jb=fixture()
    before=copy.deepcopy(covariance)
    add_cross_phase_covariance(rows,covariance)
    expected=ja@np.asarray(covariance["covMatrix"])[:2,2:]@jb.T
    np.testing.assert_allclose(rows[0]["cross_phase_cell_covariance"]["b"],expected)
    np.testing.assert_allclose(rows[1]["cross_phase_cell_covariance"]["a"],expected.T)
    assert covariance==before


@pytest.mark.parametrize("invalid",["stale","duplicate_variable","non_psd","asymmetric","nonfinite","bad_jacobian","unqualified","duplicate_phase"])
def test_invalid_joint_information_cannot_produce_ratio_error_evidence(invalid):
    rows,covariance,_,_=fixture()
    if invalid=="stale":covariance["freshCOV"]=False
    if invalid=="duplicate_variable":covariance["varyList"][1]=covariance["varyList"][0]
    if invalid=="non_psd":covariance["covMatrix"][0][2]=covariance["covMatrix"][2][0]=1.
    if invalid=="asymmetric":covariance["covMatrix"][0][2]=1.
    if invalid=="nonfinite":covariance["covMatrix"][0][2]=float("nan")
    if invalid=="bad_jacobian":rows[0]["cell_jacobian"]=None
    if invalid=="unqualified":rows[0]["covariance_status"]="unavailable"
    if invalid=="duplicate_phase":rows.append(copy.deepcopy(rows[0]))
    add_cross_phase_covariance(rows,covariance)
    assert not any(r["cross_phase_cell_covariance"] for r in rows)
