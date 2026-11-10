"""Public artificial strain maps; independent equations and strict bindings."""
import copy
import hashlib

import numpy as np
import pytest

from gpx_recipe.preparation.structural_strain import cell_metric
from gpx_recipe.preparation.structural_strain_cif import read,_transform,from_metadata
from gpx_recipe.preparation.structural_mode_families import family_label,STRAIN_NORMALIZATION


CIF='''data_child
_iso_strainmode_number 3
loop_
_iso_strainmode_ID
_iso_strainmode_label
_iso_strainmode_value
1 P4/nmm[0,0,0]GM4+(a)strain(a) 0
2 P4/nmm[0,0,0]GM1+(a)strain_1(a) 0
3 P4/nmm[0,0,0]GM1+(a)strain_2(a) 0
loop_
_iso_strainmodenorm_ID
_iso_strainmodenorm_value
1 1.4142135623730951
2 0.7071067811865475
3 1.0
loop_
_iso_strain_ID
_iso_strain_label
1 E_1
2 E_2
3 E_3
4 E_4
5 E_5
6 E_6
loop_
_iso_strainmodematrix_row
_iso_strainmodematrix_col
_iso_strainmodematrix_value
1 2 1
2 2 1
3 3 1
6 1 1
_iso_parentcell_length_a 4
_iso_parentcell_length_b 4
_iso_parentcell_length_c 9
_iso_parentcell_angle_alpha 90
_iso_parentcell_angle_beta 90
_iso_parentcell_angle_gamma 90
_iso_parent-to-child.transform_Pp_abc a+b,-a+b,c;0,-1/2,0
'''


def fixture(tmp_path,text=CIF):
    path=tmp_path/'child.cif';path.write_text(text)
    candidate=dict(id='public-strain',transformation_matrix=[[1,1,0],[-1,1,0],[0,0,1]],
        origin_shift=[0,.5,0],child=dict(path='child.cif',sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return path,candidate


def model(tmp_path,text=CIF):
    path,candidate=fixture(tmp_path,text)
    return read(path,parent_sha256='a'*64,candidate=candidate)


def test_complete_iso_map_retains_signed_coordinates_and_typed_physical_families(tmp_path):
    value=model(tmp_path);m=np.array([.012,-.004,.006])
    child=value.projection.forward(m)
    result=value.project(child[None])
    np.testing.assert_allclose(result['native_amplitudes'][0],m,atol=1e-14)
    by_irrep={f['irrep']:result['amplitudes'][0,i] for i,f in enumerate(value.families)}
    assert by_irrep==pytest.approx({'GM4+':.012,'GM1+':np.hypot(.004,.006)})
    assert result['normalization']==STRAIN_NORMALIZATION and result['units']=='dimensionless'
    assert result['known'].all() and not result['candidate_pruning_authorized']
    assert not value.metadata['observations_included']


def test_only_explicit_parent_reference_removes_ordinary_thermal_change(tmp_path):
    value=model(tmp_path);parent=cell_metric([4.01,4.01,9.02,90,90,90])
    child=value.projection.basis@parent@value.projection.basis.T
    absolute=value.project(child[None]);current=value.project(child[None],parent_metrics=parent[None])
    gm4=next(i for i,r in enumerate(value.families) if r['irrep']=='GM4+')
    assert absolute['amplitudes'][0,gm4]<1e-12
    assert np.max(absolute['amplitudes'])>1e-3
    np.testing.assert_allclose(current['amplitudes'],0,atol=1e-12)


def test_displacement_and_strain_of_same_irrep_never_share_a_target_identity():
    displacement=family_label('P4/nmm[0,0,0]GM1+(a)[Pr1:c:dsp]A1(a)',parent_sha256='a'*64)
    strain=family_label('P4/nmm[0,0,0]GM1+(a)strain_1(a)',parent_sha256='a'*64,quantity='lattice_strain')
    assert displacement['id']!=strain['id'] and strain['quantity']=='lattice_strain'
    assert 'units' not in displacement and strain['units']=='dimensionless'
    assert family_label('P4/nmm[1/2,0,0]M1+(a)strain(a)',parent_sha256='a'*64,quantity='lattice_strain') is None


def test_unknown_family_label_stays_unknown_without_losing_a_strain_coordinate(tmp_path):
    value=model(tmp_path,CIF.replace('P4/nmm[0,0,0]GM4+(a)strain(a)','unknown-strain-coordinate'))
    assert value.projection.dimensions==3 and len(value.families)==1
    assert value.metadata['opaque_native_axes']==[dict(native_index=0,label='unknown-strain-coordinate')]
    result=value.project(value.projection.forward([.012,0,0])[None])
    assert result['native_amplitudes'][0,0]==pytest.approx(.012)
    assert result['amplitudes'][0,0]<1e-12


@pytest.mark.parametrize('change',[
    lambda s:s.replace('3 1.0','3 0'),
    lambda s:s.replace('3 1.0','3 nan'),
    lambda s:s.replace('3 1.0','2 1.0'),
    lambda s:s.replace('6 1 1','6 1 1\n6 1 1'),
    lambda s:s.replace('6 1 1','7 1 1'),
    lambda s:s.replace('6 1 1','6 4 1'),
    lambda s:s.replace('6 1 1',''),
    lambda s:s.replace('6 E_6','6 E_5'),
    lambda s:s.replace('a+b,-a+b,c;0,-1/2,0','a+b,a+b,c;0,-1/2,0'),
    lambda s:s.replace('a+b,-a+b,c;0,-1/2,0','a+b,-a+b,c;0,1/4,0'),
    lambda s:s.replace('a+b,-a+b,c;0,-1/2,0','a+b,-a+b,c+1;0,-1/2,0'),
    lambda s:s.replace('_iso_parentcell_length_c 9','_iso_parentcell_length_c 0'),
])
def test_bad_or_incomplete_metadata_never_becomes_zero_or_missing_targets(tmp_path,change):
    with pytest.raises((ValueError,KeyError)):model(tmp_path,change(CIF))


def test_absent_strain_table_remains_unavailable_but_partial_table_is_an_error(tmp_path):
    path,candidate=fixture(tmp_path,'data_child\n_iso_strainmode_number 0\n')
    assert read(path,parent_sha256='a'*64,candidate=candidate) is None
    path,candidate=fixture(tmp_path,'data_child\n_iso_strainmode_number 0\n_iso_strainmode_label missing\n')
    with pytest.raises(ValueError,match='Partial'):read(path,parent_sha256='a'*64,candidate=candidate)


@pytest.mark.parametrize('case',['hash','parent','basis','origin','magnetic'])
def test_other_input_context_cannot_borrow_the_map(tmp_path,case):
    path,candidate=fixture(tmp_path);parent='a'*64
    if case=='hash':candidate['child']['sha256']='b'*64
    elif case=='parent':parent='invalid'
    elif case=='basis':candidate['transformation_matrix'][0]=[2,0,0]
    elif case=='origin':candidate['origin_shift'][0]=.25
    else:candidate['magnetic_space_group']='P1'
    with pytest.raises(ValueError):read(path,parent_sha256=parent,candidate=candidate)


def test_indexed_loop_permutations_preserve_the_same_complete_map(tmp_path):
    text=CIF.replace('1 1.4142135623730951\n2 0.7071067811865475\n3 1.0',
                     '3 1.0\n1 1.4142135623730951\n2 0.7071067811865475')
    value=model(tmp_path,text)
    result=value.project(value.projection.forward([.012,-.004,.006])[None])
    np.testing.assert_allclose(result['native_amplitudes'],[[.012,-.004,.006]],atol=1e-14)


def test_rational_transform_parser_rejects_non_linear_and_executable_input():
    basis,origin=_transform('a/2+b, -a+b/2, c; 0, -1/2, 0')
    np.testing.assert_allclose(basis,[[.5,1,0],[-1,.5,0],[0,0,1]])
    np.testing.assert_allclose(origin,[0,-.5,0])
    for text in ['__import__(os),b,c;0,0,0','a*b,b,c;0,0,0','a/0,b,c;0,0,0','a,b,c;0,0',
                 'a,b,c;0,nan,0','a,b,c;0,0,0;0','(a+b),b,c;0,0,0']:
        with pytest.raises(ValueError):_transform(text)


def test_native_centering_translation_can_relate_reviewed_and_iso_origins(tmp_path):
    text=CIF.replace('a+b,-a+b,c;0,-1/2,0','a+b,-a+b,c;4/3,5/3,2/3')
    path,candidate=fixture(tmp_path,text);candidate['origin_shift']=[0,0,0]
    centers=[[0,0,0],[2/3,1/3,1/3],[1/3,2/3,2/3]]
    with pytest.raises(ValueError,match='embedding'):
        read(path,parent_sha256='a'*64,candidate=candidate)
    value=read(path,parent_sha256='a'*64,candidate=candidate,parent_translations=centers)
    assert value.metadata['origin_equivalence']['parent_translations']==centers
    restored=from_metadata(value.metadata,parent_sha256='a'*64,candidate=candidate)
    m=np.array([.012,-.004,.006])
    np.testing.assert_allclose(restored.projection.project(restored.projection.forward(m))['amplitudes'],m,atol=1e-14)
    for bad in ([.25,0,0],[1/3,1/3,2/3]):
        changed=copy.deepcopy(value.metadata);changed['origin_equivalence']['iso_origin']=bad
        with pytest.raises(ValueError,match='centering'):from_metadata(changed)


@pytest.mark.parametrize('centers',[
    [],[[0,0,0],[.25,0,0]],[[0,0,0],[0,0,0]],[[.5,0,0]],
    [[0,0,0],[float('nan'),0,0]],[[0,0]],
])
def test_partial_or_invalid_parent_centering_cannot_weaken_origin_binding(tmp_path,centers):
    path,candidate=fixture(tmp_path)
    with pytest.raises(ValueError,match='centering'):
        read(path,parent_sha256='a'*64,candidate=candidate,parent_translations=centers)


def test_bounded_complete_batches_cannot_mutate_maps_or_input_metrics(tmp_path):
    value=model(tmp_path);child=value.projection.forward([.012,0,0])[None];before=child.copy()
    value.project(child);np.testing.assert_array_equal(child,before)
    with pytest.raises(ValueError):value.projection.matrix.flags.writeable=True
    for metrics in [child[0],np.empty((0,3,3)),np.repeat(child,257,axis=0),np.full((1,3,3),np.nan)]:
        with pytest.raises(ValueError):value.project(metrics)


def test_physical_strain_norm_uses_parent_metric_and_is_invariant_to_lab_rotation(tmp_path):
    d=copy.deepcopy(model(tmp_path).metadata)
    d.update(parent_cell=[4,5,7,78,94,103],basis=np.eye(3).tolist(),matrix=np.eye(6).tolist(),
        normalizations=[1.]*6,native_labels=[f'P1[0,0,0]GM1+(a)strain_{i}(a)' for i in range(1,7)])
    d['candidate'].update(transformation_matrix=np.eye(3).tolist(),origin_shift=[0,0,0])
    value=from_metadata(d);e=np.array([.001,-.002,.003,.004,-.005,.006])
    tensor=np.diag(e[:3]);tensor[1,2]=tensor[2,1]=e[3]/2
    tensor[0,2]=tensor[2,0]=e[4]/2;tensor[0,1]=tensor[1,0]=e[5]/2
    parent=cell_metric(d['parent_cell']);child=(np.eye(3)+tensor)@parent@(np.eye(3)+tensor)
    result=value.project(child[None]);b=np.linalg.cholesky(parent).T
    cartesian=b@tensor@np.linalg.inv(b);physical=(cartesian+cartesian.T)/2
    expected=np.linalg.norm(physical)
    assert result['amplitudes'][0,0]==pytest.approx(expected,rel=1e-10)
    rotation,_=np.linalg.qr(np.array([[1.,2.,3.],[-1.,2.,5.],[3.,1.,4.]]))
    rotated=rotation@b@tensor@np.linalg.inv(rotation@b)
    assert np.linalg.norm((rotated+rotated.T)/2)==pytest.approx(expected,rel=1e-12)
    assert not np.isclose(expected,np.linalg.norm(e))
