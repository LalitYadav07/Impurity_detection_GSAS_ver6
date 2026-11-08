"""Irrep-family targets retain exact parent k, secondary modes and unknowns."""
import json
import shutil

import numpy as np
import pytest

from gpx_recipe.preparation import structural_mode_projection as projection
from gpx_recipe.preparation.structural_mode_families import FamilyProjection,family_label,NORMALIZATION
from gpx_recipe.preparation.structural_paired_blocks import read,pack_block
from gpx_recipe.symmetry.compiler import file_sha
from test_structural_paired_blocks import block
from test_structural_mode_projection import attach,rewrite_projection


def label(irrep='GM1+',*,k='0,0,0',site='Ba1',opd='a'):
    return f'R-3m[{k}]{irrep}({opd})[{site}:c:dsp]A1(a)'


def native_map(root,mapping,gauges,gram,labels):
    path=block(root);attach(path)
    def arrays(a):a.update(physical_to_native=np.asarray(mapping,np.float64),
        gauge_basis=np.asarray(gauges,np.float64),physical_gram=np.asarray(gram,np.float64))
    def metadata(d):d.update(native_dimensions=len(labels),physical_dimensions=np.shape(mapping)[1],
        native_mode_labels=labels,native_normalization=[1.]*len(labels))
    rewrite_projection(path,change=metadata,arrays=arrays)
    return projection.read(path)


def test_one_family_contains_every_site_component_and_keeps_opd_variants(tmp_path):
    value=native_map(tmp_path/'map',np.eye(3),np.empty((3,0)),np.eye(3),
        [label(site='Ba1'),label(site='Ba2',opd='a,0'),label('GM3-',site='O1')])
    model=FamilyProjection(value)
    result=model.project(np.array([[.1,.2,.3],[0.,0.,0.]]))
    by_irrep={r['irrep']:i for i,r in enumerate(model.families)}
    first=next(r for r in model.families if r['irrep']=='GM1+')
    assert first['native_indices']==[0,1] and first['order_parameter_directions']==['a','a,0']
    np.testing.assert_allclose(result['amplitudes'][:,by_irrep['GM1+']],[np.sqrt(.05),0.])
    np.testing.assert_allclose(result['amplitudes'][:,by_irrep['GM3-']],[.3,0.])
    assert result['known'].all() and result['normalization']==NORMALIZATION
    assert model.metadata['all_native_axes_accounted']
    assert not result['experimental_mode_absence_qualified']


def test_nonorthogonal_physical_metric_is_not_replaced_by_native_amplitude_units(tmp_path):
    value=native_map(tmp_path/'map',np.diag([2.,3.]),np.empty((2,0)),[[4.,.5],[.5,1.]],
        [label(),label('GM3-')])
    model=FamilyProjection(value);point=np.array([[.1,.2]])
    np.testing.assert_allclose(model.native_amplitudes(point),[[.2,.6]])
    np.testing.assert_allclose(model.project(point)['amplitudes'],[[.2,.2]])
    # Cross-family Gram terms mean these marginal norms need not add in
    # quadrature to the complete displacement norm.
    assert not np.isclose(np.sum(model.project(point)['amplitudes']**2),point[0]@value.arrays['physical_gram']@point[0])


def test_origin_only_family_is_explicitly_not_an_absence_target(tmp_path):
    value=native_map(tmp_path/'map',[[1.],[0.]],[[0.],[1.]],[[4.]],
        [label(),label('GM3-')])
    model=FamilyProjection(value);result=model.project(np.array([[.05]]))
    by_irrep={r['irrep']:i for i,r in enumerate(model.families)}
    assert result['known'][0,by_irrep['GM1+']]
    assert not result['known'][0,by_irrep['GM3-']]
    assert next(r for r in model.families if r['irrep']=='GM3-')['pure_origin_gauge']


def test_a_family_mixed_with_origin_freedom_remains_unresolved(tmp_path):
    value=native_map(tmp_path/'map',[[1.],[1.]],[[1.],[-1.]],[[1.]],
        [label(),label('GM3-')])
    model=FamilyProjection(value);result=model.project(np.array([[.1]]))
    assert not result['known'].any()
    assert all(not r['origin_gauge_invariant'] for r in model.families)
    assert all(not r['pure_origin_gauge'] for r in model.families)


def test_native_gauge_roundoff_outside_a_physical_family_is_not_order_one_leakage(tmp_path):
    value=native_map(tmp_path/'map',[[1.,0.],[0.,1.],[0.,0.]],[[1e-16],[0.],[1.]],np.eye(2),
        [label(),label('GM3-'),label('GM2-')])
    model=FamilyProjection(value);rows={r['irrep']:r for r in model.families}
    assert rows['GM1+']['origin_gauge_invariant'] and rows['GM3-']['origin_gauge_invariant']
    assert not rows['GM2-']['amplitude_target_available']
    assert rows['GM1+']['gauge_reconstruction_residual']<=1e-16


def test_weak_native_units_do_not_hide_large_physical_gauge_coupling(tmp_path):
    value=native_map(tmp_path/'map',[[1e-12],[1e-12]],[[1.],[-1.]],[[1.]],
        [label(),label('GM3-')])
    model=FamilyProjection(value)
    assert not any(r['origin_gauge_invariant'] for r in model.families)
    assert not model.project(np.array([[.1]]))['known'].any()


def test_complete_irrep_family_can_include_origin_directions_without_losing_physical_modes(tmp_path):
    value=native_map(tmp_path/'map',[[1.],[1.]],[[1.],[-1.]],[[1.]],
        [label(),label(site='Ba2')])
    model=FamilyProjection(value);result=model.project(np.array([[.1]]))
    assert result['known'].all() and result['amplitudes'][0,0]==pytest.approx(.1)


def test_opaque_native_labels_are_retained_without_guessed_irreps(tmp_path):
    value=native_map(tmp_path/'map',np.eye(2),np.empty((2,0)),np.eye(2),[label(),'unrecognized native label'])
    model=FamilyProjection(value)
    assert model.metadata['recognized_native_axes']==1 and model.metadata['native_axes']==2
    assert model.opaque_native_axes[0]['native_index']==1
    assert model.opaque_native_axes[0]['label']=='unrecognized native label'
    assert model.project(np.array([[.1,.2]]))['known'].shape==(1,1)


def test_exact_k_is_retained_in_centered_parent_conventions():
    first=family_label(label('F1-',k='0,1/2,1'),parent_sha256='a'*64)
    second=family_label(label('F1-',k='0,1/2,0'),parent_sha256='a'*64)
    assert first['id']!=second['id'] and first['k']==['0','1/2','1']
    equivalent=family_label(label('F1-',k='0,0.5,1'),parent_sha256='a'*64)
    assert equivalent['id']==first['id']
    assert family_label(label('F1-',k='0,1/2,1'),parent_sha256='b'*64)['id']!=first['id']


def test_extreme_native_units_and_tiny_physical_modes_are_not_dropped(tmp_path):
    value=native_map(tmp_path/'map',np.diag([1e-12,1e12]),np.empty((2,0)),np.eye(2),[label(),label('GM3-')])
    result=FamilyProjection(value).project(np.array([[1e-200,2e-200]]))
    np.testing.assert_allclose(result['amplitudes']/1e-200,[[1.,2.]],rtol=1e-12,atol=0)
    assert result['known'].all()


def test_projection_batches_are_bounded_without_dropping_any_coordinate(tmp_path):
    value=native_map(tmp_path/'map',np.eye(2),np.empty((2,0)),np.eye(2),[label(),label('GM3-')])
    model=FamilyProjection(value)
    for points in [np.zeros((257,2)),np.zeros((1,1)),np.zeros((1,2),int),np.full((1,2),np.nan)]:
        with pytest.raises(ValueError):model.project(points)


@pytest.mark.parametrize('packed',[False,True])
def test_binding_recovered_coordinates_keeps_all_original_numerical_file_bytes(tmp_path,packed):
    raw=block(tmp_path/'raw')
    source=raw
    if packed:
        pack_block(raw,tmp_path/'packed');source=tmp_path/'packed'
    maps=tmp_path/'maps';shutil.copytree(source,maps);mapped=attach(maps)
    before={name:(source/name).read_bytes() for name in ['pairs.json','pairs.npz']}
    target=tmp_path/'bound'
    restored=projection.bind_block(source,target,maps,names='pairs',stamp=mapped.mode_projection.stamp)
    assert restored.manifest['mode_projection_source_manifest_sha256']==file_sha(source/'pairs.json')
    assert (target/'pairs.npz').read_bytes()==before['pairs.npz']
    assert before=={name:(source/name).read_bytes() for name in before}
    for key,value in read(source).arrays.items():assert value.tobytes()==restored.arrays[key].tobytes()
    with pytest.raises(ValueError):projection.bind_block(target,tmp_path/'twice',maps,names='pairs',stamp=mapped.mode_projection.stamp)


def test_binding_rejects_a_different_child_before_creating_output(tmp_path):
    source=block(tmp_path/'source');maps=block(tmp_path/'maps');mapped=attach(maps)
    d=json.loads((source/'pairs.json').read_text());d['candidate']['origin_shift']=[.5,0,0]
    (source/'pairs.json').write_text(json.dumps(d))
    with pytest.raises(ValueError,match='different parent'):
        projection.bind_block(source,tmp_path/'rejected',maps,names='pairs',stamp=mapped.mode_projection.stamp)
    assert not (tmp_path/'rejected').exists()
