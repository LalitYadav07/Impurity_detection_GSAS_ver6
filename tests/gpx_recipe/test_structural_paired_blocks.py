import hashlib
import json
import numpy as np
import pytest

from gpx_recipe.preparation.structural_paired_blocks import read,SCHEMA


def block(root,change=None,metadata=None):
    root.mkdir();q=np.linspace(.5,4.,101);background=np.full(101,3.);n=17
    trajectory=np.r_[-1,np.repeat(np.arange(8),2)];roles=np.r_[0,np.tile([1,2],8)]
    parents=np.array([background+np.exp(-((q-(1+.001*max(i,0)))/.05)**2) for i in trajectory])
    curves=parents.copy();curves[roles==2]+=np.exp(-((q-2.2)/.05)**2)*.03
    labels=np.where(roles==2,12,166).astype(np.int32)
    arrays=dict(q=q,theory_intensity=curves,background=background,parent_intensity=parents,
        parent_background=np.broadcast_to(background,(n,101)).copy(),
        parent_a=np.tile([.04,.04,.01,0,0,0],(n,1)),parameters=np.zeros((n,3)),
        trajectory=trajectory,role=roles,geometry_groups=np.repeat(labels[:,None],2,axis=1),
        ambiguous_geometry=np.zeros(n,bool),displacement_rms=np.zeros(n),relative_metric_norm=np.zeros(n))
    if change:change(arrays)
    np.savez_compressed(root/'pairs.npz',**arrays)
    document=dict(schema=SCHEMA,observations_included=False,scientific_result=False,
        pairs_sha256=hashlib.sha256((root/'pairs.npz').read_bytes()).hexdigest(),
        source_sha256='a'*64,instrument_sha256='b'*64,child_sha256='c'*64,seed=1,parent_seed=2,
        mode_dimensions=1,cell_dimensions=2,examples=n,bins=101,trajectories=8,steps=1,trajectory_slice=[0,8],
        candidate=dict(id='child',transformation_matrix=np.eye(3).tolist(),origin_shift=[0,0,0],
            child=dict(path='child.cif',sha256='c'*64)),declared_child_group=dict(number=12),
        groups=['parent:'+'a'*64]+[f'parent:{"a"*64}:1:2:trajectory-{i}' for i in trajectory[1:]])
    if metadata:metadata(document)
    (root/'pairs.json').write_text(json.dumps(document));return root


def test_closed_reader_owns_pairs_and_keeps_parent_only_controls_with_their_children(tmp_path):
    value=read(block(tmp_path/'valid'))
    assert value.arrays['theory_intensity'].shape==(17,101)
    assert value.manifest['candidate']['id']=='child'
    with pytest.raises(ValueError):value.arrays['parent_intensity'].flags.writeable=True


def test_byte_bounded_pairs_do_not_drop_or_limit_weak_high_dimensional_coordinates(tmp_path):
    def change(a):a['parameters']=np.zeros((17,4098))
    def metadata(d):d['mode_dimensions']=4096
    value=read(block(tmp_path/'many-modes',change=change,metadata=metadata))
    assert value.arrays['parameters'].shape==(17,4098)


@pytest.mark.parametrize('case',['roles','changed_reference','scale','nonpositive_cell','missing_array','nonfinite','parameters','labels'])
def test_bound_bytes_do_not_override_invalid_pair_semantics(tmp_path,case):
    def change(a):
        if case=='roles':a['role'][2]=1
        elif case=='changed_reference':a['parent_a'][2,0]*=1.01
        elif case=='scale':a['theory_intensity'][1]=a['background']+2*(a['parent_intensity'][1]-a['background'])
        elif case=='nonpositive_cell':a['parent_a'][0,0]=-1
        elif case=='missing_array':del a['parent_background']
        elif case=='nonfinite':a['theory_intensity'][2,10]=np.nan
        elif case=='parameters':a['parameters']=np.zeros((17,7))
        else:a['geometry_groups'][2,0]=999
    with pytest.raises(ValueError):read(block(tmp_path/case,change=change))


@pytest.mark.parametrize('case',['checksum','groups','slice','observations','child'])
def test_wrong_input_binding_or_physical_groups_are_rejected(tmp_path,case):
    def change(d):
        if case=='checksum':d['pairs_sha256']='0'*64
        elif case=='groups':d['groups'][2]='independent-child'
        elif case=='slice':d['trajectory_slice']=[0,4]
        elif case=='observations':d['observations_included']=True
        else:d['candidate']['child']['sha256']='f'*64
    with pytest.raises(ValueError):read(block(tmp_path/case,metadata=change))
