"""Authored arrays exercise sparse lineage; they are not native accuracy data."""
import copy
import hashlib
import json

import numpy as np
import pytest

from gpx_recipe.preparation import structural_supplement_pairs as sparse
from gpx_recipe.preparation import structural_jobs as jobs
from gpx_recipe.preparation.structural_cohort import assemble_pairs
from gpx_recipe.preparation.structural_paired_blocks import read,pack_block,source_paths
from gpx_recipe.preparation.structural_mode_directions import from_basis
from gpx_recipe.symmetry.compiler import file_sha
from test_structural_paired_preparation import inputs
from test_structural_mode_projection import attach


def fixture(root):
    request,source=inputs(root);attach(source,names='examples')
    return request,source


def native_arrays(source,*,parents=(6,1),labels=12):
    block=read(source,names='examples');d=block.manifest;a=block.arrays;m=d['mode_dimensions']
    projection=block.mode_projection;v=projection.arrays
    directions=from_basis(v['physical_to_native'],v['gauge_basis'],v['physical_gram'],projection.metadata['native_mode_labels'])
    rays=[dict(source=0,parent=parent,axis=0,sign=sign,amplitudes=[.02,.05])
          for parent,sign in zip(parents,[-1,1])]
    values={k:[a[k][0].copy()] for k in ('theory_intensity','parent_intensity','parent_background','parent_a','parameters','role','geometry_groups')}
    for ray in rays:
        row=next(i for i,(p,r) in enumerate(zip(a['trajectory'],a['role'])) if p==ray['parent'] and r==1)
        for amplitude in [0.,*ray['amplitudes']]:
            for k in values:values[k].append(a[k][row].copy())
            if amplitude:
                values['role'][-1]=np.int8(2);values['geometry_groups'][-1]=np.full(2,labels,np.int32)
                values['parameters'][-1][:m]=directions.physical[ray['axis']]*ray['sign']*amplitude
                values['theory_intensity'][-1]+=np.exp(-((a['q']-2.65)/.04)**2)*amplitude**2
    return {**{k:np.asarray(v) for k,v in values.items()},'q':a['q'].copy(),'background':a['background'].copy()},rays


def addition(source,destination,**options):
    arrays,rays=native_arrays(source,**options)
    result=sparse.write([source],arrays,rays,destination,direction_basis=sparse.NAMED_BASIS,native_protocol_sha256='d'*64,
        native_curves_sha256='e'*64,native_result_sha256='9'*64,producer_implementation_sha256='f'*64)
    return result,arrays,rays


def rewrite(path,change=None,metadata=None):
    data=read(path,names='examples');a={k:v.copy() for k,v in data.arrays.items()};d=copy.deepcopy(data.manifest)
    if change:change(a)
    if metadata:metadata(d)
    np.savez_compressed(path/'examples.npz',**a);d['pairs_sha256']=file_sha(path/'examples.npz')
    (path/'examples.json').write_text(json.dumps(d))


def test_explicit_sparse_rows_preserve_original_context_ids_every_array_and_fresh_labels(tmp_path):
    _,source=fixture(tmp_path);before={str(p):file_sha(p) for p in source_paths(source,names='examples')}
    added,arrays,rays=addition(source,tmp_path/'added',labels=13)
    assert added.manifest['schema']==sparse.RAW_SCHEMA and 'trajectory_slice' not in added.manifest
    assert added.arrays['trajectory'].tolist()==[-1,6,6,6,1,1,1]
    assert added.manifest['supplement']['rays']==rays and np.all(added.arrays['geometry_groups'][added.arrays['role']==2]==13)
    for k,v in arrays.items():assert added.arrays[k].dtype==v.dtype and added.arrays[k].tobytes()==v.tobytes()
    packed=pack_block(tmp_path/'added',tmp_path/'packed',source_names='examples',names='examples')
    assert packed.manifest['schema']==sparse.PACKED_SCHEMA
    for k,v in added.arrays.items():assert packed.arrays[k].tobytes()==v.tobytes()
    assert before=={str(p):file_sha(p) for p in source_paths(source,names='examples')}


@pytest.mark.parametrize('change',['none','within_ray','whole_ray'])
def test_nonzero_original_cell_coordinates_are_retained_without_fabricated_summaries(tmp_path,change):
    _,source=fixture(tmp_path)
    block=read(source,names='examples');d=copy.deepcopy(block.manifest)
    arrays={k:v.copy() for k,v in block.arrays.items()};m=d['mode_dimensions']
    # Authored transport control: the real native corpus likewise expresses
    # the reference-cell perturbation in every child/control coordinate tail.
    for parent in set(arrays['trajectory'])-{-1}:
        arrays['parameters'][arrays['trajectory']==parent,m:]=(.0002+int(parent)*.00001)
    from gpx_recipe.preparation import structural_mode_projection as projection
    d['schema']=projection.RAW_SCHEMA;d.pop('array_encoding',None)
    np.savez_compressed(source/'examples.npz',**arrays);d['pairs_sha256']=file_sha(source/'examples.npz')
    (source/'examples.json').write_text(json.dumps(d))
    native,rays=native_arrays(source)
    if change=='within_ray':native['parameters'][2,m:]*=1.01
    elif change=='whole_ray':native['parameters'][1:4,m:]*=1.01
    if change!='none':
        with pytest.raises(ValueError,match='cell coordinates'):
            sparse.write([source],native,rays,tmp_path/'added',direction_basis=sparse.NAMED_BASIS,native_protocol_sha256='d'*64,
                native_curves_sha256='e'*64,native_result_sha256='9'*64,producer_implementation_sha256='f'*64)
        return
    result=sparse.write([source],native,rays,tmp_path/'added',direction_basis=sparse.NAMED_BASIS,native_protocol_sha256='d'*64,
        native_curves_sha256='e'*64,native_result_sha256='9'*64,producer_implementation_sha256='f'*64)
    assert not sparse.OMITTED_SUMMARIES.intersection(result.arrays)
    assert np.any(result.arrays['parameters'][1:,m:]!=0)
    for k,v in native.items():assert result.arrays[k].tobytes()==v.tobytes()
    assemble_pairs([read(source,names='examples'),result],bins=128)


@pytest.mark.parametrize('basis',[sparse.NAMED_BASIS,sparse.WHITENED_BASIS])
def test_named_modes_and_mixed_physical_directions_cannot_be_relabelled_as_each_other(tmp_path,basis):
    from gpx_recipe.preparation.structural_mode_projection import Projection,descriptor
    from gpx_recipe.structural_live.optimization.physical_prior_geometry import orthonormal_lift
    _,source=fixture(tmp_path);added,_,_=addition(source,tmp_path/'added')
    metadata=copy.deepcopy(added.mode_projection.metadata)
    metadata.update(native_dimensions=3,physical_dimensions=2,
                    native_mode_labels=['first','second','origin'],native_normalization=[1.,1.,1.])
    gram=np.array([[2.,.5],[.5,1.]])
    coordinates=dict(physical_to_native=np.array([[1.,0.],[0.,1.],[0.,0.]]),
                     gauge_basis=np.array([[0.],[0.],[1.]]),physical_gram=gram)
    projection=Projection(metadata,coordinates,added.mode_projection.stamp)
    d=copy.deepcopy(added.manifest);d['mode_dimensions']=2
    d['supplement']['sources'][0]['projection_identity']=descriptor(projection)['identity']
    d['supplement']['direction_basis']=basis
    arrays={k:v.copy() for k,v in added.arrays.items()};arrays['parameters']=np.zeros((d['examples'],4))
    directions=(from_basis(coordinates['physical_to_native'],coordinates['gauge_basis'],gram,metadata['native_mode_labels']).physical
                if basis==sparse.NAMED_BASIS else orthonormal_lift(gram).T)
    cursor=1
    for ray in d['supplement']['rays']:
        direction=directions[ray['axis']];direction=direction/np.sqrt(direction@gram@direction)
        count=len(ray['amplitudes'])
        arrays['parameters'][cursor+1:cursor+1+count,:2]=np.asarray(ray['amplitudes'])[:,None]*ray['sign']*direction
        cursor+=1+count
    sparse.validate_arrays(d,arrays,projection)
    d['supplement']['direction_basis']=sparse.WHITENED_BASIS if basis==sparse.NAMED_BASIS else sparse.NAMED_BASIS
    with pytest.raises(ValueError,match='physical basis'):sparse.validate_arrays(d,arrays,projection)


@pytest.mark.parametrize('storage',['dense','file_backed','native_pair_replay'])
def test_all_storage_paths_retain_every_original_row_and_fold_with_noncontiguous_additions(tmp_path,storage):
    _,source=fixture(tmp_path);addition(source,tmp_path/'added')
    before=assemble_pairs([read(source,names='examples')],bins=128)
    paths=[tmp_path/'added',source]
    if storage=='dense':after=assemble_pairs([read(p,names='examples') for p in paths],bins=128)
    elif storage=='file_backed':
        from gpx_recipe.preparation.structural_paired_store import build
        after=build(paths,tmp_path/'cache',names='examples',bins=128)
    else:
        from gpx_recipe.preparation.structural_replay import build
        after=build(paths,names='examples',bins=128)
    from gpx_recipe.preparation.structural_training_data import feature_rows
    try:
        actual=feature_rows(after['features']).take(np.arange(len(after['groups']),dtype=np.int64))
        locations={hashlib.sha256(v.tobytes()).digest():i for i,v in enumerate(actual)}
        selected=np.array([locations[hashlib.sha256(v.tobytes()).digest()] for v in before['features']])
        np.testing.assert_array_equal(actual[selected],before['features'])
        np.testing.assert_array_equal(after['split'][selected],before['split'])
        np.testing.assert_array_equal(after['compatible_labels'][selected],before['compatible_labels'])
        np.testing.assert_array_equal(after['reference_features'],before['reference_features'])
        assert after['partition_anchor']['schema']=='radar-owned-original-additive-partition/v1'
        assert len(after['additive_pair_inventory'])==1 and set(after['groups'])==set(before['groups'])
    finally:
        if storage!='dense':after['features'].close()


def test_additive_plan_binds_original_sources_coordinates_and_all_four_files(tmp_path):
    payload,source=fixture(tmp_path);addition(source,tmp_path/'added')
    target=tmp_path/'plan.json';plan=jobs.create_plan(payload['source'],[source,tmp_path/'added'],target)
    assert plan['schema']==jobs.ADDITIVE_PLAN_SCHEMA and plan['parent_strata'] is None
    assert set(plan['additive_sources'])=={'added'} and jobs.read_plan(target)==plan
    assert len(jobs.plan_inputs(target))==8
    for full in (True,False):assert jobs.checked_blocks(target,plan,validate_arrays=full)==[source,tmp_path/'added']
    assert jobs.create_plan(payload['source'],[source,tmp_path/'added'],target,recover_identical=True)==plan
    d=json.loads(target.read_text());d['additive_sources']['added']['sources'][0]['pairs_sha256']='0'*64
    target.write_text(json.dumps(d))
    with pytest.raises(ValueError):jobs.read_plan(target)


@pytest.mark.parametrize('case',['disguised','missing_amplitude','basis','cell_parameter','stats','projection','gauge','duplicate_ray'])
def test_sparse_metadata_cannot_erase_steps_relabel_gauges_or_rotate_native_coordinates(tmp_path,case):
    _,source=fixture(tmp_path);addition(source,tmp_path/'added');path=tmp_path/'added'
    def arrays(a):
        if case=='basis':a['parameters'][2,0]*=1.01
        elif case=='cell_parameter':a['parameters'][2,-1]=.001
        elif case=='stats':a['displacement_rms']=np.zeros(len(a['role']))
    def metadata(d):
        if case=='disguised':d['schema']='radar-structural-ranking-paired-pilot/v3'
        elif case=='missing_amplitude':d['supplement']['rays'][0]['amplitudes'].pop()
        elif case=='projection':d['supplement']['sources'][0]['projection_identity']='0'*64
        elif case=='gauge':d['supplement']['rays'][0]['axis']=1
        elif case=='duplicate_ray':d['supplement']['rays'][1]=copy.deepcopy(d['supplement']['rays'][0])
    rewrite(path,change=arrays,metadata=metadata)
    with pytest.raises(ValueError):read(path,names='examples')


@pytest.mark.parametrize('case',['missing_original','original_identity','parent_cell'])
def test_assembly_requires_exact_original_source_and_parent_context(tmp_path,case):
    _,source=fixture(tmp_path);addition(source,tmp_path/'added');path=tmp_path/'added'
    if case=='original_identity':rewrite(path,metadata=lambda d:d['supplement']['sources'][0].update(manifest_identity='0'*64))
    elif case=='parent_cell':
        def change(a):a['parent_a'][1:4]*=1.001
        rewrite(path,change=change)
    blocks=[read(path,names='examples')]
    if case!='missing_original':blocks.insert(0,read(source,names='examples'))
    with pytest.raises(ValueError):assemble_pairs(blocks,bins=128)


def test_new_labels_without_original_holdout_support_fail_instead_of_reshuffling(tmp_path):
    _,source=fixture(tmp_path);addition(source,tmp_path/'added',labels=13)
    with pytest.raises(ValueError,match='retained original holdouts'):
        assemble_pairs([read(source,names='examples'),read(tmp_path/'added',names='examples')],bins=128)


def test_legacy_training_does_not_allocate_a_new_supplement_context_index(tmp_path,monkeypatch):
    request,source=fixture(tmp_path)
    before=assemble_pairs([read(source,names='examples')],bins=128)
    monkeypatch.setattr(sparse,'MAX_CONTEXT_BYTES',0)
    after=assemble_pairs([read(source,names='examples')],bins=128)
    np.testing.assert_array_equal(before['features'],after['features'])
    np.testing.assert_array_equal(before['split'],after['split'])
    plan_file=tmp_path/'plan.json';plan=jobs.create_plan(request['source'],[source],plan_file)
    assert jobs.checked_blocks(plan_file,plan)==[source]


def test_metadata_preflight_does_not_keep_supplement_arrays_alive(tmp_path):
    import weakref
    _,source=fixture(tmp_path);added,arrays,_=addition(source,tmp_path/'added')
    catalog=sparse.Catalog(arrays=False)
    catalog.add(read(source,names='examples').manifest,name='original')
    ref=weakref.ref(arrays['theory_intensity'])
    catalog.add(added.manifest,name='added',arrays=arrays,projection=added.mode_projection)
    del arrays
    assert ref() is None and set(catalog.finish())=={'added'}


def diagnostic(root,source):
    arrays,rays=native_arrays(source);d=read(source,names='examples').manifest
    ray_records=[];origins=[dict(kind='original_anchor')];groups=['parent:'+d['source_sha256']];cursor=1
    for ray in rays:
        group=f'parent:{d["source_sha256"]}:{d["seed"]}:{d["parent_seed"]}:trajectory-{ray["parent"]}'
        native=dict(axis=ray['axis'],sign=ray['sign'],original_trajectory=ray['parent'],
            group=group,parent_a=arrays['parent_a'][cursor].tolist())
        ray_records.append(native);origins.append(dict(kind='parent_control',**native))
        origins.extend(dict(kind='fresh_native_hypothesis',amplitude=v,**native) for v in ray['amplitudes'])
        groups.extend([group]*(1+len(ray['amplitudes'])));cursor+=1+len(ray['amplitudes'])
    root.mkdir();protocol=root/'protocol.json';native=root/'unit';native.mkdir()
    hashes=[d[k] for k in ('source_sha256','instrument_sha256','child_sha256')]
    document=dict(schema='radar-native-signed-mode-supplement-protocol/v1',direction_basis=sparse.NAMED_BASIS,units=[dict(index=0,rays=ray_records)],
        amplitudes=rays[0]['amplitudes'],input_sha256=hashes,package_sha256='f'*64,
        request=dict(grid=d['grid'],settings=dict(candidate=d['candidate'])))
    protocol.write_text(json.dumps(document));np.savez_compressed(native/'curves.npz',**arrays)
    detail=dict(schema='radar-structural-native-signed-mode-supplement/v1',unit_index=0,package_sha256='f'*64,
        protocol_sha256=file_sha(protocol),input_sha256=hashes,candidate=d['candidate'],mode_dimensions=d['mode_dimensions'],
        cell_dimensions=d['cell_dimensions'],origins=origins,groups=groups,points=len(origins),new_native_hypotheses=4,
        independent_native_checks=5,curves_sha256=file_sha(native/'curves.npz'),observations_included=False,
        scientific_result=False,candidate_pruning_authorized=False,test_curves_scored=0,training_pairs_added=0,
        parent_controls=[dict(row=i,intensity_scale=1.,relative_error=0.) for i,o in enumerate(origins)
                         if o['kind']!='fresh_native_hypothesis'])
    (native/'result.json').write_text(json.dumps(detail));return native,protocol,arrays


def test_complete_native_diagnostic_adapts_without_changing_any_input_point(tmp_path):
    _,source=fixture(tmp_path);native,protocol,arrays=diagnostic(tmp_path/'native',source)
    result=sparse.admit_diagnostic(source,native,protocol,tmp_path/'admitted')
    for k,v in arrays.items():assert result.arrays[k].tobytes()==v.tobytes()
    assert result.manifest['supplement']['native_result_sha256']==file_sha(native/'result.json')
    assert result.manifest['supplement']['native_protocol_sha256']==file_sha(protocol)


@pytest.mark.parametrize('case',['producer','protocol','point','origin','group','control','native_checks','curve_bytes','native_cell','candidate','missing_basis'])
def test_native_adaptation_cannot_shrink_or_reassign_the_sealed_scope(tmp_path,case):
    _,source=fixture(tmp_path);native,protocol,arrays=diagnostic(tmp_path/'native',source)
    result=native/'result.json';d=json.loads(result.read_text())
    if case=='producer':d['package_sha256']='0'*64
    elif case=='protocol':d['protocol_sha256']='0'*64
    elif case=='point':d['points']-=1
    elif case=='origin':d['origins'].pop()
    elif case=='group':d['groups'][2]='invented-independent-point'
    elif case=='control':d['parent_controls'][0]['relative_error']=.01
    elif case=='native_checks':d['independent_native_checks']-=1
    elif case=='curve_bytes':
        with (native/'curves.npz').open('ab') as stream:stream.write(b'changed')
    elif case=='native_cell':
        arrays['parent_a'][2,0]*=1.001;np.savez_compressed(native/'curves.npz',**arrays)
        d['curves_sha256']=file_sha(native/'curves.npz')
    elif case=='candidate':d['candidate']['id']='different'
    else:
        p=json.loads(protocol.read_text());p.pop('direction_basis');protocol.write_text(json.dumps(p))
        d['protocol_sha256']=file_sha(protocol)
    result.write_text(json.dumps(d))
    with pytest.raises(ValueError):sparse.admit_diagnostic(source,native,protocol,tmp_path/'admitted')
