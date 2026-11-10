"""Numerical transport and inference controls; these random weights are not science."""
import hashlib
import json

import numpy as np
import pytest

torch=pytest.importorskip('torch')
from gpx_recipe.preparation.structural_attention import network
from gpx_recipe.preparation import structural_attention_weights as w
from gpx_recipe.preparation.structural_features import three_curve_features
from gpx_recipe.preparation.structural_ranker import Ranker
from gpx_recipe.preparation.structural_support import STREAM_METHOD
from gpx_recipe.live.array_limits import validate_npz_bytes


def artifact(root,*,architecture=None):
    root.mkdir();torch.set_num_threads(1);torch.manual_seed(65173)
    architecture=w.ARCHITECTURE if architecture is None else architecture
    model=network(2,architecture=architecture);model.eval();plane,layout=w.pack(model)
    q=np.linspace(.5,4.,301);edges=np.linspace(.5,4.,129);bg=np.full(len(q),3.)
    parent=bg+np.exp(-((q-1.2)/.08)**2);current=parent+.15*np.exp(-((q-2.3)/.06)**2)
    feature=three_curve_features(q,current,parent,edges,background=bg,reference_background=bg)['features']
    reference=three_curve_features(q,parent,parent,edges,background=bg,reference_background=bg)['features']
    arrays=dict(parameters=plane,q_edges=edges,normalization=np.r_[np.zeros(3),np.ones(3),1e-5],
        classes=np.array([12,166],np.int32),reference_features=reference,
        support_prototypes=feature.reshape(1,-1))
    np.savez_compressed(root/'weights.npz',**arrays)
    candidates=[dict(id=key,transformation_matrix=np.eye(3).tolist(),origin_shift=[0,0,0],
        child=dict(path=key+'.cif',sha256='c'*64)) for key in ['child','parent']]
    metadata=dict(schema=w.SCHEMA,architecture=architecture,representation=w.REPRESENTATION,
        channels=3,reference_kind=w.REFERENCE_KIND,normalization_method=w.NORMALIZATION,
        weights_sha256=hashlib.sha256((root/'weights.npz').read_bytes()).hexdigest(),
        classes=[12,166],bins=128,candidates=candidates,candidate_groups=dict(child=12,parent=166),
        class_candidate_ids={'12':['child'],'166':['parent']},
        parent_sha256='a'*64,instrument_sha256='b'*64,background_required=True,
        support_method=STREAM_METHOD,reference_feature_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
        tensor_layout_sha256=layout,screening_qualified=True)
    (root/'model.json').write_text(json.dumps(metadata))
    request=dict(background=bg,parent_sha256='a'*64,instrument_sha256='b'*64,
        reference=dict(q=q,intensity=parent,background=bg))
    return model,q,current,request,arrays,metadata


@pytest.mark.parametrize('architecture',sorted(w.CLASS_ARCHITECTURES))
def test_attention_roundtrip_keeps_exact_logits_with_six_data_arrays(tmp_path,architecture):
    model,q,current,request,arrays,d=artifact(tmp_path/'model',architecture=architecture)
    raw=(tmp_path/'model/weights.npz').read_bytes();validate_npz_bytes(raw,maximum_bytes=32*1024**2)
    assert len(arrays)==6 and len(model.state_dict())>16
    loaded=Ranker(tmp_path/'model')
    feature=arrays['support_prototypes'].reshape(1,3,128)
    with torch.no_grad():
        torch.testing.assert_close(model(torch.from_numpy(feature)),loaded.model(torch.from_numpy(feature)),rtol=0,atol=0)
    first=loaded.rank(q,current,**request);second=Ranker(tmp_path/'model').rank(q,current,**request)
    assert first==second and first['ranking_available']
    assert json.loads(json.dumps(first))==first
    assert set(first['candidate_order'])=={'child','parent'}
    assert first['candidate_pruning_authorized'] is first['screening_qualified'] is False
    assert first['complete_native_search_required'] is True
    assert first['support']['statistical_coverage_qualified'] is False
    outside=loaded.rank(q,current+np.sin(q*37),**request)
    assert not outside['ranking_available'] and outside['candidate_order']==['child','parent']
    assert json.loads(json.dumps(outside))==outside


def test_a_dead_legacy_readout_never_advertises_pattern_based_ranking(tmp_path):
    model,q,current,request,arrays,d=artifact(tmp_path/'model',architecture=w.LEGACY_ARCHITECTURE)
    with torch.no_grad():model.head[1].weight.zero_();model.head[1].bias.fill_(-100.)
    arrays['parameters'],d['tensor_layout_sha256']=w.pack(model)
    root=tmp_path/'model';np.savez_compressed(root/'weights.npz',**arrays)
    d['weights_sha256']=hashlib.sha256((root/'weights.npz').read_bytes()).hexdigest()
    (root/'model.json').write_text(json.dumps(d))
    result=Ranker(root).rank(q,current,**request)
    assert not result['ranking_available'] and 'inactive readout' in result['reason']
    assert result['candidate_order']==['child','parent'] and result['complete_native_search_required']
    assert all(row['model_score'] is None for row in result['families'])
    assert result['support']['status']=='within_empirical_synthetic_support'


def test_an_architecture_label_cannot_reinterpret_older_numeric_weights(tmp_path):
    _,_,_,_,_,d=artifact(tmp_path/'model',architecture=w.LEGACY_ARCHITECTURE)
    d['architecture']=w.ARCHITECTURE
    (tmp_path/'model/model.json').write_text(json.dumps(d))
    with pytest.raises(ValueError,match='installed architecture'):Ranker(tmp_path/'model')


def test_a_leading_parent_family_does_not_swallow_the_entire_child_search_order(tmp_path):
    _,q,current,request,_,d=artifact(tmp_path/'model')
    extra=dict(d['candidates'][0],id='other',child=dict(path='other.cif',sha256='d'*64))
    d['candidates'].append(extra);d['candidate_groups']['other']=1
    d['class_candidate_ids']={'12':['child','other'],'166':['other','child','parent']}
    (tmp_path/'model/model.json').write_text(json.dumps(d));loaded=Ranker(tmp_path/'model')
    with torch.no_grad():
        loaded.model.head[-1].weight.zero_();loaded.model.head[-1].bias.copy_(torch.tensor([0.,5.]))
    result=loaded.rank(q,current,**request)
    assert result['ranking_available']
    assert [(row['space_group'],row['candidate_ids']) for row in result['families']]==[(166,['parent']),(12,['child'])]
    assert result['candidate_order']==['parent','child','other']
    assert result['candidate_pruning_authorized'] is False and result['complete_native_search_required']


@pytest.mark.parametrize('case',['missing','q_grid','reference_mask','mask','foreign_context'])
def test_inference_requires_the_exact_native_parent_and_covered_range(tmp_path,case):
    _,q,current,request,_,_=artifact(tmp_path/'model');loaded=Ranker(tmp_path/'model')
    if case=='missing':request['reference']=None
    elif case=='q_grid':request['reference']['q']=q+1e-8
    elif case=='reference_mask':request['reference']['valid']=np.ones(len(q))
    elif case=='mask':request['valid']=np.ones(len(q),bool);request['valid'][100:120]=False
    else:request['parent_sha256']='f'*64
    with pytest.raises(ValueError):loaded.rank(q,current,**request)


@pytest.mark.parametrize('case',['plane_length','plane_dtype','plane_nan','layout','normalization','support','reference','extra_array'])
def test_hash_valid_corrupt_numerical_models_cannot_reach_inference(tmp_path,case):
    _,_,_,_,arrays,d=artifact(tmp_path/'model');root=tmp_path/'model'
    if case=='plane_length':arrays['parameters']=arrays['parameters'][:-1]
    elif case=='plane_dtype':arrays['parameters']=arrays['parameters'].astype(np.float64)
    elif case=='plane_nan':arrays['parameters'][10]=np.nan
    elif case=='layout':d['tensor_layout_sha256']='f'*64
    elif case=='normalization':arrays['normalization'][3]=1e100
    elif case=='support':arrays['support_prototypes']=np.zeros((1,8*128),np.float32)
    elif case=='reference':arrays['reference_features']=np.zeros((8,128),np.float32)
    else:arrays['python_program']=np.zeros(1)
    np.savez_compressed(root/'weights.npz',**arrays)
    d['weights_sha256']=hashlib.sha256((root/'weights.npz').read_bytes()).hexdigest()
    (root/'model.json').write_text(json.dumps(d))
    with pytest.raises(ValueError):Ranker(root)


def test_portable_model_dispatch_keeps_asset_names_confined(tmp_path):
    artifact(tmp_path/'model')
    with pytest.raises(ValueError,match='captured directory'):
        Ranker(tmp_path/'model',metadata_name='../model/model.json')
    (tmp_path/'bad.json').write_text('[]')
    with pytest.raises(ValueError,match='metadata object'):Ranker(tmp_path,metadata_name='bad.json')


@pytest.mark.parametrize('changed',['examples','original_examples','independent_groups','partition_rows'])
def test_optional_dataset_counts_must_match_the_published_model_totals(tmp_path,changed):
    from gpx_recipe.preparation.structural_training_counts import describe
    _,_,_,_,_,d=artifact(tmp_path/'model')
    d.update(dataset=describe(['a','b','c'],[0,1,2],[True,True,True],
        generated_pairs=3,simulated_observations=0),examples=3,original_examples=3,
        independent_groups=3,partition_rows=[1,1,1])
    path=tmp_path/'model/model.json';path.write_text(json.dumps(d))
    assert Ranker(tmp_path/'model').metadata['dataset']['unique_clean_pairs']==3
    d[changed]=[2,1,1] if changed=='partition_rows' else 4
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='training-count receipt'):Ranker(tmp_path/'model')


@pytest.mark.parametrize('changed',[None,'class','membership','missing','bool'])
def test_optional_family_counts_are_bound_to_exact_native_class_memberships(tmp_path,changed):
    from gpx_recipe.preparation.structural_training_counts import describe
    _,_,_,_,_,d=artifact(tmp_path/'model')
    dataset=describe(['a','b','c'],[0,1,2],[True]*3,generated_pairs=3,
        simulated_observations=0,classes=[12,166],compatible_labels=[[True,True]]*3)
    d.update(dataset=dataset,examples=3,original_examples=3,independent_groups=3,
        partition_rows=[1,1,1],counts={p:[1,1] for p in ('train','validation','test')})
    if changed=='class':dataset['family_coverage']['classes']=[12,167]
    elif changed=='membership':d['counts']['train']=[1,0]
    elif changed=='missing':del d['counts']
    elif changed=='bool':d['counts']['train']=[True,1]
    path=tmp_path/'model/model.json';path.write_text(json.dumps(d))
    if changed is None:assert Ranker(tmp_path/'model').metadata['dataset']==dataset
    else:
        with pytest.raises(ValueError):Ranker(tmp_path/'model')
