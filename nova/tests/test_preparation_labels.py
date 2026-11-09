"""Structural children of the same irrep remain distinguishable and searchable."""
import json
from types import SimpleNamespace

import pytest

from radar_pd_nova.preparation_labels import entry_label,scope_labels,native_dimensions,training_description,training_validation


def test_children_of_one_irrep_show_their_declared_subgroup_and_k():
    one=dict(kind='structural',settings=dict(candidate=dict(id='one',irrep='GM3+',modes=['a']*16)))
    two=dict(kind='structural',settings=dict(candidate=dict(id='two',irrep='GM3+',modes=['a']*27)))
    labels={('structural','one'):[dict(k=['0','0','0'],label='P1 (a,0) 12 C2/m, basis={(1,2,0),(-1,0,0),(-1/3,-2/3,1/3)}')],
            ('structural','two'):[dict(k=['0','0','0'],label='C1 (a,b) 2 P-1, basis={(1,0,0),(1,1,0),(2/3,1/3,1/3)}')]}
    left=entry_label(one,labels);right=entry_label(two,labels)
    assert left['model']=='C2/m · GM3+' and right['model']=='P-1 · GM3+'
    assert left['k']==right['k']=='0, 0, 0'
    assert left['coordinates']=='16 displacement modes in discovery; native count pending'
    assert right['coordinates']=='27 displacement modes in discovery; native count pending'
    assert left['candidate']!=right['candidate']


def test_training_pattern_shards_keep_the_model_scope_visible():
    model=dict(kind='structural-examples',settings=dict(candidate=dict(id='one',irrep='GM3+',modes=['a']),
        trajectory_slice=[8,16],trajectories=64))
    labels={('structural','one'):[dict(k=['0','0','0'],label='P1 (a,0) 12 C2/m, basis={(1,2,0),(-1,0,0),(-1/3,-2/3,1/3)}')]}
    result=entry_label(model,labels)
    assert result['model']=='C2/m · GM3+ · training patterns'
    assert result['k']=='0, 0, 0'
    assert result['coordinates'].startswith('Trajectories 9–16 of 64')


def test_legacy_and_ambiguous_scope_never_invent_a_propagation_vector():
    model=dict(kind='structural',settings=dict(candidate=dict(id='one',irrep='GM3+',modes=[])))
    assert entry_label(model,{})['k']==''
    labels={('structural','one'):[dict(k=['0','0','0'],label='one'),dict(k=['0','0','1/2'],label='two')]}
    assert entry_label(model,labels)['k']==''


def test_magnetic_scope_uses_its_explicit_vector_and_preserves_bns_label():
    model=dict(kind='magnetic',settings=dict(k=['0','0','1/2'],candidate=dict(id='same',magnetic_space_group='P_S1')))
    labels={('magnetic','same'):[dict(k=['0','0','0'],label='other'),dict(k=['0','0','1/2'],label='chosen')]}
    assert entry_label(model,labels)['model']=='P_S1'
    assert entry_label(model,labels)['k']=='0, 0, 1/2'


@pytest.mark.parametrize('modes',[None,[], 'unavailable'])
def test_missing_discovery_count_never_claims_zero_native_coordinates(modes):
    model=dict(kind='structural',settings=dict(candidate=dict(id='one',modes=modes)))
    assert entry_label(model,{})['coordinates']=='Native coordinate count unavailable'
    assert entry_label(model,{},dimensions=(2,3))['coordinates']==(
        '2 native displacement coordinates · 3 lattice coordinates')


def test_completed_dimensions_use_only_checked_matching_entry_metadata(tmp_path):
    model=dict(kind='structural',settings=dict(candidate=dict(id='one',modes=[])),grid={'bank':1})
    name='entries/one/attempt-1/entry.json';path=tmp_path/name;path.parent.mkdir(parents=True)
    document=dict(schema='radar-experiment-preparation-entry/v1',kind='structural',
        settings=model['settings'],grid=model['grid'],detail=dict(mode_dimensions=2,cell_dimensions=3))
    path.write_text(json.dumps(document));seen=[]
    def verify(names):seen.extend(names);return {name:path}
    library=SimpleNamespace(root=tmp_path,verify_assets=verify)
    entry=dict(state='prepared',model=model,files=[dict(path=name)])
    assert native_dimensions(library,entry)==(2,3)
    assert seen==[name]
    assert native_dimensions(library,dict(entry,state='pending')) is None
    document['settings']={};path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match='differs from this model'):native_dimensions(library,entry)


@pytest.mark.parametrize('counts',[(True,3),(-1,3),(2,None),(2,'3')])
def test_untyped_completed_counts_are_unavailable(tmp_path,counts):
    model=dict(kind='structural',settings={},grid={});name='entries/one/entry.json'
    path=tmp_path/name;path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(schema='radar-experiment-preparation-entry/v1',kind='structural',
        settings={},grid={},detail=dict(mode_dimensions=counts[0],cell_dimensions=counts[1]))))
    library=SimpleNamespace(root=tmp_path,verify_assets=lambda names:{name:path})
    assert native_dimensions(library,dict(state='prepared',model=model,files=[dict(path=name)])) is None


def test_completed_training_shows_clean_splits_without_calling_simulated_rows_independent_pairs(tmp_path):
    from gpx_recipe.preparation.structural_training_counts import describe
    dataset=describe(['a','a','b','c'],[0,0,1,2],[True,False,True,True],generated_pairs=5,simulated_observations=2)
    model=dict(kind='structural-training',settings={},grid={});name='entries/train/entry.json'
    path=tmp_path/name;path.parent.mkdir(parents=True)
    document=dict(schema='radar-experiment-preparation-entry/v1',kind='structural-training',
        settings={},grid={},detail=dict(dataset=dataset))
    path.write_text(json.dumps(document));seen=[]
    def verify(names):seen.extend(names);return {name:path}
    library=SimpleNamespace(root=tmp_path,verify_assets=verify)
    entry=dict(state='prepared',model=model,files=[dict(path=name)])
    result=training_description(library,entry)
    assert '3 unique clean pairs' in result and 'train / validation / test: 1 / 1 / 1' in result
    assert '1 additional simulated rows' in result and '3 separate trajectory groups' in result
    assert seen==[name]
    assert training_description(library,dict(entry,state='pending')) is None
    document['detail']={};path.write_text(json.dumps(document))
    assert training_description(library,entry) is None
    document['detail']={'dataset':dict(dataset,unique_clean_pairs=30)};path.write_text(json.dumps(document))
    with pytest.raises(ValueError):training_description(library,entry)


def test_completed_training_exposes_thin_family_coverage_alongside_total_pairs(tmp_path):
    from gpx_recipe.preparation.structural_training_counts import describe
    dataset=describe(['a','a','a','b','c'],[0,0,0,1,2],[True,True,False,True,True],
        generated_pairs=4,simulated_observations=1,classes=[6,8],
        compatible_labels=[[True,True],[True,False],[False,True],[True,True],[True,True]])
    name='entries/train/entry.json';path=tmp_path/name;path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(schema='radar-experiment-preparation-entry/v1',
        kind='structural-training',settings={},grid={},detail=dict(dataset=dataset))))
    library=SimpleNamespace(root=tmp_path,verify_assets=lambda names:{name:path})
    entry=dict(state='prepared',model=dict(kind='structural-training',settings={},grid={}),files=[dict(path=name)])
    text=training_description(library,entry)
    assert '4 unique clean pairs' in text
    assert '2 symmetry families; least-covered SG 8: 1 clean training pair across 1 trajectory' in text
    assert 'independent experiments' not in text


def validation_entry(tmp_path, partitions):
    name='entries/train/entry.json';path=tmp_path/name;path.parent.mkdir(parents=True)
    document=dict(schema='radar-experiment-preparation-entry/v1',kind='structural-training',
        settings={},grid={},detail=dict(test=dict(observation_metrics=dict(partitions=partitions))))
    path.write_text(json.dumps(document))
    seen=[]
    def verify(names):seen.extend(names);return {name:path}
    library=SimpleNamespace(root=tmp_path,verify_assets=verify)
    entry=dict(state='prepared',model=dict(kind='structural-training',settings={},grid={}),files=[dict(path=name)])
    return library,entry,seen


def test_validation_keeps_high_clean_recall_separate_from_poor_perturbed_recall(tmp_path):
    library,entry,seen=validation_entry(tmp_path,dict(
        clean_native=dict(rows=100,top5=.97),simulated_observation=dict(rows=900,top5=.35)))
    rows=training_validation(library,entry)
    assert [(r['value'],r['count']) for r in rows]==[
        ('97.0%','100 held-out patterns'),('35.0%','900 held-out patterns')]
    assert [r['title'] for r in rows]==['Clean native simulations','Simulated observation variants']
    assert seen==['entries/train/entry.json']
    assert training_validation(library,dict(entry,state='preparing'))==[]


def test_zero_observation_partition_is_not_reported_as_zero_accuracy(tmp_path):
    library,entry,_=validation_entry(tmp_path,dict(
        clean_native=dict(rows=20,top5=.9),simulated_observation=dict(rows=0,top5=None)))
    assert len(training_validation(library,entry))==1


def test_common_classes_cannot_hide_weak_class_recovery_in_validation(tmp_path):
    library,entry,_=validation_entry(tmp_path,dict(clean_native=dict(
        rows=1000,top5=.98,macro_top5=.51,worst_class_top5=.02)))
    row=training_validation(library,entry)[0]
    assert row['value']=='98.0%'
    assert row['detail']=='Class-balanced compatible hit: 51.0% · Lowest class compatible hit: 2.0%'


def test_ambiguous_pattern_hit_cannot_hide_loss_of_a_compatible_label(tmp_path):
    import numpy as np
    from gpx_recipe.preparation.structural_label_retention import LabelRetention
    meter=LabelRetention([1,2,3,4,5,6],5)
    meter.add(np.array([[True,False,False,False,False,True]]),np.array([[0,1,2,3,4,5]]))
    library,entry,_=validation_entry(tmp_path,dict(clean_native=dict(
        rows=1,top5=1.,macro_top5=1.,worst_class_top5=1.,
        label_retention=dict(top5=meter.finish()))))
    row=training_validation(library,entry)[0]
    assert row['value']=='100.0%'
    assert row['detail']=='Each compatible label: 50.0% average · Lowest label: 0.0% · All labels retained: 0.0% of patterns'
    path=tmp_path/'entries/train/entry.json'
    document=json.loads(path.read_text())
    document['detail']['test']['observation_metrics']['partitions']['clean_native']['top5']=.5
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match='same top-five counts'):
        training_validation(library,entry)


@pytest.mark.parametrize('bad',[True,float('nan'),-.1,1.1])
def test_invalid_class_balanced_diagnostics_cannot_be_shown_as_accuracy(tmp_path,bad):
    library,entry,_=validation_entry(tmp_path,dict(clean_native=dict(rows=20,top5=.9,macro_top5=bad)))
    with pytest.raises(ValueError,match='Per-class'):training_validation(library,entry)


@pytest.mark.parametrize('row',[dict(rows=True,top5=.9),dict(rows=-1,top5=.9),
    dict(rows=2,top5=True),dict(rows=2,top5=1.1),dict(rows=2,top5=float('nan')),
    dict(rows=2,top5=None)])
def test_invalid_validation_values_cannot_be_displayed_as_accuracy(tmp_path,row):
    library,entry,_=validation_entry(tmp_path,dict(clean_native=row))
    with pytest.raises(ValueError):training_validation(library,entry)


def test_scope_labels_verify_the_captured_metadata_and_only_include_selected_models(tmp_path):
    path=tmp_path/'scope.json';document=dict(schema='radar-preparation-declared-scope/v1',searches=[
        dict(kind='structural',k=['0','0','0'],selected_candidates=['chosen'],coverage=dict(candidates=[
            dict(id='chosen',label='A'),dict(id='deferred',label='B')]))])
    path.write_text(json.dumps(document));seen=[]
    def verified(names):seen.extend(names);return {'scope.json':path}
    library=SimpleNamespace(root=tmp_path,document=dict(specification=dict(scope='scope.json')),verify_assets=verified)
    assert scope_labels(library)=={('structural','chosen'):[dict(k=['0','0','0'],label='A')]}
    assert seen==['scope.json']
    path.write_bytes(b' '* (4*1024*1024+1))
    with pytest.raises(ValueError,match='metadata budget'):scope_labels(library)


@pytest.mark.parametrize('searches',[None,17,[None],[dict(kind='structural',k='000')],
    [dict(kind='structural',k=['0','0','0'],selected_candidates=[{}],coverage={})]])
def test_malformed_optional_display_metadata_cannot_invent_model_labels(tmp_path,searches):
    path=tmp_path/'scope.json';path.write_text(json.dumps(dict(schema='radar-preparation-declared-scope/v1',searches=searches)))
    library=SimpleNamespace(root=tmp_path,document=dict(specification=dict(scope='scope.json')),
        verify_assets=lambda names:{'scope.json':path})
    assert scope_labels(library)=={}
