"""Structural children of the same irrep remain distinguishable and searchable."""
import json
from types import SimpleNamespace

import pytest

from radar_pd_nova.preparation_labels import entry_label,scope_labels,native_dimensions


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
