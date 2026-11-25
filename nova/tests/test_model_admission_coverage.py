"""Admission control-flow tests; native physics is covered by the native suite.

Only the native readers are replaced here. Real Q projection, file identities,
complete child iteration, numerical k checks and receipt validation are used.
"""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from radar_pd_nova import model_admission as admission


def features(kind='full'):
    q=np.linspace(1.,3.,201)
    valid=np.ones(q.shape,dtype=bool)
    if kind=='edge':q[0]=1.001
    elif kind=='gap':valid[90:111]=False
    return dict(q=q,intensity=2.+np.exp(-20.*(q-2.)**2),background=np.zeros_like(q),valid=valid)


@pytest.fixture
def native_bridge(tmp_path,monkeypatch):
    from gpx_recipe.preparation import reference,structural_domain,structural_ranker,structural_ranking_hints,structural_ranking_native
    from gpx_recipe import snapshot
    from gpx_recipe.symmetry import native_audit
    paths={}
    for role in admission.PATH_FIELDS:
        path=tmp_path/role;path.write_text('captured '+role);paths[role]=str(path)
    children=[];candidates=[]
    for index in range(6):
        path=tmp_path/f'child-{index}.cif';path.write_text(f'exact child {index}')
        children.append(str(path))
        candidates.append(dict(id=f'candidate-{index}',child=dict(sha256=admission.file_hash(path)),modes=[f'mode-{index}']))
    metadata={key+'_sha256':admission.file_hash(paths[key]) for key in ('parent','instrument','weights')}
    payload=dict(paths,children=children,phase='parent',histogram='PWDR native',k=['0','0','0'],
        grid=dict(kind='PNT',lower=3000.,upper=40000.,bins=701,bank=1),library_id='a'*24,
        destination=str(tmp_path/'admitted'))
    state=SimpleNamespace(current=features(),reference=features(),failure=None,imported=[],labels={},mutate=False)
    hist=SimpleNamespace(name=payload['histogram'])
    phase=SimpleNamespace(name=payload['phase'],data={'Histograms':{hist.name:{'Use':True}}})
    class Project:
        def __init__(self,**kwargs):pass
        def phase(self,name):return phase
        def phases(self):return [phase]
        def histogram(self,name):return hist
        def histograms(self):return [hist]
    def compatible(*args,**kwargs):
        if state.failure:raise ValueError(state.failure)
        return .0001
    def unconstrained(project):
        if state.failure=='unsupported constraints':raise ValueError(state.failure)
    def observed(source,histogram):
        return state.reference if source==payload.get('reference') else state.current
    def add(project,path,**kwargs):
        index=children.index(path);state.imported.append(index)
        if state.mutate:Path(payload['source']).write_text('changed during admission')
        return (SimpleNamespace(data={'ISODISTORT':{'G2ModeList':[SimpleNamespace(name=f'mode-{index}')]}}),
            dict(source_mode_labels=[state.labels.get(index,'Parent[0,0,0]mode')]))
    monkeypatch.setattr(structural_ranking_hints,'_metadata',lambda path:(metadata,candidates))
    monkeypatch.setattr(structural_domain,'maximum_parent_metric',lambda metadata:.01)
    monkeypatch.setattr(structural_ranker,'Ranker',lambda *a,**k:SimpleNamespace(edges=np.linspace(1.,3.,65)))
    monkeypatch.setattr(reference,'create_reference',lambda *a,**k:dict(path='generated-reference',parent_phase=phase.name,histogram=hist.name))
    monkeypatch.setattr(snapshot,'gsas',lambda:SimpleNamespace(G2Project=Project))
    monkeypatch.setattr(native_audit,'inspect_source',lambda *a,**k:dict(phases=[dict(setting_sha256='b'*64,
        operations=[dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation=[0,0,0])])]))
    monkeypatch.setattr(structural_ranking_native,'require_compatible',compatible)
    monkeypatch.setattr(structural_ranking_native,'require_unconstrained_project',unconstrained)
    monkeypatch.setattr(structural_ranking_native,'observed_features',observed)
    monkeypatch.setitem(sys.modules,'gsas_isodistort_import',SimpleNamespace(add_isodistort_phase=add))
    return payload,state,candidates


@pytest.mark.parametrize('kind',['full','edge','gap'])
def test_coverage_does_not_replace_complete_native_candidate_checks(native_bridge,kind):
    payload,state,candidates=native_bridge;state.current=features(kind)
    result=admission.execute(payload);admission.validate_receipt(result,payload)
    assert result['compatible'] and result['no_refinement'] and result['inputs_unchanged']
    assert state.imported==list(range(6))
    assert result['candidate_ids']==result['eligible_candidate_ids']==[c['id'] for c in candidates]
    assert len(result['candidate_checks'])==6 and result['k_verified'] and result['mode_inventory_verified']
    applicability=result['ranking_applicability'];window=applicability['sources'][0]
    assert window['source_sha256']==admission.file_hash(payload['source'])
    assert applicability['status']==('coverage_passed' if kind=='full' else 'unavailable')
    assert window['coverage_complete'] is (kind=='full')
    assert applicability['minimum_required_coverage']==.999
    assert applicability['ranking_evaluated'] is applicability['candidate_pruning_authorized'] is False
    if kind!='full':assert window['minimum_bin_coverage']<.999 and window['incomplete_bins']>0


def test_reference_coverage_is_independent_and_bound_to_its_exact_source(native_bridge,tmp_path):
    payload,state,_=native_bridge
    before=tmp_path/'before.gpx';before.write_text('before-transition scan');payload['reference']=str(before)
    state.reference=features('gap')
    result=admission.execute(payload);admission.validate_receipt(result,payload)
    assert len(state.imported)==6 and result['compatible']
    current,reference=result['ranking_applicability']['sources']
    assert current['coverage_complete'] and not reference['coverage_complete']
    assert reference['source_role']=='reference' and reference['source_sha256']==admission.file_hash(before)
    assert result['ranking_applicability']['status']=='unavailable'


@pytest.mark.parametrize('failure',['instrument profile differs','nuclear chemistry differs','cell outside prepared domain','unsupported constraints'])
def test_missing_ml_coverage_cannot_bypass_native_context_guards(native_bridge,failure):
    payload,state,_=native_bridge;state.current=features('edge');state.failure=failure
    result=admission.execute(payload)
    assert not result['compatible'] and failure in result['reason'] and not state.imported
    with pytest.raises(ValueError,match='incompatible'):admission.validate_receipt(result,payload)


@pytest.mark.parametrize('role',['child','weights','instrument'])
def test_missing_ml_coverage_cannot_bypass_exact_input_integrity(native_bridge,role):
    payload,state,_=native_bridge;state.current=features('gap')
    Path(payload['children'][0] if role=='child' else payload[role]).write_text('corrupted')
    result=admission.execute(payload)
    assert not result['compatible'] and 'differs' in result['reason']
    assert not state.imported


def test_missing_ml_coverage_still_checks_out_of_scope_children(native_bridge):
    payload,state,candidates=native_bridge;state.current=features('edge')
    state.labels[0]='Parent[0,0,1/2]mode'
    result=admission.execute(payload);admission.validate_receipt(result,payload)
    assert state.imported==list(range(6)) and len(result['candidate_checks'])==6
    assert result['excluded_candidate_ids']==[candidates[0]['id']]
    assert result['eligible_candidate_ids']==[c['id'] for c in candidates[1:]]
    assert result['ranking_applicability']['status']=='unavailable'


def test_mutated_source_never_becomes_an_accepted_reuse_receipt(native_bridge):
    payload,state,_=native_bridge;state.current=features('edge');state.mutate=True
    result=admission.execute(payload)
    assert not result['compatible'] and not result['inputs_unchanged']
    assert 'changed during native verification' in result['reason']


@pytest.mark.parametrize('damage',['missing','foreign-source','false-pass','pruning','counts','nonfinite'])
def test_receipt_cannot_overstate_or_detach_ranking_coverage(native_bridge,damage):
    payload,state,_=native_bridge;state.current=features('gap')
    result=admission.execute(payload)
    value=result['ranking_applicability'];window=value['sources'][0]
    if damage=='missing':result.pop('ranking_applicability')
    elif damage=='foreign-source':window['source_sha256']='f'*64
    elif damage=='false-pass':value['status']='coverage_passed'
    elif damage=='pruning':value['candidate_pruning_authorized']=True
    elif damage=='counts':window['incomplete_bins']=0
    else:window['minimum_bin_coverage']=float('nan')
    with pytest.raises(ValueError,match='ranking'):admission.validate_receipt(result,payload)
