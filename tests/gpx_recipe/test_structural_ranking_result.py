"""Ranking changes scheduling only; source, scope and native work stay bound."""
import copy

import pytest

from gpx_recipe.preparation.structural_ranking_result import unavailable,validate,prioritize


def protocol():
    return dict(kind='structural',source=dict(sha256='a'*64),implementation_sha256='b'*64,
                candidates=[dict(id='first'),dict(id='second'),dict(id='third')])


def ranked(p):
    value=unavailable(p,'Advisory priority only',snapshot='c'*64)
    value.update(ranking_available=True,candidate_order=['second','first','third'],entry_id='d'*24,bank='2',
        checks=[dict(entry='d'*24,status='compatible',reason='Native context matched')],
        support=dict(status='within_empirical_synthetic_support',nearest_training_rms=.1,
            validation_maximum_rms=.2,statistical_coverage_qualified=False),
        families=[dict(space_group=12,model_score=.4,candidate_ids=['second','first'])])
    return value


def test_priority_is_stable_and_retains_every_candidate_and_start():
    p=protocol();r=ranked(p);validate(r,p,snapshot='c'*64)
    rows=[dict(candidate=c,seed=s) for s in ('x','y') for c in ('first','second','third')]
    ordered=prioritize(rows,r,p)
    assert [(r['candidate'],r['seed']) for r in ordered]==[
        ('second','x'),('second','y'),('first','x'),('first','y'),('third','x'),('third','y')]
    assert len(ordered)==len(rows) and rows[0]['candidate']=='first'
    assert prioritize(rows,unavailable(p,'Not prepared'),p)==rows


@pytest.mark.parametrize('change',['source','snapshot','drop','prune','qualified','support','foreign','repeat','unchecked','status_type','support_type'])
def test_ranking_cannot_redirect_or_qualify_scientific_work(change):
    p=protocol();r=ranked(p)
    if change=='source':r['source_sha256']='e'*64
    elif change=='snapshot':r['preparation_snapshot_sha256']='e'*64
    elif change=='drop':r['candidate_order'].pop()
    elif change=='prune':r['candidate_pruning_authorized']=True
    elif change=='qualified':r['screening_qualified']=True
    elif change=='support':r['support']['status']='outside_prepared_support'
    elif change=='foreign':r['families'][0]['candidate_ids']=['other']
    elif change=='repeat':r['candidate_order']=['second','first','first']
    elif change=='unchecked':r['checks']=[]
    elif change=='status_type':r['checks'][0]['status']=[]
    else:r['support']['status']=[]
    with pytest.raises(ValueError):validate(r,p,snapshot='c'*64)


def test_receipt_from_a_previous_source_revision_is_not_reused():
    p=protocol();r=ranked(p);later=copy.deepcopy(p);later['source']['sha256']='e'*64
    with pytest.raises(ValueError):prioritize([dict(candidate='first')],r,later)


def test_unavailable_receipt_cannot_sneak_in_an_order_or_score():
    p=protocol();r=unavailable(p,'No supported model');r['candidate_order'].reverse()
    with pytest.raises(ValueError):validate(r,p)


def test_live_ranking_cannot_be_reused_for_a_completed_gpx_or_another_snapshot():
    from gpx_recipe.preparation.structural_ranking_result import SNAPSHOT_SCHEMA
    p=protocol();r=ranked(p)
    observation=dict(snapshot_sha256='f'*64,calculation_source_sha256='e'*64,
        acquisition_sha256='d'*64,revision=2)
    r.update(schema=SNAPSHOT_SCHEMA,observation=observation)
    assert validate(r,p,observation=observation) is r
    with pytest.raises(ValueError):validate(r,p)
    for key in observation:
        changed=dict(observation);changed[key]=3 if key=='revision' else '0'*64
        with pytest.raises(ValueError):validate(r,p,observation=changed)
    rows=[dict(candidate=key) for key in ('first','second','third')]
    assert [row['candidate'] for row in prioritize(rows,r,p,observation=observation)]==['second','first','third']
    assert unavailable(p,'Unavailable',observation=observation)['schema']==SNAPSHOT_SCHEMA


@pytest.mark.parametrize('observation',[{},None,dict(snapshot_sha256='f'*64,
    calculation_source_sha256='e'*64,acquisition_sha256='d'*64,revision=True)])
def test_live_binding_requires_exact_hashes_and_integer_revision(observation):
    from gpx_recipe.preparation.structural_ranking_result import observation_binding
    with pytest.raises(ValueError):observation_binding(observation)


def work_set(classes):
    from gpx_recipe.preparation.structural_adaptive_sets import SCHEMA,ADJACENT_SCORE_RATIO
    return dict(schema=SCHEMA,retained_space_groups=classes[:1],deferred_space_groups=classes[1:],
        calibrated_maximum_gap=.4,calibration_groups=20,error_rate=.05,adjacent_score_ratio=ADJACENT_SCORE_RATIO,
        reason='Contract fixture, not a scientific prediction',model_scores_are_probabilities=False,
        experimental_coverage_qualified=False,candidate_pruning_authorized=False,native_refinement_required=True)


@pytest.mark.parametrize('live',[False,True])
def test_adaptive_evidence_keeps_source_binding_and_all_native_work(live):
    from gpx_recipe.preparation.structural_ranking_result import attach_adaptive,ADAPTIVE_SCHEMA,ADAPTIVE_SNAPSHOT_SCHEMA
    p=protocol();r=ranked(p);observation=None
    if live:
        observation=dict(snapshot_sha256='f'*64,calculation_source_sha256='e'*64,acquisition_sha256='d'*64,revision=3)
        r.update(schema='radar-structural-snapshot-ranking-evidence/v1',observation=observation)
    value=attach_adaptive(r,work_set([12]))
    assert value['schema']==(ADAPTIVE_SNAPSHOT_SCHEMA if live else ADAPTIVE_SCHEMA)
    assert validate(value,p,observation=observation) is value
    assert value['candidate_order']==r['candidate_order'] and len(value['candidate_order'])==3
    changed=copy.deepcopy(value);changed['adaptive_work_set']['retained_space_groups']=[166]
    with pytest.raises(ValueError):validate(changed,p,observation=observation)
    changed=copy.deepcopy(value);changed['schema']=r['schema']
    with pytest.raises(ValueError):validate(changed,p,observation=observation)
    if live:
        with pytest.raises(ValueError):validate(value,p)


def test_unavailable_evidence_cannot_hide_an_adaptive_priority():
    from gpx_recipe.preparation.structural_ranking_result import attach_adaptive
    p=protocol();r=unavailable(p,'Unavailable')
    with pytest.raises(ValueError):attach_adaptive(r,work_set([12]))
