"""Calibration semantics, independent groups and unknown-mode preservation."""
import copy

import numpy as np
import pytest

from gpx_recipe.preparation import structural_joint_modes as joint
from gpx_recipe.preparation import structural_mode_calibration as modes
from gpx_recipe.preparation.structural_mode_families import NORMALIZATION
from gpx_recipe.symmetry.contracts import digest

IDENTITIES = dict(zip(modes.IDENTITIES, ['a'*64, 'b'*64, 'c'*64, 'd'*64]))


def spec():
    families = []
    for label in ['GM1+', 'GM2+']:
        row = dict(parent_sha256='b'*64, parent_convention='test parent convention', k=['0', '0', '0'], irrep=label)
        families.append(dict(id=digest(row), **row))
    return joint.validate(dict(schema=joint.SCHEMA, classes=[12, 166], families=sorted(families, key=lambda f:f['id']),
        normalization=NORMALIZATION, scales=[.01, .02], known_intervals=[[2, 0], [2, 2]],
        targets_identity='e'*64, fitting_split_sha256='f'*64, fitting_alias_groups=2, quantiles=[.1, .9],
        mode_head_trained=True, intervals_calibrated=False, experimental_mode_absence_qualified=False,
        candidate_pruning_authorized=False))


def batch(n, *, c=0, f=0, lower=.25, upper=.5):
    return dict(row=np.arange(n, dtype=np.int64), space_group=np.full(n, c, np.int64),
        family=np.full(n, f, np.int64), lower=np.full(n, lower, np.float64), upper=np.full(n, upper, np.float64))


def calibration(n=19, duplicate=1, *, c=0, f=0):
    groups=[f'cal-{i:02}' for i in range(n)]
    value=modes.Calibration(spec(), identities=IDENTITIES, groups=groups, used_groups=['fit', 'test', 'validation'])
    for _ in range(duplicate):
        value.add(np.tile([.3, .4], (n, 1)), batch(n, c=c, f=f), groups)
    return value.finish()


def test_independent_calibration_expands_to_contain_every_native_interval_endpoint():
    value=calibration()
    condition=value['conditions'][0]
    assert condition['quantile_rank']==19 and condition['expansion']==pytest.approx(.1)
    result=modes.expanded(np.array([[.3, .4]]), batch(1), ['new-test'], value, spec(), identities=IDENTITIES)
    assert result['available'].tolist()==[True]
    assert result['bounds'][0]==pytest.approx([.2, .5])
    assert not result['experimental_coverage_qualified'] and not result['experimental_mode_absence_qualified']
    assert not result['candidate_pruning_authorized']


def test_many_correlated_variants_do_not_manufacture_independent_trajectories():
    value=calibration(n=1, duplicate=100)
    condition=value['conditions'][0]
    assert condition['queries']==100 and len(condition['group_scores'])==1 and condition['expansion'] is None
    result=modes.expanded(np.array([[.3, .4]]), batch(1), ['test'], value, spec(), identities=IDENTITIES)
    assert not result['available'].any() and np.isnan(result['bounds']).all()
    twice=calibration(duplicate=2)
    assert twice['conditions'][0]['group_scores']==calibration()['conditions'][0]['group_scores']


def test_a_rare_condition_cannot_borrow_other_families_calibration_groups():
    groups=[f'g{i:02}' for i in range(19)]
    value=modes.Calibration(spec(), identities=IDENTITIES, groups=groups, used_groups=[])
    value.add(np.tile([.3, .4], (19, 1)), batch(19), groups)
    value.add(np.array([[.3, .4]]), batch(1, c=1, f=1), groups[:1])
    result=value.finish()
    assert result['conditions'][0]['expansion'] is not None
    assert result['conditions'][1]['expansion'] is None


def test_missing_fitting_support_stays_unknown_even_with_enough_calibration_targets():
    value=calibration(c=0, f=1)
    assert len(value['conditions'][0]['group_scores'])==19
    assert value['conditions'][0]['expansion'] is None


def test_all_aliases_and_variants_contribute_the_worst_error_within_a_group():
    groups=['a', 'b', 'c']
    value=modes.Calibration(spec(), identities=IDENTITIES, groups=groups, used_groups=[], error_rate=.5)
    value.add(np.tile([.25, .5], (3, 1)), batch(3), groups)
    value.add(np.array([[.1, .2]]), batch(1, upper=.9), ['a'])
    result=value.finish()['conditions'][0]
    assert result['group_scores'][0][0]==0 and result['group_scores'][0][1]==pytest.approx(.7)
    assert result['group_scores'][1:]==[[1, 0.], [2, 0.]]
    # The finite-sample median rank is two for three independent groups.
    assert result['quantile_rank']==2 and result['expansion']==0.


def test_unknown_targets_are_processed_without_inventing_zero_amplitudes():
    value=modes.Calibration(spec(), identities=IDENTITIES, groups=['a'], used_groups=[])
    empty=batch(0)
    value.add(np.empty((0, 2), np.float64), empty, ['a'])
    assert value.finish()['conditions']==[]
    with pytest.raises(ValueError, match='Every declared'):
        modes.Calibration(spec(), identities=IDENTITIES, groups=['a'], used_groups=[]).finish()


def test_sub_float32_targets_and_ambiguous_native_bounds_are_not_erased():
    groups=[f'g{i:02}' for i in range(19)]
    value=modes.Calibration(spec(), identities=IDENTITIES, groups=groups, used_groups=[])
    value.add(np.zeros((19, 2), np.float32), batch(19, lower=0., upper=1e-250), groups)
    assert 1e-250 <= value.finish()['conditions'][0]['expansion'] <= np.nextafter(1e-250,np.inf)


def test_outside_prepared_support_never_reuses_a_calibrated_bound():
    value=calibration()
    result=modes.expanded(np.array([[.3, .4]]), batch(1), ['test'], value, spec(), identities=IDENTITIES, within_support=False)
    assert not result['available'].any() and np.isnan(result['bounds']).all()


def test_heldout_report_separates_within_group_failures_from_correlated_row_counts():
    value=calibration();evaluation=modes.Evaluation(value,spec(),identities=IDENTITIES)
    groups=['test-a','test-a','test-b']
    evaluation.add(np.tile([.3,.4],(3,1)),batch(3),groups)
    evaluation.add(np.array([[.3,.4]]),batch(1,upper=.9),['test-a'])
    report=evaluation.finish();row=report['conditions'][0]
    assert report['rows']==4 and report['physical_groups']==2
    assert row['interval_available_queries']==4 and row['interval_contained_queries']==3
    assert row['target_interval_containment']==.75
    assert row['whole_group_containment']==.5 and row['wholly_contained_groups']==1
    assert not report['experimental_mode_absence_qualified']


def test_heldout_report_keeps_unknown_conditions_and_refuses_calibration_leakage():
    value=calibration();evaluation=modes.Evaluation(value,spec(),identities=IDENTITIES)
    evaluation.add(np.array([[.3,.4]]),batch(1,c=1,f=1),['new-test'])
    report=evaluation.finish();row=report['conditions'][0]
    assert report['known_queries']==1 and report['interval_available_queries']==0
    assert row['target_interval_containment'] is None and row['whole_group_containment'] is None
    with pytest.raises(ValueError,match='own calibration'):
        evaluation.add(np.array([[.3,.4]]),batch(1),value['groups'][:1])


def test_calibration_owns_its_frozen_readout_identity():
    readout=spec();original=copy.deepcopy(readout)
    value=modes.Calibration(readout,identities=IDENTITIES,groups=['cal'],used_groups=[])
    readout['scales'][0]*=2
    value.add(np.array([[.3,.4]]),batch(1),['cal'])
    assert value.finish()['mode_readout_identity']==digest(original)


def test_streamed_evaluation_audits_and_owns_calibration_once(tmp_path,monkeypatch):
    value=calibration();calls=[];original=modes.validate
    def counted(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(modes,'validate',counted)
    evaluation=modes.Evaluation(value,spec(),identities=IDENTITIES)
    value['conditions'][0]['expansion']=100.
    for group in ['test-a','test-b']:
        evaluation.add(np.array([[.3,.4]]),batch(1),[group])
    assert calls==[1]
    assert evaluation.finish()['conditions'][0]['interval_contained_queries']==2


@pytest.mark.parametrize('case', ['valid', 'condition', 'queries', 'available', 'group_count',
                                 'coverage', 'calibration_identity', 'authority'])
def test_test_receipt_is_bound_to_every_native_mode_query_and_independent_group(case):
    value=calibration();evaluation=modes.Evaluation(value,spec(),identities=IDENTITIES)
    evaluation.add(np.tile([.3,.4],(2,1)),batch(2),['test-a','test-b'])
    report=evaluation.finish()
    native=dict(schema=joint.METRICS,classes=spec()['classes'],families=[f['id'] for f in spec()['families']],
        rows=2,known_rows=2,known_queries=2,conditions=[dict(space_group_index=0,family_index=0,
            queries=2,pinball_loss=.1,target_interval_containment=.5)],available=True,loss=.1,
        experimental_mode_absence_qualified=False,candidate_pruning_authorized=False)
    if case=='condition':report['conditions']=[]
    elif case=='queries':report['conditions'][0]['queries']-=1
    elif case=='available':report['conditions'][0]['interval_available_queries']=0
    elif case=='group_count':report['physical_groups']=1
    elif case=='coverage':report['conditions'][0]['whole_group_containment']=.5
    elif case=='calibration_identity':report['calibration_sha256']='0'*64
    elif case=='authority':report['experimental_mode_absence_qualified']=True
    kwargs=dict(identities=IDENTITIES,native_metrics=native,rows=2,groups=2)
    if case=='valid':assert modes.validate_evaluation(report,value,spec(),**kwargs)==report
    else:
        with pytest.raises(ValueError):modes.validate_evaluation(report,value,spec(),**kwargs)


@pytest.mark.parametrize('case', ['weights', 'parent', 'instrument', 'dataset', 'mode_scales', 'authority',
                                 'fake_group', 'narrowed_radius', 'duplicate_group', 'missing_condition'])
def test_imported_calibration_cannot_change_scope_or_claim_experimental_authority(case):
    value=calibration();expected=copy.deepcopy(IDENTITIES);readout=spec()
    if case in ['weights', 'parent', 'instrument', 'dataset']:
        key={'weights':'weights_sha256','parent':'parent_sha256','instrument':'instrument_sha256','dataset':'dataset_sha256'}[case]
        expected[key]='0'*64
    elif case=='mode_scales':readout['scales'][0]*=2
    elif case=='authority':value['experimental_mode_absence_qualified']=True
    elif case=='fake_group':value['conditions'][0]['group_scores'].append([19, 0.])
    elif case=='narrowed_radius':value['conditions'][0]['expansion']=0.
    elif case=='duplicate_group':value['conditions'][0]['group_scores'].append([0, .1])
    else:value['conditions'].append(copy.deepcopy(value['conditions'][0]))
    with pytest.raises(ValueError):modes.validate(value,readout,identities=expected)


@pytest.mark.parametrize('case', ['fitting_overlap', 'undeclared_group', 'float32_truth', 'invalid_bounds',
                                 'missing_endpoint', 'duplicate_query', 'integer_prediction', 'nonfinite_prediction'])
def test_incomplete_or_leaking_native_inputs_are_rejected(case):
    groups=['a'];used=[];target=batch(1);p=np.array([[.3, .4]])
    if case=='fitting_overlap':used=['a']
    value=None
    if case!='fitting_overlap':value=modes.Calibration(spec(),identities=IDENTITIES,groups=groups,used_groups=used)
    if case=='undeclared_group':groups=['b']
    elif case=='float32_truth':target['lower']=target['lower'].astype(np.float32)
    elif case=='invalid_bounds':target['upper'][:]=0.
    elif case=='missing_endpoint':target.pop('upper')
    elif case=='duplicate_query':
        target={k:np.repeat(v,2) for k,v in target.items()};p=np.tile(p,(2,1))
    elif case=='integer_prediction':p=p.astype(int)
    elif case=='nonfinite_prediction':p[0,0]=np.nan
    with pytest.raises(ValueError):
        if case=='fitting_overlap':modes.Calibration(spec(),identities=IDENTITIES,groups=groups,used_groups=used)
        else:value.add(p,target,groups)
