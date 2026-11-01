"""Controlled local-signal evidence without phase labels or facility arrays."""
from dataclasses import replace

import numpy as np
import pytest

from radar_pd_nova.pattern_transition import prepare_pattern, compare_observed_patterns
from radar_pd_nova.transition_signals import BoundaryPolicy, apply_policy, measure_signals
from test_pattern_transition import observed


def compare(a,b):
    a,b=prepare_pattern(a),prepare_pattern(b)
    report=compare_observed_patterns(a,b)
    result=apply_policy(measure_signals(a,b,report),BoundaryPolicy(selected=('appearance',)))
    return report,result


def support_at(report,position):
    return next(row['source_support'] for row in report['unmatched']
                if row['side']=='after' and abs(row['position']/position-1)<.003)


@pytest.mark.parametrize('seed',range(8))
def test_preexisting_weak_motion_and_better_counts_are_not_new_signal(seed):
    a=observed(new=[(2700,1.2,.0025)],noise=.35,seed=seed)
    b=observed(new=[(2740.5,1.2,.0025)],noise=.1,seed=seed+100)
    report,result=compare(a,b)
    assert not result['indicators']['appearance']['detected'],result
    assert support_at(report,2740.5)['status'] in {'preexisting','unresolved'}


@pytest.mark.parametrize('seed',range(4))
def test_new_weak_peak_can_be_detected_next_to_an_already_matched_strong_peak(seed):
    a=observed(noise=.15,seed=seed)
    b=observed(new=[(2337,3.,.0012)],noise=.15,seed=seed+100)
    report,result=compare(a,b)
    support=support_at(report,2337)
    assert support['matched_source_peaks']>=1
    assert support['status']=='new',support
    assert result['indicators']['appearance']['detected']
    assert support['prior_overlap_possible'] is True


@pytest.mark.parametrize('seed',range(4))
def test_true_new_isolated_weak_peak_retains_appearance_evidence(seed):
    report,result=compare(observed(noise=.15,seed=seed),
                          observed(new=[(2700,2.,.0025)],noise=.15,seed=seed+100))
    assert support_at(report,2700)['status']=='new'
    assert support_at(report,2700)['appearance_available'] is True
    assert result['indicators']['appearance']['detected']


@pytest.mark.parametrize('seed',range(4))
def test_growth_of_an_old_subthreshold_profile_is_not_phase_appearance(seed):
    report,result=compare(observed(new=[(2700,.5,.0025)],noise=.15,seed=seed),
                          observed(new=[(2700,1.5,.0025)],noise=.15,seed=seed+100))
    assert support_at(report,2700)['status'] in {'preexisting','unresolved'}
    assert not result['indicators']['appearance']['detected']


@pytest.mark.parametrize('seed',range(3))
def test_a_masked_possible_predecessor_is_unresolved_not_absent(seed):
    a=observed(new=[(2700,1.2,.0025)],noise=.35,seed=seed)
    mask=a.valid.copy();mask[(a.x>2688)&(a.x<2712)]=False
    a=replace(a,valid=mask)
    b=observed(new=[(2740.5,1.2,.0025)],noise=.1,seed=seed+100)
    report,result=compare(a,b)
    support=support_at(report,2740.5)
    assert support['status']=='unresolved'
    assert support['missing_source_support'] is True
    assert not result['indicators']['appearance']['detected']
    assert result['status']=='uncertain'
    assert not result['indicators']['appearance']['available']


@pytest.mark.parametrize('seed',range(3))
def test_unresolved_prior_blend_is_not_forced_to_new_or_preexisting(seed):
    report,result=compare(observed(new=[(2305,10,.001)],noise=.1,seed=seed),
                          observed(new=[(2320,10,.001)],noise=.1,seed=seed+100))
    support=support_at(report,2320)
    assert support['status']=='unresolved'
    assert support['prior_overlap_possible'] is True
    assert not result['indicators']['appearance']['detected']
    assert result['status']=='uncertain'


@pytest.mark.parametrize('seed',range(3))
def test_independent_anisotropic_motion_does_not_require_a_global_deformation(seed):
    a=observed(new=[(2700,1.2,.0025)],noise=.35,seed=seed)
    b=observed(new=[(2740.5,1.2,.0025)],noise=.1,seed=seed+100,gain=2.,
               shifts=[-.008,.007,-.006,.004,-.005,.009,.002,-.003])
    report,result=compare(a,b)
    assert support_at(report,2740.5)['status'] in {'preexisting','unresolved'}
    assert not result['indicators']['appearance']['detected']


def test_existing_resolved_peak_intensity_change_is_separate_from_appearance():
    report,result=compare(observed(new=[(2700,2.,.0025)],noise=.15,seed=1),
                          observed(new=[(2740.5,6.,.0025)],noise=.15,seed=101))
    assert not result['indicators']['appearance']['detected']
    assert any('relative_intensity_redistribution' in r['kinds'] for r in report['regions'])


def test_known_new_signal_survives_an_independent_unresolved_masked_region():
    a=observed(new=[(2700,1.2,.0025)],noise=.35,seed=1)
    mask=a.valid.copy();mask[(a.x>2688)&(a.x<2712)]=False
    a=replace(a,valid=mask)
    b=observed(new=[(2740.5,1.2,.0025),(6000,5.,.0025)],noise=.1,seed=101)
    report,result=compare(a,b)
    assert support_at(report,2740.5)['status']=='unresolved'
    assert support_at(report,6000)['status']=='new'
    assert result['status']=='transition_evidence'
    assert result['indicators']['appearance']['available']
    assert result['indicators']['appearance']['unresolved_regions']


def test_compact_policy_report_retains_three_way_source_evidence():
    from radar_pd_nova.transition_tracking import TransitionTrackingEngine
    a=observed(new=[(2700,1.2,.0025)],noise=.35,seed=1)
    b=observed(new=[(2740.5,1.2,.0025)],noise=.1,seed=101)
    engine=TransitionTrackingEngine(policy=BoundaryPolicy(selected=('appearance',)))
    report=engine.compare_evidence(prepare_pattern(a),prepare_pattern(b))
    assert report['source_support']
    assert all(row['status'] in {'new','preexisting','unresolved'} for row in report['source_support'])


def test_competing_dense_assignment_cannot_manufacture_new_signal_after_rigid_motion():
    rng=np.random.default_rng(1)
    centers=2000+np.cumsum(rng.uniform(20,75,10))
    peaks=[(p,float(height),float(width)/p) for p,height,width in
           zip(centers,rng.uniform(8,90,10),rng.uniform(2,6,10))]
    a=observed(new=peaks,noise=.1,seed=1)
    b=replace(a,x=a.x*1.015,intensity=a.intensity/1.015,sigma=a.sigma/1.015)
    report,result=compare(a,b)
    assert not result['indicators']['appearance']['detected']
    assert report['metrics']['matched_features']==report['metrics']['before_features']==report['metrics']['after_features']
    assert result['status']=='no_resolved_change'


def test_an_extra_indistinguishable_branch_does_not_reuse_an_assigned_predecessor():
    report,result=compare(observed(noise=.1,seed=1),
                          observed(new=[(2337,100.,.0025)],noise=.1,seed=101))
    support=support_at(report,2337)
    assert support['status']=='unresolved'
    assert support['competing_assigned_positions']
    assert not result['indicators']['appearance']['detected']


def test_local_completion_never_uses_a_forbidden_correspondence():
    from radar_pd_nova.pattern_transition import _complete_feature_assignment,partial_assignment,TransitionSettings
    settings=TransitionSettings()
    costs=np.array([[.7,.01,np.inf],[np.inf,.7,.01],[np.inf,np.inf,.7]])
    original=partial_assignment(costs,settings.unmatched_cost)
    assert len(original[0])==2
    completed=_complete_feature_assignment(original,costs,settings)
    assert completed==([(0,0),(1,1),(2,2)],[],[])
    costs[2,2]=2*settings.unmatched_cost
    original=partial_assignment(costs,settings.unmatched_cost)
    completed=_complete_feature_assignment(original,costs,settings)
    assert len(completed[0])==2 and (2,2) not in completed[0]


def test_complete_noise_limited_support_can_be_negative_without_claiming_origin():
    report,result=compare(observed(new=[(2700,1.2,.0025)],noise=.8,seed=1),
                          observed(new=[(2740.5,1.2,.0025)],noise=.1,seed=101))
    support=support_at(report,2740.5)
    assert support['status']=='unresolved'
    assert support['appearance_available'] is True
    assert support['appearance_status']=='no_resolved_change'
    assert support['reason']=='consistent_with_source_uncertainty'
    assert result['status']=='no_resolved_change'
    assert result['indicators']['appearance']['consistent_regions']


def test_exported_tracking_reports_identify_the_comparison_and_signal_methods():
    from radar_pd_nova.pattern_transition import ENGINE_VERSION
    from radar_pd_nova.transition_signals import SIGNAL_VERSION
    from radar_pd_nova.transition_input import observation
    from radar_pd_nova.transition_tracking import TransitionTrackingEngine
    pattern=observed(ident='one')
    rows=[{'run_id':'one','run_number':1,'measured':pattern}]
    loader=lambda row:observation(row['measured'])
    legacy=TransitionTrackingEngine(loader=loader).evaluate(rows)
    online=TransitionTrackingEngine(loader=loader,policy=BoundaryPolicy()).evaluate(rows)
    assert legacy['engine']==online['engine']==ENGINE_VERSION
    assert online['signal_engine']==SIGNAL_VERSION
