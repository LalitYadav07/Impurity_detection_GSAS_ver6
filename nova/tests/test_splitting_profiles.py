"""Controlled component evolution, without phase labels or hkl knowledge."""
from dataclasses import replace

import numpy as np
import pytest

from radar_pd_nova.pattern_transition import prepare_pattern, compare_observed_patterns
from radar_pd_nova.splitting_profiles import component_profile
from radar_pd_nova.transition_signals import BoundaryPolicy, apply_policy, measure_signals
from radar_pd_nova.transition_regions import localized_regions
from test_pattern_transition import observed


def doublet(gap=24, *, amplitudes=(60,50), shift=0, width=.0015, gain=1, seed=2):
    return observed(factors=[1,1,1,1,0,1,1,1],
        new=[(4000+shift-gap/2,amplitudes[0],width),
             (4000+shift+gap/2,amplitudes[1],width)],
        noise=.1, gain=gain, seed=seed)


def measurement(a,b):
    before,after=map(prepare_pattern,(a,b))
    measured=measure_signals(before,after,compare_observed_patterns(before,after))
    return measured


def test_existing_doublet_separation_is_measured_even_with_the_same_maximum_count():
    measured=measurement(doublet(24),doublet(32,seed=7))
    persistent=[v for v in measured["splitting"] if v.get("kind")=="local_component_separation_change"]
    assert persistent
    event=next(v for v in persistent if v["interval"][0]<4000<v["interval"][1])
    assert event["before_profile"]["separation"]==pytest.approx(24,abs=.15)
    assert event["after_profile"]["separation"]==pytest.approx(32,abs=.15)
    assert event["separation_change_fwhm"]>.5
    result=apply_policy(measured,BoundaryPolicy(selected=("splitting",)))
    assert result["indicators"]["splitting"]["detected"]
    assert "undetermined" in event["physical_cause"]


@pytest.mark.parametrize("options",[
    {"seed":7}, {"amplitudes":(90,20),"seed":7},
    {"shift":4,"gain":3,"seed":7}, {"width":.0018,"seed":7},
])
def test_constant_separation_does_not_vote_for_splitting_from_noise_gain_motion_or_width(options):
    result=apply_policy(measurement(doublet(),doublet(**options)),BoundaryPolicy(selected=("splitting",)))
    assert not result["indicators"]["splitting"]["detected"],result


def test_recognized_doublet_below_threshold_is_visible_without_a_boundary_vote():
    b=doublet(24.4,seed=7)
    result=apply_policy(measurement(doublet(),b),BoundaryPolicy(selected=("splitting",)))
    assert not result["indicators"]["splitting"]["detected"]
    regions,_=localized_regions(result,["splitting"],b,"tof_us")
    assert any(r["interval"][0]<4000<r["interval"][1] and r["status"]=="recognized_split" for r in regions)
    assert not localized_regions(result,["splitting"],b,"tof_us",include_split_features=False)[0]


@pytest.mark.parametrize("reverse",[False,True])
def test_resolved_split_and_merge_are_symmetric_changes(reverse):
    a,b=doublet(0),doublet(24,seed=7)
    if reverse:
        a,b=b,a
    events=measurement(a,b)["splitting"]
    component=[v for v in events if v.get("kind")=="local_resolved_component_count_change"]
    assert component
    assert component[0]["before_components"]==(2 if reverse else 1)
    assert component[0]["after_components"]==(1 if reverse else 2)


def test_asymmetric_single_peak_competes_against_the_two_component_model():
    p=doublet(0)
    x=p.x
    background=15+.0005*x
    asym=100*np.exp(-.5*((x-4000)/np.where(x<4000,5,9))**2)
    # Fit only the known local asymmetric feature, retaining native errors.
    p=replace(p,intensity=background+asym+np.random.default_rng(8).normal(0,.1,len(x)))
    fit=component_profile(prepare_pattern(p),3940,4060)
    assert fit and not fit["two_component_qualified"]


def test_masked_component_window_is_withheld():
    p=doublet();valid=p.valid.copy();valid[(p.x>3990)&(p.x<3995)]=False
    assert component_profile(prepare_pattern(replace(p,valid=valid)),3940,4060) is None


def test_split_rule_rejects_invalid_settings():
    for value in [True,".1",float("nan"),.001,3]:
        with pytest.raises(ValueError):
            BoundaryPolicy(split_separation_fwhm=value)
