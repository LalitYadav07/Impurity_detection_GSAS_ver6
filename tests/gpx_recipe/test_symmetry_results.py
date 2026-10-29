import pytest
from gpx_recipe.symmetry_results import summarize_portfolio


def valid(uid,rwp):
    return dict(id=uid,status='valid',rwp=rwp,effects_verified=True,inputs_verified=True)


def test_a_better_aborted_solution_prevents_promoting_a_worse_local_minimum():
    result=summarize_portfolio(['good-local','aborted-global'],[
        valid('good-local',8.19),dict(id='aborted-global',status='optimization_unresolved',rwp=.00073)],fit_band_percent=.25)
    assert result['status']=='optimization_unresolved'
    assert result['unresolved_competitors']==['aborted-global']
    assert result['scientific_assignment']=='not_established'


def test_hard_unphysical_solution_cannot_displace_a_qualified_fit():
    result=summarize_portfolio(['physical','unphysical'],[valid('physical',6.3),
        dict(id='unphysical',status='physical_rejection',rwp=1.2)],fit_band_percent=.25)
    assert result['status']=='reviewable_portfolio'
    assert result['competitive_trial_ids']==['physical']


@pytest.mark.parametrize('state',['technical_failure','deferred'])
def test_failed_or_unexplored_branches_are_not_counted_as_physical_rejections(state):
    result=summarize_portfolio(['a','b'],[valid('a',6.3),dict(id='b',status=state)],fit_band_percent=.25)
    assert result['status']=='incomplete'


def test_many_equivalent_fits_are_kept_without_choosing_a_structure():
    result=summarize_portfolio(['a','b'],[valid('a',6.2985),valid('b',6.2985)],fit_band_percent=.25)
    assert result['competitive_trial_ids']==['a','b']
    assert result['scientific_assignment']=='not_established'


def test_a_missing_job_remains_visible_after_other_jobs_finish():
    assert summarize_portfolio(['a','b'],[valid('a',6.3)],fit_band_percent=.25)['missing']==['b']


def test_a_claimed_valid_fit_needs_the_actual_parameter_effect_audit():
    with pytest.raises(ValueError,match='unverified'):
        summarize_portfolio(['a'],[dict(id='a',status='valid',rwp=1.2)],fit_band_percent=.25)
