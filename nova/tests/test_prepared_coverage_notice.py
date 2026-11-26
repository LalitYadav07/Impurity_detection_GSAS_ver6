"""Explain bounded native coverage receipts without altering their meaning."""
import copy
import pytest
from radar_pd_nova.prepared_structural_discovery import ranking_notice


def receipt():
    def row(role):
        return dict(source_role=role,incomplete_bins=1,total_bins=4096,
            valid_q_min=.5920735589810109,valid_q_max=12.557776575155614,
            trained_q_min=.5927975960140495,trained_q_max=12.559903694732162)
    return dict(ranking_applicability=dict(status='unavailable',sources=[row('current'),row('reference')],
        reason='ML ranking is unavailable for this window. The complete native candidate set is retained.'))


def test_one_edge_bin_reports_counts_and_distinct_measured_and_trained_ranges():
    value=receipt();before=copy.deepcopy(value);text=ranking_notice(value)
    assert 'Trained Q 0.592798–12.5599 Å⁻¹' in text
    assert 'target: 1 / 4,096 bins incomplete; valid Q 0.592074–12.5578 Å⁻¹' in text
    assert 'reference: 1 / 4,096 bins incomplete' in text
    assert 'common valid Q coverage' in text
    assert text.endswith('The complete native candidate set is retained.')
    assert value==before


@pytest.mark.parametrize('change',['absent','nonfinite','boolean','negative','duplicate_role','changed_grid','no_target','no_incomplete_bins'])
def test_unusable_optional_details_keep_the_native_fallback_message(change):
    value=receipt();data=value['ranking_applicability'];row=data['sources'][0]
    if change=='absent':data.pop('sources')
    elif change=='nonfinite':row['valid_q_max']=float('nan')
    elif change=='boolean':row['incomplete_bins']=True
    elif change=='negative':row['incomplete_bins']=-1
    elif change=='duplicate_role':data['sources'][1]['source_role']='current'
    elif change=='changed_grid':data['sources'][1]['trained_q_max']=11.
    elif change=='no_target':data['sources']=data['sources'][1:]
    else:
        for source in data['sources']:source['incomplete_bins']=0
    assert ranking_notice(value)=='Native candidates reused; '+data['reason']


def test_completed_coverage_does_not_claim_inference_has_run():
    value=receipt();value['ranking_applicability']['status']='coverage_passed'
    assert ranking_notice(value)==''
    assert ranking_notice({})==''
