"""Visible adaptive counts follow exact selected protocols, not a top-five cap."""
import copy

from gpx_recipe.preparation.structural_ranking_result import attach_adaptive
from gpx_recipe.symmetry.campaign import Campaign
from gpx_recipe.symmetry.contracts import digest
from radar_pd_nova.structural_ranking_labels import describe,checked_receipts,visible_rows
from test_structural_ranking_dispatch import setup_inputs,receipt
from test_structural_ranking_result import work_set


def test_descriptions_show_the_actual_adaptive_size_and_do_not_call_scores_probabilities():
    classes=list(range(1,13))
    adaptive=work_set(classes);adaptive['retained_space_groups']=classes[:8];adaptive['deferred_space_groups']=classes[8:]
    evidence=dict(ranking_available=True,families=[dict(space_group=c) for c in classes],adaptive_work_set=adaptive)
    text=describe(evidence)
    assert '8 / 12' in text and 'SG 8' in text and '4 classes deferred' in text
    assert 'every declared native model remains scheduled' in text
    legacy=describe(dict(ranking_available=True,families=evidence['families']))
    assert '12 symmetry classes' in legacy and 'No calibrated first work set' in legacy


def test_campaign_restoration_rejects_changed_scope_and_selected_scan_cannot_reuse_other_priority(tmp_path):
    inputs,p,document=setup_inputs(tmp_path)
    campaign=Campaign(tmp_path/'campaign',shared_inputs=True)
    campaign.enroll(p,inputs,approved_hash=digest(p))
    evidence=attach_adaptive(receipt(p,document),work_set([65,63]));sha=digest(p)
    rows=[dict(protocol=sha,source='scan-a',metadata=dict(scan='Run 1'))]*100
    campaign.set_setting('structural_rankings',{sha:evidence})
    checked=checked_receipts(campaign,rows)
    shown=visible_rows(checked,rows)
    assert len(shown)==1 and shown[0]['scan']=='Run 1' and '1 / 2' in shown[0]['summary']
    assert visible_rows(checked,[dict(protocol='f'*64,source='scan-b')])==[]
    damaged=copy.deepcopy(evidence);damaged['adaptive_work_set']['candidate_pruning_authorized']=True
    campaign.set_setting('structural_rankings',{sha:damaged})
    assert checked_receipts(campaign,rows)=={}
    # Nothing in this read-only rendering changes the finite scientific work.
    assert len(campaign.trials())==4


def test_visible_scan_priorities_keep_independent_protocols_and_do_not_reload_the_ledger():
    first=dict(ranking_available=False,reason='First scan outside prepared support')
    second=dict(ranking_available=False,reason='Second scan has no matching profile')
    saved={'a':first,'b':second}
    assert visible_rows(saved,[dict(protocol='b',source='scan-b')])==[
        dict(id='b',scan='scan-b',summary=second['reason'])]
