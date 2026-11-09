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
    # Optional advice cannot break recovery of older/unlinked display rows.
    mixed=[dict(source='legacy'),dict(protocol=[],source='unlinked'),*rows]
    assert checked_receipts(campaign,mixed)==checked
    assert visible_rows(checked,mixed)==shown
    damaged=copy.deepcopy(evidence);damaged['adaptive_work_set']['candidate_pruning_authorized']=True
    campaign.set_setting('structural_rankings',{sha:damaged})
    assert checked_receipts(campaign,rows)=={}
    campaign.set_setting('structural_rankings',{sha:[]})
    assert checked_receipts(campaign,rows)=={}
    # Nothing in this read-only rendering changes the finite scientific work.
    assert len(campaign.trials())==4


def test_visible_scan_priorities_keep_independent_protocols_and_do_not_reload_the_ledger():
    first=dict(ranking_available=False,reason='First scan outside prepared support')
    second=dict(ranking_available=False,reason='Second scan has no matching profile')
    saved={'a':first,'b':second}
    assert visible_rows(saved,[dict(protocol='b',source='scan-b')])==[
        dict(id='b',scan='scan-b',summary=second['reason'])]


def test_conditional_modes_remain_unresolved_and_follow_the_selected_scan_only():
    from radar_pd_nova.structural_ranking_labels import mode_rows
    from test_structural_ranking_result import mode_estimates
    estimates=mode_estimates([12]);evidence=dict(ranking_available=True,families=[dict(space_group=12)],mode_estimates=estimates)
    result=mode_rows(evidence)
    assert result[0]['family']=='GM1+' and result[0]['k']=='0, 0, 0'
    assert result[0]['status']=='Unresolved · uncalibrated'
    estimates['rows'][0].update(lower_estimate=None,upper_estimate=None,reason='No fitting support')
    assert mode_rows(evidence)[0]['amplitude']=='Not learned'
    shown=visible_rows({'one':evidence},[dict(protocol='one',source='Run 1')])
    assert shown[0]['mode_rows']==mode_rows(evidence)
    assert visible_rows({'one':evidence},[dict(protocol='other',source='Run 2')])==[]
    estimates['rows'][0]['status']='absent'
    assert mode_rows(evidence)==[]


def test_mode_view_defaults_to_ranked_symmetry_and_preserves_unknown_conditions():
    from radar_pd_nova.structural_ranking_labels import mode_rows,mode_view
    from test_structural_ranking_result import mode_estimates
    value=mode_estimates([12,166]);value['rows'][1].update(lower_estimate=None,upper_estimate=None,reason='No fitting support')
    evidence=dict(families=[dict(space_group=166),dict(space_group=12)],mode_estimates=value)
    rows=mode_rows(evidence);view=mode_view(rows)
    assert view['mode_groups']==['SG 166','SG 12'] and view['mode_group']=='SG 166'
    assert len(view['mode_rows'])==2 and len(view['mode_visible_rows'])==1
    assert view['mode_visible_rows'][0]['amplitude']=='Not learned'
    assert view['mode_notes']==['No fitting support'] and '1 without fitting support' in view['mode_summary']
    chosen=mode_view(rows,'SG 12');assert chosen['mode_group']=='SG 12'
    assert chosen['mode_visible_rows'][0]['status']=='Unresolved · uncalibrated'
    # A selection is presentation state; unsupported/missing groups cannot add rows.
    assert mode_view(rows,'SG 230')['mode_group']=='SG 166'
    assert mode_view([],'SG 12')['mode_visible_rows']==[]


def test_refresh_retains_mode_selection_only_for_the_same_visible_protocol():
    from radar_pd_nova.structural_ranking_labels import mode_view,mode_rows
    from test_structural_ranking_result import mode_estimates
    evidence=dict(ranking_available=True,families=[dict(space_group=166),dict(space_group=12)],
                  mode_estimates=mode_estimates([12,166]))
    saved={'one':evidence,'two':evidence};rows=[dict(protocol='one',source='Run 1')]
    old=visible_rows(saved,rows);old[0].update(mode_view(mode_rows(evidence),'SG 12'))
    assert visible_rows(saved,rows,previous=old)[0]['mode_group']=='SG 12'
    assert visible_rows(saved,[dict(protocol='two',source='Run 2')],previous=old)[0]['mode_group']=='SG 166'


def test_campaign_and_live_mode_controls_only_change_the_display():
    from types import SimpleNamespace
    from radar_pd_nova.structural_ranking_labels import mode_view,mode_rows
    from radar_pd_nova.symmetry_controller import SymmetryMixin
    from radar_pd_nova.webmon.structural_ui import StructuralLiveMixin
    from test_structural_ranking_result import mode_estimates
    evidence=dict(families=[dict(space_group=166),dict(space_group=12)],mode_estimates=mode_estimates([12,166]))
    table=mode_view(mode_rows(evidence));other=dict(id='other',**copy.deepcopy(table))
    state=SimpleNamespace(sx_structural_priority_rows=[dict(id='one',**table),other],
                          webmon_structural_modes=copy.deepcopy(table))
    app=SimpleNamespace(server=SimpleNamespace(state=state))
    SymmetryMixin.select_structural_mode_group(app,'one','SG 12')
    assert state.sx_structural_priority_rows[0]['mode_group']=='SG 12'
    assert state.sx_structural_priority_rows[1]==other
    StructuralLiveMixin.webmon_select_structural_mode_group(app,'SG 12')
    assert state.webmon_structural_modes['mode_group']=='SG 12'
    assert state.webmon_structural_modes['mode_rows']==table['mode_rows']
